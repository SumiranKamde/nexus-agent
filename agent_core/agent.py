import json
import os
import shutil
import sys
from dataclasses import dataclass
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq, BadRequestError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_PATH = os.path.join(PROJECT_ROOT, "sandbox_files")
TODO_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_servers", "todo_server.py")
STATE_CHANGING_TOOLS = {
    "write_file", "edit_file", "create_directory", "move_file",
    "delete_file", "delete_directory",
    "add_task", "complete_task", "undo_last_action", "remember", "forget",
    "create_notification", "mark_notifications_seen", "send_whatsapp",
    "send_file_via_whatsapp",
}
# Tools the proactive check is allowed to call directly (notifications + reads)
PROACTIVE_ALLOWED_STATE_TOOLS = {
    "create_notification", "send_whatsapp", "send_file_via_whatsapp",
}
SYSTEM_PROMPT = (
    "You are Nexus, a friendly personal productivity assistant. You have access to a set of "
    "tools for managing files, a to-do list (including an audit log and undo), searching the "
    "live web, getting the real current date/time, remembering facts about the user across "
    "conversations (remember, list_memories, forget), and sending WhatsApp messages to the user's "
    "phone. Always check the full list of tools provided to you for this request before deciding "
    "whether one applies — do not assume a request has no matching tool just because it isn't "
    "explicitly named here.\n\n"
    "If the user shares a lasting preference, habit, or personal detail (e.g. 'I prefer evening "
    "reminders', 'my name is Sumiran', 'I usually add groceries on Fridays'), use the remember "
    "tool to save it — but don't save trivial one-off details.\n\n"
    "IMPORTANT: Your own knowledge has a training cutoff and is NOT reliable for anything "
    "time-sensitive. For the current date or time, you MUST use the get_current_time tool. For "
    "prices, exchange rates, news, or current events, you MUST use the search tool. For questions "
    "about past actions Nexus has taken, you MUST use the undo_last_action or get_audit_log tools "
    "rather than guessing. Never answer any of these from memory.\n\n"
    "If a request has no matching tool, say so plainly and briefly — do not invent a result.\n\n"
    "For everything else — general knowledge, casual conversation, quick math, jokes — just "
    "answer directly in plain text without using a tool."
)

PROACTIVE_PROMPT = (
    "This is an automatic background check, not a live conversation with the user. Review "
    "the user's current to-do list and anything you remember about them. Decide whether "
    "there is something worth proactively reminding them about right now (e.g. tasks piling "
    "up, something time-sensitive). If so, call create_notification with a short, friendly "
    "message so it shows in the app, AND call send_whatsapp with the same message so it reaches "
    "their phone even if the app isn't open. If there is genuinely nothing worth surfacing right "
    "now, do not call any tool."
)

@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = None

@dataclass
class PlanResult:
    text: str
    function_calls: list
    provider: str
    state: dict
    trail: list = None

def _console_script(name: str) -> str:
    """Locate a console script installed alongside the running interpreter.

    Looking next to sys.executable first means the venv's copy wins even when
    the venv isn't activated; PATH and then the bare name are fallbacks.
    """
    scripts_dir = os.path.dirname(sys.executable)
    return shutil.which(name, path=scripts_dir) or shutil.which(name) or name


def _safe_text(response, fallback="I'm not able to do that with my current tools — could you rephrase, or ask for something else?"):
    try:
        return response.text or fallback
    except Exception:
        return fallback

# How many earlier chat turns to replay into the prompt. Keeps follow-ups like
# "add the second one to my list" working without unbounded token growth.
MAX_HISTORY_MESSAGES = 10

def _history_pairs(history, limit=MAX_HISTORY_MESSAGES):
    """Normalize UI chat history into [(role, text)], oldest first.

    Skips blank/None content (a plan can return text=None) and drops any
    leading assistant turns, since both providers expect the replayed
    conversation to open with a user message.
    """
    pairs = [
        (msg["role"], msg["content"])
        for msg in history or []
        if msg.get("role") in ("user", "assistant")
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    ]
    pairs = pairs[-limit:]
    while pairs and pairs[0][0] == "assistant":
        pairs.pop(0)
    return pairs

class NexusAgent:
    def __init__(self):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        self.gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.sessions = {}
        self._stack = AsyncExitStack()

    async def plan(self, user_message: str, history=None) -> PlanResult:
        """Plan a response to user_message.

        history is the prior chat as [{"role": "user"|"assistant", "content": str}],
        oldest first, excluding user_message itself.
        """
        if self.gemini_client:
            try:
                return await self._plan_gemini(user_message, history)
            except Exception as e:
                print(f"[Nexus] Gemini unavailable ({e}); falling back to Groq Llama...")
        if self.groq_client:
            return await self._plan_groq(user_message, history)
        return PlanResult(
            text="No AI provider is configured. Add GEMINI_API_KEY or GROQ_API_KEY to .env.",
            function_calls=[], provider="none", state={}, trail=[],
        )

    async def _plan_gemini(self, user_message: str, history=None) -> PlanResult:
        all_tools, tool_owner = await self._merged_tools()
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [
            types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)])
            for role, text in _history_pairs(history)
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        system_prompt = await self._get_system_prompt()

        plan_trail = []
        for turn in range(5):
            response = await self.gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=contents,
                config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools], system_instruction=system_prompt),
            )
            contents.append(response.candidates[0].content)
            calls = list(response.function_calls or [])

            if not calls:
                return PlanResult(text=_safe_text(response), function_calls=[], provider="gemini", state={}, trail=plan_trail)

            risky = [fc for fc in calls if fc.name in STATE_CHANGING_TOOLS]
            if risky:
                tool_calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in calls]
                return PlanResult(text=None, function_calls=tool_calls, provider="gemini", state={"contents": contents, "tool_owner": tool_owner}, trail=plan_trail)

            # All read-only this round — execute automatically and let it keep reasoning
            tool_response_parts = []
            for fc in calls:
                args = fc.args or {}
                print(f"[Nexus/Gemini] Auto-executing read-only {fc.name}({args})")
                result = await tool_owner[fc.name].call_tool(fc.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"[Nexus/Gemini] -> {result_text}")
                plan_trail.append((fc.name, args, result_text))
                tool_response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result_text}))
            contents.append(types.Content(role="user", parts=tool_response_parts))

        return PlanResult(text="That request needed too many steps — could you break it into smaller parts?", function_calls=[], provider="gemini", state={}, trail=plan_trail)

    async def _plan_groq(self, user_message: str, history=None) -> PlanResult:
        all_tools, tool_owner = await self._merged_tools()
        tools_schema = [{"type": "function", "function": t} for t in all_tools]
        system_prompt = await self._get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": role, "content": text} for role, text in _history_pairs(history)]
        messages.append({"role": "user", "content": user_message})

        plan_trail = []
        for _ in range(5):
            response = None
            for attempt in range(2):
                try:
                    response = self.groq_client.chat.completions.create(
                        model=GROQ_FALLBACK_MODEL, messages=messages, tools=tools_schema, tool_choice="auto", temperature=0,
                    )
                    break
                except BadRequestError as e:
                    print(f"[Nexus/Groq] Malformed tool call (attempt {attempt + 1}/2): {e}")
            if response is None:
                return PlanResult(text="I'm having trouble planning that request — could you try rephrasing?", function_calls=[], provider="groq", state={}, trail=[])
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                return PlanResult(text=msg.content or "I'm not sure how to respond to that — could you rephrase?", function_calls=[], provider="groq", state={}, trail=plan_trail)

            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls],
            })

            risky = [tc for tc in tool_calls if tc.function.name in STATE_CHANGING_TOOLS]
            if risky:
                calls = [ToolCall(tc.function.name, json.loads(tc.function.arguments or "{}"), id=tc.id) for tc in tool_calls]
                return PlanResult(text=None, function_calls=calls, provider="groq", state={"messages": messages, "tool_owner": tool_owner, "tools_schema": tools_schema}, trail=plan_trail)

            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                print(f"[Nexus/Groq] Auto-executing read-only {tc.function.name}({args})")
                result = await tool_owner[tc.function.name].call_tool(tc.function.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"[Nexus/Groq] -> {result_text}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

        return PlanResult(text="That request needed too many steps — could you break it into smaller parts?", function_calls=[], provider="groq", state={}, trail=plan_trail)

    async def execute_calls(self, plan_result: PlanResult):
        if plan_result.provider == "gemini":
            return await self._execute_calls_gemini(plan_result)
        return await self._execute_calls_groq(plan_result)

    async def _execute_calls_gemini(self, plan_result: PlanResult):
        contents = plan_result.state["contents"]
        tool_owner = plan_result.state["tool_owner"]
        trail = list(plan_result.trail or [])

        tool_response_parts = []
        for fc in plan_result.function_calls:
            result = await tool_owner[fc.name].call_tool(fc.name, fc.args)
            result_text = result.content[0].text if result.content else "[]"
            trail.append((fc.name, fc.args, result_text))
            tool_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result_text})
            )
        contents.append(types.Content(role="user", parts=tool_response_parts))

        # Give it a few turns to chain safe (non-state-changing) follow-ups,
        # e.g. reading back what was just written — without needing a second
        # confirmation. The system prompt is carried through so the user's
        # remembered facts and the "don't answer from memory" rules still apply.
        safe_tools, safe_owner = await self._merged_tools(exclude_state_changing=True)
        gemini_safe_tools = types.Tool(function_declarations=safe_tools)
        system_prompt = await self._get_system_prompt()

        try:
            for _ in range(3):
                response = await self.gemini_client.aio.models.generate_content(
                    model=GEMINI_MODEL, contents=contents,
                    config=types.GenerateContentConfig(temperature=0, tools=[gemini_safe_tools], system_instruction=system_prompt),
                )
                contents.append(response.candidates[0].content)
                calls = list(response.function_calls or [])
                if not calls:
                    return _safe_text(response), trail

                follow_parts = []
                for fc in calls:
                    args = fc.args or {}
                    result = await safe_owner[fc.name].call_tool(fc.name, args)
                    result_text = result.content[0].text if result.content else "[]"
                    trail.append((fc.name, args, result_text))
                    follow_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result_text}))
                contents.append(types.Content(role="user", parts=follow_parts))

            response = await self.gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=contents,
                config=types.GenerateContentConfig(temperature=0, system_instruction=system_prompt),
            )
            return _safe_text(response), trail

        except Exception as e:
            print(f"[Nexus] Gemini unavailable for summary ({e}); asking Groq to summarize instead...")
            summary_prompt = (
                "I just performed these actions on the user's behalf:\n" +
                "\n".join(f"- {name}({args}) -> {result}" for name, args, result in trail) +
                "\nWrite a short, friendly 1-2 sentence reply confirming what was done."
            )
            completion = self.groq_client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL, messages=[{"role": "user", "content": summary_prompt}],
            )
            return completion.choices[0].message.content, trail

    async def _execute_calls_groq(self, plan_result: PlanResult):
        messages = plan_result.state["messages"]
        tool_owner = plan_result.state["tool_owner"]
        trail = list(plan_result.trail or [])
        for fc in plan_result.function_calls:
            result = await tool_owner[fc.name].call_tool(fc.name, fc.args)
            result_text = result.content[0].text if result.content else "[]"
            print(f"[Nexus/Execute] {fc.name}({fc.args}) -> {result_text}")
            trail.append((fc.name, fc.args, result_text))
            messages.append({"role": "tool", "tool_call_id": fc.id, "content": result_text})

        # No 'tools' passed here on purpose — same trick as the Gemini path.
        # Without tools attached, the model can't chain another call and leave
        # .content empty; it's forced to answer in plain text.
        response = self.groq_client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=messages,
            temperature=0,
        )
        text = response.choices[0].message.content or "Done."
        return text, trail

    async def connect_servers(self):
        # Filesystem server — Windows needs npx launched through cmd /c
        fs_params = StdioServerParameters(
            command="cmd",
            args=["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", SANDBOX_PATH],
        )
        read, write = await self._stack.enter_async_context(stdio_client(fs_params))
        fs_session = await self._stack.enter_async_context(ClientSession(read, write))
        await fs_session.initialize()
        self.sessions["filesystem"] = fs_session

        # To-Do server — sys.executable, not "python": a bare "python" resolves
        # through PATH and can land on a system interpreter without our deps
        # installed, which shows up as an opaque "Connection closed".
        todo_params = StdioServerParameters(command=sys.executable, args=[TODO_SERVER_PATH])
        read, write = await self._stack.enter_async_context(stdio_client(todo_params))
        todo_session = await self._stack.enter_async_context(ClientSession(read, write))
        await todo_session.initialize()
        self.sessions["todo"] = todo_session

        # Web Search server — DuckDuckGo, no API key needed. Installed as a
        # console script in the venv, so resolve it the same way as the
        # interpreter rather than trusting PATH.
        search_params = StdioServerParameters(command=_console_script("duckduckgo-mcp-server"), args=[])
        read, write = await self._stack.enter_async_context(stdio_client(search_params))
        search_session = await self._stack.enter_async_context(ClientSession(read, write))
        await search_session.initialize()
        self.sessions["websearch"] = search_session

        for name, session in self.sessions.items():
            tools = (await session.list_tools()).tools
            print(f"[Nexus] Connected to '{name}' server — tools: {[t.name for t in tools]}")

    async def close(self):
        await self._stack.aclose()

    async def _merged_tools(self, exclude_state_changing=False, allowed_state_tools=None):
        """Return (tool_declarations, tool_owner_map).

        Args:
            exclude_state_changing: If True, omit state-changing tools.
            allowed_state_tools:   If given (set), these state-changing tools
                                   are kept even when exclude_state_changing is True.
        """
        allowed = allowed_state_tools or set()
        all_tools = []
        tool_owner = {}
        for session in self.sessions.values():
            mcp_tools = (await session.list_tools()).tools
            for t in mcp_tools:
                if exclude_state_changing and t.name in STATE_CHANGING_TOOLS and t.name not in allowed:
                    continue
                clean_schema = {k: v for k, v in t.inputSchema.items() if k not in ("additionalProperties", "$schema")}
                all_tools.append({"name": t.name, "description": t.description or "", "parameters": clean_schema})
                tool_owner[t.name] = session
        return all_tools, tool_owner

    async def _get_system_prompt(self) -> str:
        try:
            result = await self.sessions["todo"].call_tool("list_memories", {})
            memories_text = result.content[0].text if result.content else "[]"
            memories = json.loads(memories_text)
        except Exception:
            memories = []

        if memories:
            facts = "\n".join(f"- {m['fact']}" for m in memories)
            return SYSTEM_PROMPT + f"\n\nHere is what you currently remember about this user:\n{facts}\n\nUse this to personalize your responses where relevant."
        return SYSTEM_PROMPT

    async def ask(self, user_message: str) -> str:
        """Compatibility wrapper that never bypasses the confirmation workflow."""
        result = await self.plan(user_message)
        if result.function_calls:
            return "This request needs confirmation in the Nexus UI before I can make changes."
        return result.text

    async def proactive_check(self):
        try:
            await self._proactive_gemini()
        except Exception as e:
            print(f"[Nexus/Proactive] Gemini unavailable ({e}); trying Groq...")
            try:
                await self._proactive_groq()
            except Exception as e2:
                print(f"[Nexus/Proactive] Skipped this cycle — both providers failed: {e2}")

    async def _proactive_gemini(self):
        all_tools, tool_owner = await self._merged_tools(
            exclude_state_changing=True,
            allowed_state_tools=PROACTIVE_ALLOWED_STATE_TOOLS,
        )
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [types.Content(role="user", parts=[types.Part(text=PROACTIVE_PROMPT)])]

        response = await self.gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL, contents=contents,
            config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools], system_instruction=await self._get_system_prompt()),
        )
        contents.append(response.candidates[0].content)

        notified = False
        turns = 0
        while response.function_calls and turns < 4:
            turns += 1
            tool_response_parts = []
            for fc in response.function_calls:
                args = fc.args or {}
                print(f"[Nexus/Proactive] Calling {fc.name}({args})")
                result = await tool_owner[fc.name].call_tool(fc.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"[Nexus/Proactive] -> {result_text}")
                if fc.name == "create_notification":
                    notified = True
                tool_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"result": result_text})
                )
            contents.append(types.Content(role="user", parts=tool_response_parts))
            response = await self.gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=contents,
                config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools], system_instruction=await self._get_system_prompt()),
            )
            contents.append(response.candidates[0].content)

        if not notified:
            print("[Nexus/Proactive] Cycle complete — nothing surfaced.")

    async def _proactive_groq(self):
        all_tools, tool_owner = await self._merged_tools(
            exclude_state_changing=True,
            allowed_state_tools=PROACTIVE_ALLOWED_STATE_TOOLS,
        )
        tools_schema = [{"type": "function", "function": t} for t in all_tools]
        messages = [{"role": "system", "content": await self._get_system_prompt()}, {"role": "user", "content": PROACTIVE_PROMPT}]

        response = self.groq_client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL, messages=messages, tools=tools_schema, tool_choice="auto", temperature=0,
        )
        msg = response.choices[0].message

        notified = False
        turns = 0
        while msg.tool_calls and turns < 4:
            turns += 1
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                print(f"[Nexus/Proactive] Calling {call.function.name}({args})")
                result = await tool_owner[call.function.name].call_tool(call.function.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"[Nexus/Proactive] -> {result_text}")
                if call.function.name == "create_notification":
                    notified = True
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})
            response = self.groq_client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL, messages=messages, tools=tools_schema, tool_choice="auto", temperature=0,
            )
            msg = response.choices[0].message

        if not notified:
            print("[Nexus/Proactive] Cycle complete — nothing surfaced.")

    async def send_task_summary_whatsapp(self):
        try:
            result = await self.sessions["todo"].call_tool("list_tasks", {})
            tasks = json.loads(result.content[0].text) if result.content else []
            if not tasks:
                message = "📋 Daily check-in: your to-do list is empty. Nice work!"
            else:
                lines = [f"{i+1}. {t['task']}" + (f" (due {t['due']})" if t.get('due') else "") for i, t in enumerate(tasks)]
                message = "📋 Your to-do list today:\n" + "\n".join(lines)
            send_result = await self.sessions["todo"].call_tool("send_whatsapp", {"message": message})
            print(f"[Nexus/Scheduled] Task summary: {send_result.content[0].text if send_result.content else send_result}")
        except Exception as e:
            print(f"[Nexus/Scheduled] Failed to send task summary: {e}")
