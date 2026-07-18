import asyncio
import json
import os
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
    "add_task", "complete_task", "undo_last_action",
}
SYSTEM_PROMPT = (
    "You are Nexus, a friendly personal productivity assistant. You have access to a set of "
    "tools for managing files, a to-do list (including viewing an audit log of past actions and "
    "undoing the most recent one), searching the live web, and getting the real current date/time. "
    "Always check the full list of tools provided to you for this request before deciding whether "
    "one applies — do not assume a request has no matching tool just because it isn't explicitly "
    "named in this instruction; new tools may have been added since this was written.\n\n"
    "IMPORTANT: Your own knowledge has a training cutoff and is NOT reliable for anything "
    "time-sensitive. For the current date or time, you MUST use the get_current_time tool. For "
    "prices, exchange rates, news, or current events, you MUST use the search tool. For questions "
    "about past actions Nexus has taken (e.g. 'undo that', 'show my audit log'), you MUST use the "
    "undo_last_action or get_audit_log tools rather than guessing. Never answer any of these from memory.\n\n"
    "If a request has no matching tool (e.g. deleting a file, when no delete tool exists), say so "
    "plainly and briefly — do not invent a result.\n\n"
    "For everything else — general knowledge, casual conversation, quick math, jokes — just "
    "answer directly in plain text without using a tool."
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
    provider: str   # "gemini" or "groq" — lets execute_calls() know which conversation state to use
    state: dict


class NexusAgent:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.sessions = {}
        self._stack = AsyncExitStack()

    async def plan(self, user_message: str) -> PlanResult:
        try:
            return await self._plan_gemini(user_message)
        except Exception as e:
            print(f"[Nexus] Gemini unavailable ({e}); falling back to Groq Llama...")
            return await self._plan_groq(user_message)

    async def _plan_gemini(self, user_message: str) -> PlanResult:
        all_tools, tool_owner = await self._merged_tools()
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

        response = await self.gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools], system_instruction=SYSTEM_PROMPT),
        )
        contents.append(response.candidates[0].content)
        if response.function_calls:
            print(f"[Nexus/Gemini] Plan for {user_message!r}:")
            for fc in response.function_calls:
                print(f"  -> wants to call {fc.name}({dict(fc.args or {})})")
        else:
            print(f"[Nexus/Gemini] Plan for {user_message!r}: no tool call, direct text = {response.text!r}")

        calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in (response.function_calls or [])]
        return PlanResult(
            text=response.text if not calls else None,
            function_calls=calls,
            provider="gemini",
            state={"contents": contents, "tool_owner": tool_owner},
        )

    async def _plan_groq(self, user_message: str) -> PlanResult:
        all_tools, tool_owner = await self._merged_tools()
        tools_schema = [{"type": "function", "function": t} for t in all_tools]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]

        for attempt in range(2):
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0,  # deterministic output — cuts down on malformed tool-call tokens
                )
                msg = response.choices[0].message
                if msg.tool_calls:
                    print(f"[Nexus/Groq] Plan for {user_message!r}:")
                    for tc in msg.tool_calls:
                        print(f"  -> wants to call {tc.function.name}({tc.function.arguments})")
                else:
                    print(f"[Nexus/Groq] Plan for {user_message!r}: no tool call, direct text = {msg.content!r}")

                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in (msg.tool_calls or [])
                    ],
                })
                calls = [ToolCall(tc.function.name, json.loads(tc.function.arguments or "{}"), id=tc.id)
                         for tc in (msg.tool_calls or [])]
                return PlanResult(
                    text=(msg.content or "I'm not sure how to respond to that — could you rephrase?") if not calls else None,
                    function_calls=calls,
                    provider="groq",
                    state={"messages": messages, "tool_owner": tool_owner, "tools_schema": tools_schema},
                )
            except BadRequestError as e:
                print(f"[Nexus] Groq produced a malformed tool call (attempt {attempt + 1}/2); retrying...")
                last_error = e

        # Both attempts failed — degrade gracefully instead of crashing the app
        return PlanResult(
            text=("I'm having trouble planning that request right now — both Gemini and Groq hit issues. "
                  "Could you try rephrasing, or try again in a moment?"),
            function_calls=[],
            provider="groq",
            state={},
        )
    async def execute_calls(self, plan_result: PlanResult):
        if plan_result.provider == "gemini":
            return await self._execute_calls_gemini(plan_result)
        return await self._execute_calls_groq(plan_result)

    async def _execute_calls_gemini(self, plan_result: PlanResult):
        contents = plan_result.state["contents"]
        tool_owner = plan_result.state["tool_owner"]
        tool_response_parts = []
        trail = []
        for fc in plan_result.function_calls:
            result = await tool_owner[fc.name].call_tool(fc.name, fc.args)
            result_text = result.content[0].text if result.content else "[]"
            print(f"[Nexus/Execute] {fc.name}({fc.args}) -> {result_text}")
            trail.append((fc.name, fc.args, result_text))
            tool_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result_text})
            )
        contents.append(types.Content(role="user", parts=tool_response_parts))

        try:
            response = await self.gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=contents,
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text, trail
        except Exception as e:
            print(f"[Nexus] Gemini unavailable for summary ({e}); asking Groq to summarize instead...")
            summary_prompt = (
                "I just performed these actions on the user's behalf:\n" +
                "\n".join(f"- {name}({args}) -> {result}" for name, args, result in trail) +
                "\nWrite a short, friendly 1-2 sentence reply confirming what was done."
            )
            completion = self.groq_client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            return completion.choices[0].message.content, trail

    async def _execute_calls_groq(self, plan_result: PlanResult):
        messages = plan_result.state["messages"]
        tool_owner = plan_result.state["tool_owner"]
        trail = []
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

        # To-Do server — plain python executable, no wrapper needed
        todo_params = StdioServerParameters(command="python", args=[TODO_SERVER_PATH])
        read, write = await self._stack.enter_async_context(stdio_client(todo_params))
        todo_session = await self._stack.enter_async_context(ClientSession(read, write))
        await todo_session.initialize()
        self.sessions["todo"] = todo_session

        # Web Search server — DuckDuckGo, no API key needed
        search_params = StdioServerParameters(command="duckduckgo-mcp-server", args=[])
        read, write = await self._stack.enter_async_context(stdio_client(search_params))
        search_session = await self._stack.enter_async_context(ClientSession(read, write))
        await search_session.initialize()
        self.sessions["websearch"] = search_session

        for name, session in self.sessions.items():
            tools = (await session.list_tools()).tools
            print(f"[Nexus] Connected to '{name}' server — tools: {[t.name for t in tools]}")

    async def close(self):
        await self._stack.aclose()

    async def _merged_tools(self):
        """Combine tool lists from every connected server into one manifest,
        and remember which session owns each tool name so we can route calls back."""
        all_tools = []
        tool_owner = {}
        for session in self.sessions.values():
            mcp_tools = (await session.list_tools()).tools
            for t in mcp_tools:
                clean_schema = {k: v for k, v in t.inputSchema.items() if k not in ("additionalProperties", "$schema")}
                all_tools.append({"name": t.name, "description": t.description or "", "parameters": clean_schema})
                tool_owner[t.name] = session
        return all_tools, tool_owner

    async def ask(self, user_message: str) -> str:
        try:
            return await self._ask_gemini(user_message)
        except Exception as gemini_error:
            print(f"[Nexus] Gemini unavailable ({gemini_error}); falling back to Groq Llama...")
            return await self._ask_groq(user_message)

    async def _ask_gemini(self, user_message: str) -> str:
        all_tools, tool_owner = await self._merged_tools()
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

        response = await self.gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools]),
        )
        contents.append(response.candidates[0].content)

        print("\n--- Think -> Act -> Observe trail (Gemini) ---")
        turns = 0
        while response.function_calls and turns < 5:
            turns += 1
            tool_response_parts = []
            for fc in response.function_calls:
                args = fc.args or {}
                print(f"  [Act] Calling {fc.name}({args})")
                result = await tool_owner[fc.name].call_tool(fc.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"  [Observe] {result_text}")
                tool_response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"result": result_text})
                )
            contents.append(types.Content(role="user", parts=tool_response_parts))
            response = await self.gemini_client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools]),
            )
            contents.append(response.candidates[0].content)
        print("--- end trail ---\n")

        return response.text

    async def _ask_groq(self, user_message: str) -> str:
        all_tools, tool_owner = await self._merged_tools()
        tools_schema = [{"type": "function", "function": t} for t in all_tools]
        messages = [{"role": "user", "content": user_message}]

        response = self.groq_client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = response.choices[0].message

        print("\n--- Think -> Act -> Observe trail (Groq fallback) ---")
        turns = 0
        while msg.tool_calls and turns < 5:
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
                print(f"  [Act] Calling {call.function.name}({args})")
                result = await tool_owner[call.function.name].call_tool(call.function.name, args)
                result_text = result.content[0].text if result.content else "[]"
                print(f"  [Observe] {result_text}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})
            response = self.groq_client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL, messages=messages, tools=tools_schema, tool_choice="auto",
            )
            msg = response.choices[0].message
        print("--- end trail ---\n")

        return msg.content


async def main():
    agent = NexusAgent()
    await agent.connect_servers()
    try:
        reply = await agent.ask("Add 'buy groceries' to my to-do list, then list everything on it.")
        print("Nexus:", reply)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())