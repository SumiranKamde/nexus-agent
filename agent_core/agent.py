import asyncio
import base64
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq, BadRequestError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# On Streamlit Community Cloud, secrets are injected via st.secrets.
# Mirror them into os.environ so the rest of the code can use os.environ.get
# without caring whether we're running locally or in the cloud.
try:
    import streamlit as st
    _st_secrets = st.secrets.to_dict() if hasattr(st, "secrets") else {}
    for _k, _v in _st_secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # streamlit not available or secrets not configured — ignore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
# Groq retires hosted models without warning, and a retired id fails as a 404 on
# every call — i.e. the whole fallback path goes dead silently. If you see
# model_not_found, run tests/manual/check_providers.py and pick a live id from
# the list it prints; it must support tool calling.
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_PATH = os.path.join(PROJECT_ROOT, "sandbox_files")
UPLOAD_PATH = os.path.join(SANDBOX_PATH, "uploads")
TODO_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_servers", "todo_server.py")
STATE_CHANGING_TOOLS = {
    "write_file", "edit_file", "create_directory", "move_file",
    "delete_file", "delete_directory",
    "add_task", "complete_task", "delete_task", "undo_last_action", "remember", "forget",
    "create_notification", "mark_notifications_seen", "send_whatsapp",
    "send_file_via_whatsapp", "mark_task_notified",
}
# Tools the proactive check is allowed to call directly (notifications + reads)
PROACTIVE_ALLOWED_STATE_TOOLS = {
    "create_notification", "send_whatsapp", "send_file_via_whatsapp", "mark_task_notified",
}
SYSTEM_PROMPT = (
    "You are Nexus, a friendly personal productivity assistant. You have access to a set of "
    "tools for managing files, a to-do list (including an audit log and undo), searching the "
    "live web, getting the real current date/time, remembering facts about the user across "
    "conversations (remember, list_memories, forget), and sending WhatsApp messages to the user's "
    "phone. The user can also attach images and PDF documents for you to analyze — describe, "
    "summarize, extract data from, or answer questions about them. "
    "Always check the full list of tools provided to you for this request before deciding "
    "whether one applies — do not assume a request has no matching tool just because it isn't "
    "explicitly named here.\n\n"
    "CRITICAL TOOL EXECUTION MANDATE:\n"
    "1. When the user asks you to perform ANY real action (such as adding a task, completing a task, "
    "deleting a task, editing/writing a file, saving a memory, or sending a message), you MUST invoke "
    "the appropriate tool. Never merely claim or say in conversational text that you have performed an "
    "action without calling the tool. Text alone CANNOT modify database state.\n"
    "2. When the user asks to complete, delete, or modify an existing task by description (e.g. 'mark presentation done', "
    "'remove grocery task'), you MUST first inspect tasks with list_tasks if you don't know the exact ID, "
    "and then call complete_task or delete_task with the exact task_id or matching task title. Never invent task IDs.\n\n"
    "When adding a task that has a deadline or time-based reminder (e.g. 'remind me at 5pm', "
    "'buy groceries tomorrow at 10am', 'submit report in 30 minutes'), you MUST call get_current_time "
    "first to get today's date and time, and format the 'due' parameter as 'YYYY-MM-DD HH:MM' in "
    "Asia/Kolkata timezone. This enables the background monitor to trigger a live WhatsApp notification "
    "precisely when the task is due.\n\n"
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
class FileAttachment:
    """An uploaded file to be included in the next LLM request."""
    filename: str
    mime_type: str
    data: bytes

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

# Accepted upload MIME types — images go to vision, PDFs get extracted.
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
PDF_MIME_TYPE = "application/pdf"

def _console_script(name: str) -> str:
    """Locate a console script installed alongside the running interpreter.

    Looking next to sys.executable first means the venv's copy wins even when
    the venv isn't activated; PATH and then the bare name are fallbacks.
    """
    scripts_dir = os.path.dirname(sys.executable)
    return shutil.which(name, path=scripts_dir) or shutil.which(name) or name


def _extract_pdf_text(data: bytes, filename: str, max_chars: int = 12000) -> str:
    """Extract plain text from a PDF using PyMuPDF.

    Falls back to a short error note (never raises) so that a corrupt file
    doesn't kill the whole request.
    """
    try:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        total = 0
        for page in doc:
            text = page.get_text()
            if total + len(text) > max_chars:
                pages.append(text[:max_chars - total] + "\n...(truncated)")
                break
            pages.append(text)
            total += len(text)
        doc.close()
        body = "\n--- page break ---\n".join(pages).strip()
        return f"[Extracted text from {filename}]:\n{body}" if body else f"[{filename} contained no extractable text]"
    except Exception as e:
        return f"[Could not extract text from {filename}: {e}]"


def save_uploads(files):
    """Persist uploaded files into sandbox_files/uploads/ and return their paths."""
    if not files:
        return []
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    saved = []
    for f in files:
        dest = os.path.join(UPLOAD_PATH, f.filename)
        with open(dest, "wb") as fp:
            fp.write(f.data)
        saved.append(dest)
        print(f"[Nexus] Saved upload: {dest} ({len(f.data)} bytes)")
    return saved


def _safe_text(response, fallback="I'm not able to do that with my current tools — could you rephrase, or ask for something else?"):
    try:
        return response.text or fallback
    except Exception:
        return fallback


def _describe_trail(trail):
    """Summarize executed tool calls without an LLM.

    Used when both providers are unreachable *after* tools have already run —
    the actions are done, so the user needs to be told what happened even if
    nothing is available to phrase it nicely.
    """
    if not trail:
        return "Done."
    lines = "\n".join(f"- {name}({args})" for name, args, _result in trail)
    return f"Done — I ran these, but couldn't reach a model to summarize:\n{lines}"

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
        self.gemini_client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=30000)) if gemini_key else None
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.sessions = {}
        self._stack = AsyncExitStack()
        self._cached_tools_raw = None
        self._cached_memories = None

    async def plan(self, user_message: str, history=None, files=None) -> PlanResult:
        """Plan a response to user_message.

        history is the prior chat as [{"role": "user"|"assistant", "content": str}],
        oldest first, excluding user_message itself.
        files is an optional list of FileAttachment objects.
        """
        # Persist uploads so the filesystem MCP server can access them later.
        save_uploads(files)

        if self.gemini_client:
            models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
            for m in models_to_try:
                try:
                    return await self._plan_gemini(user_message, history, files, model=m)
                except Exception as e:
                    print(f"[Nexus] Gemini model '{m}' unavailable ({e}); trying next...")
        if self.groq_client:
            try:
                return await self._plan_groq(user_message, history, files)
            except Exception as e:
                # Last line of defence: both providers are down. Without this the
                # exception lands in Streamlit's script thread as a raw traceback.
                print(f"[Nexus] Groq fallback also failed ({e}).")
                return PlanResult(
                    text="Both AI providers are unreachable right now, so I can't plan that. "
                         "Check the terminal for the error — if it mentions model_not_found, "
                         "GROQ_FALLBACK_MODEL needs updating.",
                    function_calls=[], provider="none", state={}, trail=[],
                )
        return PlanResult(
            text="No AI provider is configured. Add GEMINI_API_KEY or GROQ_API_KEY to .env.",
            function_calls=[], provider="none", state={}, trail=[],
        )

    async def _plan_gemini(self, user_message: str, history=None, files=None, model=None) -> PlanResult:
        active_model = model or GEMINI_MODEL
        all_tools, tool_owner = await self._merged_tools()
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [
            types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)])
            for role, text in _history_pairs(history)
        ]
        # Build the user turn: text + any attached files as inline parts.
        user_parts = [types.Part(text=user_message)]
        for f in (files or []):
            user_parts.append(types.Part.from_bytes(data=f.data, mime_type=f.mime_type))
            print(f"[Nexus/Gemini] Attached {f.filename} ({f.mime_type}, {len(f.data)} bytes)")
        contents.append(types.Content(role="user", parts=user_parts))
        system_prompt = await self._get_system_prompt()

        plan_trail = []
        for turn in range(5):
            response = await self.gemini_client.aio.models.generate_content(
                model=active_model, contents=contents,
                config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools], system_instruction=system_prompt),
            )
            contents.append(response.candidates[0].content)
            calls = list(response.function_calls or [])

            if not calls:
                return PlanResult(text=_safe_text(response), function_calls=[], provider="gemini", state={}, trail=plan_trail)

            risky = [fc for fc in calls if fc.name in STATE_CHANGING_TOOLS]
            if risky:
                tool_calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in calls]
                return PlanResult(text=None, function_calls=tool_calls, provider="gemini", state={"contents": contents, "tool_owner": tool_owner, "model": active_model}, trail=plan_trail)

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

    async def _plan_groq(self, user_message: str, history=None, files=None) -> PlanResult:
        all_tools, tool_owner = await self._merged_tools()
        tools_schema = [{"type": "function", "function": t} for t in all_tools]
        system_prompt = await self._get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": role, "content": text} for role, text in _history_pairs(history)]

        # Groq uses the OpenAI multimodal content format: images go as base64
        # data URLs, PDFs are converted to extracted text since Groq doesn't
        # accept raw PDF uploads.
        files = files or []
        image_files = [f for f in files if f.mime_type in IMAGE_MIME_TYPES]
        pdf_files = [f for f in files if f.mime_type == PDF_MIME_TYPE]

        if image_files:
            # Build a multimodal content array for the user message.
            content_parts = [{"type": "text", "text": user_message}]
            for f in image_files:
                b64 = base64.b64encode(f.data).decode("ascii")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{f.mime_type};base64,{b64}"},
                })
                print(f"[Nexus/Groq] Attached image {f.filename} ({len(f.data)} bytes)")
            # Append PDF extracted text inline.
            for f in pdf_files:
                content_parts[0]["text"] += "\n\n" + _extract_pdf_text(f.data, f.filename)
                print(f"[Nexus/Groq] Extracted text from {f.filename}")
            messages.append({"role": "user", "content": content_parts})
        elif pdf_files:
            # Text-only path: append extracted PDF text to the user message.
            extra = "\n\n".join(_extract_pdf_text(f.data, f.filename) for f in pdf_files)
            for f in pdf_files:
                print(f"[Nexus/Groq] Extracted text from {f.filename}")
            messages.append({"role": "user", "content": user_message + "\n\n" + extra})
        else:
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
                plan_trail.append((tc.function.name, args, result_text))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

        return PlanResult(text="That request needed too many steps — could you break it into smaller parts?", function_calls=[], provider="groq", state={}, trail=plan_trail)

    def _resolve_tool_session(self, tool_name: str, tool_owner: dict = None):
        if tool_owner and tool_name in tool_owner:
            return tool_owner[tool_name]
        for sname, session in self.sessions.items():
            summary = getattr(self, "tool_summary", {}).get(sname, [])
            if tool_name in summary:
                return session
        if tool_name in ("add_task", "complete_task", "delete_task", "list_tasks", "get_due_tasks",
                         "mark_task_notified", "undo_last_action", "get_audit_log",
                         "remember", "list_memories", "forget", "create_notification",
                         "list_notifications", "mark_notifications_seen", "send_whatsapp",
                         "send_file_via_whatsapp", "get_current_time"):
            return self.sessions.get("todo")
        return self.sessions.get("filesystem")

    def _clean_tool_args(self, tool_name: str, args: dict) -> dict:
        clean = dict(args or {})
        for int_key in ("task_id", "memory_id", "limit"):
            if int_key in clean and clean[int_key] is not None:
                try:
                    clean[int_key] = int(clean[int_key])
                except (ValueError, TypeError):
                    pass
        return clean

    async def execute_calls(self, plan_result: PlanResult):
        if plan_result.provider == "gemini":
            return await self._execute_calls_gemini(plan_result)
        return await self._execute_calls_groq(plan_result)

    async def _execute_calls_gemini(self, plan_result: PlanResult):
        contents = plan_result.state["contents"]
        tool_owner = plan_result.state.get("tool_owner", {})
        trail = list(plan_result.trail or [])

        tool_response_parts = []
        for fc in plan_result.function_calls:
            session = self._resolve_tool_session(fc.name, tool_owner)
            clean_args = self._clean_tool_args(fc.name, fc.args)
            try:
                result = await session.call_tool(fc.name, clean_args)
                result_text = result.content[0].text if result and result.content else "[]"
            except Exception as e:
                print(f"[Nexus/Execute] Error executing {fc.name}({clean_args}): {e}")
                result_text = f"Error executing {fc.name}: {e}"

            trail.append((fc.name, clean_args, result_text))
            tool_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result_text})
            )
            if fc.name in ("remember", "forget"):
                self._cached_memories = None
        contents.append(types.Content(role="user", parts=tool_response_parts))

        # Give it a few turns to chain safe (non-state-changing) follow-ups,
        # e.g. reading back what was just written — without needing a second
        # confirmation. The system prompt is carried through so the user's
        # remembered facts and the "don't answer from memory" rules still apply.
        safe_tools, safe_owner = await self._merged_tools(exclude_state_changing=True)
        gemini_safe_tools = types.Tool(function_declarations=safe_tools)
        system_prompt = await self._get_system_prompt()

        active_model = plan_result.state.get("model", GEMINI_MODEL)
        try:
            for _ in range(3):
                response = await self.gemini_client.aio.models.generate_content(
                    model=active_model, contents=contents,
                    config=types.GenerateContentConfig(temperature=0, tools=[gemini_safe_tools], system_instruction=system_prompt),
                )
                contents.append(response.candidates[0].content)
                calls = list(response.function_calls or [])
                if not calls:
                    return _safe_text(response), trail

                follow_parts = []
                for fc in calls:
                    args = fc.args or {}
                    session = self._resolve_tool_session(fc.name, safe_owner)
                    clean_args = self._clean_tool_args(fc.name, args)
                    try:
                        result = await session.call_tool(fc.name, clean_args)
                        result_text = result.content[0].text if result and result.content else "[]"
                    except Exception as e:
                        result_text = f"Error executing {fc.name}: {e}"
                    trail.append((fc.name, clean_args, result_text))
                    follow_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result_text}))
                contents.append(types.Content(role="user", parts=follow_parts))

            response = await self.gemini_client.aio.models.generate_content(
                model=active_model, contents=contents,
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
            try:
                completion = self.groq_client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL, messages=[{"role": "user", "content": summary_prompt}],
                )
                return completion.choices[0].message.content, trail
            except Exception as e2:
                # The tools have already run by this point, so failing here would
                # leave the user unsure whether their action landed. The trail is
                # the ground truth — report it directly rather than not at all.
                print(f"[Nexus] Groq summary also failed ({e2}); reporting the trail directly.")
                return _describe_trail(trail), trail

    async def _execute_calls_groq(self, plan_result: PlanResult):
        messages = plan_result.state["messages"]
        tool_owner = plan_result.state.get("tool_owner", {})
        trail = list(plan_result.trail or [])
        for fc in plan_result.function_calls:
            session = self._resolve_tool_session(fc.name, tool_owner)
            clean_args = self._clean_tool_args(fc.name, fc.args)
            try:
                result = await session.call_tool(fc.name, clean_args)
                result_text = result.content[0].text if result and result.content else "[]"
            except Exception as e:
                print(f"[Nexus/Execute] Error executing {fc.name}({clean_args}): {e}")
                result_text = f"Error executing {fc.name}: {e}"

            print(f"[Nexus/Execute] {fc.name}({clean_args}) -> {result_text}")
            trail.append((fc.name, clean_args, result_text))
            messages.append({"role": "tool", "tool_call_id": fc.id, "content": result_text})
            if fc.name in ("remember", "forget"):
                self._cached_memories = None

        # No 'tools' passed here on purpose — same trick as the Gemini path.
        # Without tools attached, the model can't chain another call and leave
        # .content empty; it's forced to answer in plain text.
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=messages,
                temperature=0,
            )
            text = response.choices[0].message.content or "Done."
        except Exception as e:
            # As in the Gemini path: the tools already ran, so report the trail
            # rather than raising and leaving the user guessing.
            print(f"[Nexus/Groq] Summary call failed ({e}); reporting the trail directly.")
            text = _describe_trail(trail)
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

        self._cached_tools_raw = []
        for name, session in self.sessions.items():
            tools = (await session.list_tools()).tools
            print(f"[Nexus] Connected to '{name}' server — tools: {[t.name for t in tools]}")
            for t in tools:
                clean_schema = {k: v for k, v in t.inputSchema.items() if k not in ("additionalProperties", "$schema")}
                self._cached_tools_raw.append({
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": clean_schema,
                    "owner": session,
                })

    async def close(self):
        await self._stack.aclose()

    async def _merged_tools(self, exclude_state_changing=False, allowed_state_tools=None):
        """Return (tool_declarations, tool_owner_map).

        Uses cached tool schemas from connect_servers to eliminate redundant stdio roundtrips.
        """
        allowed = allowed_state_tools or set()
        all_tools = []
        tool_owner = {}

        cached = getattr(self, "_cached_tools_raw", None)
        if cached is None:
            raw_tools = []
            for session in self.sessions.values():
                mcp_tools = (await session.list_tools()).tools
                for t in mcp_tools:
                    clean_schema = {k: v for k, v in t.inputSchema.items() if k not in ("additionalProperties", "$schema")}
                    raw_tools.append({
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": clean_schema,
                        "owner": session,
                    })
        else:
            raw_tools = cached

        for item in raw_tools:
            name = item["name"]
            if exclude_state_changing and name in STATE_CHANGING_TOOLS and name not in allowed:
                continue
            all_tools.append({
                "name": name,
                "description": item["description"],
                "parameters": item["parameters"],
            })
            tool_owner[name] = item["owner"]

        return all_tools, tool_owner

    async def _get_system_prompt(self, force_refresh=False) -> str:
        cached_mem = getattr(self, "_cached_memories", None)
        if cached_mem is None or force_refresh:
            try:
                result = await self.sessions["todo"].call_tool("list_memories", {})
                memories_text = result.content[0].text if result.content else "[]"
                cached_mem = json.loads(memories_text)
            except Exception:
                cached_mem = []
            self._cached_memories = cached_mem

        if cached_mem:
            facts = "\n".join(f"- {m['fact']}" for m in cached_mem)
            return SYSTEM_PROMPT + f"\n\nHere is what you currently remember about this user:\n{facts}\n\nUse this to personalize your responses where relevant."
        return SYSTEM_PROMPT

    async def get_dashboard_data(self):
        """Fetch memories, tasks, audit log, and notifications in parallel in a single async roundtrip."""
        todo_session = self.sessions.get("todo")
        if not todo_session:
            return {"memories": [], "tasks": [], "recent_activity": [], "notifications": []}

        mem_f = todo_session.call_tool("list_memories", {})
        task_f = todo_session.call_tool("list_tasks", {})
        audit_f = todo_session.call_tool("get_audit_log", {"limit": 5})
        notif_f = todo_session.call_tool("list_notifications", {"unseen_only": True})

        results = await asyncio.gather(mem_f, task_f, audit_f, notif_f, return_exceptions=True)

        def _safe_json(res):
            if isinstance(res, Exception) or not res or not getattr(res, "content", None):
                return []
            try:
                return json.loads(res.content[0].text)
            except Exception:
                return []

        memories = _safe_json(results[0])
        tasks = _safe_json(results[1])
        recent_activity = _safe_json(results[2])
        notifications = _safe_json(results[3])

        self._cached_memories = memories

        return {
            "memories": memories,
            "tasks": tasks,
            "recent_activity": recent_activity,
            "notifications": notifications,
        }

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

    async def check_live_task_reminders(self):
        """Check for open tasks that have hit their due time and dispatch WhatsApp notifications."""
        try:
            todo_session = self.sessions.get("todo")
            if not todo_session:
                return

            result = await todo_session.call_tool("get_due_tasks", {"timezone_name": "Asia/Kolkata"})
            due_tasks = json.loads(result.content[0].text) if result.content else []

            if not due_tasks:
                return

            print(f"[Nexus/Scheduler] Found {len(due_tasks)} task(s) due for notification.")

            for t in due_tasks:
                task_id = t["id"]
                task_text = t["task"]
                due_info = t.get("due_formatted") or t.get("due") or ""

                alert_msg = f"Task Reminder: {task_text}" + (f" (Due: {due_info})" if due_info else "")
                wa_msg = f"Task Reminder: {task_text}\nDue: {due_info}" if due_info else f"Task Reminder: {task_text}"

                # 1. Post notification banner for UI
                try:
                    await todo_session.call_tool("create_notification", {"message": alert_msg})
                except Exception as ne:
                    print(f"[Nexus/Scheduler] Failed to create in-app notification: {ne}")

                # 2. Send live WhatsApp message to phone
                try:
                    wa_res = await todo_session.call_tool("send_whatsapp", {"message": wa_msg})
                    wa_text = wa_res.content[0].text if wa_res.content else str(wa_res)
                    print(f"[Nexus/Scheduler] Sent WhatsApp alert for task {task_id}: {wa_text}")
                except Exception as we:
                    print(f"[Nexus/Scheduler] WhatsApp notification failed for task {task_id}: {we}")

                # 3. Mark as notified so it never triggers again
                try:
                    await todo_session.call_tool("mark_task_notified", {"task_id": task_id})
                except Exception as me:
                    print(f"[Nexus/Scheduler] Failed to mark task {task_id} notified: {me}")

        except Exception as e:
            print(f"[Nexus/Scheduler] Error in check_live_task_reminders: {e}")

