import asyncio
import json
import os
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANDBOX_PATH = os.path.join(PROJECT_ROOT, "sandbox_files")
TODO_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_servers", "todo_server.py")
STATE_CHANGING_TOOLS = {
    "write_file", "edit_file", "create_directory", "move_file",  # filesystem
    "add_task", "complete_task",                                  # todo
}


class NexusAgent:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.sessions = {}
        self._stack = AsyncExitStack()

    async def plan(self, user_message: str):
        """Ask Gemini what it wants to do, but don't execute anything yet."""
        all_tools, tool_owner = await self._merged_tools()
        gemini_tools = types.Tool(function_declarations=all_tools)
        contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

        response = await self.gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0, tools=[gemini_tools]),
        )
        contents.append(response.candidates[0].content)
        return response, contents, tool_owner

    async def execute_calls(self, function_calls, tool_owner, contents):
        """Actually run approved tool calls, then get one final summarizing reply."""
        tool_response_parts = []
        trail = []
        for fc in function_calls:
            args = fc.args or {}
            result = await tool_owner[fc.name].call_tool(fc.name, args)
            result_text = result.content[0].text if result.content else "[]"
            trail.append((fc.name, args, result_text))
            tool_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result_text})
            )
        contents.append(types.Content(role="user", parts=tool_response_parts))

        response = await self.gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0),  # no tools this round — force a plain-text reply
        )
        return response.text, trail

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