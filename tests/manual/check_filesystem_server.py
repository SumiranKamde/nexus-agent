"""Manual check: does the Filesystem MCP server come up and see the sandbox?

    python tests/manual/check_filesystem_server.py

Read-only — lists the sandbox and reads one file out of it.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_core.agent import SANDBOX_PATH

# On Windows, npx must be launched through cmd /c — calling it directly
# from Python fails with a "file not found" style error.
server_params = StdioServerParameters(
    command="cmd",
    args=["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", SANDBOX_PATH],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("Tools this server offers:")
            for t in tools:
                print(" -", t.name)

            tool_names = [t.name for t in tools]

            list_tool = "list_directory" if "list_directory" in tool_names else tool_names[0]
            result = await session.call_tool(list_tool, {"path": SANDBOX_PATH})
            print(f"\n{list_tool} result:")
            print(result.content[0].text)

            read_tool = "read_text_file" if "read_text_file" in tool_names else "read_file"
            sandbox_files = sorted(p.name for p in Path(SANDBOX_PATH).glob("*.txt"))
            if read_tool in tool_names and sandbox_files:
                target = str(Path(SANDBOX_PATH) / sandbox_files[0])
                file_result = await session.call_tool(read_tool, {"path": target})
                print(f"\n{read_tool} on {sandbox_files[0]}:")
                print(file_result.content[0].text)
            else:
                print("\nNo .txt files in the sandbox to read.")


asyncio.run(main())
