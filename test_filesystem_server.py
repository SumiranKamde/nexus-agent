import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SANDBOX_PATH = r"C:\Users\skamd\Projects\nexus-agent\sandbox_files"

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
            if read_tool in tool_names:
                file_result = await session.call_tool(read_tool, {"path": SANDBOX_PATH + r"\test.txt"})
                print(f"\n{read_tool} result:")
                print(file_result.content[0].text)

asyncio.run(main())