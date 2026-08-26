import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp_servers/todo_server.py"],
)

def show(result):
    """Safely print a tool result, whether or not it has content blocks."""
    if result.content:
        print(result.content[0].text)
    elif getattr(result, "structured_content", None) is not None:
        print(result.structured_content)
    else:
        print("[]")

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("Tools this server offers:")
            for t in tools:
                print(" -", t.name)

            print("\nAdding a task...")
            result = await session.call_tool("add_task", {"task": "Buy groceries", "due": "tomorrow"})
            show(result)

            print("\nListing tasks...")
            result = await session.call_tool("list_tasks", {})
            show(result)

            print("\nCompleting the task we just saw above...")
            tasks = await session.call_tool("list_tasks", {})
            # (we already know task 1 exists from your last run — safe to reuse)
            result = await session.call_tool("complete_task", {"task_id": 1})
            show(result)

            print("\nListing tasks again (should be empty)...")
            result = await session.call_tool("list_tasks", {})
            show(result)

asyncio.run(main())