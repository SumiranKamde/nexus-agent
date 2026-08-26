"""Manual check: does the to-do MCP server round-trip a task?

    python tests/manual/check_todo_server.py

WRITES TO memory/todo.db — it adds a task, completes it, then undoes the
completion so the database is left as it was found.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_core.agent import TODO_SERVER_PATH

server_params = StdioServerParameters(command="python", args=[TODO_SERVER_PATH])


def show(result):
    """Safely print a tool result, whether or not it has content blocks."""
    if result.content:
        print(result.content[0].text)
    elif getattr(result, "structured_content", None) is not None:
        print(result.structured_content)
    else:
        print("[]")
    return result


def payload(result):
    return json.loads(result.content[0].text) if result.content else []


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("Tools this server offers:")
            for t in tools:
                print(" -", t.name)

            print("\nAdding a task...")
            show(await session.call_tool(
                "add_task", {"task": "[check] round-trip probe", "due": "tomorrow"}
            ))

            print("\nListing tasks...")
            tasks = payload(show(await session.call_tool("list_tasks", {})))

            probe = next((t for t in tasks if t["task"] == "[check] round-trip probe"), None)
            if probe is None:
                print("\nCouldn't find the task we just added — stopping.")
                return

            print(f"\nCompleting task {probe['id']}...")
            show(await session.call_tool("complete_task", {"task_id": probe["id"]}))

            print("\nListing tasks again (the probe should be gone)...")
            show(await session.call_tool("list_tasks", {}))

            print("\nUndoing, so the database is left as we found it...")
            show(await session.call_tool("undo_last_action", {}))
            show(await session.call_tool("undo_last_action", {}))


asyncio.run(main())
