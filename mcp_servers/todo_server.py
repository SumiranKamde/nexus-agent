from fastmcp import FastMCP
import sqlite3
import os

mcp = FastMCP("todo-server")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "todo.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            due TEXT,
            done INTEGER DEFAULT 0
        )
    """)
    return conn

@mcp.tool()
def add_task(task: str, due: str = "") -> str:
    """Add a new to-do item."""
    conn = get_conn()
    conn.execute("INSERT INTO tasks (task, due) VALUES (?, ?)", (task, due))
    conn.commit()
    conn.close()
    return f"Added: {task}"

@mcp.tool()
def list_tasks() -> list:
    """List all open (not yet completed) to-do items."""
    conn = get_conn()
    rows = conn.execute("SELECT id, task, due FROM tasks WHERE done = 0").fetchall()
    conn.close()
    return [{"id": r[0], "task": r[1], "due": r[2]} for r in rows]

@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a to-do item as done."""
    conn = get_conn()
    conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return f"Task {task_id} marked complete"

if __name__ == "__main__":
    mcp.run()