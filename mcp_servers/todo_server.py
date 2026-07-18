from fastmcp import FastMCP
import sqlite3
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from datetime import datetime, timezone

RISK_LEVELS = {
    "add_task": "low",
    "complete_task": "medium",
}

def log_action(tool_name: str, arguments: dict, result: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log (timestamp, tool_name, arguments, result, risk_level) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), tool_name, json.dumps(arguments), result, RISK_LEVELS.get(tool_name, "low")),
    )
    conn.commit()
    conn.close()

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            result TEXT NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)
    return conn



@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get the current real date and time. Defaults to India Standard Time if no timezone is given."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz)
    return now.strftime("%A, %d %B %Y, %I:%M %p %Z")

@mcp.tool()
def add_task(task: str, due: str = "") -> str:
    """Add a new to-do item."""
    conn = get_conn()
    cursor = conn.execute("INSERT INTO tasks (task, due) VALUES (?, ?)", (task, due))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    result = f"Added: {task} (id {new_id})"
    log_action("add_task", {"task": task, "due": due}, result)
    return result

@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a to-do item as done."""
    conn = get_conn()
    conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    result = f"Task {task_id} marked complete"
    log_action("complete_task", {"task_id": task_id}, result)
    return result

@mcp.tool()
def undo_last_action() -> str:
    """Undo the most recent add_task or complete_task action."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, tool_name, arguments FROM audit_log WHERE tool_name IN ('add_task','complete_task') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        conn.close()
        return "Nothing to undo."

    log_id, tool_name, args_json = row
    args = json.loads(args_json)

    if tool_name == "add_task":
        # Undo an add by deleting the most recently added task with matching text
        conn.execute("DELETE FROM tasks WHERE id = (SELECT id FROM tasks WHERE task = ? ORDER BY id DESC LIMIT 1)", (args["task"],))
        outcome = f"Removed task: {args['task']}"
    elif tool_name == "complete_task":
        conn.execute("UPDATE tasks SET done = 0 WHERE id = ?", (args["task_id"],))
        outcome = f"Task {args['task_id']} marked incomplete again"
    else:
        outcome = "Nothing to undo."

    conn.execute("DELETE FROM audit_log WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return outcome

@mcp.tool()
def get_audit_log(limit: int = 10) -> list:
    """View the most recent logged actions, most recent first."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT timestamp, tool_name, arguments, result, risk_level FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"time": r[0], "tool": r[1], "args": r[2], "result": r[3], "risk": r[4]} for r in rows]
@mcp.tool()
def list_tasks() -> list:
    """List all open (not yet completed) to-do items."""
    conn = get_conn()
    rows = conn.execute("SELECT id, task, due FROM tasks WHERE done = 0").fetchall()
    conn.close()
    return [{"id": r[0], "task": r[1], "due": r[2]} for r in rows]


if __name__ == "__main__":
    mcp.run()