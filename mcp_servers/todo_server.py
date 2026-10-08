from fastmcp import FastMCP
import sqlite3
import os
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from twilio.rest import Client as TwilioClient

load_dotenv()

RISK_LEVELS = {
    "add_task": "low",
    "complete_task": "medium",
    "delete_task": "medium",
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

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            due TEXT,
            done INTEGER DEFAULT 0,
            notified INTEGER DEFAULT 0
        )
    """)
    try:
        conn.execute("ALTER TABLE tasks ADD COLUMN notified INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            seen INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_done_notified ON tasks(done, notified)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_seen ON notifications(seen)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_tool ON audit_log(tool_name, id DESC)")
    conn.commit()
    conn.close()

init_db()


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _parse_due_time(due_str: str, tz: ZoneInfo) -> datetime | None:
    """Parse common date/time formats into an aware datetime in timezone tz."""
    if not due_str or not due_str.strip():
        return None
    raw = due_str.strip()
    s = raw.lower()
    now = datetime.now(tz)
    base_date = now.date()

    if "tomorrow" in s:
        base_date = now.date() + timedelta(days=1)
        s = s.replace("tomorrow", "").strip()
    elif "today" in s:
        base_date = now.date()
        s = s.replace("today", "").strip()

    if s.startswith("at "):
        s = s[3:].strip()
    s = s.replace(" at ", " ").strip()

    candidate_strings = [s, raw]
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M%p",
        "%Y-%m-%d %I %p",
        "%Y-%m-%d %I%p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %I:%M %p",
        "%d/%m/%Y %I:%M %p",
        "%Y-%m-%d",
    ]
    for c in candidate_strings:
        if not c:
            continue
        for fmt in formats:
            try:
                dt = datetime.strptime(c, fmt)
                if fmt == "%Y-%m-%d":
                    dt = dt.replace(hour=23, minute=59, second=59)
                return dt.replace(tzinfo=tz)
            except ValueError:
                pass

    time_formats = [
        "%H:%M:%S",
        "%H:%M",
        "%I:%M %p",
        "%I:%M%p",
        "%I %p",
        "%I%p",
    ]
    for c in candidate_strings:
        if not c:
            continue
        for fmt in time_formats:
            try:
                t = datetime.strptime(c.upper(), fmt).time()
                return datetime.combine(base_date, t, tzinfo=tz)
            except ValueError:
                pass

    return None


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
    cursor = conn.execute("INSERT INTO tasks (task, due, notified) VALUES (?, ?, 0)", (task, due))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    result = f"Added: {task} (id {new_id})" + (f" [due: {due}]" if due else "")
    log_action("add_task", {"task": task, "due": due, "task_id": new_id}, result)
    return result

@mcp.tool()
def complete_task(task_id: int = 0, task: str = "") -> str:
    """Mark a to-do item as done by its ID or matching description."""
    conn = get_conn()
    target_id = None
    target_task = ""

    try:
        task_id = int(task_id) if task_id else 0
    except (ValueError, TypeError):
        task_id = 0

    if task_id > 0:
        row = conn.execute("SELECT id, task FROM tasks WHERE id = ? AND done = 0", (task_id,)).fetchone()
        if row:
            target_id, target_task = row[0], row[1]

    if not target_id and task:
        clean_name = task.strip().lower()
        rows = conn.execute("SELECT id, task FROM tasks WHERE done = 0").fetchall()
        for rid, rtask in rows:
            if clean_name == rtask.strip().lower() or clean_name in rtask.lower() or rtask.lower() in clean_name:
                target_id, target_task = rid, rtask
                break

    if not target_id:
        conn.close()
        ident = f"id {task_id}" if task_id else f"description '{task}'"
        return f"No open task found with {ident}"

    conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (target_id,))
    conn.commit()
    conn.close()
    result = f"Task {target_id} ('{target_task}') marked complete"
    log_action("complete_task", {"task_id": target_id, "task": target_task}, result)
    return result

@mcp.tool()
def delete_task(task_id: int = 0, task: str = "") -> str:
    """Delete a to-do item permanently by its ID or matching description."""
    conn = get_conn()
    target_id = None
    target_task = ""
    target_due = ""

    try:
        task_id = int(task_id) if task_id else 0
    except (ValueError, TypeError):
        task_id = 0

    if task_id > 0:
        row = conn.execute("SELECT id, task, due FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row:
            target_id, target_task, target_due = row[0], row[1], row[2] or ""

    if not target_id and task:
        clean_name = task.strip().lower()
        rows = conn.execute("SELECT id, task, due FROM tasks").fetchall()
        for rid, rtask, rdue in rows:
            if clean_name == rtask.strip().lower() or clean_name in rtask.lower() or rtask.lower() in clean_name:
                target_id, target_task, target_due = rid, rtask, rdue or ""
                break

    if not target_id:
        conn.close()
        ident = f"id {task_id}" if task_id else f"description '{task}'"
        return f"No task found with {ident}"

    conn.execute("DELETE FROM tasks WHERE id = ?", (target_id,))
    conn.commit()
    conn.close()
    result = f"Task {target_id} ('{target_task}') deleted permanently"
    log_action("delete_task", {"task_id": target_id, "task": target_task, "due": target_due}, result)
    return result

@mcp.tool()
def undo_last_action() -> str:
    """Undo the most recent add_task, complete_task, or delete_task action."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, tool_name, arguments FROM audit_log WHERE tool_name IN ('add_task','complete_task','delete_task') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        conn.close()
        return "Nothing to undo."

    log_id, tool_name, args_json = row
    args = json.loads(args_json)

    if tool_name == "add_task":
        task_id = args.get("task_id")
        if task_id is None:
            conn.close()
            return "This older task action cannot be safely undone."
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (int(task_id),))
        outcome = f"Removed task {task_id}" if cursor.rowcount else f"Task {task_id} no longer exists"
    elif tool_name == "complete_task":
        task_id = args.get("task_id")
        if task_id is not None:
            conn.execute("UPDATE tasks SET done = 0 WHERE id = ?", (int(task_id),))
            outcome = f"Task {task_id} marked incomplete again"
        else:
            outcome = "Missing task_id to undo completion."
    elif tool_name == "delete_task":
        task_text = args.get("task", "")
        task_due = args.get("due", "")
        if task_text:
            cursor = conn.execute("INSERT INTO tasks (task, due, done, notified) VALUES (?, ?, 0, 0)", (task_text, task_due))
            restored_id = cursor.lastrowid
            outcome = f"Restored deleted task '{task_text}' (new id {restored_id})"
        else:
            outcome = "Missing details to restore deleted task."
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
    rows = conn.execute("SELECT id, task, due, notified FROM tasks WHERE done = 0").fetchall()
    conn.close()
    return [{"id": r[0], "task": r[1], "due": r[2], "notified": bool(r[3])} for r in rows]

@mcp.tool()
def get_due_tasks(timezone_name: str = "Asia/Kolkata") -> list:
    """List open tasks that are due for notification right now (due time has arrived and not yet notified)."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz)

    conn = get_conn()
    rows = conn.execute("SELECT id, task, due FROM tasks WHERE done = 0 AND (notified IS NULL OR notified = 0)").fetchall()
    conn.close()

    due_tasks = []
    for r in rows:
        task_id, task_text, due_str = r[0], r[1], r[2]
        if not due_str:
            continue
        due_dt = _parse_due_time(due_str, tz)
        if due_dt and due_dt <= now:
            due_tasks.append({
                "id": task_id,
                "task": task_text,
                "due": due_str,
                "due_formatted": due_dt.strftime("%d %b %Y, %I:%M %p"),
            })
    return due_tasks

@mcp.tool()
def mark_task_notified(task_id: int) -> str:
    """Mark a task as having had its live due-time notification sent."""
    conn = get_conn()
    cursor = conn.execute("UPDATE tasks SET notified = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        return f"Task {task_id} not found"
    return f"Task {task_id} marked as notified"

@mcp.tool()
def remember(fact: str) -> str:
    """Save a lasting fact or preference about the user for future conversations."""
    conn = get_conn()
    conn.execute("INSERT INTO memories (fact, created_at) VALUES (?, ?)", (fact, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    result = f"Remembered: {fact}"
    log_action("remember", {"fact": fact}, result)
    return result

@mcp.tool()
def list_memories() -> list:
    """List everything currently remembered about the user, with each memory's id."""
    conn = get_conn()
    rows = conn.execute("SELECT id, fact, created_at FROM memories ORDER BY id").fetchall()
    conn.close()
    return [{"id": r[0], "fact": r[1], "since": r[2]} for r in rows]

@mcp.tool()
def forget(memory_id: int) -> str:
    """Delete a specific remembered fact by its id (use list_memories first to find the id)."""
    conn = get_conn()
    cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    if cursor.rowcount == 0:
        return f"Memory {memory_id} was not found"
    result = f"Forgot memory {memory_id}"
    log_action("forget", {"memory_id": memory_id}, result)
    return result

@mcp.tool()
def create_notification(message: str) -> str:
    """Create a proactive notification for the user to see in the UI."""
    conn = get_conn()
    conn.execute("INSERT INTO notifications (message, created_at) VALUES (?, ?)", (message, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return f"Notification created: {message}"

@mcp.tool()
def list_notifications(unseen_only: bool = True) -> list:
    """List notifications. By default, only unseen ones."""
    conn = get_conn()
    query = "SELECT id, message, created_at FROM notifications" + (" WHERE seen = 0" if unseen_only else "")
    rows = conn.execute(query).fetchall()
    conn.close()
    return [{"id": r[0], "message": r[1], "created_at": r[2]} for r in rows]

@mcp.tool()
def mark_notifications_seen() -> str:
    """Mark all notifications as seen/dismissed."""
    conn = get_conn()
    cursor = conn.execute("UPDATE notifications SET seen = 1 WHERE seen = 0")
    conn.commit()
    conn.close()
    return f"Marked {cursor.rowcount} notification(s) as seen"

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_WHATSAPP_FROM")
TWILIO_TO = os.environ.get("TWILIO_WHATSAPP_TO")

@mcp.tool()
def send_whatsapp(message: str) -> str:
    """Send a WhatsApp message to the user's phone via Twilio."""
    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        return "WhatsApp is not configured — missing Twilio credentials in .env"
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
        result = f"WhatsApp message sent (sid: {msg.sid})"
    except Exception as e:
        result = f"Failed to send WhatsApp message: {e}"
    log_action("send_whatsapp", {"message": message}, result)
    return result

SANDBOX_PATH = os.path.join(os.path.dirname(__file__), "..", "sandbox_files")

@mcp.tool()
def send_file_via_whatsapp(file_name: str) -> str:
    """Read a text file from the sandbox folder and send its contents to the user's WhatsApp."""
    file_path = os.path.join(SANDBOX_PATH, file_name)
    sandbox_root = os.path.realpath(SANDBOX_PATH)
    resolved_path = os.path.realpath(file_path)
    if os.path.commonpath([sandbox_root, resolved_path]) != sandbox_root:
        return "Refused: file must be inside the sandbox folder."
    if not os.path.isfile(resolved_path):
        return f"File not found: {file_name}"
    try:
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Could not read file: {e}"

    original_len = len(content)
    prefix = f"\U0001f4c4 {file_name}:\n\n"
    suffix = "\n...(truncated, file exceeds WhatsApp's 1600-character message limit)"
    max_total = 1550  # safety margin below Twilio's hard 1600-char limit
    max_content_len = max_total - len(prefix) - len(suffix)

    truncated = original_len > max_content_len
    if truncated:
        content = content[:max_content_len] + suffix
    message = prefix + content

    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO]):
        return "WhatsApp is not configured — missing Twilio credentials in .env"
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
        note = " (truncated to fit WhatsApp's limit)" if truncated else ""
        result = f"Sent {file_name} via WhatsApp (sid: {msg.sid}){note}"
    except Exception as e:
        result = f"Failed to send file via WhatsApp: {e}"
    log_action("send_file_via_whatsapp", {"file_name": file_name}, result)
    return result

if __name__ == "__main__":
    mcp.run()
