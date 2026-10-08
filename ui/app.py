import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import base64
import concurrent.futures
import html
import json
import threading

import streamlit as st
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from streamlit_autorefresh import st_autorefresh

from agent_core.agent import NexusAgent, FileAttachment, IMAGE_MIME_TYPES, PDF_MIME_TYPE

# ── Config ──────────────────────────────────────────────────────
SCHEDULER_TIMEZONE = "Asia/Kolkata"
PROACTIVE_CHECK_HOUR, PROACTIVE_CHECK_MINUTE = 8, 0
TASK_SUMMARY_HOUR, TASK_SUMMARY_MINUTE = 8, 15
PLAN_TIMEOUT_SECONDS = 120
UI_TIMEOUT_SECONDS = 20

ACTION_LABELS = {
    "add_task": ("Add task", "Create a new to-do item"),
    "complete_task": ("Complete task", "Mark a task as done"),
    "delete_task": ("Delete task", "Remove a task permanently"),
    "write_file": ("Write file", "Create or overwrite a file"),
    "edit_file": ("Edit file", "Modify an existing file"),
    "move_file": ("Move file", "Move or rename a file"),
    "delete_file": ("Delete file", "Remove a file"),
    "delete_directory": ("Delete folder", "Remove a directory"),
    "create_directory": ("Create folder", "Make a new directory"),
    "send_whatsapp": ("Send WhatsApp", "Message your phone"),
    "send_file_via_whatsapp": ("Send file via WhatsApp", "Send an attachment to your phone"),
    "remember": ("Save memory", "Remember a fact about you"),
    "forget": ("Forget memory", "Remove a remembered fact"),
    "create_notification": ("Create alert", "Post a notification"),
    "mark_notifications_seen": ("Dismiss alerts", "Mark notifications read"),
    "undo_last_action": ("Undo", "Revert the last change"),
    "mark_task_notified": ("Mark notified", "Record that task reminder was dispatched"),
}


# ── Background event loop ──────────────────────────────────────
class BackgroundLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout=UI_TIMEOUT_SECONDS):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Timed out after {timeout}s.")


@st.cache_resource
def get_agent():
    bg = BackgroundLoop()
    agent = NexusAgent()
    bg.run(agent.connect_servers(), timeout=90)

    agent.tool_summary = {}
    for name, session in agent.sessions.items():
        tools = bg.run(session.list_tools())
        agent.tool_summary[name] = [t.name for t in tools.tools]

    async def _start_scheduler():
        scheduler = AsyncIOScheduler(
            event_loop=asyncio.get_running_loop(), timezone=SCHEDULER_TIMEZONE
        )
        scheduler.add_job(
            agent.proactive_check, "cron",
            hour=PROACTIVE_CHECK_HOUR, minute=PROACTIVE_CHECK_MINUTE,
            id="proactive_check", coalesce=True, misfire_grace_time=3600,
        )
        scheduler.add_job(
            agent.send_task_summary_whatsapp, "cron",
            hour=TASK_SUMMARY_HOUR, minute=TASK_SUMMARY_MINUTE,
            id="task_summary", coalesce=True, misfire_grace_time=3600,
        )
        remind_fn = getattr(agent, "check_live_task_reminders", None)
        if not remind_fn and hasattr(NexusAgent, "check_live_task_reminders"):
            remind_fn = NexusAgent.check_live_task_reminders.__get__(agent, NexusAgent)
        if remind_fn:
            scheduler.add_job(
                remind_fn, "interval",
                minutes=1, id="live_task_reminders",
                coalesce=True, misfire_grace_time=60,
            )
        scheduler.start()
        return scheduler

    agent.scheduler = bg.run(_start_scheduler())
    for job in agent.scheduler.get_jobs():
        print(f"[Nexus] Scheduled '{job.id}' -- next run {job.next_run_time}")

    return agent, bg


agent, bg = get_agent()
st.set_page_config(page_title="Nexus", page_icon="N", layout="wide")
st_autorefresh(interval=30_000, key="notif_poll")


# ── Minimal Futuristic Design System ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #09090B;
    --surface: #111113;
    --surface-2: #18181B;
    --border: #27272A;
    --border-subtle: #1E1E22;
    --text: #FAFAFA;
    --text-muted: #A1A1AA;
    --text-dim: #71717A;
    --accent: #22C55E;
    --amber: #F59E0B;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272A; border-radius: 99px; }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0C0C0E;
    border-right: 1px solid var(--border);
}

/* Tabs */
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-dim);
    padding: 6px 12px;
    border-radius: 4px 4px 0 0;
}
[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
    color: var(--text);
    background: var(--surface);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
}

/* Chat Input */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3F3F46;
}

/* Buttons */
[data-testid="stButton"] button {
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    transition: background 0.12s ease, border-color 0.12s ease;
}
[data-testid="stButton"] button:hover {
    background: var(--surface-2);
    border-color: #3F3F46;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* Action Cards */
.qa-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.qa-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.qa-card-desc {
    font-size: 0.76rem;
    color: var(--text-dim);
    line-height: 1.4;
}

/* Approval Panel */
.approval-box {
    background: var(--surface);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 1rem 1.15rem;
    margin: 0.8rem 0;
}

/* Metric Boxes */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.55rem 0.85rem;
}

/* Status Dot */
.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.dot-green {
    background: var(--accent);
}
</style>
""", unsafe_allow_html=True)


# ── Fast Parallel Data Fetch ────────────────────────────────────
try:
    dashboard_data = bg.run(agent.get_dashboard_data(), timeout=15)
except Exception:
    dashboard_data = {"memories": [], "tasks": [], "recent_activity": [], "notifications": []}

memories = dashboard_data.get("memories", [])
tasks = dashboard_data.get("tasks", [])
recent_activity = dashboard_data.get("recent_activity", [])
notifications = dashboard_data.get("notifications", [])
total_tools = sum(len(v) for v in agent.tool_summary.values())


# ── Session State Initialization ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "scheduled_prompt" not in st.session_state:
    st.session_state.scheduled_prompt = None
if "scheduled_attachments" not in st.session_state:
    st.session_state.scheduled_attachments = None
if "scheduled_files_meta" not in st.session_state:
    st.session_state.scheduled_files_meta = None


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 0.4rem;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #FAFAFA; letter-spacing: -0.02em;">NEXUS</div>
        <div style="font-size: 0.74rem; color: #71717A;">Autonomous Workspace Agent</div>
    </div>
    <div style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #71717A; margin-bottom: 1rem;">
        <span class="dot dot-green"></span> Online &amp; Ready
    </div>
    """, unsafe_allow_html=True)

    tab_tasks, tab_tools, tab_memory = st.tabs(["Tasks", "Tools", "Memory"])

    # ── Tasks Tab ──
    with tab_tasks:
        if tasks:
            for t in tasks:
                title = str(t.get("task", t.get("title", t)))
                task_id = t.get("id", "")
                due = t.get("due", "")
                c_title, c_btn = st.columns([4, 2])
                with c_title:
                    due_badge = f" <span style='font-size:0.7rem;color:#F59E0B;font-family:monospace;'>[{html.escape(due)}]</span>" if due else ""
                    st.markdown(f"<div style='font-size:0.8rem;color:#FAFAFA;padding:4px 0;'>{html.escape(title)}{due_badge}</div>", unsafe_allow_html=True)
                with c_btn:
                    if st.button("Done", key=f"done_{task_id}_{title[:8]}", use_container_width=True):
                        try:
                            tid = int(task_id) if str(task_id).isdigit() else 0
                            bg.run(agent.sessions["todo"].call_tool("complete_task", {"task_id": tid, "task": title}))
                        except Exception as e:
                            st.error(f"Error completing task: {e}")
                        st.rerun()
        else:
            st.caption("No pending tasks.")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        with st.form("sidebar_add_task_form", clear_on_submit=True):
            in_task = st.text_input("New task", placeholder="Add to-do item...", label_visibility="collapsed")
            if st.form_submit_button("Add Task", use_container_width=True) and in_task.strip():
                try:
                    bg.run(agent.sessions["todo"].call_tool("add_task", {"task": in_task.strip()}))
                except Exception as e:
                    st.error(f"Error adding task: {e}")
                st.rerun()

    # ── Tools Tab ──
    with tab_tools:
        for sname, stools in agent.tool_summary.items():
            st.caption(f"{sname.upper()} ({len(stools)})")
            with st.expander(f"View {sname}", expanded=False):
                for tn in stools:
                    st.code(tn, language=None)

    # ── Memory Tab ──
    with tab_memory:
        if memories:
            for m in memories:
                fact = m.get("fact", "")
                mid = m.get("id", "")
                c_fact, c_del = st.columns([4, 2])
                with c_fact:
                    st.markdown(f"<div style='font-size:0.8rem;color:#FAFAFA;padding:4px 0;'>{html.escape(fact)}</div>", unsafe_allow_html=True)
                with c_del:
                    if st.button("Forget", key=f"forget_{mid}", use_container_width=True):
                        try:
                            mid_int = int(mid) if str(mid).isdigit() else 0
                            bg.run(agent.sessions["todo"].call_tool("forget", {"memory_id": mid_int}))
                            agent._cached_memories = None
                        except Exception as e:
                            st.error(f"Error forgetting memory: {e}")
                        st.rerun()
        else:
            st.caption("No saved memories.")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        with st.form("sidebar_add_mem_form", clear_on_submit=True):
            in_mem = st.text_input("New memory", placeholder="Remember preference...", label_visibility="collapsed")
            if st.form_submit_button("Save Memory", use_container_width=True) and in_mem.strip():
                try:
                    bg.run(agent.sessions["todo"].call_tool("remember", {"fact": in_mem.strip()}))
                    agent._cached_memories = None
                except Exception as e:
                    st.error(f"Error saving memory: {e}")
                st.rerun()

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    has_user_msg = any(m.get("role") == "user" for m in st.session_state.messages)
    c_clr, c_rty = st.columns(2) if has_user_msg else (st.container(), None)
    with c_clr:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending = None
            st.session_state.scheduled_prompt = None
            st.session_state.scheduled_attachments = None
            st.session_state.scheduled_files_meta = None
            st.rerun()
    if c_rty is not None:
        with c_rty:
            if st.button("🔄 Retry last", use_container_width=True, help="Re-run the last prompt"):
                last_u = None
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    if st.session_state.messages[i].get("role") == "user":
                        last_u = st.session_state.messages.pop(i)
                        break
                if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
                    st.session_state.messages.pop()
                st.session_state.pending = None
                if last_u:
                    st.session_state.scheduled_prompt = last_u.get("content")
                    st.session_state.scheduled_attachments = last_u.get("raw_attachments")
                    st.session_state.scheduled_files_meta = last_u.get("files")
                st.rerun()


# ── Render Helpers ─────────────────────────────────────────────
def _render_attachments(files_meta):
    if not files_meta:
        return
    cols = st.columns(min(len(files_meta), 4))
    for i, fm in enumerate(files_meta):
        mime = fm.get("mime_type", "")
        name = fm.get("name", "file")
        with cols[i % len(cols)]:
            if mime in IMAGE_MIME_TYPES and fm.get("data_b64"):
                st.image(f"data:{mime};base64,{fm['data_b64']}", caption=name, width=160)
            else:
                size_kb = fm.get("size", 0) / 1024
                st.caption(f"[doc] {name} ({size_kb:.0f} KB)")


def _render_trace(trail):
    if not trail:
        return
    with st.expander(f"Execution trace ({len(trail)} steps)", expanded=False):
        for i, item in enumerate(trail, 1):
            name = item[0]
            args = item[1] if len(item) > 1 else {}
            result = item[2] if len(item) > 2 else ""
            st.markdown(f"**{i}.** `{name}`")
            if args:
                st.json(args, expanded=False)
            if result:
                if isinstance(result, (dict, list)):
                    st.json(result, expanded=False)
                else:
                    try:
                        st.json(json.loads(str(result)), expanded=False)
                    except Exception:
                        r = str(result)
                        st.code(r[:1000] + ("..." if len(r) > 1000 else ""), language=None)


# ── Notifications ──────────────────────────────────────────────
if notifications:
    for n in notifications:
        st.info(n["message"])
    if st.button("Dismiss notifications", key="dismiss_notifs"):
        try:
            bg.run(agent.sessions["todo"].call_tool("mark_notifications_seen", {}))
        except Exception:
            pass
        st.rerun()


# ── Message History ────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("files"):
            _render_attachments(msg["files"])
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("trail"):
            _render_trace(msg["trail"])

        # Retry option for every prompt's assistant response
        if msg.get("role") == "assistant" and not st.session_state.pending:
            prev_user_idx = None
            for p_i in range(idx - 1, -1, -1):
                if st.session_state.messages[p_i].get("role") == "user":
                    prev_user_idx = p_i
                    break
            if prev_user_idx is not None:
                r_col, _ = st.columns([1.5, 8])
                with r_col:
                    if st.button("🔄 Retry", key=f"retry_turn_{idx}", help="Re-run this prompt"):
                        target_user = st.session_state.messages[prev_user_idx]
                        st.session_state.scheduled_prompt = target_user.get("content")
                        st.session_state.scheduled_attachments = target_user.get("raw_attachments")
                        st.session_state.scheduled_files_meta = target_user.get("files")
                        st.session_state.messages.pop(idx)
                        st.session_state.messages.pop(prev_user_idx)
                        st.rerun()


# ── Empty State: Dashboard + Quick Actions ─────────────────────
if not st.session_state.messages and not st.session_state.pending:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open Tasks", len(tasks))
    m2.metric("Saved Memories", len(memories))
    m3.metric("Available Tools", total_tools)
    m4.metric("Active Servers", len(agent.tool_summary))

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.84rem;font-weight:600;color:#A1A1AA;margin-bottom:0.6rem;'>Quick Actions</div>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Review Tasks</div><div class="qa-card-desc">Inspect pending items with due dates &amp; priority</div></div>', unsafe_allow_html=True)
        if st.button("Run Review", key="qa_tasks", use_container_width=True):
            st.session_state.scheduled_prompt = "List all my pending tasks with priority and due dates."
            st.rerun()
    with r1c2:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Web Research</div><div class="qa-card-desc">Search the live internet via Brave Search</div></div>', unsafe_allow_html=True)
        if st.button("Run Search", key="qa_search", use_container_width=True):
            st.session_state.scheduled_prompt = "Search the web for the latest tech and AI news today."
            st.rerun()
    with r1c3:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Browse Files</div><div class="qa-card-desc">Inspect sandbox workspace files and directory tree</div></div>', unsafe_allow_html=True)
        if st.button("Browse Workspace", key="qa_files", use_container_width=True):
            st.session_state.scheduled_prompt = "Show me the directory tree of my sandbox workspace."
            st.rerun()

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Daily Digest</div><div class="qa-card-desc">Get a morning briefing of open tasks and audit history</div></div>', unsafe_allow_html=True)
        if st.button("Generate Digest", key="qa_digest", use_container_width=True):
            st.session_state.scheduled_prompt = "Give me a daily digest: summarize open tasks, recent audit log actions, and my memory preferences."
            st.rerun()
    with r2c2:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Undo Last Action</div><div class="qa-card-desc">Roll back the most recent state change</div></div>', unsafe_allow_html=True)
        if st.button("Undo Change", key="qa_undo", use_container_width=True):
            st.session_state.scheduled_prompt = "Undo the last action that was performed."
            st.rerun()
    with r2c3:
        st.markdown('<div class="qa-card"><div class="qa-card-title">Send WhatsApp</div><div class="qa-card-desc">Dispatch a quick message directly to your phone</div></div>', unsafe_allow_html=True)
        with st.form("qa_wa_form", clear_on_submit=True):
            wa_text = st.text_input("Message text", placeholder="Type message to phone...", label_visibility="collapsed")
            if st.form_submit_button("Send to Phone", use_container_width=True) and wa_text.strip():
                st.session_state.scheduled_prompt = f"Send this WhatsApp message to me: {wa_text.strip()}"
                st.rerun()


# ── Action Approval ────────────────────────────────────────────
if st.session_state.scheduled_prompt and st.session_state.pending:
    st.session_state.pending = None

if st.session_state.pending:
    calls = st.session_state.pending.function_calls

    st.markdown("""
    <div class="approval-box">
        <div style="font-size:0.88rem;font-weight:600;color:#F59E0B;margin-bottom:0.25rem;">Action Confirmation Required</div>
        <div style="font-size:0.82rem;color:#A1A1AA;">Nexus needs your approval before performing the following changes:</div>
    </div>
    """, unsafe_allow_html=True)

    for i, call in enumerate(calls, 1):
        name = str(call.name)
        label, desc = ACTION_LABELS.get(name, (name, ""))
        st.markdown(f"**{i}. {label}** -- *{desc}*")
        if call.args:
            st.json(call.args, expanded=True)

    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("Confirm", type="primary", use_container_width=True):
            try:
                with st.spinner("Executing..."):
                    text, trail = bg.run(
                        agent.execute_calls(st.session_state.pending),
                        timeout=PLAN_TIMEOUT_SECONDS,
                    )
            except TimeoutError as e:
                text, trail = f"Execution timed out ({e}).", st.session_state.pending.trail
            except Exception as e:
                text, trail = f"Execution encountered an error: {e}", st.session_state.pending.trail
            st.session_state.messages.append({"role": "assistant", "content": text, "trail": trail})
            st.session_state.pending = None
            st.rerun()
    with c2:
        if st.button("Cancel", type="secondary", use_container_width=True):
            st.session_state.messages.append({"role": "assistant", "content": "Action cancelled. No changes were made."})
            st.session_state.pending = None
            st.rerun()
    with c3:
        if st.button("🔄 Retry", type="secondary", use_container_width=True, help="Cancel plan and re-run this prompt"):
            last_u = None
            for p_i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[p_i].get("role") == "user":
                    last_u = st.session_state.messages.pop(p_i)
                    break
            st.session_state.pending = None
            if last_u:
                st.session_state.scheduled_prompt = last_u.get("content")
                st.session_state.scheduled_attachments = last_u.get("raw_attachments")
                st.session_state.scheduled_files_meta = last_u.get("files")
            st.rerun()


# ── Chat Input & Agent Dispatch ────────────────────────────────
else:
    chat_val = st.chat_input(
        "Ask Nexus to check tasks, files, the web, or attach documents...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp", "gif", "pdf"],
    )

    scheduled = st.session_state.scheduled_prompt
    st.session_state.scheduled_prompt = None

    prompt = None
    files_in = []

    if chat_val:
        if hasattr(chat_val, "text"):
            prompt = chat_val.text
            files_in = chat_val.files or []
        elif isinstance(chat_val, dict):
            prompt = chat_val.get("text", "")
            files_in = chat_val.get("files", [])
        else:
            prompt = str(chat_val)
    elif scheduled:
        prompt = scheduled

    if prompt:
        attachments = []
        files_meta = []
        if st.session_state.scheduled_attachments:
            attachments = st.session_state.scheduled_attachments
            st.session_state.scheduled_attachments = None
            if st.session_state.scheduled_files_meta:
                files_meta = st.session_state.scheduled_files_meta
                st.session_state.scheduled_files_meta = None
        else:
            for uf in files_in:
                data = uf.getvalue()
                attachments.append(FileAttachment(filename=uf.name, mime_type=uf.type, data=data))
                meta = {"name": uf.name, "mime_type": uf.type, "size": len(data)}
                if uf.type in IMAGE_MIME_TYPES and len(data) < 500_000:
                    meta["data_b64"] = base64.b64encode(data).decode("ascii")
                files_meta.append(meta)

        history = list(st.session_state.messages)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "files": files_meta or None,
            "raw_attachments": attachments or None,
        })

        try:
            with st.spinner("Reasoning..."):
                result = bg.run(
                    agent.plan(prompt, history, files=attachments or None),
                    timeout=PLAN_TIMEOUT_SECONDS,
                )
        except TimeoutError as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Request timed out ({e}). Please try again."})
            st.rerun()
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred while processing: {e}"})
            st.rerun()

        if result.function_calls:
            st.session_state.pending = result
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": result.text, "trail": result.trail,
            })
        st.rerun()
