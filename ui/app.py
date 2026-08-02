import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import threading

import streamlit as st
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from streamlit_autorefresh import st_autorefresh

from agent_core.agent import NexusAgent, STATE_CHANGING_TOOLS


class BackgroundLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


@st.cache_resource
def get_agent():
    bg = BackgroundLoop()
    agent = NexusAgent()
    bg.run(agent.connect_servers())

    # Snapshot tool counts once, for the sidebar — avoids a round trip on every rerun
    agent.tool_summary = {}
    for name, session in agent.sessions.items():
        tools = bg.run(session.list_tools())
        agent.tool_summary[name] = [t.name for t in tools.tools]

    async def _start_scheduler():
        scheduler = AsyncIOScheduler(event_loop=asyncio.get_event_loop(), timezone="Asia/Kolkata")
        scheduler.add_job(agent.proactive_check, "cron", hour=8, minute=0)
        scheduler.add_job(agent.send_task_summary_whatsapp, "cron", hour=8, minute=0)
        scheduler.start()

    bg.run(_start_scheduler())
    return agent, bg


agent, bg = get_agent()
st_autorefresh(interval=30_000, key="notif_poll")

st.set_page_config(page_title="Nexus", page_icon="🟣", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🟣 Nexus")
    st.caption("MCP-Powered Personal Productivity Agent")
    st.divider()

    st.markdown("**Connected Tools**")
    for server_name, tools in agent.tool_summary.items():
        with st.expander(f"{server_name} ({len(tools)})"):
            for t in tools:
                st.caption(f"• {t}")

    st.divider()
    try:
        tasks_result = bg.run(agent.sessions["todo"].call_tool("list_tasks", {}))
        tasks = json.loads(tasks_result.content[0].text) if tasks_result.content else []
    except Exception:
        tasks = []
    st.metric("Open tasks", len(tasks))

    try:
        audit_result = bg.run(agent.sessions["todo"].call_tool("get_audit_log", {"limit": 3}))
        recent = json.loads(audit_result.content[0].text) if audit_result.content else []
    except Exception:
        recent = []
    if recent:
        st.markdown("**Recent activity**")
        for entry in recent:
            st.caption(f"`{entry['tool']}` — {entry['time'][11:16]} UTC")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

# ---------- Main chat ----------
st.title("Nexus")
st.caption("Ask it to check your tasks, files, or the web — it plans, asks before acting, and shows its work.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def render_trail(trail):
    if not trail:
        return
    with st.expander("🧠 Reasoning trail", expanded=False):
        for name, args, result in trail:
            st.markdown(f"**→ {name}**`({args})`")
            st.code(str(result), language=None)


for msg in st.session_state.messages:
    avatar = "🟣" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        render_trail(msg.get("trail"))

if notifications_result := None:
    pass  # placeholder to keep structure explicit

try:
    notif_result = bg.run(agent.sessions["todo"].call_tool("list_notifications", {"unseen_only": True}))
    notifications = json.loads(notif_result.content[0].text) if notif_result.content else []
except Exception:
    notifications = []

if notifications:
    for n in notifications:
        st.info(f"🔔 {n['message']}")
    if st.button("Dismiss all"):
        bg.run(agent.sessions["todo"].call_tool("mark_notifications_seen", {}))
        st.rerun()

if st.session_state.pending:
    calls = st.session_state.pending.function_calls
    with st.container(border=True):
        st.markdown("**⚠️ Nexus wants to:**")
        for c in calls:
            st.markdown(f"- `{c.name}`{c.args}")
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm", use_container_width=True):
            with st.spinner("Running approved action..."):
                text, exec_trail = bg.run(agent.execute_calls(st.session_state.pending))
            full_trail = (st.session_state.pending.trail or []) + exec_trail
            st.session_state.messages.append({"role": "assistant", "content": text, "trail": full_trail})
            st.session_state.pending = None
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state.messages.append({"role": "assistant", "content": "Okay, I didn't make any changes."})
            st.session_state.pending = None
            st.rerun()
else:
    if prompt := st.chat_input("Ask Nexus to check your tasks or files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            plan_result = bg.run(agent.plan(prompt))

        if plan_result.function_calls:
            st.session_state.pending = plan_result
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": plan_result.text, "trail": plan_result.trail,
            })
        st.rerun()