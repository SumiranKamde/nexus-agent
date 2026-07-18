import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from streamlit_autorefresh import st_autorefresh

import asyncio
import threading

import streamlit as st
from agent_core.agent import NexusAgent, STATE_CHANGING_TOOLS


class BackgroundLoop:
    """Runs one asyncio event loop forever in its own thread, so our MCP
    connections stay open across Streamlit's script reruns instead of
    being torn down and rebuilt on every click."""
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

    async def _start_scheduler():
        scheduler = AsyncIOScheduler(event_loop=asyncio.get_event_loop())
        # Demo interval — fires every 2 minutes so you can see it live.
        # For real use, swap to a daily check instead:
        #   scheduler.add_job(agent.proactive_check, "cron", hour=8, minute=0)
        scheduler.add_job(agent.proactive_check, "cron", hour=8, minute=0)  # once daily, 8 AM
        scheduler.start()

    bg.run(_start_scheduler())
    return agent, bg


agent, bg = get_agent()

st.title("🟣 Nexus — Personal Productivity Agent")

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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.pending:
    calls = st.session_state.pending.function_calls
    desc = ", ".join(f"{c.name}({c.args})" for c in calls)
    st.warning(f"Nexus wants to: {desc}")
    col1, col2 = st.columns(2)

    if col1.button("✅ Confirm"):
        with st.spinner("Running approved action..."):
            text, trail = bg.run(agent.execute_calls(st.session_state.pending))
        st.session_state.messages.append({"role": "assistant", "content": text})
        st.session_state.pending = None
        st.rerun()

    if col2.button("❌ Cancel"):
        st.session_state.messages.append({"role": "assistant", "content": "Okay, I didn't make any changes."})
        st.session_state.pending = None
        st.rerun()

else:
    if prompt := st.chat_input("Ask Nexus to check your tasks or files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            plan_result = bg.run(agent.plan(prompt))

        if plan_result.function_calls:
            if any(c.name in STATE_CHANGING_TOOLS for c in plan_result.function_calls):
                st.session_state.pending = plan_result
            else:
                text, trail = bg.run(agent.execute_calls(plan_result))
                st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            st.session_state.messages.append({"role": "assistant", "content": plan_result.text})

        st.rerun()