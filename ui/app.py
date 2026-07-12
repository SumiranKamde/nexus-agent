import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    return agent, bg


agent, bg = get_agent()

st.title("🟣 Nexus — Personal Productivity Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.pending:
    calls = st.session_state.pending["function_calls"]
    desc = ", ".join(f"{fc.name}({dict(fc.args or {})})" for fc in calls)
    st.warning(f"Nexus wants to: {desc}")
    col1, col2 = st.columns(2)

    if col1.button("✅ Confirm"):
        pending = st.session_state.pending
        with st.spinner("Running approved action..."):
            text, trail = bg.run(agent.execute_calls(
                pending["function_calls"], pending["tool_owner"], pending["contents"]
            ))
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
            response, contents, tool_owner = bg.run(agent.plan(prompt))

        if response.function_calls:
            if any(fc.name in STATE_CHANGING_TOOLS for fc in response.function_calls):
                st.session_state.pending = {
                    "function_calls": response.function_calls,
                    "contents": contents,
                    "tool_owner": tool_owner,
                }
            else:
                text, trail = bg.run(agent.execute_calls(response.function_calls, tool_owner, contents))
                st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            st.session_state.messages.append({"role": "assistant", "content": response.text})

        st.rerun()