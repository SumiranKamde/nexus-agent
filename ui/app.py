import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import html
import json
import threading

import streamlit as st
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

    agent.tool_summary = {}
    for name, session in agent.sessions.items():
        tools = bg.run(session.list_tools())
        agent.tool_summary[name] = [t.name for t in tools.tools]

    return agent, bg


agent, bg = get_agent()
st.set_page_config(page_title="Nexus", page_icon="◈", layout="wide")
st_autorefresh(interval=30_000, key="notif_poll")

# ============================================================= SIGNAL CONSOLE THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --void: #0B0D17;
    --panel: #141726;
    --panel-2: #1B1F33;
    --signal: #8B5CF6;
    --pulse: #22D3EE;
    --amber: #F5A623;
    --mist: #8B93A8;
    --text: #E7E9F3;
    --border: #262B44;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Wordmark */
.nexus-wordmark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: 0.02em;
    background: linear-gradient(90deg, var(--signal), var(--pulse), var(--signal));
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shimmer 6s linear infinite;
    margin-bottom: 0;
}
@keyframes shimmer { to { background-position: 200% center; } }

.nexus-caption {
    font-family: 'JetBrains Mono', monospace;
    color: var(--mist);
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-top: -0.3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

.console-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: var(--mist);
    text-transform: uppercase;
    margin: 1.1rem 0 0.5rem 0;
}

.status-row {
    display: flex; align-items: center; gap: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: var(--text);
    padding: 0.25rem 0;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--pulse);
    box-shadow: 0 0 6px var(--pulse), 0 0 12px var(--pulse);
    animation: pulse 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.8); }
}

/* Chat messages — custom console-log rendering */
.msg-row { display: flex; margin: 1.1rem 0; }
.msg-row.user { justify-content: flex-end; }
.msg-bubble {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: 4px;
    font-size: 0.95rem; line-height: 1.5;
}
.msg-bubble.assistant {
    background: var(--panel);
    border-left: 3px solid var(--signal);
    color: var(--text);
}
.msg-bubble.user {
    background: var(--panel-2);
    border-right: 3px solid var(--mist);
    color: var(--text);
}
.msg-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.1em;
    color: var(--mist); text-transform: uppercase;
    margin-bottom: 0.35rem; display: block;
}

/* Trace log (reasoning trail) */
[data-testid="stExpander"] {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: var(--pulse);
}

/* Action request panel */
.action-panel {
    background: var(--panel);
    border-left: 3px solid var(--amber);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}
.action-panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.12em;
    color: var(--amber); text-transform: uppercase;
    margin-bottom: 0.5rem; display: block;
}
.action-call {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: var(--text);
    margin: 0.15rem 0;
}

/* Buttons */
[data-testid="stButton"] button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 3px;
    border: 1px solid var(--border);
}
[data-testid="stButton"] button:hover {
    border-color: var(--signal);
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.4);
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 4px;
}

/* Metric */
[data-testid="stMetric"] {
    background: var(--panel-2);
    border-radius: 4px; padding: 0.6rem 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================= SIDEBAR
with st.sidebar:
    st.markdown('<div class="nexus-wordmark" style="font-size:1.6rem;">◈ NEXUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="nexus-caption">agent console</div>', unsafe_allow_html=True)

    st.markdown('<div class="console-label">Connected Servers</div>', unsafe_allow_html=True)
    for server_name, tools in agent.tool_summary.items():
        st.markdown(
            f'<div class="status-row"><span class="status-dot"></span>{server_name} '
            f'<span style="color:var(--mist)">· {len(tools)} tools</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("tools", expanded=False):
            for t in tools:
                st.caption(f"› {t}")

    st.markdown('<div class="console-label">Status</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="console-label">Recent Activity</div>', unsafe_allow_html=True)
        for entry in recent:
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:var(--mist);">'
                f'{entry["time"][11:16]} · {entry["tool"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear console", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

# ============================================================= MAIN
st.markdown('<div class="nexus-wordmark">◈ NEXUS</div>', unsafe_allow_html=True)
st.markdown('<div class="nexus-caption">// plans → acts → observes → confirms before anything real changes</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def render_message(role, content, trail=None):
    tag = "nexus" if role == "assistant" else "you"
    safe_content = html.escape(str(content)).replace("\n", "<br>")
    st.markdown(
        f'<div class="msg-row {role}"><div class="msg-bubble {role}">'
        f'<span class="msg-tag">{tag}</span>{safe_content}</div></div>',
        unsafe_allow_html=True,
    )
    if trail:
        with st.expander("◈ trace log"):
            for name, args, result in trail:
                st.markdown(f"**{name}**`{args}`")
                st.code(str(result), language=None)


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("trail"))

try:
    notif_result = bg.run(agent.sessions["todo"].call_tool("list_notifications", {"unseen_only": True}))
    notifications = json.loads(notif_result.content[0].text) if notif_result.content else []
except Exception:
    notifications = []

if notifications:
    for n in notifications:
        st.info(f"◈ {n['message']}")
    if st.button("Dismiss all"):
        bg.run(agent.sessions["todo"].call_tool("mark_notifications_seen", {}))
        st.rerun()

if st.session_state.pending:
    calls = st.session_state.pending.function_calls
    calls_html = "".join(
        f'<div class="action-call">→ {html.escape(str(c.name))}({html.escape(str(c.args))})</div>'
        for c in calls
    )
    st.markdown(
        f'<div class="action-panel"><span class="action-panel-label">Action Request</span>{calls_html}</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    if col1.button("Confirm", use_container_width=True):
        with st.spinner("Executing..."):
            text, exec_trail = bg.run(agent.execute_calls(st.session_state.pending))
        full_trail = (st.session_state.pending.trail or []) + exec_trail
        st.session_state.messages.append({"role": "assistant", "content": text, "trail": full_trail})
        st.session_state.pending = None
        st.rerun()
    if col2.button("Cancel", use_container_width=True):
        st.session_state.messages.append({"role": "assistant", "content": "Okay, I didn't make any changes."})
        st.session_state.pending = None
        st.rerun()
else:
    if prompt := st.chat_input("Ask Nexus to check your tasks, files, or the web..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Reasoning..."):
            plan_result = bg.run(agent.plan(prompt))

        if plan_result.function_calls:
            st.session_state.pending = plan_result
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": plan_result.text, "trail": plan_result.trail,
            })
        st.rerun()
