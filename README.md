# NEXUS — MCP-Powered Personal Productivity Agent

An AI agent that plans, calls real tools (files, to-do list, web search, WhatsApp), and asks for confirmation before any action that changes something real.

## Prerequisites

- **Python 3.10+**
- **Node.js** (LTS) — required for the Filesystem MCP server
- **Git**

Check you have these:
```powershell
python --version
node --version
git --version
```

## Setup (first time only)

1. **Clone the repo**
```powershell
   git clone https://github.com/<your-username>/nexus-agent.git
   cd nexus-agent
```

2. **Create and activate a virtual environment**

   Windows (PowerShell):
```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
```
   If PowerShell blocks the activation script:
```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
   then re-run the activate command.

   Mac/Linux:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

   Your terminal prompt should now start with `(venv)`.

3. **Install dependencies**
```powershell
   pip install --upgrade pip
   pip install -r requirements.in
```

4. **Set up your API keys**
```powershell
   copy .env.example .env      # Windows
   # cp .env.example .env      # Mac/Linux
```
   Open `.env` and fill in:
   - `GEMINI_API_KEY` — free from [aistudio.google.com](https://aistudio.google.com)
   - `GROQ_API_KEY` — free from [console.groq.com](https://console.groq.com)
   - `TWILIO_*` — only needed if you're testing WhatsApp features. Ask the project lead for shared sandbox credentials, or leave blank (everything else still works — `send_whatsapp` will just report it isn't configured instead of failing).

## Running it

```powershell
streamlit run ui/app.py
```

**First run, check your terminal for:**

- `[Nexus] Connected to 'filesystem' server — tools: [...]` — the filesystem MCP server is up
- `[Nexus] Connected to 'todo' server — tools: [...]` — the to-do MCP server is up
- `[Nexus] Connected to 'websearch' server — tools: [...]` — the DuckDuckGo MCP server is up
- `[Nexus] Scheduled 'proactive_check' — next run ...` — background jobs are armed

These appear when the first browser tab connects, not at launch.

If any server fails to connect, check:
- Node.js is installed and `npx` is on your PATH (for the filesystem server)
- `duckduckgo-mcp-server` was installed via `pip install -r requirements.in`

## Project Structure

```
nexus-agent/
├── agent_core/
│   └── agent.py          # Core agent: plan → confirm → execute loop
├── mcp_servers/
│   └── todo_server.py    # To-do, memory, notifications, WhatsApp MCP server
├── ui/
│   └── app.py            # Streamlit chat UI, confirmation workflow, scheduler
├── sandbox_files/         # Files the agent can read/write (sandboxed)
├── memory/
│   └── todo.db           # SQLite database (auto-created)
├── tests/
│   ├── test_agent.py     # Unit tests (no API keys or network needed)
│   └── manual/           # Live checks you run by hand — see below
├── requirements.in        # Application dependencies (install from this)
├── requirements.txt       # Full pip freeze (for reproducibility)
├── .env.example           # Template for API keys
└── .streamlit/
    └── config.toml        # Dark theme configuration
```

## How it Works

1. **You type a message** in the Streamlit chat UI
2. **Nexus plans** — the agent reasons over your message, calling read-only tools automatically (list tasks, search the web, check the time)
3. **If changes are needed** — Nexus shows you exactly what it wants to do (e.g. "add_task", "send_whatsapp") and waits for your **Confirm** or **Cancel**
4. **After confirmation** — Nexus executes the tools and summarizes what happened

The last few turns of the conversation are replayed into each plan, so follow-ups
work: "list my tasks" → "add the second one to my calendar". Anything you want it
to keep permanently goes through the `remember` tool instead, and is injected into
the system prompt on every request.

### Background schedule

While the app is running, two jobs fire daily (Asia/Kolkata):

| Time | Job | What it does |
|------|-----|--------------|
| 08:00 | `proactive_check` | Looks at your tasks and memories, and if something is worth surfacing, writes a notification (shown in the app) and sends it to WhatsApp |
| 08:15 | `send_task_summary_whatsapp` | Sends your open to-do list to WhatsApp |

The schedule is set by the constants at the top of `ui/app.py`. These only run
while `streamlit run` is up — close the app and nothing fires.

## Tests

Unit tests — fast, no API keys, no network:

```powershell
python -m unittest discover -s tests -t . -p "test_*.py"
```

Live checks in `tests/manual/` are run individually and hit real services:

```powershell
python tests/manual/check_providers.py          # one API call per provider
python tests/manual/check_filesystem_server.py  # read-only
python tests/manual/check_todo_server.py        # writes to memory/todo.db, then undoes it
```

Run `check_providers.py` first if anything is behaving oddly. Groq retires hosted
models without notice, and a retired id 404s on every call — the fallback goes
dead while Gemini quietly carries all the traffic. The check detects this and
prints the models your account can actually use; put a tool-calling one in
`GROQ_FALLBACK_MODEL` in `agent_core/agent.py`.

