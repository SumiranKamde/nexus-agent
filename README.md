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
   pip install -r requirements.txt
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