# Nexus — MCP-Powered Personal Productivity Agent

## Setup
1. `python -m venv venv` then `.\venv\Scripts\Activate.ps1`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your GEMINI_API_KEY and GROQ_API_KEY
4. `streamlit run ui/app.py`

## Architecture
Chat UI (Streamlit) → Agent Core (Gemini primary, Groq Llama fallback) → MCP Client → Filesystem + To-Do MCP servers