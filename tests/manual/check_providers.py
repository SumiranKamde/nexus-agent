"""Manual check: are both LLM providers reachable with the keys in .env?

    python tests/manual/check_providers.py

Makes one real API call per provider. Models come from agent_core.agent so this
can't drift from what the agent actually uses.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from google import genai
from groq import Groq, NotFoundError

from agent_core.agent import GEMINI_MODEL, GROQ_FALLBACK_MODEL

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

if os.environ.get("GEMINI_API_KEY"):
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Reply with exactly: Gemini is working",
    )
    print(f"Gemini ({GEMINI_MODEL}) says:", response.text)
else:
    print("Skipping Gemini — no GEMINI_API_KEY in .env")

if os.environ.get("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: Groq is working"}],
        )
        print(f"Groq ({GROQ_FALLBACK_MODEL}) says:", completion.choices[0].message.content)
    except NotFoundError:
        # Groq retires hosted models with no notice, which kills the whole
        # fallback path. Print what's actually available so the fix is obvious.
        print(f"Groq: model '{GROQ_FALLBACK_MODEL}' no longer exists on this account.")
        print("Update GROQ_FALLBACK_MODEL in agent_core/agent.py to a tool-calling")
        print("model from this list:")
        for model in sorted(m.id for m in groq_client.models.list().data):
            print("  -", model)
else:
    print("Skipping Groq — no GROQ_API_KEY in .env")
