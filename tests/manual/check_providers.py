import os
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = gemini_client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Reply with exactly: Gemini is working"
)
print("Gemini says:", response.text)

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
completion = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Reply with exactly: Groq is working"}],
)
print("Groq says:", completion.choices[0].message.content)