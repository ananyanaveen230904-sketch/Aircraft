"""Quick Groq connectivity check. Reads GROQ_API_KEY from .env / environment (never hard-code keys)."""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
r = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}], max_tokens=5)
print(r.choices[0].message.content)
