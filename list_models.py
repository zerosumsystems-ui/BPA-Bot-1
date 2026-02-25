import os
from google import genai

key = None
with open(".streamlit/secrets.toml") as f:
    for line in f:
        if line.startswith("GEMINI_API_KEY"):
            key = line.split("=")[1].strip().strip('"')

client = genai.Client(api_key=key)
models = client.models.list()
for m in models:
    print(m.name)
