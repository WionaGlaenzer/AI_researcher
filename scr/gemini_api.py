# pip install google-generativeai
import os
from google import generativeai as genai

# Get API key from environment variable
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Please set it using: export GOOGLE_API_KEY='your-api-key'"
    )

genai.configure(api_key=api_key)

# Call List models
models = genai.list_models()
for m in models:
    print(m.name)

model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# Simple one-shot request
response = model.generate_content("Write a 2-line poem about the ocean.")
print(response.text)

# (Optional) Streaming response
stream = model.generate_content("Stream a fun fact about octopuses.", stream=True)
for chunk in stream:
    if chunk.text:
        print(chunk.text, end="")
print()
