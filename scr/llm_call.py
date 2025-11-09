# -----------------------
# Gemini LLM
# -----------------------
import google.generativeai as genai
from config import GEN_MODEL
from text_embeddings import ensure_gemini

def call_llm(system_prompt: str, user_prompt: str) -> str:
    ensure_gemini()
    model = genai.GenerativeModel(GEN_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.4))
    return (resp.text or "").strip()
