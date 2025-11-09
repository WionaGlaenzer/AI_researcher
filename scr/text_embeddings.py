import os
import numpy as np
import google.generativeai as genai
from typing import List
from config import EMBED_MODEL

# -----------------------
# Embeddings
# -----------------------
def ensure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")
    genai.configure(api_key=api_key)

def embed_text(text: str) -> np.ndarray:
    """Return L2-normalized embedding vector for a single string."""
    ensure_gemini()
    res = genai.embed_content(model=EMBED_MODEL, content=text)
    vec = np.array(res["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm

def build_chunk_embeddings(chunks: List[str]) -> np.ndarray:
    embs = [embed_text(ch) for ch in chunks]
    return np.vstack(embs)

def cosine_sim(q_vec: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (M @ q_vec).ravel()
