import os
import numpy as np
import google.generativeai as genai
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import EMBED_MODEL
from tqdm import tqdm

# -----------------------
# Embeddings
# -----------------------
def ensure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")
    genai.configure(api_key=api_key)

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Return L2-normalized embedding vector for a single string.
    
    Args:
        text: The text to embed
        task_type: Either "RETRIEVAL_DOCUMENT" (default) for documents or "RETRIEVAL_QUERY" for queries
    """
    ensure_gemini()
    res = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    vec = np.array(res["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm

def embed_batch(texts: List[str], batch_size: int = 100, max_workers: int = 10, task_type: str = "RETRIEVAL_DOCUMENT") -> List[np.ndarray]:
    """
    Embed multiple texts in parallel for better performance.
    Uses ThreadPoolExecutor to parallelize API calls with progress tracking.
    
    Args:
        texts: List of text strings to embed
        batch_size: Not used, kept for compatibility
        max_workers: Number of parallel threads (default: 10)
        task_type: Either "RETRIEVAL_DOCUMENT" (default) for documents or "RETRIEVAL_QUERY" for queries
    
    Returns:
        List of normalized embedding vectors in the same order as input
    """
    ensure_gemini()
    if not texts:
        return []
    
    # Use thread pool to parallelize embedding calls
    all_embeddings = [None] * len(texts)  # Pre-allocate to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all embedding tasks
        future_to_index = {
            executor.submit(embed_text, text, task_type): i 
            for i, text in enumerate(texts)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(texts), desc="Embedding chunks", unit="chunk") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    vec = future.result()
                    all_embeddings[idx] = vec
                except Exception as e:
                    print(f"[WARNING] Failed to embed chunk {idx}: {e}")
                    # Try to get a zero vector of correct dimension
                    # We'll handle this after first successful embedding
                    all_embeddings[idx] = None
                pbar.update(1)
    
    # Handle any failed embeddings by using zero vectors
    # Find first successful embedding to get dimension
    first_success = next((emb for emb in all_embeddings if emb is not None), None)
    if first_success is not None:
        for i, emb in enumerate(all_embeddings):
            if emb is None:
                all_embeddings[i] = np.zeros_like(first_success)
    else:
        raise RuntimeError("All embedding attempts failed!")
    
    return all_embeddings

def build_chunk_embeddings(chunks: List[str]) -> np.ndarray:
    """Build embeddings for all chunks using batch processing."""
    if not chunks:
        return np.array([]).reshape(0, -1)
    
    print(f"[EMBEDDING] Generating embeddings for {len(chunks)} chunks...")
    embs = embed_batch(chunks, batch_size=100)
    return np.vstack(embs)

def cosine_sim(q_vec: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (M @ q_vec).ravel()
