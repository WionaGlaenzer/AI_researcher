from typing import Any, Dict, List, Tuple
import numpy as np
import re
from prompts import TOP_K_SNIPPETS
from text_embeddings import embed_text, cosine_sim
from file_io import approx_char_start

# -----------------------
# Retrieval helpers
# -----------------------
def retrieve_snippets(state: Dict[str, Any], query: str, k: int = TOP_K_SNIPPETS):
    M = state.get("chunk_embeddings")
    chunks = state.get("paper_chunks", [])
    
    # Handle empty state - no chunks or embeddings available
    if M is None:
        state["evidence_log"].append({
            "turn": state.get("turn", 0),
            "phase": state.get("phase", "unspecified"),
            "query": query,
            "top_snippets": [],
            "error": "No embeddings available for retrieval",
        })
        return []
    
    # Check if M is a numpy array and has valid shape
    if hasattr(M, 'shape'):
        # Check if it's a 0-d array (scalar) or has 0 rows
        if M.shape == () or (len(M.shape) > 0 and M.shape[0] == 0):
            state["evidence_log"].append({
                "turn": state.get("turn", 0),
                "phase": state.get("phase", "unspecified"),
                "query": query,
                "top_snippets": [],
                "error": "Empty embeddings matrix",
            })
            return []
    
    if len(chunks) == 0:
        state["evidence_log"].append({
            "turn": state.get("turn", 0),
            "phase": state.get("phase", "unspecified"),
            "query": query,
            "top_snippets": [],
            "error": "No chunks available for retrieval",
        })
        return []
    
    q = embed_text(query, task_type="RETRIEVAL_QUERY")
    sims = cosine_sim(q, M)
    idxs = np.argsort(sims)[::-1][:k]

    results = []
    top_snips_log = []
    for i in idxs:
        i = int(i)
        text = state["paper_chunks"][i]
        score = float(sims[i])
        meta = state.get("chunk_meta", [{}])[i] if i < len(state.get("chunk_meta", [])) else {}
        results.append((i, text, score, meta))
        top_snips_log.append({
            "chunk_index": i,
            "score": round(score, 4),
            "approx_char_start": approx_char_start(i),
            "preview": state["paper_chunks"][i][:200].replace("\n", " "),
            "source": {
                "doc_id": meta.get("doc_id"),
                "origin": meta.get("origin"),
                "title": meta.get("title"),
                "arxiv_id": meta.get("arxiv_id"),
                "url": meta.get("url"),
                "txt_path": meta.get("txt_path"),
                "chunk_local_idx": meta.get("chunk_local_idx"),
            },
        })

    state["evidence_log"].append({
        "turn": state["turn"],
        "phase": state.get("phase", "unspecified"),
        "query": query,
        "top_snippets": top_snips_log,
    })
    return results

def _build_citation_link(meta: Dict[str, Any]) -> str:
    """Build a markdown citation link in format [Title](url)"""
    title = meta.get("title") or meta.get("arxiv_id") or "Unknown Source"
    url = meta.get("url")
    
    # Convert arxiv URLs to HTML format if we have an arxiv_id
    if meta.get("arxiv_id"):
        arxiv_id = meta.get("arxiv_id")
        # Convert to HTML format (e.g., 2204.09140v2 -> https://arxiv.org/html/2204.09140v2)
        url = f"https://arxiv.org/html/{arxiv_id}"
    elif url:
        # If we have a URL but it's in abs format, convert to html format
        # Match arxiv.org/abs/ pattern and convert to html
        url = re.sub(r'https?://arxiv\.org/abs/([^/]+)', r'https://arxiv.org/html/\1', url)
    
    # If still no URL, use a placeholder
    if not url:
        url = "#"
    
    return f"[{title}]({url})"

def format_snippets(snips):
    lines = []
    for rank, (idx, text, score, meta) in enumerate(snips, 1):
        citation = _build_citation_link(meta)
        text_preview = re.sub(r'\s+', ' ', text)[:300]
        lines.append(
            f"{citation} [chunk {idx}, score {score:.3f}]: {text_preview}"
        )
    return "\n".join(lines)

def build_snippet_meta(snips: List[Tuple[int, str, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    meta_rows = []
    for rank, (idx, text, score, src_meta) in enumerate(snips, 1):
        meta_rows.append({
            "rank": rank,
            "label": f"S#{rank}",
            "chunk_index": idx,
            "approx_char_start": approx_char_start(idx),
            "score": round(float(score), 4),
            "preview": re.sub(r"\s+", " ", text)[:200],
            "source": {
                "doc_id": src_meta.get("doc_id"),
                "origin": src_meta.get("origin"),
                "title": src_meta.get("title"),
                "arxiv_id": src_meta.get("arxiv_id"),
                "url": src_meta.get("url"),
                "txt_path": src_meta.get("txt_path"),
                "chunk_local_idx": src_meta.get("chunk_local_idx"),
            },
        })
    return meta_rows