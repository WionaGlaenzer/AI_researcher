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
    q = embed_text(query)
    M = state["chunk_embeddings"]
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

def format_snippets(snips):
    lines = []
    for rank, (idx, text, score, meta) in enumerate(snips, 1):
        src = meta.get("arxiv_id") or meta.get("title") or meta.get("doc_id") or "unknown"
        src_short = src if isinstance(src, str) else "unknown"
        lines.append(
            f"S#{rank} [chunk {idx}, score {score:.3f}, src {src_short}]: {re.sub(r'\\s+',' ',text)[:300]}"
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