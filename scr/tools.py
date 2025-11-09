# tools.py
import os, re, unicodedata
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import requests, certifi
import arxiv
from arxiv import UnexpectedEmptyPageError
from pdfminer.high_level import extract_text

from config import MAX_TOTAL_CHUNKS, MAX_CHARS_PER_FILE
from file_io import chunk_text
from text_embeddings import embed_text

# ---------- helpers ----------
_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","of","for","to","in","on","at","by",
    "with","without","from","into","about","over","under","between","is","are","was","were",
    "be","been","being","this","that","these","those","it","its","as","we","you","they",
    "i","our","their","your","via","using","use"
}

def _normalize_filename(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:120]

def _extract_keywords(prompt: str) -> List[str]:
    phrase = prompt.strip()
    tokens = re.findall(r"[A-Za-z0-9\-]+", prompt.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    seen, ordered = set(), []
    for t in [phrase] + tokens:
        if t and t not in seen:
            seen.add(t); ordered.append(t)
    return ordered

def _build_abs_query(keywords: List[str]) -> str:
    if not keywords: return 'all:""'
    phrase = keywords[0]
    parts = [f'abs:"{phrase}"'] + [f'abs:"{k}"' for k in keywords[1:]]
    return parts[0] if len(parts) == 1 else f'({parts[0]}) OR (' + " OR ".join(parts[1:]) + ")"

def _last_5y_range():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=5*365)
    fmt = "%Y%m%d%H%M"
    return start.strftime(fmt), now.strftime(fmt)

def _build_query(prompt: str) -> str:
    core = _build_abs_query(_extract_keywords(prompt))
    a, b = _last_5y_range()
    return f"({core}) AND submittedDate:[{a} TO {b}]"

def _download_pdf(url: str, out_path: str, timeout=30):
    r = requests.get(url, stream=True, timeout=timeout, verify=certifi.where())
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)

# ---------- public API ----------
def get_more_papers_from_arxiv(
    prompt: str,
    outdir: str = None,
    max_results: int = 5,
    candidates: int = 25,
    run_id: str = None
) -> List[Dict[str, Any]]:
    """
    Fetch arXiv papers (last 5y) by abstract keywords, download PDFs, convert to .txt.
    Returns: [{arxiv_id,title,url,pdf_url,pdf_path,txt_path}, ...]
    
    Args:
        prompt: Search query
        outdir: Base output directory (default: data/arxiv_output)
        max_results: Maximum number of papers to download
        candidates: Number of candidates to scan
        run_id: Optional run ID to create a subdirectory for this run
    """
    if outdir is None:
        # Get project root (parent of scr/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outdir = os.path.join(project_root, "data", "arxiv_output")
    
    # If run_id is provided, create a subdirectory for this run
    if run_id:
        outdir = os.path.join(outdir, run_id)
    
    query = _build_query(prompt)
    out_pdf = os.path.join(outdir, "pdf")
    out_txt = os.path.join(outdir, "txt")
    os.makedirs(out_pdf, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)

    client = arxiv.Client(page_size=25, delay_seconds=3, num_retries=5)
    search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.Relevance,
                          max_results=max(1, min(candidates, 25)))
    try:
        results = list(client.results(search))
    except UnexpectedEmptyPageError:
        search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.Relevance, max_results=10)
        results = list(client.results(search))

    out = []
    for r in results[:max_results]:
        arxiv_id = r.get_short_id()
        title = r.title or arxiv_id
        base = f"{arxiv_id}_{_normalize_filename(title)}"
        pdf_path = os.path.join(out_pdf, f"{base}.pdf")
        txt_path = os.path.join(out_txt, f"{base}.txt")
        try:
            _download_pdf(r.pdf_url, pdf_path)
        except Exception:
            continue
        try:
            # Suppress pdfminer warnings about invalid color values
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    text = extract_text(pdf_path)
                finally:
                    sys.stderr = old_stderr
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            txt_path = None
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "abstract": getattr(r, "summary", None),  # helpful fallback
        })
        
    return out

# tools.py

def ingest_txts_into_state(
    state: Dict[str, Any],
    txt_paths: List[str],
    chunk_text_fn,
    embed_text_fn,
    max_total_chunks: int = 3000,      # <-- global cap
    max_chars_per_file: int = 300_000  # <-- safety: truncate huge files (~300k chars)
):
    """
    Append new txt files into your retrieval index with hard limits.
    Mutates state['paper_chunks'] and state['chunk_embeddings'].
    """
    import numpy as np

    state.setdefault("paper_chunks", [])
    state.setdefault("chunk_embeddings", None)

    # Remaining chunk budget
    budget = max_total_chunks - len(state["paper_chunks"])
    if budget <= 0:
        return  # already at/over cap

    new_chunks, new_vecs = [], []

    for p in txt_paths:
        if budget <= 0:
            break
        if not p or not os.path.exists(p):
            continue

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        # Truncate very large files before chunking
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file]

        # Chunk and respect the remaining budget
        for ch in chunk_text_fn(text):
            if budget <= 0:
                break
            new_chunks.append(ch)
            new_vecs.append(embed_text_fn(ch))
            budget -= 1

    if not new_chunks:
        return

    # Append to state
    state["paper_chunks"].extend(new_chunks)
    new_vecs = np.vstack(new_vecs)
    state["chunk_embeddings"] = (
        new_vecs if state["chunk_embeddings"] is None
        else np.vstack([state["chunk_embeddings"], new_vecs])
    )

def ingest_txts_into_state_with_meta(
    state: Dict[str, Any],
    papers: List[Dict[str, Any]],
    chunk_text_fn,
    embed_text_fn,
    max_total_chunks: int = 3000,
    max_chars_per_file: int = 300_000,
):
    """
    Ingest papers' TXT into state with per-chunk metadata.
    'papers' entries should have: txt_path, title, arxiv_id, url (as returned by get_more_papers_from_arxiv).
    """
    import numpy as np
    state.setdefault("paper_chunks", [])
    state.setdefault("chunk_embeddings", None)
    state.setdefault("chunk_meta", [])

    # Build quick index by txt_path
    by_txt = {p.get("txt_path"): p for p in papers if p.get("txt_path")}

    # Remaining budget
    budget = max_total_chunks - len(state["paper_chunks"])
    if budget <= 0:
        return 0

    added = 0
    new_vecs = []
    for txt_path, pinfo in by_txt.items():
        if budget <= 0:
            break
        if not os.path.exists(txt_path):
            continue
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file]

        local_idx = 0
        doc_id = f"arxiv::{pinfo.get('arxiv_id') or os.path.basename(txt_path)}"
        title = pinfo.get("title") or os.path.basename(txt_path)
        for ch in chunk_text_fn(text):
            if budget <= 0:
                break
            state["paper_chunks"].append(ch)
            vec = embed_text_fn(ch)
            new_vecs.append(vec)
            state["chunk_meta"].append({
                "doc_id": doc_id,
                "origin": "arxiv",
                "title": title,
                "arxiv_id": pinfo.get("arxiv_id"),
                "url": pinfo.get("url"),
                "txt_path": os.path.abspath(txt_path),
                "chunk_local_idx": local_idx,
            })
            local_idx += 1
            added += 1
            budget -= 1

    if added:
        new_vecs = np.vstack(new_vecs)
        state["chunk_embeddings"] = (
            new_vecs if state["chunk_embeddings"] is None
            else np.vstack([state["chunk_embeddings"], new_vecs])
        )
    return added

# -----------------------
# Fallback: ingest abstracts when TXT fails
# -----------------------
def ingest_abstracts_into_state(state: Dict[str, Any], papers: List[Dict[str, Any]]) -> int:
    """
    If PDFs couldn't be converted to TXT, at least ingest abstracts (if provided by tools).
    Also appends chunk_meta entries tagged as 'abstract'.
    Returns number of chunks added.
    """
    abstracts = []
    for p in papers:
        abs_txt = p.get("abstract")
        if abs_txt and isinstance(abs_txt, str) and abs_txt.strip():
            abstracts.append((p, abs_txt.strip()))
    if not abstracts:
        return 0

    added = 0
    for p, abs_txt in abstracts:
        local_idx = 0
        for ch in chunk_text(abs_txt):
            if len(state["paper_chunks"]) >= MAX_TOTAL_CHUNKS:
                return added
            state["paper_chunks"].append(ch)
            vec = embed_text(ch)
            if state["chunk_embeddings"] is None:
                state["chunk_embeddings"] = np.expand_dims(vec, 0)
            else:
                state["chunk_embeddings"] = np.vstack([state["chunk_embeddings"], vec])

            # meta
            state["chunk_meta"].append({
                "doc_id": f"arxiv::{p.get('arxiv_id') or os.path.basename(p.get('txt_path') or '')}::abstract",
                "origin": "arxiv_abstract",
                "title": p.get("title") or "arXiv abstract",
                "arxiv_id": p.get("arxiv_id"),
                "url": p.get("url"),
                "txt_path": p.get("txt_path"),
                "chunk_local_idx": local_idx,
            })
            local_idx += 1
            added += 1
    return added

def append_meta_for_txts(state: Dict[str, Any], papers: List[Dict[str, Any]], newly_added_chunks: int):
    """
    After ingest_txts_into_state (which does not add metadata), mirror its chunking order and
    append matching chunk_meta entries for exactly 'newly_added_chunks' chunks.
    This keeps meta aligned with embeddings/chunks even with caps.
    """
    if newly_added_chunks <= 0:
        return
    remaining = newly_added_chunks
    for p in papers:
        txt = p.get("txt_path")
        if not txt or not os.path.exists(txt):
            continue
        try:
            with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if len(text) > MAX_CHARS_PER_FILE:
            text = text[:MAX_CHARS_PER_FILE]

        local_idx = 0
        for _ch in chunk_text(text):
            if remaining <= 0:
                return
            state["chunk_meta"].append({
                "doc_id": f"arxiv::{p.get('arxiv_id') or os.path.basename(txt)}",
                "origin": "arxiv",
                "title": p.get("title") or os.path.basename(txt),
                "arxiv_id": p.get("arxiv_id"),
                "url": p.get("url"),
                "txt_path": os.path.abspath(txt),
                "chunk_local_idx": local_idx,
            })
            local_idx += 1
            remaining -= 1
            if len(state["paper_chunks"]) >= MAX_TOTAL_CHUNKS:
                return


__all__ = [
    "get_more_papers_from_arxiv", 
    "ingest_txts_into_state", 
    "ingest_txts_into_state_with_meta",
    "ingest_abstracts_into_state",
    "append_meta_for_txts"
]
