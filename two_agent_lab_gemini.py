"""
LangGraph two-agent mini research lab (Gemini + Embedding Retrieval + arXiv fetch)
Agents: Innovator and Critic

Flow:
  1) Innovator proposes initial claims from evidence
  2) Critic provides feedback + retrieval guidance
  3) Innovator writes a NEW refined retrieval query based on the critique,
     fetches more papers from arXiv, ingests them (with caps), re-retrieves,
     and emits an improved Final Report.

Setup
-----
pip install -U langgraph google-generativeai numpy arxiv pdfminer.six requests certifi
export GEMINI_API_KEY=YOUR_API_KEY_HERE
# optional:
export GEMINI_MODEL="models/gemini-2.0-flash-lite"
export GEMINI_EMBED_MODEL="models/text-embedding-004"
export MAX_TOTAL_CHUNKS=3000
export MAX_CHARS_PER_FILE=300000

Run
---
python langgraph_two_agent_research_lab.py --prompt "How do RAG systems reduce hallucination?" --paper paper.txt
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# Import your tools from tools.py (must be in the same folder or on PYTHONPATH)
from tools import get_more_papers_from_arxiv, ingest_txts_into_state

# -----------------------
# Config
# -----------------------
GEN_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash-lite")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "models/text-embedding-004")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
TOP_K_SNIPPETS = int(os.environ.get("TOP_K", 4))

# ingestion caps
MAX_TOTAL_CHUNKS = int(os.environ.get("MAX_TOTAL_CHUNKS", "3000"))
MAX_CHARS_PER_FILE = int(os.environ.get("MAX_CHARS_PER_FILE", "300000"))

# -----------------------
# IO
# -----------------------
def load_paper(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i : i + size])
        i += step
    return out

def approx_char_start(chunk_index: int) -> int:
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return max(0, chunk_index * step)

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

# -----------------------
# Gemini LLM
# -----------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    ensure_gemini()
    model = genai.GenerativeModel(GEN_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt, generation_config=genai.types.GenerationConfig(temperature=0.4))
    return (resp.text or "").strip()

# -----------------------
# Agent prompts
# -----------------------
INNOVATOR_SYS = (
    "You are the Innovator. Propose insights grounded in the evidence. "
    "Cite evidence like (S#1), (S#2). Keep claims scoped and avoid over-generalization. Return 3–6 bullets."
)

CRITIC_SYS = (
    "You are the Critic. Identify unsupported assumptions, confounds, or weak evidence. "
    "Suggest 2+ specific improvements and retrieval filters or query terms to seek stronger evidence. Be concise."
)

REFINER_SYS = (
    "You are the Innovator writing a refined retrieval prompt for the next evidence pass. "
    "Using the original research prompt and the critic's feedback, write a single precise query (1–2 sentences) "
    "that targets missing controls, better baselines, or conflicting conditions. "
    "Output ONLY the refined query text, no explanations."
)

SYNTHESIS_SYS = (
    "You are the Innovator synthesizing the improved final report. Merge supported claims, downgrade weak ones, "
    "and produce 1 falsifiable hypothesis. Include an Evidence Table mapping claims to snippet refs (S#i)."
)

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

# -----------------------
# Graph nodes
# -----------------------
def innovator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["phase"] = "innovator_initial"
    query = state["messages"][-1]["content"] if state["messages"] else state["research_prompt"]
    snips = retrieve_snippets(state, query)
    user = f"Research prompt: {state['research_prompt']}\n\nEvidence:\n{format_snippets(snips)}"
    out = call_llm(INNOVATOR_SYS, user)

    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    state["turn"] += 1
    return state

def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    state["phase"] = "critic_feedback"
    innov = state["messages"][-1]["content"]
    snips = retrieve_snippets(state, innov)
    user = f"Innovator said:\n{innov}\n\nEvidence:\n{format_snippets(snips)}"
    out = call_llm(CRITIC_SYS, user)

    state["messages"].append({
        "role": "Critic",
        "turn": state["turn"],
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    state["turn"] += 1
    return state

def refine_and_synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # --- write refined query ---
    state["phase"] = "refine_query"
    last_innov = next(m for m in reversed(state["messages"]) if m["role"] == "Innovator")["content"]
    last_crit  = next(m for m in reversed(state["messages"]) if m["role"] == "Critic")["content"]

    refined_query = call_llm(
        REFINER_SYS,
        (
            f"Original research prompt: {state['research_prompt']}\n\n"
            f"Latest Innovator draft:\n{last_innov}\n\n"
            f"Critic feedback:\n{last_crit}\n\n"
            f"Write a single refined retrieval query:"
        ),
    )
    state["refined_query"] = refined_query

    # --- fetch from arXiv ---
    fetched = []
    fetch_error = None
    try:
        fetched = get_more_papers_from_arxiv(
            refined_query,
            outdir="arxiv_output",
            max_results=5,
            candidates=25
        )
    except Exception as e:
        fetch_error = str(e)

    # --- ingest .txt (with caps) ---
    before_chunks = len(state["paper_chunks"])
    ingested_txt = 0
    try:
        txts = [p["txt_path"] for p in fetched if p.get("txt_path")]
        ingest_txts_into_state(
            state,
            txts,
            chunk_text_fn=chunk_text,
            embed_text_fn=embed_text,
            max_total_chunks=MAX_TOTAL_CHUNKS,
            max_chars_per_file=MAX_CHARS_PER_FILE,
        )
        ingested_txt = len(state["paper_chunks"]) - before_chunks
        # Add matching metadata for those newly-added chunks
        append_meta_for_txts(state, fetched, ingested_txt)
    except Exception as e:
        state["evidence_log"].append({
            "turn": state["turn"],
            "phase": "ingest_error",
            "query": refined_query,
            "error": str(e),
        })

    # --- fallback: ingest abstracts if no txt ingested ---
    ingested_abs = 0
    if ingested_txt == 0 and fetched:
        ingested_abs = ingest_abstracts_into_state(state, fetched)

    after_chunks = len(state["paper_chunks"])
    state["evidence_log"].append({
        "turn": state["turn"],
        "phase": "arxiv_fetch",
        "query": refined_query,
        "fetched_count": len(fetched),
        "ingested_txt_chunks": ingested_txt,
        "ingested_abstract_chunks": ingested_abs,
        "chunks_before": before_chunks,
        "chunks_after": after_chunks,
        "error": fetch_error,
    })

    # --- refined retrieval over expanded corpus ---
    state["phase"] = "refined_retrieval"
    snips_refined = retrieve_snippets(state, refined_query)

    # --- synthesize final report ---
    state["phase"] = "synthesis"
    evidence_block = format_snippets(snips_refined)
    user = (
        f"Prompt: {state['research_prompt']}\n\n"
        f"Refined retrieval query: {refined_query}\n\n"
        f"Refined Evidence:\n{evidence_block}\n\n"
        f"Use the refined evidence plus prior discussion to craft the improved final report."
    )
    out = call_llm(SYNTHESIS_SYS, user)

    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "content": out,
        "snippets": build_snippet_meta(snips_refined),
        "refined_query": refined_query,
    })
    state["final_report"] = out
    state["turn"] += 1
    return state

# -----------------------
# Graph wiring
# -----------------------
def build_graph():
    g = StateGraph(dict)
    g.add_node("innovator", innovator_node)
    g.add_node("critic", critic_node)
    g.add_node("refine_and_synthesize", refine_and_synthesize_node)
    g.set_entry_point("innovator")
    g.add_edge("innovator", "critic")
    g.add_edge("critic", "refine_and_synthesize")
    g.add_edge("refine_and_synthesize", END)
    return g.compile()

# -----------------------
# CLI / Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--log", default="run_log.json")
    args = ap.parse_args()

    text = load_paper(args.paper)
    chunks = chunk_text(text)
    chunk_embeddings = build_chunk_embeddings(chunks)

    # Seed per-chunk metadata for the local file
    seed_doc_id = f"seed::{os.path.abspath(args.paper)}"
    chunk_meta = [
        {
            "doc_id": seed_doc_id,
            "origin": "seed",
            "title": os.path.basename(args.paper),
            "arxiv_id": None,
            "url": None,
            "txt_path": os.path.abspath(args.paper),
            "chunk_local_idx": i,   # index within this document
        } for i in range(len(chunks))
    ]

    state: Dict[str, Any] = {
        "research_prompt": args.prompt,
        "paper_chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "chunk_meta": chunk_meta,
        "turn": 0,
        "messages": [],
        "evidence_log": [],
        "final_report": None,
    }

    app = build_graph()
    state = app.invoke(state)

    print("\n=== Transcript ===\n")
    for m in state["messages"]:
        role = m.get("role")
        print(f"[{role}]\n{m.get('content','')}\n")

    print("\n=== Final Report ===\n")
    print(state.get("final_report") or "")

    # Save JSON log
    log = {
        "prompt": state["research_prompt"],
        "config": {
            "model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_snippets": TOP_K_SNIPPETS,
            "paper_path": os.path.abspath(args.paper),
            "max_total_chunks": MAX_TOTAL_CHUNKS,
            "max_chars_per_file": MAX_CHARS_PER_FILE,
        },
        "messages": state["messages"],
        "evidence_log": state["evidence_log"],
        "refined_query": state.get("refined_query"),
        "final_report": state.get("final_report"),
    }
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON log -> {os.path.abspath(args.log)}")

if __name__ == "__main__":
    main()
