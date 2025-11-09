"""
LangGraph two-agent mini research lab (Gemini + Embedding Retrieval + arXiv fetch)
Agents: Innovator and Critic

Flow:
  1) Innovator proposes initial claims from evidence
  2) Critic provides feedback + retrieval guidance
  3) Innovator writes a NEW refined retrieval query based on the critique,
     fetches more papers from arXiv, ingests them (with caps), re-retrieves,
     and emits an improved Final Report.

Run
---
python langgraph_two_agent_research_lab.py --prompt "How do RAG systems reduce hallucination?" --paper paper.txt
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np
from langgraph.graph import StateGraph, END

from config import GEN_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_SNIPPETS, MAX_TOTAL_CHUNKS, MAX_CHARS_PER_FILE
from file_io import load_paper, chunk_text
from text_embeddings import build_chunk_embeddings
from graph_nodes import innovator_node, critic_node, refine_and_synthesize_node

# -----------------------
# Graph wiring
# -----------------------
def should_continue(state: Dict[str, Any]) -> str:
    """Conditional edge: continue iterating or finish"""
    max_iterations = state.get("max_iterations", 1)
    current_iteration = state.get("iteration", 0)
    
    if current_iteration < max_iterations:
        return "continue"
    else:
        return "end"

def build_graph():
    g = StateGraph(dict)
    g.add_node("innovator", innovator_node)
    g.add_node("critic", critic_node)
    g.add_node("refine_and_synthesize", refine_and_synthesize_node)
    g.set_entry_point("innovator")
    g.add_edge("innovator", "critic")
    g.add_edge("critic", "refine_and_synthesize")
    # Conditional edge: loop back or end
    g.add_conditional_edges(
        "refine_and_synthesize",
        should_continue,
        {
            "continue": "innovator",  # Loop back to innovator for another iteration
            "end": END
        }
    )
    return g.compile()

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--iterations", type=int, default=1, 
                    help="Number of iteration cycles (Innovator->Critic->Refine). Default: 1")
    args = ap.parse_args()

    # Get project root (parent of scr/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve paper path
    if not os.path.isabs(args.paper):
        paper_path = os.path.join(project_root, "data", args.paper)
    else:
        paper_path = args.paper
    
    # Generate run ID (timestamp-based)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Resolve log path with run ID
    if args.log is None:
        log_filename = f"run_log_{run_id}.json"
        log_path = os.path.join(project_root, "output", log_filename)
    elif not os.path.isabs(args.log):
        # If user provided a filename, add run_id before .json extension
        if args.log.endswith('.json'):
            log_filename = args.log.replace('.json', f'_{run_id}.json')
        else:
            log_filename = f"{args.log}_{run_id}.json"
        log_path = os.path.join(project_root, "output", log_filename)
    else:
        # If absolute path, add run_id before .json extension
        if args.log.endswith('.json'):
            log_path = args.log.replace('.json', f'_{run_id}.json')
        else:
            log_path = f"{args.log}_{run_id}.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n=== Run ID: {run_id} ===\n")
    print(f"Paper: {paper_path}")
    print(f"Log file: {log_path}")
    print(f"Iterations: {args.iterations}")
    print(f"arXiv output: {os.path.join(project_root, 'data', 'arxiv_output', run_id)}\n")

    text = load_paper(paper_path)
    chunks = chunk_text(text)
    chunk_embeddings = build_chunk_embeddings(chunks)

    # Seed per-chunk metadata for the local file
    seed_doc_id = f"seed::{os.path.abspath(paper_path)}"
    chunk_meta = [
        {
            "doc_id": seed_doc_id,
            "origin": "seed",
            "title": os.path.basename(paper_path),
            "arxiv_id": None,
            "url": None,
            "txt_path": os.path.abspath(paper_path),
            "chunk_local_idx": i,   # index within this document
        } for i in range(len(chunks))
    ]

    state: Dict[str, Any] = {
        "research_prompt": args.prompt,
        "paper_chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "chunk_meta": chunk_meta,
        "turn": 0,
        "iteration": 0,  # Track current iteration
        "max_iterations": args.iterations,  # Maximum number of iterations
        "messages": [],
        "evidence_log": [],
        "final_report": None,
        "run_id": run_id,  # Add run_id to state
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
        "run_id": run_id,
        "prompt": state["research_prompt"],
        "config": {
            "model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_snippets": TOP_K_SNIPPETS,
            "paper_path": os.path.abspath(paper_path),
            "max_total_chunks": MAX_TOTAL_CHUNKS,
            "max_chars_per_file": MAX_CHARS_PER_FILE,
            "iterations": args.iterations,
        },
        "messages": state["messages"],
        "evidence_log": state["evidence_log"],
        "refined_query": state.get("refined_query"),
        "final_report": state.get("final_report"),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON log -> {os.path.abspath(log_path)}")

if __name__ == "__main__":
    main()
