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
python two_agent_lab_gemini2.py --prompt "How do RAG systems reduce hallucination?"
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np
from langgraph.graph import StateGraph, END

from config import GEN_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_SNIPPETS, MAX_TOTAL_CHUNKS, MAX_CHARS_PER_FILE
from file_io import chunk_text
from text_embeddings import build_chunk_embeddings, embed_text
from graph_nodes import innovator_node, critic_node, refine_and_synthesize_node
from tools import get_more_papers_from_arxiv, ingest_txts_into_state, append_meta_for_txts, ingest_abstracts_into_state
from llm_call import call_llm
from prompts import KEYWORD_EXTRACTION_SYS

# -----------------------
# Graph wiring
# -----------------------
def should_continue(state: Dict[str, Any]) -> str:
    """Conditional edge: continue iterating or finish"""
    max_iterations = state.get("max_iterations", 1)
    current_iteration = state.get("iteration", 0)
    
    print(f"[FLOW] Checking if should continue: iteration {current_iteration} / max {max_iterations}")
    
    if current_iteration < max_iterations:
        print(f"[FLOW] Continuing to next innovator->critic cycle")
        return "continue"
    else:
        print(f"[FLOW] Reached max iterations ({max_iterations}), moving to refine_and_synthesize")
        return "end"

def build_graph():
    g = StateGraph(dict)

    g.add_node("innovator", innovator_node)
    g.add_node("critic", critic_node)
    g.add_node("refine_and_synthesize", refine_and_synthesize_node)

    g.set_entry_point("innovator")

    # Always go innovator -> critic
    g.add_edge("innovator", "critic")

    # Loop between critic and innovator until should_continue says "end"
    g.add_conditional_edges(
        "critic",
        should_continue,  # returns "continue" or "end"
        {
            "continue": "innovator",             # keep looping
            "end": "refine_and_synthesize",      # exit loop to final step
        },
    )

    # After the final step, terminate
    g.add_edge("refine_and_synthesize", END)

    return g.compile()

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Research question to investigate")
    ap.add_argument("--log", default=None)
    ap.add_argument("--iterations", type=int, default=1, 
                    help="Number of iteration cycles (Innovator->Critic->Refine). Default: 1")
    ap.add_argument("--initial_papers", type=int, default=5,
                    help="Number of papers to fetch initially from arXiv. Default: 5")
    args = ap.parse_args()

    # Get project root (parent of scr/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
    print(f"Research Question: {args.prompt}")
    print(f"Log file: {log_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Initial papers to fetch: {args.initial_papers}")
    print(f"arXiv output: {os.path.join(project_root, 'data', 'arxiv_output', run_id)}\n")

    # Extract keywords from the research question using the innovator agent
    print("\n[FLOW] Initial Setup - Keyword Extraction")
    print("="*60)
    keywords_text = call_llm(KEYWORD_EXTRACTION_SYS, args.prompt, agent_name="KEYWORD_EXTRACTOR")
    # Parse comma-separated keywords and clean them up
    keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:10]
    print(f"[KEYWORD_EXTRACTOR] Extracted {len(keywords)} keywords: {keywords}")
    
    # Use keywords for arXiv search (join them into a search query)
    search_query = " ".join(keywords) if keywords else args.prompt
    print(f"Using search query: {search_query}\n")

    # Initialize empty state
    state: Dict[str, Any] = {
        "research_prompt": args.prompt,
        "extracted_keywords": keywords,  # Store extracted keywords
        "paper_chunks": [],
        "chunk_embeddings": None,  # Will be set by ingest_txts_into_state
        "chunk_meta": [],
        "turn": 0,
        "iteration": 0,
        "max_iterations": args.iterations,
        "messages": [],
        "evidence_log": [],
        "final_report": None,
        "run_id": run_id,
    }

    # Fetch initial papers from arXiv using the extracted keywords
    # Prioritize review papers for the initial search since we're creating a literature review
    print("Fetching initial papers from arXiv (prioritizing review papers)...")
    arxiv_output_dir = os.path.join(project_root, "data", "arxiv_output")
    fetched = []
    try:
        fetched = get_more_papers_from_arxiv(
            search_query,
            outdir=arxiv_output_dir,
            max_results=args.initial_papers,
            candidates=25,
            run_id=run_id,
            prioritize_reviews=True  # Prioritize review papers for initial search
        )
        print(f"Fetched {len(fetched)} papers from arXiv")
    except Exception as e:
        print(f"Error fetching papers from arXiv: {e}")

    # Ingest fetched papers into state
    if fetched:
        before_chunks = len(state["paper_chunks"])
        ingested_txt = 0
        try:
            txts = [p["txt_path"] for p in fetched if p.get("txt_path")]
            if txts:
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
                print(f"Ingested {ingested_txt} text chunks from {len(txts)} papers")
        except Exception as e:
            print(f"Error ingesting text files: {e}")

        # Fallback: ingest abstracts if no txt ingested
        ingested_abs = 0
        if ingested_txt == 0 and fetched:
            ingested_abs = ingest_abstracts_into_state(state, fetched)
            print(f"Ingested {ingested_abs} abstract chunks")
        
        if ingested_txt == 0 and ingested_abs == 0:
            print("Warning: No content was ingested from fetched papers")
    else:
        print("Warning: No papers were fetched from arXiv")

    # Check if we have any content before starting the graph
    if len(state["paper_chunks"]) == 0 or state["chunk_embeddings"] is None:
        print("\nError: No papers were successfully fetched and ingested. Cannot proceed with research.")
        print("Please check your internet connection and try again, or verify the research question.")
        return

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
            "max_total_chunks": MAX_TOTAL_CHUNKS,
            "max_chars_per_file": MAX_CHARS_PER_FILE,
            "iterations": args.iterations,
            "initial_papers": args.initial_papers,
        },
        "messages": state["messages"],
        "evidence_log": state["evidence_log"],
        "refined_query": state.get("refined_query"),
        "final_report": state.get("final_report"),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON log -> {os.path.abspath(log_path)}")
    
    # Save research report in answer format
    answer_data = [
        {
            "id": 1,
            "question": state["research_prompt"],
            "response": state.get("final_report") or ""
        }
    ]
    answer_filename = f"answer_{run_id}.json"
    answer_path = os.path.join(project_root, "data", answer_filename)
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, ensure_ascii=False, indent=4)
    print(f"Saved answer JSON -> {os.path.abspath(answer_path)}")

if __name__ == "__main__":
    main()
