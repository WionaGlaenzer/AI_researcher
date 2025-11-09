"""
LangGraph single-agent research lab (Gemini + Embedding Retrieval + arXiv fetch)
Single Agent: Innovator with self-improvement

Flow:
  1) Extract keywords and fetch initial papers from arXiv
  2) Innovator creates initial draft from evidence
  3) Innovator self-improves by reviewing its own work and creating an improved version
  4) Output final report

This is a comparison baseline to test against the multi-agent system.

Run
---
python one_agent_lab_gemini.py --prompt "How do RAG systems reduce hallucination?"
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
from tools import get_more_papers_from_arxiv, ingest_txts_into_state, append_meta_for_txts, ingest_abstracts_into_state
from llm_call import call_llm
from prompts import KEYWORD_EXTRACTION_SYS, INNOVATOR_SYS
from snippet_retrieval import retrieve_snippets, format_snippets, build_snippet_meta

# -----------------------
# Graph nodes for single-agent system
# -----------------------
def innovator_initial_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Innovator creates initial draft from evidence."""
    current_turn = state.get("turn", 0)
    
    print(f"\n{'='*60}")
    print(f"[FLOW] Turn {current_turn} - INNOVATOR INITIAL")
    print(f"{'='*60}")
    
    state["phase"] = "innovator_initial"
    query = state["research_prompt"]
    
    print(f"[INNOVATOR] Retrieving snippets with query: {query[:100]}...")
    snips = retrieve_snippets(state, query)
    print(f"[INNOVATOR] Retrieved {len(snips)} snippets")
    
    # Build context message
    context_parts = [
        f"Research prompt: {state['research_prompt']}\n\n",
        f"Evidence:\n{format_snippets(snips)}"
    ]
    user = "".join(context_parts)
    
    out = call_llm(INNOVATOR_SYS, user, agent_name="INNOVATOR")
    
    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "iteration": 0,
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    state["turn"] += 1
    print(f"[FLOW] Turn {current_turn} complete - Moving to SELF-IMPROVEMENT")
    return state

def innovator_self_improve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Innovator creates an improved version using its own first output."""
    current_turn = state.get("turn", 0)
    
    print(f"\n{'='*60}")
    print(f"[FLOW] Turn {current_turn} - INNOVATOR SELF-IMPROVEMENT")
    print(f"{'='*60}")
    
    state["phase"] = "innovator_self_improve"
    
    # Get the initial draft
    initial_draft = state["messages"][-1]["content"] if state["messages"] else ""
    initial_draft_truncated = initial_draft[:3000] + ("..." if len(initial_draft) > 3000 else "")
    
    # Retrieve snippets again (same query as before)
    query = state["research_prompt"]
    print(f"[INNOVATOR] Retrieving snippets with query: {query[:100]}...")
    snips = retrieve_snippets(state, query)
    print(f"[INNOVATOR] Retrieved {len(snips)} snippets")
    
    # Build context message with initial draft included
    context_parts = [
        f"Research prompt: {state['research_prompt']}\n\n",
        f"[Improvement Pass - Second Iteration]\n\n",
        f"Your initial draft:\n{initial_draft_truncated}\n\n",
        f"Evidence:\n{format_snippets(snips)}\n\n",
        f"Create an improved version of the literature review that builds upon your initial draft, "
        f"fills any gaps, strengthens weak areas, and better addresses the research question."
    ]
    user = "".join(context_parts)
    
    out = call_llm(INNOVATOR_SYS, user, agent_name="INNOVATOR")
    
    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "iteration": 1,
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    
    # Set as final report
    state["final_report"] = out
    state["turn"] += 1
    print(f"[FLOW] Turn {current_turn} complete - Self-improvement finished")
    return state

# -----------------------
# Graph wiring
# -----------------------
def build_graph():
    g = StateGraph(dict)
    
    g.add_node("innovator_initial", innovator_initial_node)
    g.add_node("innovator_self_improve", innovator_self_improve_node)
    
    g.set_entry_point("innovator_initial")
    g.add_edge("innovator_initial", "innovator_self_improve")
    g.add_edge("innovator_self_improve", END)
    
    return g.compile()

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Research question to investigate")
    ap.add_argument("--log", default=None)
    ap.add_argument("--initial_papers", type=int, default=25,
                    help="Number of papers to fetch initially from arXiv. Default: 25")
    args = ap.parse_args()
    
    # Get project root (parent of scr/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate run ID (timestamp-based)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Resolve log path with run ID
    if args.log is None:
        log_filename = f"run_log_single_{run_id}.json"
        log_path = os.path.join(project_root, "output", log_filename)
    elif not os.path.isabs(args.log):
        if args.log.endswith('.json'):
            log_filename = args.log.replace('.json', f'_single_{run_id}.json')
        else:
            log_filename = f"{args.log}_single_{run_id}.json"
        log_path = os.path.join(project_root, "output", log_filename)
    else:
        if args.log.endswith('.json'):
            log_path = args.log.replace('.json', f'_single_{run_id}.json')
        else:
            log_path = f"{args.log}_single_{run_id}.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"\n=== Single-Agent System - Run ID: {run_id} ===\n")
    print(f"Research Question: {args.prompt}")
    print(f"Log file: {log_path}")
    print(f"Initial papers to fetch: {args.initial_papers}")
    print(f"arXiv output: {os.path.join(project_root, 'data', 'arxiv_output', run_id)}\n")
    
    # Extract keywords from the research question
    print("\n[FLOW] Initial Setup - Keyword Extraction")
    print("="*60)
    keywords_text = call_llm(KEYWORD_EXTRACTION_SYS, args.prompt, agent_name="KEYWORD_EXTRACTOR")
    keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:10]
    print(f"[KEYWORD_EXTRACTOR] Extracted {len(keywords)} keywords: {keywords}")
    
    # Use keywords for arXiv search
    search_query = " ".join(keywords) if keywords else args.prompt
    print(f"Using search query: {search_query}\n")
    
    # Initialize empty state
    state: Dict[str, Any] = {
        "research_prompt": args.prompt,
        "extracted_keywords": keywords,
        "paper_chunks": [],
        "chunk_embeddings": None,
        "chunk_meta": [],
        "turn": 0,
        "iteration": 0,
        "messages": [],
        "evidence_log": [],
        "final_report": None,
        "run_id": run_id,
    }
    
    # Fetch initial papers from arXiv
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
            prioritize_reviews=True
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
        "system_type": "single_agent",
        "prompt": state["research_prompt"],
        "config": {
            "model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_snippets": TOP_K_SNIPPETS,
            "max_total_chunks": MAX_TOTAL_CHUNKS,
            "max_chars_per_file": MAX_CHARS_PER_FILE,
            "initial_papers": args.initial_papers,
        },
        "messages": state["messages"],
        "evidence_log": state["evidence_log"],
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
    answer_filename = f"answer_single_{run_id}.json"
    answer_path = os.path.join(project_root, "data", answer_filename)
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, ensure_ascii=False, indent=4)
    print(f"Saved answer JSON -> {os.path.abspath(answer_path)}")

if __name__ == "__main__":
    main()

