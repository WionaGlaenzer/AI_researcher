# -----------------------
# Graph nodes
# -----------------------
from typing import Any, Dict
import os
from config import MAX_TOTAL_CHUNKS, MAX_CHARS_PER_FILE
from prompts import INNOVATOR_SYS, CRITIC_SYS, REFINER_SYS, SYNTHESIS_SYS
from llm_call import call_llm
from snippet_retrieval import retrieve_snippets, format_snippets, build_snippet_meta
from tools import get_more_papers_from_arxiv, ingest_txts_into_state, append_meta_for_txts, ingest_abstracts_into_state
from file_io import chunk_text
from text_embeddings import embed_text

def innovator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1)
    
    # Determine phase based on iteration
    if current_iteration == 0:
        state["phase"] = "innovator_initial"
    else:
        state["phase"] = f"innovator_iteration_{current_iteration + 1}"
    
    # For subsequent iterations, include context from previous synthesis
    query = state["research_prompt"]
    if state["messages"]:
        # Use the latest synthesis as context for retrieval
        last_synthesis = next(
            (m["content"] for m in reversed(state["messages"]) 
             if m.get("role") == "Innovator" and "refined_query" in m),
            None
        )
        if last_synthesis:
            query = f"{state['research_prompt']}\n\nPrevious synthesis: {last_synthesis[:500]}"
    
    snips = retrieve_snippets(state, query)
    
    # Build context message
    context_parts = [f"Research prompt: {state['research_prompt']}"]
    if current_iteration > 0:
        context_parts.append(f"\n[Iteration {current_iteration + 1} of {max_iterations}]")
        # Include previous critic feedback if available
        last_crit = next(
            (m["content"] for m in reversed(state["messages"]) if m.get("role") == "Critic"),
            None
        )
        if last_crit:
            context_parts.append(f"\nPrevious Critic feedback:\n{last_crit}")
    
    context_parts.append(f"\n\nEvidence:\n{format_snippets(snips)}")
    user = "".join(context_parts)
    
    out = call_llm(INNOVATOR_SYS, user)

    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "iteration": current_iteration,
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

    current_iteration = state.get("iteration", 0)
    state["messages"].append({
        "role": "Critic",
        "turn": state["turn"],
        "iteration": current_iteration,
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
        # Get project root (parent of scr/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        arxiv_output_dir = os.path.join(project_root, "data", "arxiv_output")
        
        # Get run_id from state if available
        run_id = state.get("run_id")
        
        fetched = get_more_papers_from_arxiv(
            refined_query,
            outdir=arxiv_output_dir,
            max_results=5,
            candidates=25,
            run_id=run_id
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

    current_iteration = state.get("iteration", 0)
    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "iteration": current_iteration,
        "content": out,
        "snippets": build_snippet_meta(snips_refined),
        "refined_query": refined_query,
    })
    
    # Update final report (will be overwritten on each iteration, keeping the last one)
    state["final_report"] = out
    
    # Increment iteration counter at the end of refine_and_synthesize
    state["iteration"] = current_iteration + 1
    state["turn"] += 1
    return state