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
    current_turn = state.get("turn", 0)
    
    print(f"\n{'='*60}")
    print(f"[FLOW] Turn {current_turn} - INNOVATOR NODE")
    print(f"[FLOW] Iteration {current_iteration + 1}/{max_iterations}")
    print(f"{'='*60}")
    
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
    
    print(f"[INNOVATOR] Retrieving snippets with query: {query[:100]}...")
    snips = retrieve_snippets(state, query)
    print(f"[INNOVATOR] Retrieved {len(snips)} snippets")
    
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
            # Truncate critic feedback to avoid overly long inputs
            truncated_crit = last_crit[:2000] + ("..." if len(last_crit) > 2000 else "")
            context_parts.append(f"\nPrevious Critic feedback:\n{truncated_crit}")
    
    context_parts.append(f"\n\nEvidence:\n{format_snippets(snips)}")
    user = "".join(context_parts)
    
    out = call_llm(INNOVATOR_SYS, user, agent_name="INNOVATOR")

    state["messages"].append({
        "role": "Innovator",
        "turn": state["turn"],
        "iteration": current_iteration,
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    state["turn"] += 1
    print(f"[FLOW] Turn {current_turn} complete - Moving to CRITIC")
    return state

def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("turn", 0)
    current_iteration = state.get("iteration", 0)
    
    print(f"\n{'='*60}")
    print(f"[FLOW] Turn {current_turn} - CRITIC NODE")
    print(f"[FLOW] Iteration {current_iteration + 1}")
    print(f"{'='*60}")
    
    state["phase"] = "critic_feedback"
    innov = state["messages"][-1]["content"]
    
    # Truncate innovator content if too long
    innov_truncated = innov[:3000] + ("..." if len(innov) > 3000 else "")
    
    print(f"[CRITIC] Analyzing innovator output ({len(innov):,} chars)")
    snips = retrieve_snippets(state, innov)
    print(f"[CRITIC] Retrieved {len(snips)} snippets for analysis")
    
    user = f"Innovator said:\n{innov_truncated}\n\nEvidence:\n{format_snippets(snips)}"
    out = call_llm(CRITIC_SYS, user, agent_name="CRITIC")

    state["messages"].append({
        "role": "Critic",
        "turn": state["turn"],
        "iteration": current_iteration,
        "content": out,
        "snippets": build_snippet_meta(snips),
    })
    state["turn"] += 1
    
    # Increment iteration after each innovator->critic cycle completes
    # This prevents infinite loops when max_iterations > 1
    state["iteration"] = current_iteration + 1
    print(f"[FLOW] Turn {current_turn} complete - Iteration {current_iteration + 1} of {state.get('max_iterations', 1)} complete")
    print(f"[FLOW] Moving to next step (will check if more iterations needed)")
    return state

def refine_and_synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_turn = state.get("turn", 0)
    current_iteration = state.get("iteration", 0)
    
    print(f"\n{'='*60}")
    print(f"[FLOW] Turn {current_turn} - REFINE & SYNTHESIZE NODE")
    print(f"[FLOW] Iteration {current_iteration + 1}")
    print(f"{'='*60}")
    
    # --- write refined query ---
    state["phase"] = "refine_query"
    last_innov = next(m for m in reversed(state["messages"]) if m["role"] == "Innovator")["content"]
    last_crit  = next(m for m in reversed(state["messages"]) if m["role"] == "Critic")["content"]
    
    # Truncate inputs for query refinement
    innov_truncated = last_innov[:2000] + ("..." if len(last_innov) > 2000 else "")
    crit_truncated = last_crit[:2000] + ("..." if len(last_crit) > 2000 else "")

    print(f"[REFINER] Creating refined query from innovator ({len(last_innov):,} chars) and critic ({len(last_crit):,} chars)")
    refined_query = call_llm(
        REFINER_SYS,
        (
            f"Original research prompt: {state['research_prompt']}\n\n"
            f"Latest Innovator draft:\n{innov_truncated}\n\n"
            f"Critic feedback:\n{crit_truncated}\n\n"
            f"Write a single refined retrieval query:"
        ),
        agent_name="REFINER"
    )
    state["refined_query"] = refined_query
    print(f"[REFINER] Refined query: {refined_query}")

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

    # --- synthesize final research report ---
    state["phase"] = "synthesis"
    evidence_block = format_snippets(snips_refined)
    print(f"[SYNTHESIS] Creating final report with {len(snips_refined)} refined snippets")
    print(f"[SYNTHESIS] Evidence block length: {len(evidence_block):,} chars")
    
    user = (
        f"Research Question: {state['research_prompt']}\n\n"
        f"Refined retrieval query: {refined_query}\n\n"
        f"Refined Evidence:\n{evidence_block}\n\n"
        f"Use the refined evidence plus prior discussion to craft a comprehensive literature review "
        f"that summarizes and synthesizes existing knowledge to fully address the research question. "
        f"Focus on important concepts, findings, and insights from the literature."
    )
    out = call_llm(SYNTHESIS_SYS, user, agent_name="SYNTHESIS")

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
    
    # Note: iteration is already incremented in critic_node, so we don't increment here
    # This node runs once per iteration cycle, after the innovator->critic loop completes
    state["turn"] += 1
    report_words = len(out.split())
    print(f"[FLOW] Turn {current_turn} complete - Refine & Synthesize finished for iteration {current_iteration + 1}")
    print(f"[FLOW] Final report length: {len(out):,} chars (~{report_words:,} words, target: ~7000 words)")
    return state