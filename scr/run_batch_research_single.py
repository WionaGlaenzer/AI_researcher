"""
Batch research script that runs one_agent_lab_gemini for multiple questions.

Usage:
    python run_batch_research_single.py --questions data/questions.json --num_questions 10 --output data/batch_answers_single.json
"""
from __future__ import annotations
import os
import json
import argparse
import sys
from typing import Any, Dict, List
from datetime import datetime
import numpy as np


class Tee:
    """Write to both file and stdout/stderr"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Import from one_agent_lab_gemini
from one_agent_lab_gemini import build_graph
from config import GEN_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_SNIPPETS, MAX_TOTAL_CHUNKS, MAX_CHARS_PER_FILE
from file_io import chunk_text
from text_embeddings import embed_text
from tools import get_more_papers_from_arxiv, ingest_txts_into_state, append_meta_for_txts, ingest_abstracts_into_state
from llm_call import call_llm
from prompts import KEYWORD_EXTRACTION_SYS


def run_research_for_question(
    question: str,
    question_id: int,
    initial_papers: int = 5,
    base_run_id: str = None
) -> Dict[str, Any]:
    """
    Run the single-agent research process for a single question.
    
    Args:
        question: The research question
        question_id: ID of the question (for logging)
        initial_papers: Number of papers to fetch initially
        base_run_id: Base run ID to use (will append question number)
    
    Returns:
        Dictionary with 'id', 'question', and 'response' keys
    """
    # Get project root (parent of scr/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate run ID
    if base_run_id:
        run_id = f"{base_run_id}_q{question_id}"
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_q{question_id}"
    
    print(f"\n{'='*80}")
    print(f"Processing Question {question_id} (Single-Agent System)")
    print(f"Question: {question[:100]}...")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}\n")

    # Extract keywords from the research question
    print(f"[Q{question_id}] Extracting keywords...")
    keywords_text = call_llm(KEYWORD_EXTRACTION_SYS, question, agent_name="KEYWORD_EXTRACTOR")
    keywords = [k.strip() for k in keywords_text.split(",") if k.strip()][:10]
    print(f"[Q{question_id}] Extracted {len(keywords)} keywords: {keywords}")
    
    # Use keywords for arXiv search
    search_query = " ".join(keywords) if keywords else question

    # Initialize empty state
    state: Dict[str, Any] = {
        "research_prompt": question,
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
    print(f"[Q{question_id}] Fetching initial papers from arXiv (prioritizing review papers)...")
    arxiv_output_dir = os.path.join(project_root, "data", "arxiv_output")
    fetched = []
    try:
        fetched = get_more_papers_from_arxiv(
            search_query,
            outdir=arxiv_output_dir,
            max_results=initial_papers,
            candidates=25,
            run_id=run_id,
            prioritize_reviews=True
        )
        print(f"[Q{question_id}] Fetched {len(fetched)} papers from arXiv")
    except Exception as e:
        print(f"[Q{question_id}] Error fetching papers from arXiv: {e}")

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
                print(f"[Q{question_id}] Ingested {ingested_txt} text chunks from {len(txts)} papers")
        except Exception as e:
            print(f"[Q{question_id}] Error ingesting text files: {e}")

        # Fallback: ingest abstracts if no txt ingested
        ingested_abs = 0
        if ingested_txt == 0 and fetched:
            ingested_abs = ingest_abstracts_into_state(state, fetched)
            print(f"[Q{question_id}] Ingested {ingested_abs} abstract chunks")
        
        if ingested_txt == 0 and ingested_abs == 0:
            print(f"[Q{question_id}] Warning: No content was ingested from fetched papers")
    else:
        print(f"[Q{question_id}] Warning: No papers were fetched from arXiv")

    # Check if we have any content before starting the graph
    if len(state["paper_chunks"]) == 0 or state["chunk_embeddings"] is None:
        print(f"[Q{question_id}] Error: No papers were successfully fetched and ingested.")
        return {
            "id": question_id,
            "question": question,
            "response": f"Error: Could not fetch or ingest papers for this question. Please check your internet connection and try again."
        }

    # Run the research graph
    print(f"[Q{question_id}] Running single-agent research graph...")
    app = build_graph()
    state = app.invoke(state)

    final_report = state.get("final_report") or ""
    print(f"[Q{question_id}] Completed. Report length: {len(final_report):,} chars (~{len(final_report.split()):,} words)")

    # Save JSON run log for this question
    log_dir = os.path.join(project_root, "output")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"run_log_single_{run_id}.json"
    log_path = os.path.join(log_dir, log_filename)
    
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
            "initial_papers": initial_papers,
        },
        "messages": state["messages"],
        "evidence_log": state["evidence_log"],
        "final_report": state.get("final_report"),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[Q{question_id}] Saved JSON log -> {os.path.abspath(log_path)}")

    return {
        "id": question_id,
        "question": question,
        "response": final_report
    }


def main():
    ap = argparse.ArgumentParser(description="Run batch research for multiple questions (single-agent system)")
    ap.add_argument("--questions", required=True, help="Path to questions.json file")
    ap.add_argument("--num_questions", type=int, default=10, 
                    help="Number of questions to process (default: 10)")
    ap.add_argument("--output", default=None,
                    help="Output JSON file path (default: data/batch_answers_single_{timestamp}.json)")
    ap.add_argument("--initial_papers", type=int, default=10,
                    help="Number of papers to fetch initially per question (default: 25)")
    ap.add_argument("--start_from", type=int, default=1,
                    help="Start from question ID (default: 1)")
    args = ap.parse_args()

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve questions file path
    if not os.path.isabs(args.questions):
        questions_path = os.path.join(project_root, args.questions)
    else:
        questions_path = args.questions
    
    # Load questions
    print(f"Loading questions from: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    
    # Filter and sort questions
    questions_to_process = [
        q for q in all_questions 
        if isinstance(q, dict) and q.get("id", 0) >= args.start_from
    ]
    questions_to_process.sort(key=lambda x: x.get("id", 0))
    questions_to_process = questions_to_process[:args.num_questions]
    
    if not questions_to_process:
        print("No questions found to process.")
        return
    
    print(f"\nProcessing {len(questions_to_process)} questions with SINGLE-AGENT system")
    print(f"Question IDs: {[q.get('id') for q in questions_to_process]}")
    print(f"Initial papers per question: {args.initial_papers}\n")
    
    # Generate base run ID
    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up output file for logging
    if args.output is None:
        output_filename = f"batch_answers_single_{base_run_id}.json"
        output_path = os.path.join(project_root, "data", output_filename)
    elif not os.path.isabs(args.output):
        output_path = os.path.join(project_root, args.output)
    else:
        output_path = args.output
    
    # Create .out file path (same name as output but with .out extension)
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    if output_basename.endswith('.json'):
        out_filename = output_basename.replace('.json', '.out')
    else:
        out_filename = f"{output_basename}.out"
    out_file_path = os.path.join(output_dir, out_filename)
    
    # Save original stdout/stderr before redirecting
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Open output file and create Tee objects
    out_file = open(out_file_path, "w", encoding="utf-8")
    tee_stdout = Tee(original_stdout, out_file)
    tee_stderr = Tee(original_stderr, out_file)
    
    # Redirect to Tee
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    # Print initial message (will go to both console and file)
    print(f"Saving output log to: {os.path.abspath(out_file_path)}")
    
    try:
        # Process each question
        results = []
        for idx, q_data in enumerate(questions_to_process, 1):
            question_id = q_data.get("id", idx)
            question = q_data.get("question", "")
            
            if not question:
                print(f"[Q{question_id}] Skipping: No question text found")
                continue
            
            try:
                result = run_research_for_question(
                    question=question,
                    question_id=question_id,
                    initial_papers=args.initial_papers,
                    base_run_id=base_run_id
                )
                results.append(result)
                print(f"[Q{question_id}] ✓ Successfully completed\n")
            except Exception as e:
                print(f"[Q{question_id}] ✗ Error: {e}\n")
                results.append({
                    "id": question_id,
                    "question": question,
                    "response": f"Error processing question: {str(e)}"
                })
        
        # Save results
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"\n{'='*80}")
        print(f"Batch processing complete! (Single-Agent System)")
        print(f"Processed {len(results)} questions")
        print(f"Results saved to: {os.path.abspath(output_path)}")
        print(f"Output log saved to: {os.path.abspath(out_file_path)}")
        print(f"{'='*80}\n")
    
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        out_file.close()


if __name__ == "__main__":
    main()

