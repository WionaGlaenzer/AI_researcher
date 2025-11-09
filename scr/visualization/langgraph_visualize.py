#!/usr/bin/env python3
"""
LangGraph visualization for the AI research agent system.

What it does:
  1) Builds the complete research agent graph (including keyword extraction and arXiv tool calls)
  2) Saves Mermaid visualization as PDF to: graph_mermaid_api.pdf
"""

import sys
from typing import TypedDict, Literal
from PIL import Image
import io

# --- LangGraph imports ---
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.runnables.graph import MermaidDrawMethod
except Exception as e:
    print("Error: Requires 'langgraph' and 'langchain-core'.\n"
          "Install with: pip install langgraph langchain-core", file=sys.stderr)
    raise

# ---- State definition ----
class State(TypedDict, total=False):
    research_prompt: str
    extracted_keywords: list
    paper_chunks: list
    iteration: int
    max_iterations: int
    messages: list
    final_report: str

# ---- Node implementations (simple stubs so the script runs) ----
def keyword_extraction_node(state: State) -> State:
    """Extracts keywords from the research question using LLM."""
    prompt = state.get("research_prompt", "Research question")
    # Simulate keyword extraction
    keywords = ["keyword1", "keyword2", "keyword3"]
    return {**state, "extracted_keywords": keywords}

def initial_arxiv_fetch_node(state: State) -> State:
    """Fetches initial papers from arXiv using extracted keywords."""
    keywords = state.get("extracted_keywords", [])
    # Simulate arXiv fetch
    paper_chunks = state.get("paper_chunks", [])
    paper_chunks.extend([f"Paper chunk {i}" for i in range(5)])
    return {**state, "paper_chunks": paper_chunks}

def innovator_node(state: State) -> State:
    """Innovator agent: retrieves snippets and drafts literature review sections."""
    iteration = state.get("iteration", 0)
    prompt = state.get("research_prompt", "Research question")
    # Simulate innovator work
    idea = f"Innovator draft for iteration {iteration + 1}"
    messages = state.get("messages", [])
    messages.append({"role": "Innovator", "content": idea})
    return {**state, "messages": messages}

def critic_node(state: State) -> State:
    """Critic agent: analyzes innovator output and provides feedback."""
    iteration = state.get("iteration", 0)
    # Simulate critic feedback
    critique = f"Critic feedback for iteration {iteration + 1}"
    messages = state.get("messages", [])
    messages.append({"role": "Critic", "content": critique})
    # Increment iteration
    return {**state, "messages": messages, "iteration": iteration + 1}

def refine_and_synthesize_node(state: State) -> State:
    """Refines query, fetches more papers from arXiv, and synthesizes final report."""
    # Simulate refined query generation
    refined_query = "Refined search query"
    
    # Simulate additional arXiv fetch (happens inside this node)
    paper_chunks = state.get("paper_chunks", [])
    paper_chunks.extend([f"Additional paper chunk {i}" for i in range(3)])
    
    # Simulate final synthesis
    final_report = "Final comprehensive literature review report"
    messages = state.get("messages", [])
    messages.append({"role": "Innovator", "content": final_report, "refined_query": refined_query})
    
    return {
        **state,
        "paper_chunks": paper_chunks,
        "messages": messages,
        "final_report": final_report,
    }

def should_continue(state: State) -> Literal["continue", "end"]:
    """Conditional edge: continue iterating or finish."""
    max_iterations = state.get("max_iterations", 1)
    current_iteration = state.get("iteration", 0)
    return "continue" if current_iteration < max_iterations else "end"

# ---- Graph builder (matches your actual system structure) ----
def build_graph():
    g = StateGraph(dict)

    # Add all nodes
    g.add_node("keyword_extraction", keyword_extraction_node)
    g.add_node("initial_arxiv_fetch", initial_arxiv_fetch_node)
    g.add_node("innovator", innovator_node)
    g.add_node("critic", critic_node)
    g.add_node("refine_and_synthesize", refine_and_synthesize_node)

    # Set entry point
    g.set_entry_point("keyword_extraction")
    
    # Flow: keyword_extraction -> initial_arxiv_fetch -> innovator
    g.add_edge("keyword_extraction", "initial_arxiv_fetch")
    g.add_edge("initial_arxiv_fetch", "innovator")
    
    # Flow: innovator -> critic
    g.add_edge("innovator", "critic")
    
    # Conditional loop: critic -> (continue -> innovator, end -> refine_and_synthesize)
    g.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue": "innovator",
            "end": "refine_and_synthesize",
        },
    )
    
    # Final step: refine_and_synthesize -> END
    g.add_edge("refine_and_synthesize", END)

    return g.compile()

# ---- Visualization helper ----
MERMAID_API_PDF_PATH = "graph_mermaid_api.pdf"

def write_mermaid_api_pdf(app, path: str) -> None:
    """Generate Mermaid PDF using the API method (requires network access)."""
    try:
        # Get PNG from Mermaid API
        png_bytes = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        
        # Convert PNG to PDF using PIL
        img = Image.open(io.BytesIO(png_bytes))
        # Convert to RGB if necessary (PDF doesn't support RGBA)
        if img.mode == 'RGBA':
            # Create white background
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Upscale image for higher resolution (2x for 300 DPI equivalent)
        # This ensures crisp text and lines in the PDF
        upscale_factor = 2
        new_size = (img.size[0] * upscale_factor, img.size[1] * upscale_factor)
        img_high_res = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save as PDF with high resolution (300 DPI)
        img_high_res.save(path, 'PDF', resolution=300.0)
        print(f"✓ Wrote {path}")
    except Exception as e:
        print(f"✗ Mermaid API PDF failed (needs network access): {e}", file=sys.stderr)
        return

# ---- Main ----
def main():
    app = build_graph()
    write_mermaid_api_pdf(app, MERMAID_API_PDF_PATH)

if __name__ == "__main__":
    main()
