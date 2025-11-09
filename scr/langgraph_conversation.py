"""
Two-Agent Research Lab (Gemini Edition) — no langgraph.types.State needed
Agents:
  - Innovator -> Critic -> Innovator (Synthesis)
Run:
  pip install -U langgraph google-generativeai numpy scikit-learn
  export GEMINI_API_KEY=your_key
  python two_agent_lab_gemini.py --prompt "How do RAG systems reduce hallucination?" --paper paper.txt
"""
from __future__ import annotations
import os, re, json, argparse
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Config ----
GEN_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash-lite")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K_SNIPPETS = 4

# ---- IO ----
def load_paper(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i+size])
        i += step
    return out

# ---- Retrieval ----
def make_tfidf(chunks: List[str]):
    v = TfidfVectorizer(lowercase=True, stop_words="english")
    X = v.fit_transform(chunks)
    return v, X

def retrieve_snippets(state: Dict[str, Any], query: str, k=TOP_K_SNIPPETS):
    vectorizer, X = state["tfidf"]
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:k]
    results = [(int(i), state["paper_chunks"][int(i)], float(sims[int(i)])) for i in idxs]
    state["evidence_log"].append({
        "turn": state["turn"],
        "query": query,
        "top_snippets": [{"chunk": i, "score": s, "preview": txt[:200]} for i, txt, s in results],
    })
    return results

def format_snippets(snips):
    return "\n".join(
        f"S#{rank} [chunk {idx}, score {score:.2f}]: {re.sub(r'\\s+',' ',text)[:300]}"
        for rank, (idx, text, score) in enumerate(snips, 1)
    )

# ---- Gemini ----
def call_llm(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY first.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEN_MODEL, system_instruction=system_prompt)
    resp = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.4),
    )
    return (resp.text or "").strip()

# ---- Prompts ----
INNOVATOR_SYS = """You are the Innovator. Propose insights grounded in the evidence.
Cite evidence like (S#1), (S#2). Keep claims scoped and avoid over-generalization. Return 3–6 bullets."""
CRITIC_SYS = """You are the Critic. Identify unsupported assumptions, missing controls, or weak evidence.
Suggest 2+ specific improvements. Be concise."""
SYNTHESIS_SYS = """You are the Innovator synthesizing the final report.
Merge supported claims, downgrade weak ones, and produce 1 falsifiable hypothesis.
Include an Evidence Table mapping claims to snippet refs."""

# ---- Nodes ----
def innovator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["messages"][-1]["content"] if state["messages"] else state["research_prompt"]
    snips = retrieve_snippets(state, query)
    user = f"Research prompt: {state['research_prompt']}\n\nEvidence:\n{format_snippets(snips)}"
    out = call_llm(INNOVATOR_SYS, user)
    state["messages"].append({"role": "Innovator", "turn": state["turn"], "content": out})
    state["turn"] += 1
    return state

def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    innov = state["messages"][-1]["content"]
    snips = retrieve_snippets(state, innov)
    user = f"Innovator said:\n{innov}\n\nEvidence:\n{format_snippets(snips)}"
    out = call_llm(CRITIC_SYS, user)
    state["messages"].append({"role": "Critic", "turn": state["turn"], "content": out})
    state["turn"] += 1
    return state

def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    last_innov = next(m for m in reversed(state["messages"]) if m["role"] == "Innovator")["content"]
    last_crit = next(m for m in reversed(state["messages"]) if m["role"] == "Critic")["content"]
    snips = retrieve_snippets(state, state["research_prompt"])
    user = (
        f"Prompt: {state['research_prompt']}\n\nInnovator:\n{last_innov}\n\n"
        f"Critic:\n{last_crit}\n\nEvidence:\n{format_snippets(snips)}"
    )
    out = call_llm(SYNTHESIS_SYS, user)
    state["messages"].append({"role": "Innovator", "turn": state["turn"], "content": out})
    state["final_report"] = out
    return state

# ---- Graph ----
def build_graph():
    g = StateGraph(dict)          # <— plain dict state
    g.add_node("innovator", innovator_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesis", synthesis_node)
    g.set_entry_point("innovator")
    g.add_edge("innovator", "critic")
    g.add_edge("critic", "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()

# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--paper", required=True)
    args = ap.parse_args()

    text = load_paper(args.paper)
    chunks = chunk_text(text)
    tfidf = make_tfidf(chunks)

    state: Dict[str, Any] = {
        "research_prompt": args.prompt,
        "paper_chunks": chunks,
        "tfidf": tfidf,
        "turn": 0,
        "messages": [],
        "evidence_log": [],
        "final_report": None,
    }

    app = build_graph()
    state = app.invoke(state)

    print("\n=== Transcript ===\n")
    for m in state["messages"]:
        print(f"[{m['role']}]\n{m['content']}\n")

    print("\n=== Final Report ===\n")
    print(state["final_report"] or "")

if __name__ == "__main__":
    main()
