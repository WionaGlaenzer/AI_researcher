# viz_run_log_streamlit.py
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from graphviz import Digraph

st.set_page_config(page_title="Run Log Visualizer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def load_log(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def flatten_source(src: Dict[str, Any]) -> Dict[str, Any]:
    """Prefix all source fields with src_ for display-friendly columns."""
    if not isinstance(src, dict):
        return {}
    out = {}
    for k, v in src.items():
        out[f"src_{k}"] = v
    return out

def to_messages_df(messages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(messages):
        rows.append({
            "idx": i,
            "turn": m.get("turn", i),
            "role": m.get("role", "?"),
            "content": m.get("content", ""),
            "num_snippets": len(m.get("snippets", [])),
        })
    return pd.DataFrame(rows)

def evidence_tables(log: Dict[str, Any]) -> List[pd.DataFrame]:
    """One DataFrame per evidence_log turn, now with source columns if present."""
    tables = []
    for ev in log.get("evidence_log", []):
        rows = []
        for s in ev.get("top_snippets", []):
            base = {
                "turn": ev.get("turn"),
                "phase": ev.get("phase"),
                "query": (ev.get("query") or "")[:120] + ("…" if len(ev.get("query") or "") > 120 else ""),
                "chunk_index": s.get("chunk_index"),
                "score": s.get("score"),
                "approx_char_start": s.get("approx_char_start"),
                "preview": s.get("preview",""),
            }
            base.update(flatten_source(s.get("source", {})))
            rows.append(base)
        if rows:
            df = pd.DataFrame(rows)
            # Prefer a useful column order if present
            preferred_order = [
                "turn","phase","score","chunk_index","approx_char_start",
                "src_origin","src_arxiv_id","src_title","src_doc_id","src_chunk_local_idx","src_url","src_txt_path",
                "query","preview",
            ]
            cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
            df = df[cols]
            tables.append(df)
    return tables

def build_graph(messages: List[Dict[str, Any]]) -> Digraph:
    """
    Nodes:
      T#k  - turn nodes
      S#k:t - snippet nodes (label S#rank | chunk | score | src)
    Edges:
      T#k -> S#k:t
    """
    dot = Digraph("run_graph", graph_attr={"rankdir": "LR", "splines":"spline"})
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#eef5ff", color="#4a6ee0")

    for i, m in enumerate(messages):
        tid = f"T{i}"
        role = m.get("role", "?")
        title = f"{role} (turn {m.get('turn', i)})"
        dot.node(tid, title)

        # Link snippets as circular nodes
        for sn in m.get("snippets", []):
            lbl = sn.get("label", f"S#{sn.get('rank', '?')}")
            chunk = sn.get("chunk_index", "?")
            score = sn.get("score", 0.0)
            approx_start = sn.get("approx_char_start", None)
            src = sn.get("source", {}) or {}
            src_tag = src.get("arxiv_id") or src.get("title") or src.get("doc_id") or ""
            s_id = f"{tid}:{lbl}"
            s_label = f"{lbl}\nchunk={chunk}\nscore={score}"
            if approx_start is not None:
                s_label += f"\nstart≈{approx_start}"
            if src_tag:
                s_label += f"\nsrc={src_tag}"
            dot.node(s_id, s_label, shape="ellipse", style="filled", fillcolor="#fff7e6", color="#f0a500")
            dot.edge(tid, s_id, color="#9aa7c7")
    return dot

def build_sankey(messages: List[Dict[str, Any]], group_by: str = "chunk"):
    """
    Sankey linking Turns -> Targets
    group_by:
      - "chunk": targets are raw chunk indices
      - "source": targets are source doc ids (or titles/arxiv_id as fallback)
    """
    # Left nodes: one per turn
    turn_nodes = [f"{m.get('role','?')}|T{i}" for i, m in enumerate(messages)]

    # Right nodes depend on grouping
    if group_by == "source":
        # derive a doc identifier
        def doc_label(sn):
            src = sn.get("source", {}) or {}
            return src.get("doc_id") or src.get("arxiv_id") or src.get("title") or f"chunk:{sn.get('chunk_index')}"
        targets_list = sorted({doc_label(sn) for m in messages for sn in m.get("snippets", [])})
        target_labels = [f"Doc {t}" for t in targets_list]
        target_lookup_key = lambda sn: f"Doc {doc_label(sn)}"
    else:
        # default: chunk indices
        targets_list = sorted({sn.get("chunk_index") for m in messages for sn in m.get("snippets", []) if "chunk_index" in sn})
        target_labels = [f"Chunk {c}" for c in targets_list]
        target_lookup_key = lambda sn: f"Chunk {sn.get('chunk_index')}"

    labels = turn_nodes + target_labels
    idx_map = {lab: i for i, lab in enumerate(labels)}
    src, tgt, val = [], [], []

    for i, m in enumerate(messages):
        turn_lab = turn_nodes[i]
        for sn in m.get("snippets", []):
            target_lab = target_lookup_key(sn)
            if target_lab not in idx_map:
                continue
            score = sn.get("score", 0.1) or 0.1
            src.append(idx_map[turn_lab])
            tgt.append(idx_map[target_lab])
            val.append(float(score))

    if not src:
        return go.Figure()

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=12, thickness=14),
        link=dict(source=src, target=tgt, value=val),
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=500)
    return fig

def snippet_browser(messages: List[Dict[str, Any]]):
    st.subheader("Snippet Browser")
    # Build a flat table of snippets (including source fields)
    rows = []
    for i, m in enumerate(messages):
        for sn in m.get("snippets", []):
            base = {
                "turn_idx": i,
                "turn": m.get("turn", i),
                "role": m.get("role","?"),
                "label": sn.get("label"),
                "chunk_index": sn.get("chunk_index"),
                "score": sn.get("score"),
                "approx_char_start": sn.get("approx_char_start"),
                "preview": sn.get("preview","")
            }
            base.update(flatten_source(sn.get("source", {})))
            rows.append(base)
    if not rows:
        st.info("No snippet metadata found in messages[].snippets[].")
        return
    df = pd.DataFrame(rows)

    # quick stats
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Snippets", len(df))
    unique_docs = df["src_doc_id"].nunique() if "src_doc_id" in df.columns else 0
    col_b.metric("Unique Sources", unique_docs)
    col_c.metric("Turns with Snippets", df["turn_idx"].nunique())

    # show table
    preferred = [
        "turn","role","label","score","chunk_index","approx_char_start",
        "src_origin","src_arxiv_id","src_title","src_doc_id","src_chunk_local_idx","src_url","src_txt_path",
        "preview"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

# -----------------------------
# UI
# -----------------------------
st.title("Two-Agent Run Log Visualizer")

left, right = st.columns([2, 3])
with left:
    default_path = "run_log.json"
    run_file = st.text_input("Path to run_log.json", value=default_path)
    uploaded = st.file_uploader("...or upload a run_log.json", type=["json"])

if uploaded:
    log = json.loads(uploaded.read().decode("utf-8"))
elif run_file and os.path.exists(run_file):
    log = load_log(run_file)
else:
    st.warning("Provide a path or upload a run_log.json to begin.")
    st.stop()

# -----------------------------
# Summary
# -----------------------------
prompt = log.get("prompt", "")
config = log.get("config", {})
messages = log.get("messages", [])
final_report = log.get("final_report", "")
evidence_log = log.get("evidence_log", [])
chunk_count = sum(len(m.get("snippets", [])) for m in messages)

with right:
    st.markdown("### Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Turns", len(messages))
    s2.metric("Model", config.get("model", ""))
    s3.metric("Top-K", config.get("top_k_snippets", ""))
    s4.metric("Paper", Path(config.get("paper_path","")).name if config.get("paper_path") else "—")

    # Count unique sources across all message snippets
    all_srcs = []
    for m in messages:
        for sn in m.get("snippets", []):
            src = sn.get("source", {}) or {}
            tag = src.get("doc_id") or src.get("arxiv_id") or src.get("title")
            if tag: all_srcs.append(tag)
    st.caption(f"Indexed chunks referenced in messages: {chunk_count} | Unique snippet sources: {len(set(all_srcs))}")

    st.markdown("**Prompt**")
    st.code(prompt or "—", language="markdown")

# -----------------------------
# Final Report
# -----------------------------
st.markdown("### Final Report")
st.markdown(final_report or "_<empty>_")

# -----------------------------
# Conversation Timeline
# -----------------------------
st.markdown("### Conversation")
df_msgs = to_messages_df(messages)
for _, row in df_msgs.iterrows():
    with st.expander(f"{row.role} — turn {row.turn}  (snippets: {row.num_snippets})", expanded=False):
        st.markdown(row.content)
        # Show snippet chips if present
        msg = messages[int(row.idx)]
        if msg.get("snippets"):
            st.caption("Snippets referenced in this turn:")
            chips = pd.DataFrame(msg["snippets"])
            # Flatten nested 'source' dict into columns
            if "source" in chips.columns:
                src = pd.json_normalize(chips["source"])
                chips = pd.concat([chips.drop(columns=["source"]), src.add_prefix("src_")], axis=1)
            st.dataframe(chips, hide_index=True, use_container_width=True)

# -----------------------------
# Evidence Tables (retrieval log)
# -----------------------------
st.markdown("### Retrieval Evidence (per turn)")
tables = evidence_tables(log)
if not tables:
    st.info("No evidence_log found.")
else:
    for i, tdf in enumerate(tables):
        st.markdown(f"**Turn {int(tdf['turn'].iloc[0])}** — query & top snippets")
        st.dataframe(tdf, use_container_width=True, hide_index=True)

# -----------------------------
# Graphs
# -----------------------------
st.markdown("### Graph View")

col_g1, col_g2 = st.columns(2)
with col_g1:
    st.caption("Turn → Snippet nodes")
    dot = build_graph(messages)
    st.graphviz_chart(dot)

with col_g2:
    group_choice = st.radio("Sankey target:", ["chunk", "source"], horizontal=True, index=1)
    fig = build_sankey(messages, group_by=group_choice)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Snippet Browser
# -----------------------------
snippet_browser(messages)

st.success("Loaded and visualized run_log.json ✅")
