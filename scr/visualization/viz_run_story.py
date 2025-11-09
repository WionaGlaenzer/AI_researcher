import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import streamlit as st

# ------------------------------
# Helpers & Data Structures
# ------------------------------

@dataclass
class Source:
    doc_id: Optional[str] = None
    origin: Optional[str] = None
    title: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    txt_path: Optional[str] = None
    chunk_local_idx: Optional[int] = None

@dataclass
class Snippet:
    rank: Optional[int] = None
    label: Optional[str] = None
    chunk_index: Optional[int] = None
    approx_char_start: Optional[int] = None
    score: Optional[float] = None
    preview: Optional[str] = None
    source: Optional[Source] = None

@dataclass
class Message:
    role: str = ""
    turn: int = 0
    content: str = ""
    snippets: List[Snippet] = field(default_factory=list)
    refined_query: Optional[str] = None  # present on some messages

@dataclass
class EvidenceItem:
    turn: int
    phase: str
    query: Optional[str] = None
    top_snippets: List[Snippet] = field(default_factory=list)
    fetched_count: Optional[int] = None
    ingested_txt_chunks: Optional[int] = None
    ingested_abstract_chunks: Optional[int] = None
    chunks_before: Optional[int] = None
    chunks_after: Optional[int] = None
    error: Optional[str] = None

@dataclass
class Story:
    prompt: str
    config: Dict[str, Any]
    messages: List[Message]
    evidence_log: List[EvidenceItem]
    refined_query: Optional[str] = None
    final_report: Optional[str] = None

# ------------------------------
# Parsing functions
# ------------------------------

def parse_source(d: Dict[str, Any]) -> Source:
    if d is None:
        return Source()
    return Source(
        doc_id=d.get("doc_id"), origin=d.get("origin"), title=d.get("title"),
        arxiv_id=d.get("arxiv_id"), url=d.get("url"), txt_path=d.get("txt_path"),
        chunk_local_idx=d.get("chunk_local_idx")
    )

def parse_snippet(d: Dict[str, Any]) -> Snippet:
    if d is None:
        return Snippet()
    return Snippet(
        rank=d.get("rank"), label=d.get("label"), chunk_index=d.get("chunk_index"),
        approx_char_start=d.get("approx_char_start"), score=d.get("score"),
        preview=d.get("preview"), source=parse_source(d.get("source"))
    )

def parse_message(d: Dict[str, Any]) -> Message:
    return Message(
        role=d.get("role", ""),
        turn=int(d.get("turn", 0)),
        content=d.get("content", ""),
        snippets=[parse_snippet(s) for s in d.get("snippets", [])],
        refined_query=d.get("refined_query")
    )

def parse_evidence_item(d: Dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        turn=int(d.get("turn", 0)),
        phase=d.get("phase", ""),
        query=d.get("query"),
        top_snippets=[parse_snippet(s) for s in d.get("top_snippets", [])],
        fetched_count=d.get("fetched_count"),
        ingested_txt_chunks=d.get("ingested_txt_chunks"),
        ingested_abstract_chunks=d.get("ingested_abstract_chunks"),
        chunks_before=d.get("chunks_before"),
        chunks_after=d.get("chunks_after"),
        error=d.get("error"),
    )

def parse_story(raw: Dict[str, Any]) -> Story:
    return Story(
        prompt=raw.get("prompt", ""),
        config=raw.get("config", {}),
        messages=[parse_message(m) for m in raw.get("messages", [])],
        evidence_log=[parse_evidence_item(e) for e in raw.get("evidence_log", [])],
        refined_query=raw.get("refined_query"),
        final_report=raw.get("final_report"),
    )

# ------------------------------
# UI components
# ------------------------------

def snippet_card(sn: Snippet):
    with st.container(border=True):
        left, right = st.columns([3,1])
        with left:
            if sn.label:
                st.caption(f"**{sn.label}** | chunk #{sn.chunk_index} | start≈{sn.approx_char_start}")
            if sn.preview:
                st.markdown(sn.preview)
        with right:
            if sn.score is not None:
                st.metric("Score", f"{sn.score:.4f}")
        if sn.source:
            s = sn.source
            meta = []
            if s.origin: meta.append(f"origin: `{s.origin}`")
            if s.title: meta.append(f"title: _{s.title}_")
            if s.doc_id: meta.append(f"doc_id: `{s.doc_id}`")
            if s.arxiv_id: meta.append(f"arXiv: `{s.arxiv_id}`")
            if s.url: meta.append(f"url available")
            if s.txt_path: meta.append(f"txt_path: `{s.txt_path}`")
            st.caption(" | ".join(meta))


def show_message(msg: Message):
    role_badge = {
        "Innovator": "🧪 Innovator",
        "Critic": "🧭 Critic",
    }.get(msg.role, msg.role)
    st.subheader(f"Turn {msg.turn} — {role_badge}")
    st.markdown(msg.content)
    if msg.refined_query:
        with st.expander("Refined query suggested by this turn"):
            st.code(msg.refined_query, language="text")

    if msg.snippets:
        st.markdown("#### Snippets used in this turn")
        for sn in msg.snippets:
            snippet_card(sn)


def show_evidence_for_turn(evidence_items: List[EvidenceItem], turn: int) -> bool:
    """Return True if any error is present for this turn."""
    turn_items = [e for e in evidence_items if e.turn == turn]
    if not turn_items:
        st.info("No evidence log for this turn.")
        return False

    has_error = False
    st.markdown("### Tool calls & retrieval log")
    for e in turn_items:
        with st.container(border=True):
            top = st.columns([3,1])[0]
            with top:
                st.caption(f"Phase: `{e.phase}`")
            if e.query:
                st.code(e.query, language="text")

            kv = []
            if e.fetched_count is not None: kv.append(("fetched_count", e.fetched_count))
            if e.ingested_txt_chunks is not None: kv.append(("ingested_txt_chunks", e.ingested_txt_chunks))
            if e.ingested_abstract_chunks is not None: kv.append(("ingested_abstract_chunks", e.ingested_abstract_chunks))
            if e.chunks_before is not None: kv.append(("chunks_before", e.chunks_before))
            if e.chunks_after is not None: kv.append(("chunks_after", e.chunks_after))
            if kv:
                cols = st.columns(len(kv))
                for (k, v), c in zip(kv, cols):
                    with c:
                        st.metric(k, v)

            if e.top_snippets:
                with st.expander("Top snippets (retrieval)"):
                    for sn in e.top_snippets:
                        snippet_card(sn)

            if e.error:
                has_error = True
                st.error(e.error)
    return has_error


# ------------------------------
# Page Navigation Logic
# ------------------------------

def init_state(turn_values: List[int]):
    turn_values = sorted(turn_values) if turn_values else [0]
    if "turn_values" not in st.session_state:
        st.session_state.turn_values = turn_values
    else:
        st.session_state.turn_values = turn_values
    if "min_turn" not in st.session_state:
        st.session_state.min_turn = turn_values[0]
    else:
        st.session_state.min_turn = turn_values[0]
    if "max_turn" not in st.session_state:
        st.session_state.max_turn = turn_values[-1]
    else:
        st.session_state.max_turn = turn_values[-1]
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = st.session_state.min_turn


def goto_turn(t: int, *_ignore):
    # Accept optional extraneous args for backward compatibility
    min_turn = st.session_state.get("min_turn", 0)
    max_turn = st.session_state.get("max_turn", 0)
    new_turn = max(min_turn, min(t, max_turn))
    if new_turn != st.session_state.current_turn:
        st.session_state.current_turn = new_turn
        st.rerun()


# ------------------------------
# Main App
# ------------------------------

def main():
    st.set_page_config(page_title="AI Researcher Story Viewer", page_icon="🧬", layout="wide")
    st.title("🧬 AI Researcher Story Viewer")
    st.caption("Click through turns, see model outputs, tool calls, retrieval snippets, and follow errors to the next step.")

    # --- Upload & parse ---
    story: Optional[Story] = None
    raw_obj: Optional[Dict[str, Any]] = None

    with st.sidebar:
        st.header("Load a story JSON")
        uploaded = st.file_uploader("Upload your story JSON", type=["json"])  # noqa: F841
        if uploaded is None:
            st.info("Upload a JSON file to get started.")
        else:
            try:
                raw_obj = json.load(uploaded)
                story = parse_story(raw_obj)
            except Exception as ex:
                st.error(f"Failed to parse uploaded JSON: {ex}")

    # Stop rendering the rest of the page until we have a story
    if story is None:
        st.stop()

    # Build per-turn index
    turns = sorted({m.turn for m in story.messages}) if story.messages else [0]
    init_state(turns)

    # Sidebar info & navigation (requires story)
    with st.sidebar:
        st.divider()
        st.subheader("Prompt")
        st.code(story.prompt or "<empty>", language="text")
        st.subheader("Config")
        st.json(story.config)

        st.divider()
        st.subheader("Turn Navigation")
        if story.messages:
            current_turn_display = st.session_state.get("current_turn", turns[0])
            st.metric("Current Turn", f"{current_turn_display} / {turns[-1]}")
            st.caption(f"Turn range: {turns[0]} to {turns[-1]}")
        else:
            st.write("No turns found in this JSON.")
            if "current_turn" not in st.session_state:
                st.session_state.current_turn = 0
        st.caption("Use ←/→ buttons to navigate between turns.")

        if story.refined_query:
            st.divider()
            st.subheader("Global refined query")
            st.code(story.refined_query, language="text")

    # Main area per-turn view
    # Ensure current_turn is initialized
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = turns[0] if turns else 0
    current_turn = st.session_state.current_turn

    # Collect messages for this turn
    these_msgs = [m for m in story.messages if m.turn == current_turn]
    if not these_msgs:
        st.info("No messages for this turn.")
    else:
        role_order = {"Innovator": 0, "Critic": 1}
        these_msgs.sort(key=lambda m: (role_order.get(m.role, 99), m.role))
        for msg in these_msgs:
            show_message(msg)

    # Evidence / Tool calls
    has_error = show_evidence_for_turn(story.evidence_log, current_turn)

    # Final / Reports for last turn
    if story.final_report and (current_turn == (turns[-1] if turns else 0)):
        st.markdown("### 🧾 Final Report")
        st.markdown(story.final_report)

    # Navigation buttons
    st.divider()
    cols = st.columns([1,1,6,2])
    with cols[0]:
        if st.button("⬅️ Previous", use_container_width=True):
            goto_turn(current_turn - 1)
    with cols[1]:
        if st.button("Next ➡️", use_container_width=True):
            goto_turn(current_turn + 1)
    with cols[3]:
        if has_error and st.button("Jump to next after error ⚠️", use_container_width=True):
            goto_turn(current_turn + 1)

    # Footer
    st.caption("Tip: Upload a JSON with `messages`, `evidence_log`, and optional `final_report`. The app will auto-wire snippets and tool calls per turn.")


if __name__ == "__main__":
    main()
