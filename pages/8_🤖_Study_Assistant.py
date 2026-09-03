"""
pages/8_🤖_Study_Assistant.py
==============================
Phase 10 — RAG-Powered AI Study Assistant

This page:
1. Loads all paper texts from the database / processed files
2. Chunks them using src/chunker.py
3. Builds (or loads) a FAISS vector index using src/rag_retriever.py
4. Provides a chat interface
5. On each message, retrieves relevant chunks and optionally calls an LLM
6. Shows the answer plus source documents the answer came from

OFFLINE MODE (no API key):
  Returns the most relevant excerpts from the uploaded papers.
  The student can read the actual source material directly.

LLM MODE (API key provided):
  Returns a synthesized, student-friendly answer grounded in the papers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from database.database import get_db_connection
from src.chunker import chunk_papers
from src.rag_retriever import RAGIndex
from src.rag_assistant import ask, AssistantResponse

st.set_page_config(
    page_title="Study Assistant — QPredict",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "rag_index.pkl"
)
INDEX_CACHE_PATH = os.path.normpath(INDEX_CACHE_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_papers_from_db() -> list[dict]:
    """
    Load paper content for the RAG index.

    Primary source: processed .txt files in data/processed/.
    Fallback (Streamlit Cloud / no local files): reconstruct text from
    the questions stored in the database.

    Returns list of dicts: {paper_id, source_name, year, text}
    """
    from database.database import get_all_papers, get_questions_for_paper

    papers = get_all_papers()
    if not papers:
        return []

    processed_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )

    result = []
    for p in papers:
        paper_id = p["id"]
        filename = p["filename"]
        year     = p.get("year")

        # ── Try the saved .txt file first ────────────────────────────────
        txt_name = os.path.splitext(filename)[0] + "_extracted.txt"
        txt_path = os.path.join(processed_dir, txt_name)
        text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        # ── Fallback: build text from questions in the DB ─────────────────
        if not text.strip():
            questions = get_questions_for_paper(paper_id)
            if questions:
                lines = [f"Examination paper: {filename}  Year: {year or 'unknown'}"]
                for q in questions:
                    num  = q.get("question_number", "")
                    qtxt = q.get("question_text", "")
                    if qtxt:
                        lines.append(f"Q{num}: {qtxt}" if num else qtxt)
                text = "\n".join(lines)

        if text.strip():
            result.append({
                "paper_id"   : f"paper_{paper_id}",
                "source_name": filename,
                "year"       : year,
                "text"       : text,
            })

    return result


@st.cache_resource(show_spinner=False)
def get_or_build_index(rebuild: bool = False) -> RAGIndex:
    """
    Load the RAG index from disk if it exists, otherwise build it.

    Uses Streamlit's cache_resource so the index stays in memory across
    page refreshes (not rebuilt on every interaction).
    """
    index = RAGIndex()

    if not rebuild and os.path.exists(INDEX_CACHE_PATH):
        try:
            index = RAGIndex.load(INDEX_CACHE_PATH)
            return index
        except Exception:
            pass  # Corrupted cache — rebuild

    papers = load_papers_from_db()
    if not papers:
        return index  # Empty index

    chunks = chunk_papers(papers)
    index.build(chunks)

    # Save for next session
    try:
        os.makedirs(os.path.dirname(INDEX_CACHE_PATH), exist_ok=True)
        index.save(INDEX_CACHE_PATH)
    except Exception:
        pass  # Non-critical — app still works without saving

    return index


# ---------------------------------------------------------------------------
# Page UI
# ---------------------------------------------------------------------------

st.title("🤖 AI Study Assistant")
st.markdown(
    "Ask questions about your uploaded examination papers. "
    "The assistant searches your papers and answers based on what's in them."
)

st.info(
    "⚠️  **Grounding Notice:** Answers are based on content from your uploaded "
    "papers only. The assistant will tell you if it cannot find relevant information."
)

# ---------------------------------------------------------------------------
# Sidebar: settings + knowledge base management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("LLM Provider", ["groq", "openai"], key="ra_provider")
    api_key  = st.text_input(
        "API Key (optional)",
        type="password",
        key="ra_api_key",
        placeholder="Leave blank to use retrieval-only mode",
    )

    top_k = st.slider(
        "Passages to retrieve",
        min_value=2, max_value=10, value=5,
        help="How many document passages to use when answering.",
    )

    st.divider()
    st.header("📚 Knowledge Base")

    # Check database
    try:
        from database.database import get_all_papers
        n_papers = len(get_all_papers())
    except Exception:
        n_papers = 0

    st.metric("Uploaded papers", n_papers)

    if n_papers == 0:
        st.warning("No papers found. Upload papers first.")
        st.page_link("pages/1_📤_Upload.py", label="Go to Upload →")
    else:
        # Build or show status of index
        if "rag_index" not in st.session_state:
            with st.spinner("Loading knowledge base..."):
                st.session_state["rag_index"] = get_or_build_index()

        idx: RAGIndex = st.session_state["rag_index"]

        if idx.is_built:
            st.success(f"✅ {idx.chunk_count} passages indexed")
            st.caption(f"Search backend: {idx.backend}")
        else:
            st.warning("Index not built yet.")

        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            # Clear cache and rebuild
            get_or_build_index.clear()
            with st.spinner("Rebuilding index from uploaded papers..."):
                st.session_state["rag_index"] = get_or_build_index(rebuild=True)
            st.success("Knowledge base rebuilt!")
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used", expanded=False):
                for src in msg["sources"]:
                    year_str = f" · {src['year']}" if src.get("year") else ""
                    st.markdown(
                        f"**{src['source_name']}{year_str}** "
                        f"(relevance: {src['score']:.2f})"
                    )
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

# Chat input
if prompt := st.chat_input("Ask anything about your uploaded papers..."):
    # Check if index is ready
    idx = st.session_state.get("rag_index")
    if idx is None or not idx.is_built:
        st.error(
            "Knowledge base is not built. Upload papers and click "
            "'Rebuild Knowledge Base' in the sidebar."
        )
        st.stop()

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching your papers..."):
            response: AssistantResponse = ask(
                query    = prompt,
                index    = idx,
                top_k    = top_k,
                api_key  = api_key if api_key else None,
                provider = provider,
            )

        # Display answer
        st.markdown(response.answer)

        # Mode badge
        if response.mode == "llm":
            st.caption("🤖 AI-synthesized answer — grounded in your papers")
        else:
            st.caption("📄 Retrieval-only mode — showing relevant passages from your papers")

        # Disclaimer
        st.caption(f"ℹ️ {response.disclaimer}")

        # Sources
        source_list = []
        if response.sources:
            with st.expander("📄 Sources used", expanded=False):
                for src in response.sources:
                    year_str = f" · {src.year}" if src.year else ""
                    st.markdown(
                        f"**{src.source_name}{year_str}** "
                        f"(relevance: {src.score:.2f})"
                    )
                    preview = src.text[:300] + "..." if len(src.text) > 300 else src.text
                    st.caption(preview)
                    source_list.append({
                        "source_name": src.source_name,
                        "year"       : src.year,
                        "score"      : src.score,
                        "text"       : src.text,
                    })

    # Store in history
    st.session_state["messages"].append({
        "role"   : "assistant",
        "content": response.answer,
        "sources": source_list,
    })

# ---------------------------------------------------------------------------
# Suggested questions (shown when chat is empty)
# ---------------------------------------------------------------------------

if not st.session_state["messages"]:
    st.markdown("### 💡 Suggested questions to get started")
    suggestions = [
        "What topics appear most frequently in these papers?",
        "Explain the OSI model based on my papers.",
        "What questions have appeared about networking protocols?",
        "What are the important topics in these exam papers?",
        "Give me a summary of what I should study.",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, use_container_width=True, key=f"sugg_{i}"):
                # Inject suggestion as a message
                st.session_state["messages"].append({"role": "user", "content": s})
                st.rerun()
