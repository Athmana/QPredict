"""
tests/test_rag_assistant.py
============================
Phase 10 — Tests for src/rag_assistant.py

WHAT WE TEST
------------
1. ask() returns AssistantResponse when index is not built
2. ask() returns retrieval-only response when no API key provided
3. Response has all required fields
4. Relevant query returns non-empty answer
5. Grounding: disclaimer is always present
6. Mode is "retrieval_only" when no API key
7. Sources list is populated when chunks are found
8. No crash on edge cases (empty query, very short query)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.rag_retriever import RAGIndex
from src.rag_assistant import ask, AssistantResponse, _build_rag_prompt, _retrieval_only_response


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

CHUNKS = [
    {
        "chunk_id"   : "p1_chunk_000",
        "paper_id"   : "p1",
        "source_name": "Networks_2023.pdf",
        "year"       : 2023,
        "text"       : (
            "The OSI model is a seven-layer framework for network communication. "
            "The layers are Physical, Data Link, Network, Transport, Session, "
            "Presentation, and Application."
        ),
    },
    {
        "chunk_id"   : "p1_chunk_001",
        "paper_id"   : "p1",
        "source_name": "Networks_2023.pdf",
        "year"       : 2023,
        "text"       : (
            "TCP/IP is the foundational suite of protocols for the internet. "
            "It uses a four-layer model: Link, Internet, Transport, Application."
        ),
    },
    {
        "chunk_id"   : "p2_chunk_000",
        "paper_id"   : "p2",
        "source_name": "Networks_2024.pdf",
        "year"       : 2024,
        "text"       : (
            "Routing algorithms determine the path packets take through a network. "
            "Dijkstra and Bellman-Ford are widely used routing algorithms."
        ),
    },
]


def make_index(chunks=None) -> RAGIndex:
    idx = RAGIndex()
    idx.build(chunks or CHUNKS)
    return idx


# ---------------------------------------------------------------------------
# Tests: _build_rag_prompt()
# ---------------------------------------------------------------------------

class TestBuildRagPrompt:
    def test_returns_string(self):
        result = _build_rag_prompt("What is OSI?", CHUNKS[:2])
        assert isinstance(result, str)

    def test_contains_query(self):
        result = _build_rag_prompt("What is OSI?", CHUNKS[:2])
        assert "What is OSI?" in result

    def test_contains_source_names(self):
        result = _build_rag_prompt("OSI layers", CHUNKS[:1])
        assert "Networks_2023.pdf" in result

    def test_contains_context_text(self):
        result = _build_rag_prompt("OSI", CHUNKS[:1])
        assert "OSI model" in result

    def test_grounding_rules_present(self):
        result = _build_rag_prompt("OSI", CHUNKS[:1])
        # Must instruct model to stay within context
        assert "context" in result.lower() or "provided" in result.lower()

    def test_handles_empty_chunks(self):
        result = _build_rag_prompt("OSI?", [])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: _retrieval_only_response()
# ---------------------------------------------------------------------------

class TestRetrievalOnlyResponse:
    def _make_chunks_with_score(self):
        return [{**c, "score": 0.85} for c in CHUNKS]

    def test_returns_assistant_response(self):
        chunks = self._make_chunks_with_score()
        result = _retrieval_only_response("OSI model", chunks)
        assert isinstance(result, AssistantResponse)

    def test_mode_is_retrieval_only(self):
        chunks = self._make_chunks_with_score()
        result = _retrieval_only_response("OSI", chunks)
        assert result.mode == "retrieval_only"

    def test_sources_populated(self):
        chunks = self._make_chunks_with_score()
        result = _retrieval_only_response("OSI", chunks)
        assert len(result.sources) == len(chunks)

    def test_answer_contains_source_name(self):
        chunks = self._make_chunks_with_score()
        result = _retrieval_only_response("OSI", chunks)
        assert "Networks_2023.pdf" in result.answer

    def test_empty_chunks_returns_no_results_message(self):
        result = _retrieval_only_response("OSI", [])
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert len(result.sources) == 0

    def test_disclaimer_always_present(self):
        chunks = self._make_chunks_with_score()
        result = _retrieval_only_response("OSI", chunks)
        assert len(result.disclaimer) > 0


# ---------------------------------------------------------------------------
# Tests: ask() function
# ---------------------------------------------------------------------------

class TestAsk:
    def test_unbuilt_index_returns_response(self):
        empty_idx = RAGIndex()
        result = ask("What is OSI?", empty_idx)
        assert isinstance(result, AssistantResponse)
        assert len(result.answer) > 0

    def test_unbuilt_index_mode_retrieval(self):
        empty_idx = RAGIndex()
        result = ask("What is OSI?", empty_idx)
        assert result.mode == "retrieval_only"

    def test_no_api_key_uses_retrieval_mode(self):
        idx = make_index()
        result = ask("What is the OSI model?", idx, api_key=None)
        assert result.mode == "retrieval_only"

    def test_returns_assistant_response_type(self):
        idx = make_index()
        result = ask("OSI layers", idx)
        assert isinstance(result, AssistantResponse)

    def test_query_stored_in_response(self):
        idx = make_index()
        result = ask("explain OSI", idx)
        assert result.query == "explain OSI"

    def test_disclaimer_always_present(self):
        idx = make_index()
        result = ask("OSI model", idx)
        assert len(result.disclaimer) > 0
        # Must mention grounding / papers
        assert "paper" in result.disclaimer.lower() or "uploaded" in result.disclaimer.lower()

    def test_relevant_query_returns_sources(self):
        idx = make_index()
        result = ask("What is the OSI model?", idx, top_k=3, min_score=0.0)
        assert len(result.sources) > 0

    def test_sources_have_required_fields(self):
        idx = make_index()
        result = ask("OSI layers", idx, top_k=2, min_score=0.0)
        for src in result.sources:
            assert hasattr(src, "text")
            assert hasattr(src, "source_name")
            assert hasattr(src, "score")
            assert hasattr(src, "chunk_id")

    def test_answer_is_string(self):
        idx = make_index()
        result = ask("networking protocols", idx)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_high_min_score_may_return_empty_sources(self):
        idx = make_index()
        # With an impossibly high threshold, no chunks should pass
        result = ask("OSI model", idx, min_score=2.0)
        # Should still return a response, just with no sources
        assert isinstance(result, AssistantResponse)
        assert len(result.sources) == 0

    def test_top_k_respected(self):
        idx = make_index()
        result = ask("networking", idx, top_k=1, min_score=0.0)
        assert len(result.sources) <= 1

    def test_empty_query_no_crash(self):
        idx = make_index()
        result = ask("", idx)
        assert isinstance(result, AssistantResponse)
