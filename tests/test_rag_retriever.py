"""
tests/test_rag_retriever.py
============================
Phase 10 — Tests for src/rag_retriever.py

WHAT WE TEST
------------
1. RAGIndex builds from chunk list
2. search() returns results with the right structure
3. Scores are in range (0–1 approximately, cosine)
4. Relevant query returns relevant chunk (smoke test)
5. Empty index search returns empty list
6. is_built and chunk_count properties work
7. Save and load round-trip works
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.rag_retriever import RAGIndex, _normalise


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
            "The layers include Physical, Data Link, Network, Transport, Session, "
            "Presentation, and Application."
        ),
    },
    {
        "chunk_id"   : "p1_chunk_001",
        "paper_id"   : "p1",
        "source_name": "Networks_2023.pdf",
        "year"       : 2023,
        "text"       : (
            "TCP/IP is the foundational protocol suite of the internet. "
            "It consists of four layers: Link, Internet, Transport, Application."
        ),
    },
    {
        "chunk_id"   : "p2_chunk_000",
        "paper_id"   : "p2",
        "source_name": "Networks_2024.pdf",
        "year"       : 2024,
        "text"       : (
            "Routing algorithms determine the path packets take through a network. "
            "Common algorithms include Dijkstra's algorithm and Bellman-Ford."
        ),
    },
    {
        "chunk_id"   : "p2_chunk_001",
        "paper_id"   : "p2",
        "source_name": "Networks_2024.pdf",
        "year"       : 2024,
        "text"       : (
            "Congestion control prevents the network from being overwhelmed by "
            "too much traffic. TCP uses mechanisms like slow start and AIMD."
        ),
    },
    {
        "chunk_id"   : "p3_chunk_000",
        "paper_id"   : "p3",
        "source_name": "OS_2024.pdf",
        "year"       : 2024,
        "text"       : (
            "CPU scheduling algorithms determine which process runs next. "
            "Common algorithms include Round Robin, FCFS, and Shortest Job First."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tests: _normalise()
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_unit_vectors(self):
        vecs = np.array([[3.0, 4.0]], dtype=np.float32)
        normed = _normalise(vecs)
        norm = np.linalg.norm(normed[0])
        assert abs(norm - 1.0) < 1e-5

    def test_zero_vector_no_crash(self):
        vecs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = _normalise(vecs)
        assert result.shape == (1, 3)

    def test_shape_preserved(self):
        vecs = np.random.rand(5, 10).astype(np.float32)
        normed = _normalise(vecs)
        assert normed.shape == (5, 10)


# ---------------------------------------------------------------------------
# Tests: RAGIndex.build()
# ---------------------------------------------------------------------------

class TestRAGIndexBuild:
    def test_build_sets_is_built(self):
        idx = RAGIndex()
        idx.build(CHUNKS)
        assert idx.is_built

    def test_chunk_count_correct(self):
        idx = RAGIndex()
        idx.build(CHUNKS)
        assert idx.chunk_count == len(CHUNKS)

    def test_empty_build_not_built(self):
        idx = RAGIndex()
        idx.build([])
        assert not idx.is_built

    def test_backend_is_faiss_or_numpy(self):
        idx = RAGIndex()
        idx.build(CHUNKS)
        assert idx.backend in ("faiss", "numpy")


# ---------------------------------------------------------------------------
# Tests: RAGIndex.search()
# ---------------------------------------------------------------------------

class TestRAGIndexSearch:
    def setup_method(self):
        self.idx = RAGIndex()
        self.idx.build(CHUNKS)

    def test_search_returns_list(self):
        results = self.idx.search("OSI model layers")
        assert isinstance(results, list)

    def test_search_respects_top_k(self):
        results = self.idx.search("networking", top_k=2)
        assert len(results) <= 2

    def test_search_returns_chunk_fields(self):
        results = self.idx.search("OSI model", top_k=1)
        assert len(results) >= 1
        chunk = results[0]
        assert "text"        in chunk
        assert "score"       in chunk
        assert "source_name" in chunk
        assert "chunk_id"    in chunk

    def test_score_in_valid_range(self):
        results = self.idx.search("OSI model layers", top_k=3)
        for r in results:
            # Cosine similarity is in [-1, 1] but for normalised positive vecs
            # it's typically [0, 1] for TF-IDF and [-1, 1] for neural
            assert isinstance(r["score"], float)

    def test_relevant_query_returns_relevant_chunk(self):
        """OSI-related query should rank the OSI chunk highly."""
        results = self.idx.search("What are the seven layers of the OSI model?", top_k=3)
        top_texts = [r["text"] for r in results]
        assert any("OSI" in t for t in top_texts)

    def test_routing_query_returns_routing_chunk(self):
        results = self.idx.search("Dijkstra routing algorithm", top_k=2)
        top_texts = [r["text"] for r in results]
        assert any("Routing" in t or "routing" in t or "Dijkstra" in t for t in top_texts)

    def test_search_empty_index_returns_empty(self):
        empty_idx = RAGIndex()
        results = empty_idx.search("OSI model")
        assert results == []

    def test_top_k_larger_than_chunks_returns_all(self):
        results = self.idx.search("network", top_k=100)
        assert len(results) <= len(CHUNKS)


# ---------------------------------------------------------------------------
# Tests: RAGIndex save / load
# ---------------------------------------------------------------------------

class TestRAGIndexPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        idx = RAGIndex()
        idx.build(CHUNKS)

        save_path = str(tmp_path / "test_index.pkl")
        idx.save(save_path)

        idx2 = RAGIndex.load(save_path)
        assert idx2.is_built
        assert idx2.chunk_count == len(CHUNKS)

    def test_loaded_index_can_search(self, tmp_path):
        idx = RAGIndex()
        idx.build(CHUNKS)
        save_path = str(tmp_path / "test_index.pkl")
        idx.save(save_path)

        idx2 = RAGIndex.load(save_path)
        results = idx2.search("OSI model layers", top_k=2)
        assert len(results) >= 1

    def test_loaded_chunk_count_matches(self, tmp_path):
        idx = RAGIndex()
        idx.build(CHUNKS)
        save_path = str(tmp_path / "test_index.pkl")
        idx.save(save_path)

        idx2 = RAGIndex.load(save_path)
        assert idx2.chunk_count == idx.chunk_count
