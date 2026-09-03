"""
tests/test_chunker.py
=====================
Phase 10 — Tests for src/chunker.py

WHAT WE TEST
------------
1. chunk_text() splits text into reasonable chunks
2. Overlap works correctly (consecutive chunks share content)
3. Short texts become a single chunk
4. Empty text returns empty list
5. Chunk dict has all required fields
6. chunk_papers() handles multiple papers
7. chunk_ids are unique within a paper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.chunker import chunk_text, chunk_papers, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, MIN_CHUNK_WORDS


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SHORT_TEXT = "The OSI model has seven layers."

# A medium-length text (~120 words)
MEDIUM_TEXT = " ".join([
    "The OSI model is a conceptual framework that standardizes the functions of a",
    "telecommunication or computing system into seven abstraction layers.",
    "The layers from bottom to top are: Physical, Data Link, Network, Transport,",
    "Session, Presentation, and Application.",
    "Each layer serves the layer above it and is served by the layer below it.",
    "The Physical layer is concerned with the transmission and reception of the",
    "unstructured raw bit stream over a physical medium.",
    "The Data Link layer provides node-to-node data transfer.",
    "The Network layer provides the functional and procedural means of transferring",
    "variable length data sequences from one node to another connected in different networks.",
    "The Transport layer provides transparent transfer of data between end users.",
])

# A long text (~500+ words) to force multiple chunks
LONG_TEXT = (MEDIUM_TEXT + " ") * 5  # Repeat to get enough words


# ---------------------------------------------------------------------------
# Tests: chunk_text()
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text_returns_empty(self):
        result = chunk_text("", "paper001", "test.pdf")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = chunk_text("   \n\n  ", "paper001", "test.pdf")
        assert result == []

    def test_short_text_returns_single_chunk(self):
        result = chunk_text(SHORT_TEXT, "paper001", "test.pdf")
        assert len(result) == 1

    def test_chunk_has_required_fields(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "test.pdf", year=2023)
        assert len(result) >= 1
        chunk = result[0]
        assert "chunk_id"    in chunk
        assert "paper_id"    in chunk
        assert "source_name" in chunk
        assert "year"        in chunk
        assert "text"        in chunk
        assert "char_start"  in chunk
        assert "char_end"    in chunk

    def test_chunk_paper_id_correct(self):
        result = chunk_text(MEDIUM_TEXT, "paper_xyz", "test.pdf")
        for chunk in result:
            assert chunk["paper_id"] == "paper_xyz"

    def test_chunk_source_name_correct(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "Networks_2024.pdf")
        for chunk in result:
            assert chunk["source_name"] == "Networks_2024.pdf"

    def test_chunk_year_stored(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "test.pdf", year=2024)
        for chunk in result:
            assert chunk["year"] == 2024

    def test_chunk_year_none_when_not_provided(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "test.pdf")
        for chunk in result:
            assert chunk["year"] is None

    def test_long_text_produces_multiple_chunks(self):
        result = chunk_text(LONG_TEXT, "paper001", "test.pdf", chunk_size=100, overlap=20)
        assert len(result) > 1

    def test_chunk_ids_are_unique(self):
        result = chunk_text(LONG_TEXT, "paper001", "test.pdf", chunk_size=100, overlap=20)
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_contain_paper_id(self):
        result = chunk_text(LONG_TEXT, "paper_abc", "test.pdf", chunk_size=100, overlap=20)
        for chunk in result:
            assert "paper_abc" in chunk["chunk_id"]

    def test_chunk_text_is_nonempty(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "test.pdf")
        for chunk in result:
            assert len(chunk["text"].strip()) > 0

    def test_overlap_shares_words(self):
        """Consecutive chunks should share some words due to overlap."""
        result = chunk_text(LONG_TEXT, "paper001", "test.pdf", chunk_size=100, overlap=30)
        if len(result) >= 2:
            words_0 = set(result[0]["text"].split())
            words_1 = set(result[1]["text"].split())
            shared = words_0 & words_1
            # With 30-word overlap, there should be some shared words
            assert len(shared) > 0

    def test_custom_chunk_size(self):
        """Smaller chunk_size should produce more chunks from long text."""
        large = chunk_text(LONG_TEXT, "p", "f", chunk_size=200, overlap=40)
        small = chunk_text(LONG_TEXT, "p", "f", chunk_size=80, overlap=20)
        assert len(small) >= len(large)

    def test_char_start_is_non_negative(self):
        result = chunk_text(MEDIUM_TEXT, "paper001", "test.pdf")
        for chunk in result:
            assert chunk["char_start"] >= 0

    def test_single_sentence_text(self):
        result = chunk_text("Hello world this is a test sentence.", "p1", "f.pdf")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: chunk_papers()
# ---------------------------------------------------------------------------

class TestChunkPapers:
    def test_empty_papers_returns_empty(self):
        result = chunk_papers([])
        assert result == []

    def test_single_paper_produces_chunks(self):
        papers = [{"paper_id": "p1", "source_name": "a.pdf", "year": 2023, "text": MEDIUM_TEXT}]
        result = chunk_papers(papers)
        assert len(result) >= 1

    def test_multiple_papers_combine_chunks(self):
        papers = [
            {"paper_id": "p1", "source_name": "a.pdf", "year": 2023, "text": LONG_TEXT},
            {"paper_id": "p2", "source_name": "b.pdf", "year": 2024, "text": LONG_TEXT},
        ]
        result = chunk_papers(papers)
        paper_ids = {c["paper_id"] for c in result}
        assert "p1" in paper_ids
        assert "p2" in paper_ids

    def test_paper_with_empty_text_skipped(self):
        papers = [
            {"paper_id": "p1", "source_name": "a.pdf", "year": 2023, "text": ""},
            {"paper_id": "p2", "source_name": "b.pdf", "year": 2024, "text": MEDIUM_TEXT},
        ]
        result = chunk_papers(papers)
        ids = {c["paper_id"] for c in result}
        assert "p1" not in ids
        assert "p2" in ids

    def test_all_chunks_have_required_fields(self):
        papers = [{"paper_id": "p1", "source_name": "a.pdf", "year": 2023, "text": LONG_TEXT}]
        result = chunk_papers(papers)
        for chunk in result:
            for field in ["chunk_id", "paper_id", "source_name", "year", "text"]:
                assert field in chunk
