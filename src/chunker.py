"""
chunker.py
==========
Phase 10 — Text Chunker for RAG

WHAT THIS MODULE DOES
---------------------
Splits long text (extracted from PDFs) into smaller, overlapping pieces
called "chunks". These chunks are what gets stored in the vector index
and retrieved during a search.

WHY DO WE CHUNK?
----------------
An LLM has a limited "context window" — it can only read a certain number
of words at once. We cannot send the entire PDF every time the student
asks a question.

Instead, we:
1. Split the document into chunks of ~300–500 words
2. Each chunk overlaps slightly with the previous one (so no sentence
   is cut off at a boundary and loses context)
3. Store all chunks in the vector index
4. At query time, retrieve only the top 3–5 most relevant chunks

This is much more efficient and produces better answers than sending
the whole document.

CHUNK OVERLAP
-------------
Imagine the text is:

  [...sentence A... sentence B... sentence C... sentence D...]

With chunk_size=3 and overlap=1:

  Chunk 1: [A, B, C]
  Chunk 2: [C, D, E]      ← C is repeated in both chunks

This overlap ensures that if an important sentence sits at the boundary
between two chunks, it appears in both, so the retriever can find it.

DATA MODEL
----------
Every chunk is a dict:

  {
    "chunk_id"   : "paper001_chunk_003",
    "paper_id"   : "paper001",
    "source_name": "Computer_Networks_2024.pdf",
    "year"       : 2024,
    "text"       : "The OSI model has seven layers...",
    "char_start" : 1240,
    "char_end"   : 1780,
  }
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration defaults (all overridable by the caller)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 400      # Target number of words per chunk
DEFAULT_OVERLAP    = 80       # Number of words to overlap between chunks
MIN_CHUNK_WORDS    = 20       # Chunks shorter than this are discarded


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple rule-based approach.

    We split on:
      - Period/question mark/exclamation followed by a space and capital letter
      - Newlines (often used as sentence separators in extracted PDF text)

    We do NOT use NLTK or spaCy here to keep dependencies minimal.
    """
    # Normalise multiple newlines/spaces
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split on sentence-ending punctuation followed by whitespace + capital
    # OR on newline characters
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z])|(?<=\n)", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    paper_id: str,
    source_name: str,
    year: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Split a document's text into overlapping word-based chunks.

    Parameters
    ----------
    text        : The full extracted text of one PDF
    paper_id    : The paper's ID from the database (e.g. "paper_001")
    source_name : The PDF filename (e.g. "Computer_Networks_2024.pdf")
    year        : The examination year (optional)
    chunk_size  : Target number of words per chunk
    overlap     : Number of words to repeat between consecutive chunks

    Returns
    -------
    List of chunk dicts, each with:
      chunk_id, paper_id, source_name, year, text, char_start, char_end
    """
    if not text or not text.strip():
        return []

    # Tokenise into words (preserving position via a running index)
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        # Text too short to chunk — return as a single chunk
        return [{
            "chunk_id"   : f"{paper_id}_chunk_000",
            "paper_id"   : paper_id,
            "source_name": source_name,
            "year"       : year,
            "text"       : text.strip(),
            "char_start" : 0,
            "char_end"   : len(text),
        }]

    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        chunk_text_str = " ".join(chunk_words)

        # Only keep chunks that meet the minimum length threshold
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            # Approximate character positions (good enough for display)
            char_start = len(" ".join(words[:start]))
            char_end   = char_start + len(chunk_text_str)

            chunks.append({
                "chunk_id"   : f"{paper_id}_chunk_{chunk_index:03d}",
                "paper_id"   : paper_id,
                "source_name": source_name,
                "year"       : year,
                "text"       : chunk_text_str,
                "char_start" : char_start,
                "char_end"   : char_end,
            })
            chunk_index += 1

        # Advance by (chunk_size - overlap) so the next chunk overlaps
        step = max(1, chunk_size - overlap)
        start += step

        # If we're near the end, stop to avoid tiny trailing chunks
        if start < len(words) and len(words) - start < MIN_CHUNK_WORDS:
            break

    return chunks


def chunk_papers(papers: list[dict]) -> list[dict]:
    """
    Chunk multiple papers at once.

    Parameters
    ----------
    papers : List of paper dicts, each containing:
               - paper_id   : str
               - source_name: str  (filename)
               - year       : int  (optional)
               - text       : str  (full extracted text)

    Returns
    -------
    Flat list of all chunks across all papers
    """
    all_chunks: list[dict] = []
    for paper in papers:
        chunks = chunk_text(
            text        = paper.get("text", ""),
            paper_id    = paper.get("paper_id", "unknown"),
            source_name = paper.get("source_name", "unknown"),
            year        = paper.get("year"),
        )
        all_chunks.extend(chunks)
    return all_chunks
