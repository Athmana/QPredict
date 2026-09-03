"""
rag_assistant.py
================
Phase 10 — RAG-Powered Study Assistant

WHAT THIS MODULE DOES
---------------------
This is the brain of the AI Study Assistant. It:

1. Takes the student's question
2. Searches the RAGIndex for relevant document chunks (the "Retrieval" part)
3. Assembles a carefully structured prompt that includes:
   - The retrieved context passages
   - The student's question
   - A system instruction that tells the LLM to stay grounded
4. Calls the LLM API (the "Generation" part)
5. Returns the answer PLUS the source chunks so the student can verify

THE GROUNDING RULE
------------------
The LLM is explicitly told:
  "Answer ONLY using the provided context. If the answer is not in the
   context, say 'I don't have enough information in the uploaded papers
   to answer this.'"

This prevents hallucination. The LLM acts as a reader, not an inventor.

OFFLINE MODE
------------
If no API key is provided, the assistant returns the top retrieved chunks
as a plain-text answer. This is called "retrieval-only mode" — the student
sees the relevant passages from their own papers, without any LLM synthesis.

This means the assistant is ALWAYS useful, even without an API key.

SUPPORTED PROVIDERS
-------------------
- Groq (llama3-8b-8192) — free tier, fast
- OpenAI (gpt-3.5-turbo) — paid, higher quality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.rag_retriever import RAGIndex


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    One retrieved passage shown to the student as a source.

    Attributes
    ----------
    text        : The passage text
    source_name : PDF filename it came from
    year        : Exam year (if known)
    score       : Relevance score (0–1)
    chunk_id    : Unique identifier
    """
    text: str
    source_name: str
    year: Optional[int]
    score: float
    chunk_id: str


@dataclass
class AssistantResponse:
    """
    The complete response from the study assistant.

    Attributes
    ----------
    answer          : The synthesized answer (or retrieved passages if offline)
    sources         : List of retrieved chunks used to generate the answer
    mode            : "llm" | "retrieval_only"
    query           : The original question
    disclaimer      : Always-present grounding disclaimer
    """
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "retrieval_only"
    query: str = ""
    disclaimer: str = (
        "This answer is based on content from your uploaded examination "
        "papers. Always verify important information against your textbook "
        "or course materials."
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the prompt that combines retrieved context with the student's query.

    The prompt structure:
      [System instruction — grounding rule]
      [Context block — retrieved passages]
      [Question]

    This structure is the standard "RAG prompt template" used across the industry.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_name", "Unknown")
        year   = chunk.get("year", "")
        year_str = f" ({year})" if year else ""
        context_parts.append(
            f"[Source {i}: {source}{year_str}]\n{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful study assistant for university students.
You have been given excerpts from the student's uploaded examination papers.

IMPORTANT RULES:
1. Answer ONLY using the information provided in the context below.
2. If the answer is not clearly present in the context, say:
   "I don't have enough information in the uploaded papers to answer this."
3. Be concise and student-friendly.
4. If relevant, mention which source (paper/year) the information comes from.
5. Do NOT invent information, formulas, or definitions not in the context.

--- CONTEXT FROM UPLOADED PAPERS ---

{context_block}

--- END OF CONTEXT ---

Student's question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# Offline retrieval-only response
# ---------------------------------------------------------------------------

def _retrieval_only_response(query: str, chunks: list[dict]) -> AssistantResponse:
    """
    When no LLM is available, return the raw retrieved passages.

    The student sees the most relevant sections from their own papers.
    This is still useful — it directs them to the right parts of their material.
    """
    if not chunks:
        return AssistantResponse(
            answer=(
                "No relevant content found in the uploaded papers for this query. "
                "Try rephrasing your question or upload more papers."
            ),
            sources=[],
            mode="retrieval_only",
            query=query,
        )

    # Format the passages into a readable answer
    lines = [
        "Here are the most relevant passages from your uploaded papers:\n"
    ]
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source_name", "Unknown")
        year   = chunk.get("year", "")
        year_str = f" · {year}" if year else ""
        score  = chunk.get("score", 0.0)

        lines.append(f"**[{i}] {source}{year_str}** (relevance: {score:.2f})")
        lines.append(chunk["text"][:600])  # Trim very long passages
        lines.append("")

        sources.append(RetrievedChunk(
            text        = chunk["text"],
            source_name = source,
            year        = chunk.get("year"),
            score       = score,
            chunk_id    = chunk.get("chunk_id", ""),
        ))

    return AssistantResponse(
        answer  = "\n".join(lines),
        sources = sources,
        mode    = "retrieval_only",
        query   = query,
    )


# ---------------------------------------------------------------------------
# LLM-powered response
# ---------------------------------------------------------------------------

def _llm_response(
    query: str,
    chunks: list[dict],
    api_key: str,
    provider: str,
) -> AssistantResponse:
    """
    Call the LLM with the retrieved context and return a grounded answer.

    Falls back to retrieval-only mode on any error.
    """
    prompt = _build_rag_prompt(query, chunks)

    try:
        if provider == "groq":
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,   # Low temperature = more factual, less creative
            )
            answer_text = response.choices[0].message.content.strip()

        elif provider == "openai":
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            answer_text = response.choices[0].message.content.strip()

        else:
            return _retrieval_only_response(query, chunks)

    except Exception:
        # Any API error → fall back gracefully
        return _retrieval_only_response(query, chunks)

    sources = [
        RetrievedChunk(
            text        = c["text"],
            source_name = c.get("source_name", "Unknown"),
            year        = c.get("year"),
            score       = c.get("score", 0.0),
            chunk_id    = c.get("chunk_id", ""),
        )
        for c in chunks
    ]

    return AssistantResponse(
        answer  = answer_text,
        sources = sources,
        mode    = "llm",
        query   = query,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ask(
    query: str,
    index: RAGIndex,
    top_k: int = 5,
    api_key: Optional[str] = None,
    provider: str = "groq",
    min_score: float = 0.10,
) -> AssistantResponse:
    """
    Ask the study assistant a question.

    Parameters
    ----------
    query     : The student's question (natural language)
    index     : A built RAGIndex containing the uploaded papers
    top_k     : Number of chunks to retrieve from the index
    api_key   : Optional LLM API key; if None, uses retrieval-only mode
    provider  : "groq" | "openai"
    min_score : Minimum relevance score threshold — chunks below this
                score are excluded even if they are in the top-K.
                (0.10 is a very loose threshold; adjust if needed)

    Returns
    -------
    AssistantResponse with answer, sources, and mode
    """
    if not index.is_built:
        return AssistantResponse(
            answer=(
                "The knowledge base has not been built yet. Please upload "
                "some examination papers and click 'Build Knowledge Base' first."
            ),
            sources=[],
            mode="retrieval_only",
            query=query,
        )

    # Retrieve relevant chunks
    raw_chunks = index.search(query, top_k=top_k)

    # Filter by minimum score
    chunks = [c for c in raw_chunks if c.get("score", 0) >= min_score]

    if not chunks:
        return AssistantResponse(
            answer=(
                "No sufficiently relevant content was found in the uploaded papers "
                "for this question. Try rephrasing, or upload more papers."
            ),
            sources=[],
            mode="retrieval_only",
            query=query,
        )

    if api_key:
        return _llm_response(query, chunks, api_key, provider)

    return _retrieval_only_response(query, chunks)
