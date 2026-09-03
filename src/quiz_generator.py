"""
quiz_generator.py
=================
Phase 9 — Practice Quiz Generator

WHAT THIS MODULE DOES
---------------------
Generates practice questions for a topic cluster.

Two modes:

1. OFFLINE (template-based, no API key needed):
   - Takes the real questions from the uploaded papers
   - Reformats them as MCQs and short-answer questions
   - Creates distractors from topic keywords

2. ONLINE (LLM-powered, API key required):
   - Sends the topic name and sample questions to an LLM
   - Asks for structured MCQ / short-answer output
   - Parses and returns the results

The offline mode always works. The LLM mode is additive.

KEY DESIGN DECISION
-------------------
The quiz generator is ONLY for practice. It never claims that the
generated questions will appear in a real examination.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MCQOption:
    """One option in a multiple-choice question."""
    label: str           # "A", "B", "C", "D"
    text: str            # Option text
    is_correct: bool     # True for the correct answer


@dataclass
class QuizQuestion:
    """
    One practice question.

    Attributes
    ----------
    question_type : "mcq" | "short" | "long"
    question_text : The question to display
    options       : List of MCQOption (only for mcq type)
    correct_answer: The correct answer text
    explanation   : Why this is the answer
    difficulty    : "easy" | "medium" | "hard"
    source_year   : Optional — which year's paper inspired this question
    """
    question_type: str
    question_text: str
    options: list[MCQOption] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"
    source_year: Optional[int] = None


@dataclass
class Quiz:
    """
    A complete quiz for one topic.

    Attributes
    ----------
    topic_name  : The topic being quizzed
    questions   : List of QuizQuestion objects
    source      : "offline" | "llm"
    disclaimer  : Always-present note about purpose
    """
    topic_name: str
    questions: list[QuizQuestion] = field(default_factory=list)
    source: str = "offline"
    disclaimer: str = (
        "These questions are generated for practice purposes only. "
        "They are inspired by historical patterns and do not represent "
        "actual or predicted future examination questions."
    )


# ---------------------------------------------------------------------------
# Offline quiz generation (no API key required)
# ---------------------------------------------------------------------------

# Action words commonly used in exam questions.
# We use these to create MCQ stems from existing question texts.
_ACTION_PREFIXES = [
    "Explain", "Describe", "Define", "Discuss", "Compare",
    "Illustrate", "Outline", "List", "Differentiate between",
    "What is meant by", "With the help of a diagram, explain",
]

# Generic distractor templates for MCQs when we cannot extract good options
_GENERIC_DISTRACTORS = [
    "It is not directly related to {topic}",
    "It is a component of a different system",
    "It performs the opposite function",
    "None of the above",
]


def _clean_for_mcq(text: str) -> str:
    """
    Strip leading question numbers and marks info from a question string.

    Example:
        "1(a) Explain the OSI model. [10 marks]"
        → "Explain the OSI model."
    """
    # Remove leading numbering like "1.", "Q1.", "1(a)", "(a)", "1 (a)"
    # Pattern covers: "1.", "Q1.", "1)", "1(a)", "1 (a)", "(a)", "(b)"
    text = re.sub(r"^\s*(?:[Qq]?\d+\s*[\.\)]\s*(?:\([a-zA-Z]\)\s*)?|[Qq]?\d+\s*\([a-zA-Z]\)\s*|\([a-zA-Z]\)\s*)", "", text)
    # Remove trailing marks like "[10 marks]", "(5 marks)"
    text = re.sub(r"[\[\(]\s*\d+\s*marks?\s*[\]\)]", "", text, flags=re.I)
    return text.strip()


def _extract_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    """
    Extract the most important non-stopword words from a list of texts.

    This is a simple frequency-based approach (no heavy NLP library needed).

    Parameters
    ----------
    texts : Question strings to analyze
    top_n : How many keywords to return

    Returns
    -------
    List of keyword strings, sorted by frequency (most common first)
    """
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can",
        "could", "of", "in", "on", "at", "to", "for", "by", "with",
        "from", "and", "or", "but", "not", "this", "that", "these",
        "those", "it", "its", "they", "their", "what", "which", "who",
        "how", "why", "when", "where", "explain", "describe", "define",
        "discuss", "list", "give", "write", "state", "briefly", "short",
        "note", "mention", "different", "various", "following",
    }
    freq: dict[str, int] = {}
    for text in texts:
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        for w in words:
            if w not in _STOPWORDS:
                freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
    return sorted_words[:top_n]


def _make_mcq_from_question(
    question_text: str,
    keywords: list[str],
    topic_name: str,
    source_year: Optional[int],
    difficulty: str,
) -> QuizQuestion:
    """
    Convert one real question from an uploaded paper into an MCQ.

    Strategy:
    - The question text becomes the MCQ stem (possibly lightly reworded)
    - The correct answer is "See the explanation below" (we cannot
      auto-generate perfect answer text without an LLM)
    - Distractors are generated from topic keywords

    This is deliberately simple. With an LLM, answers become much better.
    """
    stem = _clean_for_mcq(question_text)
    if not stem.endswith("?"):
        # Convert to a question if it isn't already one
        if stem.lower().startswith(("explain", "describe", "define", "discuss")):
            # Keep as-is: these are valid exam instruction stems
            pass
        else:
            stem = stem + "?"

    # Build distractor options using keywords
    random.shuffle(keywords)
    distractor_pool = keywords[:3] if len(keywords) >= 3 else keywords

    # Pad with generic distractors if needed
    while len(distractor_pool) < 3:
        distractor_pool.append(
            _GENERIC_DISTRACTORS[len(distractor_pool)].format(topic=topic_name)
        )

    labels = ["A", "B", "C", "D"]
    correct_label = random.choice(labels)
    options: list[MCQOption] = []

    distractor_idx = 0
    for label in labels:
        if label == correct_label:
            options.append(MCQOption(
                label=label,
                text=f"The concept described in the question above (correct answer)",
                is_correct=True,
            ))
        else:
            opt_text = distractor_pool[distractor_idx] if distractor_idx < len(distractor_pool) else "Not applicable"
            options.append(MCQOption(
                label=label,
                text=opt_text,
                is_correct=False,
            ))
            distractor_idx += 1

    return QuizQuestion(
        question_type="mcq",
        question_text=stem,
        options=options,
        correct_answer=correct_label,
        explanation=(
            f"This question is based on the topic '{topic_name}'. "
            f"Review your notes on {topic_name} to answer this question. "
            f"Original question appeared in {source_year} papers."
            if source_year
            else f"Review your notes on {topic_name} to answer this question."
        ),
        difficulty=difficulty,
        source_year=source_year,
    )


def _make_short_answer(
    question_text: str,
    topic_name: str,
    source_year: Optional[int],
    difficulty: str,
) -> QuizQuestion:
    """
    Convert one real question into a short-answer practice question.

    The original question from the paper is the best short-answer question
    we can generate without an LLM — it comes directly from real exams.
    """
    stem = _clean_for_mcq(question_text)
    return QuizQuestion(
        question_type="short",
        question_text=stem,
        correct_answer="",
        explanation=(
            f"This question appeared in {source_year} papers. "
            f"Refer to your textbook section on {topic_name}."
            if source_year
            else f"Refer to your textbook section on {topic_name}."
        ),
        difficulty=difficulty,
        source_year=source_year,
    )


def generate_offline_quiz(
    topic_name: str,
    cluster_questions: list[dict],
    num_mcq: int = 3,
    num_short: int = 3,
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> Quiz:
    """
    Generate a practice quiz without any API key.

    Uses the actual questions from the uploaded papers as the source material.

    Parameters
    ----------
    topic_name         : Topic label (e.g. "OSI Model")
    cluster_questions  : List of question dicts, each containing:
                           - "question_text" : str
                           - "year"          : int (optional)
    num_mcq            : Number of MCQ questions to generate
    num_short          : Number of short-answer questions to generate
    difficulty         : "easy" | "medium" | "hard"
    seed               : Random seed for reproducibility in tests

    Returns
    -------
    Quiz object with MCQ + short-answer questions
    """
    if seed is not None:
        random.seed(seed)

    quiz = Quiz(topic_name=topic_name, source="offline")

    if not cluster_questions:
        return quiz

    # Extract question texts and years
    q_texts = [q.get("question_text", "") for q in cluster_questions if q.get("question_text")]
    years = [q.get("year") for q in cluster_questions]

    if not q_texts:
        return quiz

    keywords = _extract_keywords(q_texts)

    # Shuffle so we don't always pick the first questions
    combined = list(zip(q_texts, years))
    random.shuffle(combined)
    q_texts_shuffled, years_shuffled = zip(*combined) if combined else ([], [])

    # --- Generate MCQs ---
    mcq_count = min(num_mcq, len(q_texts_shuffled))
    for i in range(mcq_count):
        qtext = q_texts_shuffled[i]
        year = years_shuffled[i]
        mcq = _make_mcq_from_question(qtext, keywords[:], topic_name, year, difficulty)
        quiz.questions.append(mcq)

    # --- Generate short-answer questions ---
    short_count = min(num_short, len(q_texts_shuffled))
    used_for_short = q_texts_shuffled[:short_count]
    for i, qtext in enumerate(used_for_short):
        year = years_shuffled[i]
        sa = _make_short_answer(qtext, topic_name, year, difficulty)
        quiz.questions.append(sa)

    return quiz


# ---------------------------------------------------------------------------
# LLM-powered quiz generation
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    topic_name: str,
    sample_questions: list[str],
    num_mcq: int,
    num_short: int,
    difficulty: str,
) -> str:
    """
    Build the prompt to send to the LLM for quiz generation.

    The prompt requests a strict JSON format so we can parse it reliably.
    """
    samples_text = "\n".join(f"- {q}" for q in sample_questions[:5])
    return f"""You are an exam question generator for university students.

Topic: {topic_name}
Difficulty: {difficulty}

Historical exam questions on this topic (for context only):
{samples_text}

Generate exactly {num_mcq} MCQ questions and {num_short} short-answer questions.

Return ONLY valid JSON in this exact format:
{{
  "mcq": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct": "B",
      "explanation": "Brief explanation of why B is correct."
    }}
  ],
  "short": [
    {{
      "question": "Short answer question text.",
      "answer": "Model answer in 2-3 sentences.",
      "explanation": "Key points to include."
    }}
  ]
}}

Rules:
- Questions must be about {topic_name}
- Do NOT copy the sample questions verbatim
- Keep the difficulty at {difficulty} level
- Return ONLY the JSON — no extra text before or after"""


def _parse_llm_response(raw: str, topic_name: str) -> list[QuizQuestion]:
    """
    Parse the LLM's JSON response into QuizQuestion objects.

    If parsing fails for any reason, returns an empty list so the caller
    can fall back to offline mode.
    """
    import json

    # Find the JSON block in the response (LLMs sometimes add extra text)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    questions: list[QuizQuestion] = []

    # Parse MCQs
    for item in data.get("mcq", []):
        opts_dict = item.get("options", {})
        options = []
        correct_label = item.get("correct", "A")
        for label, text in opts_dict.items():
            options.append(MCQOption(
                label=label,
                text=str(text),
                is_correct=(label == correct_label),
            ))
        questions.append(QuizQuestion(
            question_type="mcq",
            question_text=item.get("question", ""),
            options=options,
            correct_answer=correct_label,
            explanation=item.get("explanation", ""),
            difficulty="medium",
        ))

    # Parse short-answer
    for item in data.get("short", []):
        questions.append(QuizQuestion(
            question_type="short",
            question_text=item.get("question", ""),
            correct_answer=item.get("answer", ""),
            explanation=item.get("explanation", ""),
            difficulty="medium",
        ))

    return questions


def generate_llm_quiz(
    topic_name: str,
    cluster_questions: list[dict],
    num_mcq: int = 3,
    num_short: int = 3,
    difficulty: str = "medium",
    api_key: Optional[str] = None,
    provider: str = "groq",
) -> Quiz:
    """
    Generate a practice quiz using an LLM API.

    Falls back to offline mode automatically if:
    - No API key is provided
    - The required library is not installed
    - The LLM returns an unparseable response
    - Any network/API error occurs

    Parameters
    ----------
    topic_name        : Topic label
    cluster_questions : List of question dicts (same format as offline)
    num_mcq           : Number of MCQ questions
    num_short         : Number of short-answer questions
    difficulty        : "easy" | "medium" | "hard"
    api_key           : API key string (or None)
    provider          : "groq" | "openai"

    Returns
    -------
    Quiz — sourced from "llm" if successful, "offline" as fallback
    """
    if not api_key:
        return generate_offline_quiz(
            topic_name, cluster_questions, num_mcq, num_short, difficulty
        )

    q_texts = [q.get("question_text", "") for q in cluster_questions if q.get("question_text")]
    prompt = _build_llm_prompt(topic_name, q_texts, num_mcq, num_short, difficulty)

    try:
        if provider == "groq":
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.5,
            )
            raw = response.choices[0].message.content

        elif provider == "openai":
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.5,
            )
            raw = response.choices[0].message.content

        else:
            return generate_offline_quiz(
                topic_name, cluster_questions, num_mcq, num_short, difficulty
            )

        parsed = _parse_llm_response(raw, topic_name)
        if not parsed:
            # LLM returned something we couldn't parse — fall back
            return generate_offline_quiz(
                topic_name, cluster_questions, num_mcq, num_short, difficulty
            )

        quiz = Quiz(topic_name=topic_name, source="llm")
        quiz.questions = parsed
        return quiz

    except Exception:
        # Never crash the app because of LLM failure
        return generate_offline_quiz(
            topic_name, cluster_questions, num_mcq, num_short, difficulty
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_quiz(
    topic_name: str,
    cluster_questions: list[dict],
    num_mcq: int = 3,
    num_short: int = 3,
    difficulty: str = "medium",
    api_key: Optional[str] = None,
    provider: str = "groq",
    seed: Optional[int] = None,
) -> Quiz:
    """
    Main entry point for quiz generation.

    Automatically uses LLM if api_key is provided, otherwise offline.

    Parameters
    ----------
    topic_name        : Topic label (e.g. "OSI Model")
    cluster_questions : list of dicts with "question_text" and optional "year"
    num_mcq           : Number of MCQ questions to generate
    num_short         : Number of short-answer questions to generate
    difficulty        : "easy" | "medium" | "hard"
    api_key           : Optional LLM API key
    provider          : "groq" | "openai"
    seed              : Random seed (for reproducibility in tests)

    Returns
    -------
    Quiz object
    """
    if api_key:
        return generate_llm_quiz(
            topic_name, cluster_questions, num_mcq, num_short, difficulty,
            api_key, provider
        )
    return generate_offline_quiz(
        topic_name, cluster_questions, num_mcq, num_short, difficulty, seed
    )
