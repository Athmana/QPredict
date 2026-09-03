"""
tests/test_quiz_generator.py
============================
Phase 9 — Tests for src/quiz_generator.py

WHAT WE TEST
------------
1. Data structures (QuizQuestion, MCQOption, Quiz)
2. _clean_for_mcq() strips numbering and marks info correctly
3. _extract_keywords() returns relevant non-stopword words
4. generate_offline_quiz() works with typical cluster data
5. Quiz has correct question counts
6. Options are correctly structured
7. Fallback behavior when no API key provided
8. Edge cases: empty clusters, single question, zero MCQ requested
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.quiz_generator import (
    MCQOption,
    QuizQuestion,
    Quiz,
    _clean_for_mcq,
    _extract_keywords,
    generate_offline_quiz,
    generate_llm_quiz,
    generate_quiz,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_question(text: str, year: int = 2023) -> dict:
    return {"question_text": text, "year": year}


SAMPLE_QUESTIONS = [
    make_question("Explain the OSI reference model.", 2021),
    make_question("Describe the seven layers of the OSI model.", 2022),
    make_question("Explain the functions of the different layers in the OSI architecture.", 2023),
    make_question("Q1. What is the role of the Transport layer? [10 marks]", 2024),
    make_question("(b) Compare OSI and TCP/IP models.", 2025),
]


# ---------------------------------------------------------------------------
# Tests: _clean_for_mcq()
# ---------------------------------------------------------------------------

class TestCleanForMcq:
    def test_removes_leading_number_dot(self):
        result = _clean_for_mcq("1. What is an OSI model?")
        assert not result.startswith("1.")

    def test_removes_leading_q_prefix(self):
        result = _clean_for_mcq("Q1. Explain TCP/IP.")
        assert "Q1" not in result

    def test_removes_bracket_number(self):
        result = _clean_for_mcq("1(a) Explain the OSI model.")
        assert "1(a)" not in result

    def test_removes_marks_annotation_square(self):
        result = _clean_for_mcq("Explain OSI model. [10 marks]")
        assert "[10 marks]" not in result

    def test_removes_marks_annotation_round(self):
        result = _clean_for_mcq("Explain OSI model. (5 marks)")
        assert "(5 marks)" not in result.lower()

    def test_preserves_question_content(self):
        result = _clean_for_mcq("1. Explain the OSI model.")
        assert "OSI" in result

    def test_empty_string_returns_empty(self):
        result = _clean_for_mcq("")
        assert result == ""

    def test_plain_question_unchanged_content(self):
        q = "What is the role of TCP in networking?"
        result = _clean_for_mcq(q)
        assert "TCP" in result


# ---------------------------------------------------------------------------
# Tests: _extract_keywords()
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_list(self):
        texts = ["Explain the OSI model", "Describe OSI layers"]
        result = _extract_keywords(texts)
        assert isinstance(result, list)

    def test_returns_top_n(self):
        texts = ["Explain the OSI model transport layer network data link", "OSI model again"]
        result = _extract_keywords(texts, top_n=3)
        assert len(result) <= 3

    def test_technical_terms_included(self):
        texts = ["Explain TCP UDP OSI HTTP SQL protocols"]
        result = _extract_keywords(texts, top_n=10)
        # At least one technical term should appear
        found = any(term in result for term in ["tcp", "udp", "osi", "http", "sql"])
        assert found

    def test_stopwords_excluded(self):
        texts = ["the OSI model is a framework"]
        result = _extract_keywords(texts, top_n=10)
        assert "the" not in result
        assert "is" not in result

    def test_empty_input_returns_empty(self):
        result = _extract_keywords([])
        assert result == []

    def test_common_word_ranked_higher(self):
        texts = ["OSI model", "OSI architecture", "OSI layers"]
        result = _extract_keywords(texts, top_n=5)
        assert result[0].lower() == "osi"


# ---------------------------------------------------------------------------
# Tests: generate_offline_quiz()
# ---------------------------------------------------------------------------

class TestGenerateOfflineQuiz:
    def test_returns_quiz_object(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        assert isinstance(quiz, Quiz)

    def test_quiz_topic_name_set(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        assert quiz.topic_name == "OSI Model"

    def test_quiz_source_is_offline(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        assert quiz.source == "offline"

    def test_mcq_count_matches_request(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=3, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        assert len(mcqs) == 3

    def test_short_count_matches_request(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_short=2, seed=42)
        shorts = [q for q in quiz.questions if q.question_type == "short"]
        assert len(shorts) == 2

    def test_mcq_has_four_options(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=2, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        for mcq in mcqs:
            assert len(mcq.options) == 4

    def test_mcq_exactly_one_correct_option(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=3, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        for mcq in mcqs:
            correct_count = sum(1 for opt in mcq.options if opt.is_correct)
            assert correct_count == 1

    def test_mcq_correct_answer_label_valid(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=3, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        for mcq in mcqs:
            assert mcq.correct_answer in ["A", "B", "C", "D"]

    def test_empty_clusters_returns_empty_quiz(self):
        quiz = generate_offline_quiz("Empty Topic", [])
        assert len(quiz.questions) == 0

    def test_disclaimer_present(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        assert len(quiz.disclaimer) > 0

    def test_disclaimer_not_prediction(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        disclaimer_lower = quiz.disclaimer.lower()
        assert "practice" in disclaimer_lower or "not" in disclaimer_lower

    def test_questions_have_question_text(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=2, seed=42)
        for q in quiz.questions:
            assert len(q.question_text) > 0

    def test_questions_have_difficulty_set(self):
        quiz = generate_offline_quiz(
            "OSI Model", SAMPLE_QUESTIONS, difficulty="hard", seed=42
        )
        for q in quiz.questions:
            assert q.difficulty == "hard"

    def test_zero_mcq_request(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=0, num_short=3, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        assert len(mcqs) == 0

    def test_zero_short_request(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=3, num_short=0, seed=42)
        shorts = [q for q in quiz.questions if q.question_type == "short"]
        assert len(shorts) == 0

    def test_single_question_cluster(self):
        single = [make_question("Explain OSI model.", 2023)]
        quiz = generate_offline_quiz("OSI Model", single, num_mcq=1, num_short=1, seed=42)
        # Should still produce something
        assert isinstance(quiz, Quiz)

    def test_option_labels_are_abcd(self):
        quiz = generate_offline_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=2, seed=42)
        for q in quiz.questions:
            if q.question_type == "mcq":
                labels = {opt.label for opt in q.options}
                assert labels == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Tests: generate_llm_quiz() fallback behavior
# ---------------------------------------------------------------------------

class TestGenerateLlmQuizFallback:
    def test_no_api_key_falls_back_to_offline(self):
        quiz = generate_llm_quiz(
            "OSI Model", SAMPLE_QUESTIONS,
            num_mcq=2, num_short=2,
            api_key=None,
        )
        # Should fall back to offline
        assert isinstance(quiz, Quiz)
        assert len(quiz.questions) > 0

    def test_invalid_provider_falls_back(self):
        quiz = generate_llm_quiz(
            "OSI Model", SAMPLE_QUESTIONS,
            num_mcq=2, num_short=1,
            api_key="fake-key",
            provider="invalid_provider",
        )
        assert isinstance(quiz, Quiz)


# ---------------------------------------------------------------------------
# Tests: generate_quiz() entry point
# ---------------------------------------------------------------------------

class TestGenerateQuiz:
    def test_no_key_uses_offline(self):
        quiz = generate_quiz("OSI Model", SAMPLE_QUESTIONS, api_key=None, seed=42)
        assert quiz.source == "offline"

    def test_returns_quiz_type(self):
        quiz = generate_quiz("OSI Model", SAMPLE_QUESTIONS, seed=42)
        assert isinstance(quiz, Quiz)

    def test_respects_num_mcq(self):
        quiz = generate_quiz("OSI Model", SAMPLE_QUESTIONS, num_mcq=2, num_short=0, seed=42)
        mcqs = [q for q in quiz.questions if q.question_type == "mcq"]
        assert len(mcqs) == 2

    def test_empty_questions_returns_empty_quiz(self):
        quiz = generate_quiz("Nothing", [], seed=42)
        assert isinstance(quiz, Quiz)
        assert len(quiz.questions) == 0
