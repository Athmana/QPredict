"""
test_similarity.py — Tests for text_cleaner.py and similarity.py

These tests verify:
  1. Text normalization preserves technical terms and removes noise
  2. TF-IDF correctly finds similar questions
  3. Cosine similarity scores are in range 0–1
  4. Union-Find grouping is transitive
  5. The full pipeline runs without errors on realistic data

HOW TO RUN:
    cd qpredict
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.text_cleaner import normalize_question, normalize_questions, is_too_short
from src.similarity import (
    build_tfidf_matrix,
    compute_similarity_matrix,
    find_similar_pairs,
    group_similar_questions,
    get_representative_question,
    run_tfidf_similarity_analysis,
    DEFAULT_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════════
# text_cleaner: normalize_question()
# ══════════════════════════════════════════════════════════════════════════════

def test_normalize_lowercases():
    result = normalize_question("Explain The OSI Model")
    assert result == result.lower()

def test_normalize_removes_stopwords():
    result = normalize_question("What is the difference between TCP and UDP?")
    assert "the" not in result.split()
    assert "is" not in result.split()
    assert "and" not in result.split()

def test_normalize_keeps_technical_terms():
    result = normalize_question("Compare TCP and UDP protocols.")
    assert "tcp" in result
    assert "udp" in result

def test_normalize_removes_marks_annotation():
    result = normalize_question("Explain OSI model. [10 Marks]")
    assert "marks" not in result
    assert "10" not in result

def test_normalize_removes_punctuation():
    result = normalize_question("What is TCP/IP? Explain.")
    assert "?" not in result
    assert "." not in result

def test_normalize_empty_string():
    assert normalize_question("") == ""

def test_normalize_preserves_ipv4():
    result = normalize_question("Explain IPv4 addressing.")
    assert "ipv4" in result

def test_normalize_collapses_hyphen():
    result = normalize_question("Explain three-way handshake.")
    assert "-" not in result

def test_normalize_batch():
    texts = ["Explain TCP.", "Describe UDP."]
    results = normalize_questions(texts)
    assert len(results) == 2
    assert "tcp" in results[0]
    assert "udp" in results[1]

def test_is_too_short_true():
    assert is_too_short("tcp") is True         # 1 token

def test_is_too_short_false():
    assert is_too_short("explain osi reference model") is False  # 4 tokens


# ══════════════════════════════════════════════════════════════════════════════
# similarity: TF-IDF matrix
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_NORMALIZED = [
    "explain osi reference model",
    "describe seven layers osi architecture",
    "explain functions osi layers",
    "routing algorithms shortest path",
    "dijkstra bellman ford routing",
]

def test_build_tfidf_matrix_shape():
    _, matrix = build_tfidf_matrix(SAMPLE_NORMALIZED)
    # Rows = number of questions, columns = vocabulary size
    assert matrix.shape[0] == len(SAMPLE_NORMALIZED)
    assert matrix.shape[1] > 0

def test_build_tfidf_empty_input():
    vectorizer, matrix = build_tfidf_matrix([])
    assert vectorizer is None
    assert matrix is None

def test_similarity_matrix_shape():
    _, matrix = build_tfidf_matrix(SAMPLE_NORMALIZED)
    sim = compute_similarity_matrix(matrix)
    n = len(SAMPLE_NORMALIZED)
    assert sim.shape == (n, n)

def test_similarity_diagonal_is_one():
    """A question compared to itself must have similarity = 1.0."""
    _, matrix = build_tfidf_matrix(SAMPLE_NORMALIZED)
    sim = compute_similarity_matrix(matrix)
    for i in range(len(SAMPLE_NORMALIZED)):
        assert abs(sim[i][i] - 1.0) < 1e-6

def test_similarity_scores_in_range():
    """All similarity scores must be between 0 and 1."""
    _, matrix = build_tfidf_matrix(SAMPLE_NORMALIZED)
    sim = compute_similarity_matrix(matrix)
    assert np.all(sim >= -1e-6)   # allow tiny floating point error below 0
    assert np.all(sim <= 1.0 + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# similarity: find_similar_pairs()
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = [
    {"question_text": "Explain the OSI reference model.", "year": 2021},
    {"question_text": "Describe the seven layers of the OSI architecture.", "year": 2022},
    {"question_text": "Explain the functions of the OSI layers.", "year": 2023},
    {"question_text": "Explain routing algorithms like Dijkstra.", "year": 2021},
    {"question_text": "Describe Dijkstra and Bellman-Ford routing.", "year": 2022},
    {"question_text": "What is congestion control in TCP?", "year": 2024},
]

def test_find_pairs_returns_list():
    from src.text_cleaner import normalize_questions
    normalized = normalize_questions([q["question_text"] for q in SAMPLE_QUESTIONS])
    pairs = find_similar_pairs(SAMPLE_QUESTIONS, normalized, threshold=0.30)
    assert isinstance(pairs, list)

def test_find_pairs_sorted_by_similarity():
    from src.text_cleaner import normalize_questions
    normalized = normalize_questions([q["question_text"] for q in SAMPLE_QUESTIONS])
    pairs = find_similar_pairs(SAMPLE_QUESTIONS, normalized, threshold=0.30)
    if len(pairs) > 1:
        for i in range(len(pairs) - 1):
            assert pairs[i].similarity >= pairs[i + 1].similarity

def test_find_pairs_no_self_pairs():
    """A question must never be paired with itself."""
    from src.text_cleaner import normalize_questions
    normalized = normalize_questions([q["question_text"] for q in SAMPLE_QUESTIONS])
    pairs = find_similar_pairs(SAMPLE_QUESTIONS, normalized, threshold=0.30)
    for p in pairs:
        assert p.idx_a != p.idx_b

def test_osi_questions_are_similar():
    """Questions sharing key rare tokens (osi, layers) should score above a low threshold."""
    from src.text_cleaner import normalize_questions
    # Use longer questions so TF-IDF bigrams have enough shared tokens
    osi_questions = [
        {"question_text": "Explain the seven layers of the OSI reference model in detail.", "year": 2021},
        {"question_text": "Describe the seven layers of the OSI architecture with examples.", "year": 2022},
        {"question_text": "What are the functions of the seven layers in the OSI model?", "year": 2023},
        {"question_text": "Compare TCP and UDP protocols in computer networks.", "year": 2024},
    ]
    normalized = normalize_questions([q["question_text"] for q in osi_questions])
    pairs = find_similar_pairs(osi_questions, normalized, threshold=0.20)
    # The OSI questions share "osi", "seven", "layers" → should form at least one pair
    assert len(pairs) > 0

def test_cross_year_detection():
    from src.text_cleaner import normalize_questions
    osi_questions = [
        {"question_text": "Explain the seven layers of the OSI reference model in detail.", "year": 2021},
        {"question_text": "Describe the seven layers of the OSI architecture with examples.", "year": 2022},
        {"question_text": "What are the functions of the seven layers in the OSI model?", "year": 2023},
        {"question_text": "Compare TCP and UDP protocols in computer networks.", "year": 2024},
    ]
    normalized = normalize_questions([q["question_text"] for q in osi_questions])
    pairs = find_similar_pairs(osi_questions, normalized, threshold=0.20)
    cross = [p for p in pairs if p.is_cross_year()]
    assert len(cross) > 0  # OSI 2021 vs OSI 2022 should be cross-year


# ══════════════════════════════════════════════════════════════════════════════
# similarity: group_similar_questions() — Union-Find
# ══════════════════════════════════════════════════════════════════════════════

def test_group_returns_list_of_lists():
    from src.text_cleaner import normalize_questions
    normalized = normalize_questions([q["question_text"] for q in SAMPLE_QUESTIONS])
    pairs = find_similar_pairs(SAMPLE_QUESTIONS, normalized, threshold=0.20)
    groups = group_similar_questions(SAMPLE_QUESTIONS, pairs)
    assert isinstance(groups, list)
    for g in groups:
        assert isinstance(g, list)
        assert len(g) >= 2  # singletons are excluded

def test_groups_no_duplicates():
    """Each question index should appear in at most one group."""
    from src.text_cleaner import normalize_questions
    normalized = normalize_questions([q["question_text"] for q in SAMPLE_QUESTIONS])
    pairs = find_similar_pairs(SAMPLE_QUESTIONS, normalized, threshold=0.20)
    groups = group_similar_questions(SAMPLE_QUESTIONS, pairs)
    all_indices = [idx for g in groups for idx in g]
    assert len(all_indices) == len(set(all_indices))


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: run_tfidf_similarity_analysis()
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_runs():
    result = run_tfidf_similarity_analysis(SAMPLE_QUESTIONS, threshold=0.20)
    assert "pairs" in result
    assert "groups" in result
    assert "sim_matrix" in result
    assert result["total_questions"] == len(SAMPLE_QUESTIONS)

def test_full_pipeline_empty_input():
    result = run_tfidf_similarity_analysis([], threshold=0.70)
    assert result["total_pairs"] == 0
    assert result["total_groups"] == 0

def test_full_pipeline_single_question():
    """A single question cannot form any pairs — pipeline should return empty results."""
    result = run_tfidf_similarity_analysis(
        [{"question_text": "Explain TCP.", "year": 2024}], threshold=0.70
    )
    # With < 2 questions, TF-IDF cannot run, so both pairs and groups must be empty
    assert result["total_pairs"] == 0
    assert result["total_groups"] == 0

def test_full_pipeline_threshold_respected():
    """Raising the threshold to near 1.0 should yield fewer (or zero) pairs."""
    result_low  = run_tfidf_similarity_analysis(SAMPLE_QUESTIONS, threshold=0.20)
    result_high = run_tfidf_similarity_analysis(SAMPLE_QUESTIONS, threshold=0.99)
    assert result_low["total_pairs"] >= result_high["total_pairs"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
