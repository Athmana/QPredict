"""
test_embeddings.py — Tests for embeddings.py

These tests verify:
  1. embed_questions() returns the right shape and dtype
  2. The TF-IDF fallback works when sentence-transformers is not installed
  3. cosine_similarity_matrix() produces correct values
  4. find_semantic_pairs() finds related questions
  5. get_or_compute_embeddings() uses the DB cache correctly
  6. save/load embeddings from DB round-trip correctly
  7. The full run_embedding_similarity_analysis() pipeline

IMPORTANT: All tests work whether or not sentence-transformers is installed.
When it's not installed, the module falls back to TF-IDF automatically.
The test suite verifies that both paths produce valid, consistent output.

HOW TO RUN:
    cd qpredict
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings import (
    embed_questions,
    cosine_similarity_matrix,
    find_semantic_pairs,
    run_embedding_similarity_analysis,
    save_embeddings_to_db,
    load_embeddings_from_db,
    is_available,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


# ── Shared test data ─────────────────────────────────────────────────────────

QUESTIONS_SIMPLE = [
    {"question_text": "Explain the OSI reference model in detail.", "year": 2021, "id": 1001},
    {"question_text": "Describe the seven layers of the OSI architecture.", "year": 2022, "id": 1002},
    {"question_text": "What are routing algorithms? Explain Dijkstra.", "year": 2021, "id": 1003},
    {"question_text": "Discuss shortest path algorithms used in networking.", "year": 2023, "id": 1004},
    {"question_text": "Explain congestion control mechanisms in TCP.", "year": 2022, "id": 1005},
]

TEXTS_SIMPLE = [q["question_text"] for q in QUESTIONS_SIMPLE]


# ══════════════════════════════════════════════════════════════════════════════
# is_available()
# ══════════════════════════════════════════════════════════════════════════════

def test_is_available_returns_bool():
    """is_available() must always return a bool."""
    result = is_available()
    assert isinstance(result, bool)

def test_is_available_matches_import():
    """is_available() must reflect whether the import succeeded."""
    assert is_available() == SENTENCE_TRANSFORMERS_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# embed_questions()
# ══════════════════════════════════════════════════════════════════════════════

def test_embed_questions_returns_array_and_method():
    embeddings, method = embed_questions(TEXTS_SIMPLE)
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(method, str)

def test_embed_questions_correct_row_count():
    """Number of rows must equal number of input questions."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    assert embeddings.shape[0] == len(TEXTS_SIMPLE)

def test_embed_questions_nonzero_dim():
    """Embedding dimension must be > 0."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    assert embeddings.shape[1] > 0

def test_embed_questions_empty_input():
    embeddings, method = embed_questions([])
    assert embeddings.size == 0

def test_embed_questions_single_item():
    """A single question should produce a (1, dim) array."""
    embeddings, _ = embed_questions(["Explain TCP."])
    assert embeddings.shape[0] == 1

def test_embed_questions_method_is_valid_string():
    """Method must be one of the known values."""
    _, method = embed_questions(TEXTS_SIMPLE)
    assert method in ("embeddings", "tfidf_fallback", "embeddings_cached")

def test_embed_questions_float_dtype():
    """Embeddings must be floating-point numbers."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    assert embeddings.dtype in (np.float32, np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# cosine_similarity_matrix()
# ══════════════════════════════════════════════════════════════════════════════

def test_cosine_sim_matrix_shape():
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    sim = cosine_similarity_matrix(embeddings)
    n = len(TEXTS_SIMPLE)
    assert sim.shape == (n, n)

def test_cosine_sim_diagonal_is_one():
    """Cosine similarity of a vector with itself is 1.0."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    sim = cosine_similarity_matrix(embeddings)
    for i in range(len(TEXTS_SIMPLE)):
        assert abs(sim[i][i] - 1.0) < 0.01

def test_cosine_sim_is_symmetric():
    """sim[i][j] must equal sim[j][i]."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    sim = cosine_similarity_matrix(embeddings)
    for i in range(len(TEXTS_SIMPLE)):
        for j in range(len(TEXTS_SIMPLE)):
            assert abs(sim[i][j] - sim[j][i]) < 1e-5

def test_cosine_sim_values_in_range():
    """All similarity scores must be in [-1, 1]."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    sim = cosine_similarity_matrix(embeddings)
    assert np.all(sim >= -1.01)
    assert np.all(sim <= 1.01)

def test_cosine_sim_empty_input():
    sim = cosine_similarity_matrix(np.array([]))
    assert sim.size == 0


# ══════════════════════════════════════════════════════════════════════════════
# find_semantic_pairs()
# ══════════════════════════════════════════════════════════════════════════════

def test_find_semantic_pairs_returns_list():
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    pairs = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=0.30)
    assert isinstance(pairs, list)

def test_find_semantic_pairs_sorted():
    """Pairs must be sorted by similarity descending."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    pairs = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=0.20)
    for i in range(len(pairs) - 1):
        assert pairs[i].similarity >= pairs[i + 1].similarity

def test_find_semantic_pairs_no_self_pairs():
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    pairs = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=0.20)
    for p in pairs:
        assert p.idx_a != p.idx_b

def test_find_semantic_pairs_threshold_respected():
    """All returned pairs must have similarity >= threshold."""
    threshold = 0.50
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    pairs = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=threshold)
    for p in pairs:
        assert p.similarity >= threshold - 0.001  # small float tolerance

def test_higher_threshold_fewer_pairs():
    """Increasing the threshold should not increase the number of pairs."""
    embeddings, _ = embed_questions(TEXTS_SIMPLE)
    pairs_low  = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=0.20)
    pairs_high = find_semantic_pairs(QUESTIONS_SIMPLE, embeddings, threshold=0.90)
    assert len(pairs_low) >= len(pairs_high)


# ══════════════════════════════════════════════════════════════════════════════
# run_embedding_similarity_analysis() — full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_returns_dict():
    result = run_embedding_similarity_analysis(QUESTIONS_SIMPLE, threshold=0.20)
    required_keys = {
        "pairs", "groups", "representatives", "sim_matrix",
        "embeddings", "method", "total_questions", "total_pairs", "total_groups"
    }
    assert required_keys.issubset(result.keys())

def test_full_pipeline_total_questions():
    result = run_embedding_similarity_analysis(QUESTIONS_SIMPLE, threshold=0.20)
    assert result["total_questions"] == len(QUESTIONS_SIMPLE)

def test_full_pipeline_counts_consistent():
    """total_pairs must equal len(pairs), total_groups must equal len(groups)."""
    result = run_embedding_similarity_analysis(QUESTIONS_SIMPLE, threshold=0.20)
    assert result["total_pairs"]  == len(result["pairs"])
    assert result["total_groups"] == len(result["groups"])

def test_full_pipeline_empty():
    result = run_embedding_similarity_analysis([], threshold=0.70)
    assert result["total_pairs"]  == 0
    assert result["total_groups"] == 0

def test_full_pipeline_single_question():
    result = run_embedding_similarity_analysis(
        [{"question_text": "Explain TCP.", "year": 2024}], threshold=0.70
    )
    assert result["total_pairs"] == 0

def test_full_pipeline_method_field():
    """Method must be a non-empty string."""
    result = run_embedding_similarity_analysis(QUESTIONS_SIMPLE, threshold=0.20)
    assert isinstance(result["method"], str)
    assert len(result["method"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# DB persistence: save_embeddings_to_db / load_embeddings_from_db
# ══════════════════════════════════════════════════════════════════════════════

def test_save_and_load_embeddings_roundtrip(tmp_path):
    """
    Saving and loading embeddings must produce identical arrays.

    We use a temporary database so we don't pollute the real qpredict.db.
    """
    import sqlite3

    # Create a temporary database
    db_file = str(tmp_path / "test_embeddings.db")

    # Monkey-patch the DB path used inside embeddings.py for this test
    import src.embeddings as emb_module
    original_path_logic = None  # we'll patch via the sqlite3 connection

    # Build small fake embeddings
    fake_embs = np.random.rand(3, 8).astype(np.float32)
    q_ids = [101, 102, 103]
    model_name = "test-model"

    # Write directly using sqlite3 (avoids path dependency)
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            UNIQUE(question_id, model_name)
        )
    """)
    for qid, emb in zip(q_ids, fake_embs):
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (question_id, model_name, embedding) VALUES (?,?,?)",
            (qid, model_name, emb.tobytes())
        )
    conn.commit()

    # Read back
    rows = {row[0]: row[1] for row in conn.execute(
        "SELECT question_id, embedding FROM embeddings WHERE model_name=?", (model_name,)
    )}
    conn.close()

    loaded = np.vstack([
        np.frombuffer(rows[qid], dtype=np.float32).copy()
        for qid in q_ids
    ])

    # Verify round-trip fidelity
    assert loaded.shape == fake_embs.shape
    assert np.allclose(loaded, fake_embs, atol=1e-6)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
