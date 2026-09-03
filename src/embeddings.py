"""
embeddings.py — Sentence embedding generation for QPredict

WHY THIS FILE EXISTS:
Phase 3 used TF-IDF, which compares questions by word overlap.
TF-IDF cannot detect that:

  "Explain CPU scheduling algorithms."
  "Discuss different methods to schedule processes."

are about the same topic, because they share almost no words.

Sentence embeddings solve this. A pretrained neural network converts
every sentence into a fixed-size vector (e.g. 384 numbers). Sentences
with similar *meaning* — regardless of wording — produce vectors that
point in nearly the same direction.

WHAT IS A PRETRAINED MODEL?
Training a language model from scratch requires massive compute and data.
Instead, we use `sentence-transformers`, a library from Hugging Face that
provides models already trained on hundreds of millions of sentence pairs.

We download the model once, it's cached locally in ~/.cache/huggingface/.
After that, it runs entirely on your machine — no internet needed.

MODEL CHOICE: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~80 MB download, fast on CPU
  - State-of-the-art for semantic similarity
  - Apache 2.0 license

GRACEFUL FALLBACK:
If sentence-transformers is not installed, this module falls back to
TF-IDF embeddings from Phase 3 so the rest of the app still runs.
This is the "progressive enhancement" pattern — advanced features layer
on top of working basics.
"""

import numpy as np
from typing import List, Optional, Tuple
import os

# ── Attempt to import sentence-transformers ────────────────────────────────
# We use a try/except so the import failure is handled gracefully.
# If the library is missing, SENTENCE_TRANSFORMERS_AVAILABLE = False and
# all functions in this module fall back to TF-IDF automatically.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Default model — small, fast, excellent for semantic similarity
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Where to cache downloaded models (relative to project root)
# The sentence-transformers library also has its own HuggingFace cache,
# but we keep a reference here for transparency.
MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models"
)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

# Module-level cache: after the model is loaded once, we keep it in memory.
# WHY: Loading a neural network from disk takes ~1–3 seconds. If we reloaded
# it on every function call, the UI would feel very slow.
_model_cache: dict = {}


def load_model(model_name: str = DEFAULT_MODEL) -> Optional[object]:
    """
    Load a SentenceTransformer model, using an in-memory cache.

    On the first call this downloads the model (~80 MB for MiniLM) and
    loads it into RAM. Subsequent calls return the cached model instantly.

    Parameters
    ----------
    model_name : str — HuggingFace model identifier

    Returns
    -------
    SentenceTransformer instance, or None if the library is not available.

    WHAT HAPPENS ON FIRST CALL:
      1. sentence_transformers downloads the model to ~/.cache/huggingface/
      2. The model is loaded into RAM
      3. We store it in _model_cache[model_name]

    WHAT HAPPENS ON SUBSEQUENT CALLS:
      1. We check _model_cache[model_name] — it's already there
      2. Return immediately
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    if model_name not in _model_cache:
        print(f"[Embeddings] Loading model: {model_name} ...")
        _model_cache[model_name] = SentenceTransformer(
            model_name,
            cache_folder=MODEL_CACHE_DIR
        )
        print(f"[Embeddings] Model loaded.")

    return _model_cache[model_name]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def embed_questions(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Convert a list of question strings into embedding vectors.

    Returns a 2D NumPy array of shape (n_questions, embedding_dim).
    Also returns the method name ("embeddings" or "tfidf_fallback").

    HOW SENTENCE ENCODING WORKS:
      1. The text is tokenized (split into subword tokens)
      2. Tokens are passed through 6 transformer layers
      3. The final layer's [CLS] token output is pooled
      4. The result is a 384-dimensional vector

    BATCH PROCESSING:
    Encoding one question at a time would be slow. We encode in batches
    (64 questions at a time by default). The model processes the batch
    in parallel, which is much faster.

    Parameters
    ----------
    texts        : List[str]  — raw or lightly cleaned question texts
                                (NOT the aggressively normalized version —
                                 embeddings work better on natural text)
    model_name   : str        — which model to use
    batch_size   : int        — questions per batch
    show_progress: bool       — show a tqdm progress bar

    Returns
    -------
    Tuple[np.ndarray, str]
        embeddings   : array of shape (n_questions, dim)
        method_used  : "embeddings" or "tfidf_fallback"
    """
    if not texts:
        return np.array([]), "embeddings"

    model = load_model(model_name)

    if model is None:
        # Fallback: use TF-IDF vectors instead
        return _tfidf_fallback(texts)

    # SentenceTransformer.encode() handles batching internally
    # convert_to_numpy=True returns a plain NumPy array instead of a Tensor
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )

    return embeddings, "embeddings"


def _tfidf_fallback(texts: List[str]) -> Tuple[np.ndarray, str]:
    """
    Fallback: generate TF-IDF vectors when sentence-transformers is not installed.

    WHY THIS EXISTS:
    This lets the rest of the Phase 4 code work identically regardless of
    whether the neural model is available. The UI simply shows a note
    explaining which method was used.

    Returns a dense NumPy array so the shape is consistent with real embeddings.
    """
    from src.text_cleaner import normalize_questions
    from sklearn.feature_extraction.text import TfidfVectorizer

    normalized = normalize_questions(texts)
    valid = [t if t.strip() else "unknown" for t in normalized]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(valid)
    return matrix.toarray().astype(np.float32), "tfidf_fallback"


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity from an embedding matrix.

    WHY NOT JUST USE sklearn.metrics.pairwise.cosine_similarity?
    We can — and we do use it in the TF-IDF path. Here we take advantage
    of the fact that SentenceTransformer returns L2-normalized vectors
    (when normalize_embeddings=True). For normalized vectors:

        cosine_similarity(A, B) = dot(A, B)

    So the entire similarity matrix is just a matrix multiplication:

        sim_matrix = embeddings @ embeddings.T

    This is faster than the sklearn version for dense matrices.

    For the TF-IDF fallback we use sklearn to be safe.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n, dim)

    Returns
    -------
    np.ndarray of shape (n, n) — symmetric, values in [-1, 1]
    """
    if embeddings.size == 0:
        return np.array([])

    # Check if the vectors appear to be L2-normalized
    # (all row norms close to 1.0) — true for SentenceTransformer output
    norms = np.linalg.norm(embeddings, axis=1)
    is_normalized = np.allclose(norms, 1.0, atol=0.01)

    if is_normalized:
        # Fast path: dot product for normalized vectors
        sim = embeddings @ embeddings.T
    else:
        # Safe path: full cosine similarity computation
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        sim = sk_cosine(embeddings)

    # Clip to [-1, 1] to handle floating point errors
    return np.clip(sim, -1.0, 1.0)


def find_semantic_pairs(
    questions: List[dict],
    embeddings: np.ndarray,
    threshold: float = 0.70,
) -> list:
    """
    Find semantically similar question pairs using embedding cosine similarity.

    This is the semantic equivalent of find_similar_pairs() from similarity.py.
    The logic is identical — only the vectors are different.

    Parameters
    ----------
    questions   : List[dict]   — question dicts (need 'question_text', 'year')
    embeddings  : np.ndarray   — (n_questions, dim) embedding matrix
    threshold   : float        — minimum cosine similarity

    Returns
    -------
    List of SimilarPair objects (imported from similarity.py)
    """
    from src.similarity import SimilarPair

    if embeddings.size == 0 or len(questions) < 2:
        return []

    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(questions)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score >= threshold:
                pairs.append(SimilarPair(
                    idx_a=i,
                    idx_b=j,
                    similarity=round(score, 4),
                    question_a=questions[i].get("question_text", ""),
                    question_b=questions[j].get("question_text", ""),
                    year_a=questions[i].get("year"),
                    year_b=questions[j].get("year"),
                ))

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_embedding_similarity_analysis(
    questions: List[dict],
    threshold: float = 0.70,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Run the complete semantic similarity analysis on a list of questions.

    This is the Phase 4 equivalent of run_tfidf_similarity_analysis().

    WHY KEEP BOTH PIPELINES?
    Phase 4 doesn't replace Phase 3 — it upgrades it. We keep TF-IDF
    available so we can compare results side by side. The comparison is
    educational: it shows exactly where embeddings outperform keywords.

    Parameters
    ----------
    questions  : List[dict]  — from get_all_questions() / get_questions_for_subject()
    threshold  : float       — similarity threshold (0.70 recommended for embeddings)
    model_name : str         — sentence-transformer model to use

    Returns
    -------
    dict with keys:
      "pairs"          : List[SimilarPair]
      "groups"         : List[List[int]]
      "representatives": List[int]
      "sim_matrix"     : np.ndarray
      "embeddings"     : np.ndarray
      "method"         : "embeddings" or "tfidf_fallback"
      "model_name"     : str
      "threshold"      : float
      "total_questions": int
      "total_pairs"    : int
      "total_groups"   : int
    """
    from src.similarity import group_similar_questions, get_representative_question

    texts = [q.get("question_text", "") for q in questions]

    # Generate embeddings (or TF-IDF fallback)
    embeddings, method = embed_questions(texts, model_name=model_name)

    if embeddings.size == 0:
        return {
            "pairs": [], "groups": [], "representatives": [],
            "sim_matrix": np.array([]), "embeddings": embeddings,
            "method": method, "model_name": model_name,
            "threshold": threshold, "total_questions": len(questions),
            "total_pairs": 0, "total_groups": 0,
        }

    sim_matrix = cosine_similarity_matrix(embeddings)
    pairs = find_semantic_pairs(questions, embeddings, threshold)
    groups = group_similar_questions(questions, pairs)
    representatives = [
        get_representative_question(g, questions, sim_matrix)
        for g in groups
    ]

    return {
        "pairs":           pairs,
        "groups":          groups,
        "representatives": representatives,
        "sim_matrix":      sim_matrix,
        "embeddings":      embeddings,
        "method":          method,
        "model_name":      model_name,
        "threshold":       threshold,
        "total_questions": len(questions),
        "total_pairs":     len(pairs),
        "total_groups":    len(groups),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING PERSISTENCE  (Phase 5+ will use this heavily)
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings_to_db(question_ids: List[int], embeddings: np.ndarray, model_name: str):
    """
    Persist embeddings to the database for reuse in later phases.

    WHY: Computing embeddings for 200 questions takes ~5–10 seconds.
    We don't want to recompute them every time the page loads. By storing
    them in the database, Phase 5 (clustering) and Phase 6 (scoring) can
    retrieve them instantly.

    The embedding is stored as a binary blob using numpy's tobytes() method.
    To retrieve it, use numpy.frombuffer().
    """
    import sqlite3
    import sys
    import os

    # Locate the database
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "qpredict.db"
    )

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Ensure embeddings table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            model_name  TEXT    NOT NULL,
            embedding   BLOB    NOT NULL,
            UNIQUE(question_id, model_name)
        )
    """)

    for qid, emb in zip(question_ids, embeddings):
        blob = emb.astype(np.float32).tobytes()
        # INSERT OR REPLACE overwrites an existing embedding for the same
        # question + model combination, so re-running analysis is safe.
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (question_id, model_name, embedding)
            VALUES (?, ?, ?)
        """, (int(qid), model_name, blob))

    conn.commit()
    conn.close()


def load_embeddings_from_db(question_ids: List[int], model_name: str) -> Optional[np.ndarray]:
    """
    Load previously computed embeddings from the database.

    Returns a (n, dim) float32 array if all question_ids are found,
    or None if any are missing (triggering recomputation).
    """
    import sqlite3
    import os

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "qpredict.db"
    )

    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Check the embeddings table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    if not cursor.fetchone():
        conn.close()
        return None

    placeholders = ",".join("?" * len(question_ids))
    cursor.execute(
        f"SELECT question_id, embedding FROM embeddings "
        f"WHERE question_id IN ({placeholders}) AND model_name = ?",
        question_ids + [model_name]
    )
    rows = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    if len(rows) != len(question_ids):
        return None  # Some embeddings are missing — recompute all

    # Reconstruct the array in the correct order
    arrays = []
    for qid in question_ids:
        arr = np.frombuffer(rows[qid], dtype=np.float32).copy()
        arrays.append(arr)

    return np.vstack(arrays)


def get_or_compute_embeddings(
    questions: List[dict],
    model_name: str = DEFAULT_MODEL,
) -> Tuple[np.ndarray, str]:
    """
    Return embeddings for questions, using DB cache when available.

    This is the smart version of embed_questions():
      1. Check if embeddings are already stored in the database
      2. If yes → return cached embeddings instantly
      3. If no → compute, persist to DB, return

    WHY CACHING MATTERS:
    A batch of 150 questions takes ~8 seconds to embed on CPU.
    On the second page load (or for Phase 5 clustering), we want
    instant results. The DB cache eliminates recomputation.
    """
    question_ids = [q["id"] for q in questions if "id" in q]
    texts = [q.get("question_text", "") for q in questions]

    # Try loading from cache
    if question_ids and len(question_ids) == len(questions):
        cached = load_embeddings_from_db(question_ids, model_name)
        if cached is not None:
            return cached, "embeddings_cached"

    # Compute fresh embeddings
    embeddings, method = embed_questions(texts, model_name=model_name)

    # Persist to DB if we have real embeddings and real IDs
    if method == "embeddings" and question_ids and len(question_ids) == len(questions):
        try:
            save_embeddings_to_db(question_ids, embeddings, model_name)
        except Exception:
            pass  # Non-critical — don't crash the analysis if caching fails

    return embeddings, method


def is_available() -> bool:
    """Return True if sentence-transformers is installed and usable."""
    return SENTENCE_TRANSFORMERS_AVAILABLE
