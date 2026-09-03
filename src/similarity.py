"""
similarity.py — TF-IDF vectorization and cosine similarity for QPredict

WHY THIS FILE EXISTS:
We have questions as strings. We need numbers to compare them.
This module:
  1. Converts all questions into TF-IDF vectors (a matrix of numbers)
  2. Computes cosine similarity between every pair of questions
  3. Returns pairs that exceed a configurable threshold

UNDERSTANDING THE PIPELINE:

  Questions (strings)
       ↓
  TF-IDF Vectorizer
       ↓
  Matrix of shape (num_questions × num_unique_words)
  Each row is one question. Each column is one word.
  The value at [i, j] is the TF-IDF weight of word j in question i.
       ↓
  Cosine Similarity
       ↓
  Matrix of shape (num_questions × num_questions)
  The value at [i, j] is how similar question i is to question j.
  Values range from 0.0 (no similarity) to 1.0 (identical).
       ↓
  Filter pairs where similarity >= THRESHOLD
       ↓
  SimilarPair objects

WHAT IS TF-IDF IN MORE DETAIL?

  TF (Term Frequency):
    tf(word, question) = (count of word in this question) / (total words in this question)

  IDF (Inverse Document Frequency):
    idf(word) = log( (total questions) / (questions containing this word) )
    Rare words get a high IDF. Common words get a low IDF.

  TF-IDF:
    tfidf(word, question) = tf × idf

  The result: words that appear frequently in one question but rarely
  across all questions get high scores. These are the "important" words
  for that question.

WHY COSINE SIMILARITY AND NOT EUCLIDEAN DISTANCE?
  Questions vary in length. A 3-word question and a 15-word question
  would look far apart in Euclidean space even if they're about the
  same topic. Cosine similarity is length-normalized — it only cares
  about the direction of the vector, not its magnitude.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# The similarity threshold: pairs with score >= this value are considered
# "related". This is configurable because the right value depends on
# your specific question bank.
#
# 0.70 is a reasonable starting point but should be tuned:
#   Too high (0.90) → misses genuinely similar questions
#   Too low  (0.40) → groups unrelated questions together
#
# Phase 4 (embeddings) will allow a lower threshold because embeddings
# capture meaning, not just word overlap.
DEFAULT_THRESHOLD = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimilarPair:
    """
    Represents two questions that have been found to be similar.

    Attributes
    ----------
    idx_a         : index of question A in the input list
    idx_b         : index of question B in the input list
    similarity    : cosine similarity score, 0.0–1.0
    question_a    : original text of question A
    question_b    : original text of question B
    year_a        : examination year of question A
    year_b        : examination year of question B
    """
    idx_a: int
    idx_b: int
    similarity: float
    question_a: str = ""
    question_b: str = ""
    year_a: Optional[int] = None
    year_b: Optional[int] = None

    def is_cross_year(self) -> bool:
        """Return True if the two questions come from different years."""
        if self.year_a is None or self.year_b is None:
            return False
        return self.year_a != self.year_b


# ══════════════════════════════════════════════════════════════════════════════
# TFIDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════

def build_tfidf_matrix(normalized_texts: List[str]):
    """
    Fit a TF-IDF vectorizer on the given texts and return the matrix.

    WHY THIS FUNCTION:
    The TfidfVectorizer needs to "learn" the vocabulary from all questions
    first (this is called fitting). Then it transforms each question into
    a vector using that vocabulary.

    This is a two-step process:
      fit()       — learn vocabulary and IDF weights from ALL questions
      transform() — convert each question to a vector

    fit_transform() does both in one call.

    Parameters
    ----------
    normalized_texts : List[str]
        Already-normalized question strings (output of text_cleaner.py)

    Returns
    -------
    tuple: (vectorizer, tfidf_matrix)
        vectorizer   : fitted TfidfVectorizer — needed to transform new questions
        tfidf_matrix : sparse matrix of shape (n_questions, n_unique_words)

    WHAT IS A SPARSE MATRIX?
    Most questions contain only a small subset of all vocabulary words.
    Instead of storing a full matrix with mostly zeros, scikit-learn uses
    a "sparse" representation that only stores the non-zero values.
    This is much more memory-efficient.
    """
    if not normalized_texts or len(normalized_texts) < 2:
        return None, None

    # Remove completely empty strings — they can't be vectorized
    valid_texts = [t if t.strip() else "unknown" for t in normalized_texts]

    vectorizer = TfidfVectorizer(
        # ngram_range=(1, 2) means we consider both single words AND
        # two-word phrases as features.
        # "OSI model" as a bigram is more informative than "OSI" and "model" alone.
        ngram_range=(1, 2),

        # min_df=1: include a term even if it appears in only 1 document.
        # For small question banks this is important — a unique technical
        # term in one question might be the key similarity signal.
        min_df=1,

        # max_df=0.95: ignore terms that appear in more than 95% of questions.
        # These are so common they carry no discriminating power.
        max_df=0.95,

        # sublinear_tf=True: use log(1 + tf) instead of raw tf.
        # This dampens the effect of a word appearing many times in one
        # long question versus once in a short question.
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    return vectorizer, tfidf_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """
    Compute pairwise cosine similarity for all questions.

    Returns a 2D NumPy array (dense) where:
      result[i][j] = cosine similarity between question i and question j

    The diagonal is always 1.0 (a question is identical to itself).
    The matrix is symmetric: result[i][j] == result[j][i].

    WHY DENSE?
    cosine_similarity() from scikit-learn returns a dense array. For
    thousands of questions this could use significant memory. In Phase 5
    (clustering) we'll introduce FAISS for scalable approximate search.
    For the MVP with tens to hundreds of questions, this is fine.
    """
    if tfidf_matrix is None:
        return np.array([])

    # cosine_similarity() from sklearn handles the math:
    # cos(θ) = (A · B) / (||A|| × ||B||)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


def find_similar_pairs(
    questions: List[dict],
    normalized_texts: List[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[SimilarPair]:
    """
    Find all pairs of questions with similarity above the threshold.

    This is the main public function of this module for Phase 3.

    HOW IT WORKS:
      1. Build TF-IDF matrix from normalized_texts
      2. Compute pairwise cosine similarity
      3. Scan the upper triangle of the matrix (avoid counting pairs twice)
      4. Collect pairs where similarity >= threshold and i != j

    Parameters
    ----------
    questions        : List[dict]  — question dicts from the database
                                     (must have 'question_text', optionally 'year')
    normalized_texts : List[str]   — pre-normalized versions of each question
    threshold        : float       — minimum similarity to consider related

    Returns
    -------
    List[SimilarPair], sorted by similarity descending
    """
    if len(questions) < 2:
        return []

    _, tfidf_matrix = build_tfidf_matrix(normalized_texts)
    if tfidf_matrix is None:
        return []

    sim_matrix = compute_similarity_matrix(tfidf_matrix)

    pairs: List[SimilarPair] = []
    n = len(questions)

    # We only scan the UPPER TRIANGLE of the matrix (where j > i).
    # WHY: The matrix is symmetric (sim[i][j] == sim[j][i]).
    # If we scanned the full matrix we'd find every pair twice.
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

    # Sort by similarity descending — highest similarity first
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# GROUPING SIMILAR QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def group_similar_questions(
    questions: List[dict],
    similar_pairs: List[SimilarPair],
) -> List[List[int]]:
    """
    Use Union-Find to group questions that are transitively similar.

    WHY UNION-FIND?
    Imagine three questions A, B, C where:
      sim(A, B) = 0.82  → above threshold → they're related
      sim(B, C) = 0.75  → above threshold → they're related
      sim(A, C) = 0.65  → below threshold → not directly paired

    Should A and C be in the same group? Yes — because they're both
    related to B. This is "transitive similarity."

    Union-Find (also called Disjoint Set Union) efficiently tracks
    which questions belong to the same connected component.

    HOW UNION-FIND WORKS:
      - Each question starts in its own group (its own "set")
      - When we find a pair (A, B) above threshold, we "union" their sets
      - After processing all pairs, questions in the same set are grouped
      - find(x) returns the "root" representative of x's group

    Parameters
    ----------
    questions     : List[dict]      — all questions (for their indices)
    similar_pairs : List[SimilarPair] — pairs above the threshold

    Returns
    -------
    List[List[int]] — each inner list is a group of question indices
                      groups of size 1 (isolated questions) are excluded
    """
    n = len(questions)
    parent = list(range(n))  # Initially each question is its own parent

    def find(x: int) -> int:
        """Find the root of x's group (with path compression)."""
        # Path compression: make every node point directly to the root
        # This keeps the tree flat and makes future find() calls faster
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        """Merge the groups containing x and y."""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x  # attach y's root under x's root

    # Process all similar pairs
    for pair in similar_pairs:
        union(pair.idx_a, pair.idx_b)

    # Collect groups: map root → list of members
    groups: dict = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Return only groups with more than one question
    # (singletons are questions with no similar matches)
    multi_member = [members for members in groups.values() if len(members) > 1]

    # Sort groups by size descending (largest groups first)
    multi_member.sort(key=len, reverse=True)
    return multi_member


def get_representative_question(group: List[int], questions: List[dict], sim_matrix: np.ndarray) -> int:
    """
    Choose the most representative question from a group.

    The representative is the question with the highest average
    similarity to all other questions in its group — the "centroid."

    This is the question QPredict will display as the primary/canonical
    version when showing a question cluster.

    Parameters
    ----------
    group       : List[int]    — question indices in this group
    questions   : List[dict]   — all questions
    sim_matrix  : np.ndarray   — full similarity matrix

    Returns
    -------
    int — index of the most representative question
    """
    if len(group) == 1:
        return group[0]

    best_idx = group[0]
    best_avg_sim = -1.0

    for i in group:
        # Average similarity to all OTHER members of the group
        sims = [sim_matrix[i][j] for j in group if j != i]
        avg = sum(sims) / len(sims)
        if avg > best_avg_sim:
            best_avg_sim = avg
            best_idx = i

    return best_idx


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_tfidf_similarity_analysis(
    questions: List[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Run the complete TF-IDF similarity analysis on a list of questions.

    This is the single entry point the app calls. It handles everything:
    normalization → vectorization → similarity → grouping → representatives.

    Parameters
    ----------
    questions  : List[dict]  — question dicts (from get_all_questions())
                               Must have keys: question_text, year (optional)
    threshold  : float       — similarity threshold

    Returns
    -------
    dict with keys:
      "pairs"          : List[SimilarPair]   — all similar pairs found
      "groups"         : List[List[int]]     — grouped question indices
      "representatives": List[int]           — one representative per group
      "sim_matrix"     : np.ndarray          — full similarity matrix
      "normalized"     : List[str]           — normalized question texts
      "threshold"      : float               — threshold used
      "total_questions": int
      "total_pairs"    : int
      "total_groups"   : int
    """
    # Import here to avoid circular dependency issues at module load time
    from src.text_cleaner import normalize_questions

    texts = [q.get("question_text", "") for q in questions]
    normalized = normalize_questions(texts)

    _, tfidf_matrix = build_tfidf_matrix(normalized)

    if tfidf_matrix is None:
        return {
            "pairs": [], "groups": [], "representatives": [],
            "sim_matrix": np.array([]), "normalized": normalized,
            "threshold": threshold, "total_questions": len(questions),
            "total_pairs": 0, "total_groups": 0,
        }

    sim_matrix = compute_similarity_matrix(tfidf_matrix)
    pairs = find_similar_pairs(questions, normalized, threshold)
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
        "normalized":      normalized,
        "threshold":       threshold,
        "total_questions": len(questions),
        "total_pairs":     len(pairs),
        "total_groups":    len(groups),
    }
