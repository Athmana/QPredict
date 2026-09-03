"""
clustering.py — Question clustering for QPredict

WHY THIS FILE EXISTS:
Phase 3 and 4 give us pairwise similarity scores between questions.
But students don't want to see a list of 500 pairs — they want to see:

  "OSI Model — appeared in 2021, 2022, 2024, 2025 (4 papers)"
  "TCP/IP — appeared in 2021, 2023, 2025 (3 papers)"
  "Routing Algorithms — appeared in 2022, 2023, 2024 (3 papers)"

Clustering converts pairwise relationships into **named groups** where
each group represents one recurring exam concept.

THE CORE ALGORITHM: Agglomerative Clustering
─────────────────────────────────────────────
Agglomerative = "building up from the bottom"

Step 1: Start with each question in its own cluster (N clusters).
Step 2: Find the two closest clusters.
Step 3: Merge them into one.
Step 4: Repeat until no two clusters are closer than `distance_threshold`.

"Closest" is defined by a linkage criterion. We use "average" linkage:
  distance(cluster_A, cluster_B) = average distance between all pairs
                                   (one from A, one from B)

This is more robust than "single" linkage (which can chain clusters
into long strings) and "complete" linkage (which can be too strict).

DISTANCE vs SIMILARITY:
  cosine_similarity is close to 1.0 for similar questions.
  agglomerative clustering needs a distance (low = close).
  We convert:   distance = 1 - cosine_similarity

So distance=0 means identical, distance=1 means completely unrelated.

TOPIC LABELING:
After clustering, each cluster needs a human-readable name.
We extract this using TF-IDF keywords from the cluster's questions.
The top 3–4 keywords become the topic label.
Example: questions about OSI → keywords "osi", "layers", "model" → "OSI Layers Model"

An LLM can generate better labels in Phase 9 — for now keywords suffice.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuestionCluster:
    """
    Represents one group of semantically related questions.

    This is the central object of Phase 5. Each cluster corresponds to
    one recurring exam topic — the "question family".

    Attributes
    ----------
    cluster_id          : int  — 0-based index
    topic_label         : str  — human-readable topic name (e.g. "OSI Model")
    member_indices      : list — indices into the questions list
    representative_idx  : int  — the most central question's index
    years               : list — sorted list of years this topic appeared
    paper_count         : int  — number of distinct papers containing it
    total_appearances   : int  — total number of related questions found
    keywords            : list — top keywords extracted from the cluster
    """
    cluster_id: int
    topic_label: str = ""
    member_indices: List[int] = field(default_factory=list)
    representative_idx: int = 0
    years: List[int] = field(default_factory=list)
    paper_count: int = 0
    total_appearances: int = 0
    keywords: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.member_indices)

    @property
    def year_coverage(self) -> str:
        """Human-readable year list, e.g. '2021, 2022, 2024'."""
        return ", ".join(str(y) for y in self.years)

    def to_dict(self) -> dict:
        return {
            "cluster_id":         self.cluster_id,
            "topic_label":        self.topic_label,
            "member_count":       self.size,
            "representative_idx": self.representative_idx,
            "years":              self.years,
            "paper_count":        self.paper_count,
            "total_appearances":  self.total_appearances,
            "keywords":           self.keywords,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DISTANCE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def embeddings_to_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert a cosine similarity matrix to a distance matrix.

    Agglomerative clustering needs distances, not similarities.
    For normalized embeddings:
        cosine_similarity  ∈ [0, 1]  (1 = identical, 0 = unrelated)
        cosine_distance    = 1 - cosine_similarity ∈ [0, 1]

    We use the precomputed metric in sklearn by passing the distance
    matrix directly, avoiding recomputation of the full embedding norms.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n, dim)

    Returns
    -------
    np.ndarray of shape (n, n) — symmetric distance matrix
    """
    from src.embeddings import cosine_similarity_matrix
    sim = cosine_similarity_matrix(embeddings)
    # Clip to avoid distances slightly below 0 due to float arithmetic
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def cluster_agglomerative(
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """
    Cluster questions using Agglomerative Clustering.

    WHY THIS ALGORITHM:
      - Does NOT require specifying n_clusters upfront
      - distance_threshold controls granularity (lower = more clusters)
      - "average" linkage is robust against chaining artifacts
      - Returns deterministic results (no randomness)

    HOW distance_threshold MAPS TO SIMILARITY:
      distance = 1 - cosine_similarity
      threshold = 0.35  →  questions must have similarity ≥ 0.65 to cluster
      threshold = 0.20  →  questions must have similarity ≥ 0.80 (stricter)
      threshold = 0.50  →  questions must have similarity ≥ 0.50 (looser)

    Parameters
    ----------
    embeddings         : np.ndarray (n, dim)
    distance_threshold : float  — max distance within a cluster
    min_cluster_size   : int    — clusters smaller than this are discarded

    Returns
    -------
    np.ndarray of shape (n,) — cluster label for each question.
    Questions not in any cluster (singletons) get label -1.
    """
    if len(embeddings) < 2:
        return np.array([-1] * len(embeddings))

    dist_matrix = embeddings_to_distance_matrix(embeddings)

    model = AgglomerativeClustering(
        n_clusters=None,              # let distance_threshold decide
        distance_threshold=distance_threshold,
        metric="precomputed",         # we supply our own distance matrix
        linkage="average",            # average distance between cluster members
    )

    labels = model.fit_predict(dist_matrix)

    # Mark small clusters as -1 (noise / singleton)
    from collections import Counter
    counts = Counter(labels)
    for i, label in enumerate(labels):
        if label != -1 and counts[label] < min_cluster_size:
            labels[i] = -1

    return labels


def cluster_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.35,
    min_samples: int = 2,
) -> np.ndarray:
    """
    Cluster questions using DBSCAN.

    WHY THIS ALTERNATIVE:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    finds clusters based on local density. It naturally handles outliers
    — questions that are genuinely unique get labelled -1 (noise).

    HOW eps RELATES TO SIMILARITY:
      eps is the maximum distance between two points in the same neighborhood.
      eps = 0.35  →  similarity ≥ 0.65 required to be neighbors

    Parameters
    ----------
    embeddings  : np.ndarray (n, dim)
    eps         : float  — neighborhood radius (distance, not similarity)
    min_samples : int    — minimum points to form a dense region

    Returns
    -------
    np.ndarray of shape (n,) — cluster labels.
    -1 means the question is an outlier (no cluster).
    """
    if len(embeddings) < 2:
        return np.array([-1] * len(embeddings))

    dist_matrix = embeddings_to_distance_matrix(embeddings)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = model.fit_predict(dist_matrix)
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC LABEL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_topic_label(
    question_texts: List[str],
    top_n: int = 4,
    title_case: bool = True,
) -> tuple:
    """
    Extract a topic label for a cluster using TF-IDF keyword extraction.

    HOW IT WORKS:
    We treat the cluster's questions as one mini corpus. TF-IDF identifies
    which words are most characteristic of these questions compared to
    common English words.

    WHY TF-IDF FOR LABELING (not embeddings)?
    Embeddings capture meaning but don't give us discrete keywords.
    TF-IDF gives us actual words — exactly what we need for a label.

    Later (Phase 9), an LLM will generate more natural topic labels.

    Parameters
    ----------
    question_texts : List[str]  — the questions in this cluster
    top_n          : int        — how many keywords to include in label
    title_case     : bool       — whether to Title Case the result

    Returns
    -------
    tuple: (label_str, keywords_list)
      label_str    : "OSI Reference Model Layers"
      keywords_list: ["osi", "reference", "model", "layers"]
    """
    if not question_texts:
        return "Unknown Topic", []

    # If only one question, just use first N meaningful words
    if len(question_texts) == 1:
        from src.text_cleaner import normalize_question
        words = normalize_question(question_texts[0]).split()
        keywords = words[:top_n]
        label = " ".join(w.title() for w in keywords) if title_case else " ".join(keywords)
        return label, keywords

    from src.text_cleaner import normalize_questions, STOPWORDS

    # Additional domain-generic words to exclude from topic labels
    # These describe question format, not topic content
    FORMAT_WORDS = {
        "explain", "describe", "discuss", "define", "compare",
        "differentiate", "elaborate", "illustrate", "analyze",
        "analyse", "examine", "evaluate", "briefly", "detail",
        "short", "note", "write", "give", "state", "list",
        "example", "examples", "different", "various", "types",
        "type", "method", "methods", "concept", "concepts",
        "term", "terms", "use", "used", "using", "important",
        "explain", "way", "ways", "role", "roles", "function",
        "advantage", "advantages", "disadvantage", "disadvantages",
    }
    all_stopwords = STOPWORDS | FORMAT_WORDS

    normalized = normalize_questions(question_texts, remove_stopwords=True)
    # Further remove format words
    cleaned = [
        " ".join(w for w in text.split() if w not in FORMAT_WORDS)
        for text in normalized
    ]
    # If everything got removed, fall back to normalized text
    cleaned = [c if c.strip() else n for c, n in zip(cleaned, normalized)]

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),   # single words for labels (bigrams look bad)
            min_df=1,
            max_df=0.99,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(cleaned)
        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF score across all questions in the cluster
        avg_scores = np.asarray(matrix.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices
                    if feature_names[i] not in all_stopwords]

    except Exception:
        # If TF-IDF fails (e.g. all texts empty), extract first words
        words = " ".join(normalized).split()
        keywords = list(dict.fromkeys(
            w for w in words if w not in all_stopwords
        ))[:top_n]

    if not keywords:
        keywords = ["Topic"]

    label = " ".join(w.upper() if len(w) <= 3 else w.title() for w in keywords[:top_n])
    return label, keywords


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_clusters(
    questions: List[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    sim_matrix: np.ndarray,
) -> List[QuestionCluster]:
    """
    Convert raw cluster labels into rich QuestionCluster objects.

    This function takes the raw integer labels from the clustering algorithm
    and builds fully populated QuestionCluster objects with:
      - member question indices
      - representative question (most central)
      - year coverage
      - topic label
      - keyword list

    Parameters
    ----------
    questions  : List[dict]   — question dicts (need 'year', 'question_text')
    embeddings : np.ndarray   — (n, dim) — used to find centroids
    labels     : np.ndarray   — (n,) cluster label per question (-1 = noise)
    sim_matrix : np.ndarray   — (n, n) pairwise similarity

    Returns
    -------
    List[QuestionCluster], sorted by total_appearances descending
    """
    from collections import defaultdict

    # Group question indices by cluster label
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:  # skip outliers
            label_to_indices[int(label)].append(idx)

    clusters = []
    for cluster_id, member_indices in label_to_indices.items():
        # ── Representative question ───────────────────────────────────
        # The representative is the question closest to the cluster centroid.
        # Centroid = average of all member embeddings.
        member_embeddings = embeddings[member_indices]
        centroid = member_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)  # normalize

        # Find the member whose embedding is closest to the centroid
        similarities_to_centroid = member_embeddings @ centroid
        rep_local_idx = int(np.argmax(similarities_to_centroid))
        rep_idx = member_indices[rep_local_idx]

        # ── Year coverage ──────────────────────────────────────────────
        years = sorted({
            questions[i].get("year")
            for i in member_indices
            if questions[i].get("year")
        })

        # ── Paper count ────────────────────────────────────────────────
        # Number of distinct papers (by paper_id) containing cluster members
        paper_ids = {
            questions[i].get("paper_id")
            for i in member_indices
            if questions[i].get("paper_id")
        }

        # ── Topic label ────────────────────────────────────────────────
        member_texts = [questions[i].get("question_text", "") for i in member_indices]
        topic_label, keywords = extract_topic_label(member_texts)

        cluster = QuestionCluster(
            cluster_id=cluster_id,
            topic_label=topic_label,
            member_indices=member_indices,
            representative_idx=rep_idx,
            years=years,
            paper_count=len(paper_ids),
            total_appearances=len(member_indices),
            keywords=keywords,
        )
        clusters.append(cluster)

    # Sort by most appearances first (most recurring topics first)
    clusters.sort(key=lambda c: c.total_appearances, reverse=True)

    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(
    questions: List[dict],
    embeddings: np.ndarray,
    algorithm: str = "agglomerative",
    distance_threshold: float = 0.35,
    min_cluster_size: int = 2,
) -> dict:
    """
    Run the complete clustering pipeline and return all results.

    This is the single function the app calls for Phase 5.

    Parameters
    ----------
    questions          : List[dict]   — from get_questions_for_subject()
    embeddings         : np.ndarray   — (n, dim) from get_or_compute_embeddings()
    algorithm          : str          — "agglomerative" or "dbscan"
    distance_threshold : float        — 0.35 is a good default
    min_cluster_size   : int          — discard clusters smaller than this

    Returns
    -------
    dict with keys:
      "clusters"          : List[QuestionCluster]
      "labels"            : np.ndarray — raw label per question
      "sim_matrix"        : np.ndarray — pairwise similarity
      "total_questions"   : int
      "clustered"         : int  — questions assigned to a cluster
      "unclustered"       : int  — questions not in any cluster (noise)
      "total_clusters"    : int
      "algorithm"         : str
      "distance_threshold": float
    """
    from src.embeddings import cosine_similarity_matrix

    if len(questions) < 2 or embeddings.size == 0:
        return {
            "clusters": [], "labels": np.array([]),
            "sim_matrix": np.array([]),
            "total_questions": len(questions), "clustered": 0,
            "unclustered": len(questions), "total_clusters": 0,
            "algorithm": algorithm, "distance_threshold": distance_threshold,
        }

    # Step 1: run chosen clustering algorithm
    if algorithm == "dbscan":
        labels = cluster_dbscan(embeddings, eps=distance_threshold, min_samples=min_cluster_size)
    else:
        labels = cluster_agglomerative(embeddings, distance_threshold, min_cluster_size)

    # Step 2: compute similarity matrix for representative selection
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Step 3: build rich cluster objects
    clusters = build_clusters(questions, embeddings, labels, sim_matrix)

    clustered   = int(np.sum(labels != -1))
    unclustered = int(np.sum(labels == -1))

    return {
        "clusters":           clusters,
        "labels":             labels,
        "sim_matrix":         sim_matrix,
        "total_questions":    len(questions),
        "clustered":          clustered,
        "unclustered":        unclustered,
        "total_clusters":     len(clusters),
        "algorithm":          algorithm,
        "distance_threshold": distance_threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_clusters_to_db(clusters: List[QuestionCluster], questions: List[dict]):
    """
    Persist cluster results to the database.

    WHY: Phase 6 (scoring) needs to read cluster membership to compute
    frequency and year coverage scores. Saving clusters avoids rerunning
    the clustering algorithm every time.

    Tables used:
      clusters       — one row per cluster (name, representative, score)
      cluster_members — one row per (cluster, question) pair
    """
    import sqlite3
    import os
    import json

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "qpredict.db"
    )
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Ensure tables exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_label             TEXT,
            representative_question TEXT,
            representative_question_id INTEGER,
            total_appearances       INTEGER,
            paper_count             INTEGER,
            years_json              TEXT,
            keywords_json           TEXT,
            priority_score          REAL DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_members (
            cluster_id  INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, question_id),
            FOREIGN KEY (cluster_id)  REFERENCES clusters(id),
            FOREIGN KEY (question_id) REFERENCES questions(id)
        )
    """)

    saved_cluster_ids = []

    for cluster in clusters:
        rep_q = questions[cluster.representative_idx]
        rep_text = rep_q.get("question_text", "")
        rep_id   = rep_q.get("id")

        cursor.execute("""
            INSERT INTO clusters
                (topic_label, representative_question, representative_question_id,
                 total_appearances, paper_count, years_json, keywords_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cluster.topic_label,
            rep_text,
            rep_id,
            cluster.total_appearances,
            cluster.paper_count,
            json.dumps(cluster.years),
            json.dumps(cluster.keywords),
        ))
        db_cluster_id = cursor.lastrowid
        saved_cluster_ids.append(db_cluster_id)

        for member_idx in cluster.member_indices:
            q = questions[member_idx]
            q_id = q.get("id")
            if q_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO cluster_members (cluster_id, question_id)
                    VALUES (?, ?)
                """, (db_cluster_id, q_id))

    conn.commit()
    conn.close()
    return saved_cluster_ids


def get_clusters_from_db(subject: str) -> list:
    """
    Load saved clusters for a subject from the database.

    Returns list of dicts — one per cluster, with member question texts.
    Used by Phase 6 scoring and Phase 7 dashboard.
    """
    import sqlite3
    import os
    import json

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "qpredict.db"
    )
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check tables exist
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
    )
    if not cursor.fetchone():
        conn.close()
        return []

    cursor.execute("""
        SELECT c.*
        FROM clusters c
        JOIN cluster_members cm ON c.id = cm.cluster_id
        JOIN questions q ON cm.question_id = q.id
        JOIN papers p ON q.paper_id = p.id
        WHERE p.subject = ?
        GROUP BY c.id
        ORDER BY c.total_appearances DESC
    """, (subject,))

    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        d["years"] = json.loads(d.get("years_json") or "[]")
        d["keywords"] = json.loads(d.get("keywords_json") or "[]")
        result.append(d)
    return result
