"""
syllabus_mapper.py — Syllabus extraction and question-to-unit mapping

WHY THIS FILE EXISTS:
A syllabus organises exam content into units and topics.
By mapping QPredict's question clusters onto the syllabus, we can answer:
  "Which unit has the most historically tested topics?"
  "How much of Unit 3 has appeared in past exams?"
  "Are there syllabus topics that have never been tested?"

THIS MODULE DOES THREE THINGS:
1. Parse a syllabus PDF into a structured list of SyllabusUnit objects
   (unit number, unit name, list of topics).
2. Embed each topic using the same embedding model as questions.
3. Map each question cluster to its nearest syllabus unit using
   cosine similarity between cluster embedding and topic embeddings.

SYLLABUS FORMAT HANDLING:
Syllabuses vary enormously. We handle the most common patterns:
  "Unit 1 — Introduction to Computer Networks"
  "MODULE 2: Data Link Layer"
  "UNIT III: Network Layer Protocols"
  "Chapter 4 — Transport Layer"
  "1. Introduction"

Each unit is followed by topic lines that are NOT themselves unit headers.
We collect those lines as the unit's topic list.

MAPPING STRATEGY:
For each question cluster we have:
  - A topic label (e.g. "OSI Model")
  - A set of member question texts

For each syllabus unit we have:
  - A unit name (e.g. "Network Layer")
  - A list of topic strings

We compute:
  similarity(cluster, unit) = max cosine_similarity(
      cluster_centroid_embedding,
      each_topic_embedding_in_unit
  )

The unit with the highest similarity is assigned to the cluster.
We also keep the raw scores so the UI can show confidence.
"""

import re
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyllabusUnit:
    """
    One unit from a syllabus document.

    Attributes
    ----------
    unit_number : str  — "1", "II", "3", etc.
    unit_name   : str  — "Introduction to Computer Networks"
    topics      : list — list of topic strings found under this unit
    raw_text    : str  — original text block for this unit (for debugging)
    """
    unit_number: str
    unit_name:   str
    topics:      List[str] = field(default_factory=list)
    raw_text:    str = ""

    @property
    def full_label(self) -> str:
        """e.g. 'Unit 1 — Introduction to Computer Networks'"""
        return f"Unit {self.unit_number} — {self.unit_name}"

    @property
    def all_text(self) -> str:
        """Unit name + all topics joined — used as the embedding input."""
        parts = [self.unit_name] + self.topics
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "unit_number": self.unit_number,
            "unit_name":   self.unit_name,
            "topics":      self.topics,
            "full_label":  self.full_label,
        }


@dataclass
class UnitMapping:
    """
    The result of mapping one question cluster to a syllabus unit.

    Attributes
    ----------
    cluster_id         : int   — which cluster
    cluster_label      : str   — topic label of the cluster
    assigned_unit_idx  : int   — index into the SyllabusUnit list
    assigned_unit_name : str   — name of the assigned unit
    similarity_score   : float — how confident is the assignment (0–1)
    all_unit_scores    : list  — similarity to every unit (for debugging)
    """
    cluster_id:         int
    cluster_label:      str
    assigned_unit_idx:  int
    assigned_unit_name: str
    similarity_score:   float
    all_unit_scores:    List[float] = field(default_factory=list)


@dataclass
class UnitAnalysis:
    """
    Aggregated analysis for one syllabus unit.

    Contains all clusters mapped to it, total questions, and a
    unit-level priority score (average of cluster scores).
    """
    unit:               SyllabusUnit
    mapped_clusters:    list   = field(default_factory=list)   # List[ScoredCluster]
    total_questions:    int    = 0
    total_appearances:  int    = 0
    avg_priority_score: float  = 0.0
    max_priority_score: float  = 0.0
    years_covered:      List[int] = field(default_factory=list)

    @property
    def cluster_count(self) -> int:
        return len(self.mapped_clusters)


# ══════════════════════════════════════════════════════════════════════════════
# SYLLABUS PARSING
# ══════════════════════════════════════════════════════════════════════════════

# Regex patterns that identify the start of a new unit/module/chapter
UNIT_HEADER_PATTERNS = [
    # "Unit 1", "UNIT 1", "UNIT I", "unit-1"
    re.compile(r"^\s*UNIT\s*[-–—]?\s*([IVXLC\d]+)\s*[-–—:.]?\s*(.+)$", re.IGNORECASE),
    # "Module 2 — ..."
    re.compile(r"^\s*MODULE\s*[-–—]?\s*([IVXLC\d]+)\s*[-–—:.]?\s*(.+)$", re.IGNORECASE),
    # "Chapter 3 — ..."
    re.compile(r"^\s*CHAPTER\s*[-–—]?\s*([IVXLC\d]+)\s*[-–—:.]?\s*(.+)$", re.IGNORECASE),
    # "1. Introduction to Networks"  (numbered with period, at line start)
    re.compile(r"^\s*(\d{1,2})\.\s{1,4}([A-Z].{5,})$"),
    # "SECTION A — ..."
    re.compile(r"^\s*SECTION\s+([A-Z\d]+)\s*[-–—:.]?\s*(.+)$", re.IGNORECASE),
]

# Lines that are clearly NOT topic content
NOISE_PATTERNS_SYLLABUS = [
    re.compile(r"^\s*page\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),                       # standalone numbers
    re.compile(r"^\s*[-─═]{3,}\s*$"),                 # horizontal rules
    re.compile(r"credit\s*hours?", re.IGNORECASE),
    re.compile(r"contact\s*hours?", re.IGNORECASE),
    re.compile(r"^\s*(total|marks?|hrs?|hours?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*sl\.?\s*no\.?", re.IGNORECASE),
    re.compile(r"^\s*reference\s*books?", re.IGNORECASE),
    re.compile(r"^\s*text\s*books?", re.IGNORECASE),
    re.compile(r"^\s*prescribed", re.IGNORECASE),
]


def extract_syllabus_units(full_text: str) -> List[SyllabusUnit]:
    """
    Parse raw syllabus text into a list of SyllabusUnit objects.

    HOW IT WORKS:
    We walk through the text line by line. When a line matches a
    unit header pattern, we start a new unit and collect subsequent
    lines as its topics — until the next unit header.

    This is the same state-machine pattern used in the question
    extractor (Phase 2): maintain current state, flush when a new
    unit starts.

    Parameters
    ----------
    full_text : str — raw text extracted from the syllabus PDF

    Returns
    -------
    List[SyllabusUnit] — may be empty if no unit patterns matched
    """
    lines   = full_text.split("\n")
    units   = []

    current_number = None
    current_name   = None
    current_topics = []
    current_raw    = []

    def flush():
        """Save the accumulated unit."""
        nonlocal current_number, current_name, current_topics, current_raw
        if current_number and current_name:
            unit = SyllabusUnit(
                unit_number=current_number,
                unit_name=current_name.strip(),
                topics=[t for t in current_topics if t.strip()],
                raw_text="\n".join(current_raw),
            )
            units.append(unit)
        current_number = None
        current_name   = None
        current_topics = []
        current_raw    = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip noise lines
        if any(p.search(stripped) for p in NOISE_PATTERNS_SYLLABUS):
            continue

        # Check whether this line is a unit header
        matched_header = False
        for pattern in UNIT_HEADER_PATTERNS:
            m = pattern.match(stripped)
            if m:
                flush()                          # save previous unit
                current_number = m.group(1)
                current_name   = m.group(2) if len(m.groups()) >= 2 else stripped
                current_raw    = [stripped]
                matched_header = True
                break

        if not matched_header and current_number is not None:
            # This is a topic line within the current unit
            # Filter out very short lines (likely headers/labels, not topics)
            if len(stripped) >= 5:
                current_topics.append(stripped)
                current_raw.append(stripped)

    flush()  # save the last unit

    return units


def units_from_manual_input(unit_text: str) -> List[SyllabusUnit]:
    """
    Parse manually entered unit/topic text from the UI textarea.

    Expected format (student types this):
      Unit 1: Introduction
        - OSI Model
        - TCP/IP Suite
      Unit 2: Data Link Layer
        - Framing
        - Error Detection

    This is a fallback when the PDF parser doesn't detect units.
    Each line starting with "Unit" or a number creates a new unit.
    Indented lines (or lines starting with - or *) become topics.
    """
    lines = unit_text.strip().split("\n")
    units = []
    current_number = None
    current_name   = None
    current_topics = []

    def flush():
        nonlocal current_number, current_name, current_topics
        if current_number:
            units.append(SyllabusUnit(
                unit_number=current_number,
                unit_name=(current_name or "").strip(),
                topics=[t.lstrip("-•* ").strip() for t in current_topics if t.strip()],
            ))
        current_number = None
        current_name   = None
        current_topics = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Look for "Unit N:" or "N." header
        m = re.match(r"^(?:unit\s*)?(\d+|[IVXLC]+)[:\.\-–]\s*(.*)$", stripped, re.IGNORECASE)
        if m:
            flush()
            current_number = m.group(1)
            current_name   = m.group(2) or f"Unit {m.group(1)}"
        elif current_number:
            current_topics.append(stripped)

    flush()
    return units


# ══════════════════════════════════════════════════════════════════════════════
# UNIT EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def embed_syllabus_units(
    units: List[SyllabusUnit],
    model_name: str = None,
) -> np.ndarray:
    """
    Generate one embedding vector per syllabus unit.

    We embed the unit's full text (name + all topics concatenated).
    This gives a semantic representation of what topics the unit covers.

    Parameters
    ----------
    units      : List[SyllabusUnit]
    model_name : str  — sentence-transformer model (or None for default)

    Returns
    -------
    np.ndarray of shape (n_units, embedding_dim)
    """
    from src.embeddings import embed_questions, DEFAULT_MODEL

    if not units:
        return np.array([])

    unit_texts = [u.all_text for u in units]
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    else:
        kwargs["model_name"] = DEFAULT_MODEL

    embeddings, _ = embed_questions(unit_texts, **kwargs)
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER → UNIT MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def map_clusters_to_units(
    scored_clusters:   list,       # List[ScoredCluster]
    questions:         list,       # List[dict] — original question bank
    units:             List[SyllabusUnit],
    unit_embeddings:   np.ndarray,
    cluster_embeddings: np.ndarray,
) -> List[UnitMapping]:
    """
    Assign each question cluster to its best-matching syllabus unit.

    HOW IT WORKS:
    1. For each cluster, compute the centroid of its member question embeddings.
    2. Compute cosine similarity between this centroid and every unit embedding.
    3. Assign the cluster to the unit with the highest similarity.

    WHY CENTROID?
    The centroid (mean of member embeddings) represents the "centre of gravity"
    of the cluster's questions. It's a better representative of the whole
    cluster than any single question.

    NOTE ON DIMENSION MISMATCH:
    When sentence-transformers is NOT installed, the TF-IDF fallback builds
    separate vocabularies for unit texts vs question texts, producing vectors
    with different widths. We handle this by re-embedding everything together
    in one shared TF-IDF space when the dimensions don't match.

    Parameters
    ----------
    scored_clusters    : List[ScoredCluster]  — from score_all_clusters()
    questions          : List[dict]           — full question bank
    units              : List[SyllabusUnit]
    unit_embeddings    : np.ndarray  (n_units, dim)
    cluster_embeddings : np.ndarray  (n_questions, dim)

    Returns
    -------
    List[UnitMapping] — one per cluster, in the same order as scored_clusters
    """
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    if not units or unit_embeddings.size == 0 or cluster_embeddings.size == 0:
        return []

    # ── Dimension mismatch guard ───────────────────────────────────────────────
    # Happens when TF-IDF fallback builds separate vocabularies for units and
    # questions. Fix: re-embed ALL texts together in one shared TF-IDF space.
    if unit_embeddings.shape[1] != cluster_embeddings.shape[1]:
        unit_embeddings, cluster_embeddings = _reembed_in_shared_space(
            units, questions
        )
        if unit_embeddings.size == 0 or cluster_embeddings.size == 0:
            return []

    mappings = []

    for sc in scored_clusters:
        # Compute centroid of this cluster's question embeddings
        member_embs = cluster_embeddings[sc.member_indices]
        centroid    = member_embs.mean(axis=0, keepdims=True)   # shape (1, dim)

        # Cosine similarity to each unit — sk_cosine handles normalization
        # shape: (n_units, 1) → flatten to list
        unit_scores = sk_cosine(unit_embeddings, centroid).flatten().tolist()

        best_idx   = int(np.argmax(unit_scores))
        best_score = float(unit_scores[best_idx])

        mappings.append(UnitMapping(
            cluster_id=sc.cluster_id,
            cluster_label=sc.topic_label,
            assigned_unit_idx=best_idx,
            assigned_unit_name=units[best_idx].full_label,
            similarity_score=round(best_score, 4),
            all_unit_scores=[round(s, 4) for s in unit_scores],
        ))

    return mappings


def _reembed_in_shared_space(
    units: List[SyllabusUnit],
    questions: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-embed units and questions in a single shared TF-IDF vocabulary.

    WHY THIS EXISTS:
    The TF-IDF fallback in embed_questions() fits a new vectorizer each
    time it is called, producing different vocabulary sizes. When unit
    texts and question texts are embedded separately the resulting arrays
    have mismatched column counts.

    Solution: fit ONE vectorizer on the union of all texts, then transform
    units and questions separately using that shared vocabulary.

    Returns (unit_embeddings, question_embeddings) — same column count.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.text_cleaner import normalize_question

    unit_texts  = [u.all_text for u in units]
    q_texts     = [q.get("question_text", "") for q in questions]
    all_texts   = unit_texts + q_texts
    normalized  = [normalize_question(t) or "unknown" for t in all_texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    try:
        matrix     = vectorizer.fit_transform(normalized).toarray().astype(np.float32)
    except Exception:
        return np.array([]), np.array([])

    unit_embs = matrix[:len(unit_texts)]
    q_embs    = matrix[len(unit_texts):]
    return unit_embs, q_embs


# ══════════════════════════════════════════════════════════════════════════════
# UNIT-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_unit_analyses(
    units:           List[SyllabusUnit],
    scored_clusters: list,           # List[ScoredCluster]
    mappings:        List[UnitMapping],
) -> List[UnitAnalysis]:
    """
    Aggregate cluster-level results into per-unit summaries.

    For each unit, collect:
      - All clusters mapped to it
      - Total question count
      - Average and max priority scores
      - Years covered

    Parameters
    ----------
    units           : List[SyllabusUnit]
    scored_clusters : List[ScoredCluster]
    mappings        : List[UnitMapping]   — output of map_clusters_to_units()

    Returns
    -------
    List[UnitAnalysis], sorted by avg_priority_score descending
    """
    # Build an index: cluster_id → ScoredCluster
    cluster_by_id = {sc.cluster_id: sc for sc in scored_clusters}

    # Group mappings by unit index
    unit_to_cluster_ids: Dict[int, List[int]] = {}
    for m in mappings:
        unit_to_cluster_ids.setdefault(m.assigned_unit_idx, []).append(m.cluster_id)

    analyses = []
    for unit_idx, unit in enumerate(units):
        cluster_ids = unit_to_cluster_ids.get(unit_idx, [])
        mapped_clusters = [cluster_by_id[cid] for cid in cluster_ids if cid in cluster_by_id]

        total_q     = sum(sc.total_appearances for sc in mapped_clusters)
        all_years   = sorted({y for sc in mapped_clusters for y in sc.years})
        avg_score   = (sum(sc.priority_score for sc in mapped_clusters) / len(mapped_clusters)
                       if mapped_clusters else 0.0)
        max_score   = max((sc.priority_score for sc in mapped_clusters), default=0.0)

        analyses.append(UnitAnalysis(
            unit=unit,
            mapped_clusters=mapped_clusters,
            total_questions=total_q,
            total_appearances=total_q,
            avg_priority_score=round(avg_score, 1),
            max_priority_score=round(max_score, 1),
            years_covered=all_years,
        ))

    # Sort by average priority score descending
    analyses.sort(key=lambda a: a.avg_priority_score, reverse=True)
    return analyses


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_syllabus_analysis(
    syllabus_text:     str,
    scored_clusters:   list,       # List[ScoredCluster] from Phase 6
    questions:         list,       # List[dict]
    cluster_embeddings: np.ndarray,
    manual_units_text: str = "",
) -> dict:
    """
    Full syllabus analysis pipeline — single entry point.

    Steps:
      1. Parse syllabus text → SyllabusUnit list
      2. If parsing fails, try manual input
      3. Embed all units
      4. Map clusters → units
      5. Aggregate into UnitAnalysis objects

    Parameters
    ----------
    syllabus_text      : str          — raw text from syllabus PDF
    scored_clusters    : list         — from score_all_clusters()
    questions          : list         — original question dicts
    cluster_embeddings : np.ndarray   — (n_questions, dim) from get_or_compute_embeddings()
    manual_units_text  : str          — fallback if PDF parsing finds no units

    Returns
    -------
    dict with keys:
      "units"           : List[SyllabusUnit]
      "unit_analyses"   : List[UnitAnalysis]
      "mappings"        : List[UnitMapping]
      "unit_embeddings" : np.ndarray
      "total_units"     : int
      "unmapped_clusters": int
      "parse_method"    : str  — "pdf" or "manual"
    """
    # Step 1: parse
    units = extract_syllabus_units(syllabus_text)
    parse_method = "pdf"

    # Step 2: fallback to manual input
    if not units and manual_units_text.strip():
        units = units_from_manual_input(manual_units_text)
        parse_method = "manual"

    if not units:
        return {
            "units": [], "unit_analyses": [], "mappings": [],
            "unit_embeddings": np.array([]),
            "total_units": 0, "unmapped_clusters": len(scored_clusters),
            "parse_method": "none",
        }

    # Step 3: embed units
    unit_embeddings = embed_syllabus_units(units)

    # Step 4: map clusters → units
    mappings = map_clusters_to_units(
        scored_clusters, questions, units, unit_embeddings, cluster_embeddings
    )

    # Step 5: aggregate
    unit_analyses = build_unit_analyses(units, scored_clusters, mappings)

    unmapped = sum(1 for sc in scored_clusters
                   if sc.cluster_id not in {m.cluster_id for m in mappings})

    return {
        "units":            units,
        "unit_analyses":    unit_analyses,
        "mappings":         mappings,
        "unit_embeddings":  unit_embeddings,
        "total_units":      len(units),
        "unmapped_clusters": unmapped,
        "parse_method":     parse_method,
    }
