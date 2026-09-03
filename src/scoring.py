"""
scoring.py — Historical Priority Score for QPredict

WHY THIS FILE EXISTS:
After clustering (Phase 5), we have groups of related questions.
This module answers: "Which clusters should a student prioritize?"

The answer comes from four historical signals combined into one score.

IMPORTANT DISCLAIMER (mirrored in the UI):
  The Historical Priority Score reflects patterns in uploaded exam papers.
  It is NOT a prediction or probability of future exam questions.
  Use it as a guide based on historical evidence, not as a guarantee.

SCORE DESIGN PRINCIPLES:
  1. Transparent — every component is explainable separately.
  2. Normalized — final score is always 0–100.
  3. Configurable — weights are not hardcoded; they can be tuned.
  4. Honest — no component claims predictive power it doesn't have.

WEIGHTS (default):
  Frequency     40%  — how often has this topic appeared?
  Year Coverage 25%  — how many years has it appeared across?
  Recency       20%  — did it appear in recent papers?
  Consistency   15%  — did it appear in consecutive years?

These weights are a reasonable starting point. With real data and
a labelled validation set, you could optimize them (Phase 10+).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "frequency":    0.40,
    "year_coverage": 0.25,
    "recency":       0.20,
    "consistency":   0.15,
}

# The year considered "current" for recency calculations.
# Defaults to the most recent year found in the data, but can be overridden.
CURRENT_YEAR_FALLBACK = 2025


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreBreakdown:
    """
    The individual component scores that make up the Priority Score.

    WHY: Transparency. A student should be able to see exactly why a topic
    has a high or low score — not just the final number.

    Each component is 0–100 before weighting.
    """
    frequency_score:     float = 0.0
    year_coverage_score: float = 0.0
    recency_score:       float = 0.0
    consistency_score:   float = 0.0
    priority_score:      float = 0.0   # weighted combination, 0–100
    weights_used:        dict  = field(default_factory=dict)

    def explanation_lines(self) -> List[str]:
        """
        Return human-readable lines explaining the score.
        Used by the UI to show a "Why this score?" section.
        """
        lines = []
        if self.frequency_score >= 70:
            lines.append("✓ This topic appears frequently across papers.")
        elif self.frequency_score >= 40:
            lines.append("~ This topic appears occasionally across papers.")
        else:
            lines.append("✗ This topic rarely appears across papers.")

        if self.year_coverage_score >= 70:
            lines.append("✓ This topic has been tested in many different years.")
        elif self.year_coverage_score >= 40:
            lines.append("~ This topic has appeared in some years.")
        else:
            lines.append("✗ This topic has only appeared in very few years.")

        if self.recency_score >= 70:
            lines.append("✓ This topic appeared in recent papers.")
        elif self.recency_score >= 40:
            lines.append("~ This topic appeared some years ago.")
        else:
            lines.append("✗ This topic has not appeared recently.")

        if self.consistency_score >= 70:
            lines.append("✓ This topic appeared consistently without long gaps.")
        elif self.consistency_score >= 40:
            lines.append("~ This topic appeared somewhat consistently.")
        else:
            lines.append("✗ This topic appeared sporadically with gaps.")

        return lines


@dataclass
class ScoredCluster:
    """
    A QuestionCluster augmented with its Historical Priority Score.

    This is the final output object of Phase 6 — the thing the dashboard
    displays as a "topic card" with a priority score.
    """
    # Core cluster data
    cluster_id:          int
    topic_label:         str
    representative_text: str
    member_indices:      List[int]
    years:               List[int]
    paper_count:         int
    total_appearances:   int
    keywords:            List[str]

    # Scoring
    score:     ScoreBreakdown = field(default_factory=ScoreBreakdown)
    trend:     str = ""       # e.g. "Frequently Recurring", "Consistent"

    @property
    def priority_score(self) -> float:
        return self.score.priority_score

    @property
    def year_coverage(self) -> str:
        return ", ".join(str(y) for y in sorted(self.years))

    def to_dict(self) -> dict:
        return {
            "cluster_id":          self.cluster_id,
            "topic_label":         self.topic_label,
            "representative_text": self.representative_text,
            "total_appearances":   self.total_appearances,
            "paper_count":         self.paper_count,
            "years":               self.years,
            "priority_score":      round(self.priority_score, 1),
            "trend":               self.trend,
            "keywords":            self.keywords,
            "frequency_score":     round(self.score.frequency_score, 1),
            "year_coverage_score": round(self.score.year_coverage_score, 1),
            "recency_score":       round(self.score.recency_score, 1),
            "consistency_score":   round(self.score.consistency_score, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SCORE COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_frequency_score(
    total_appearances: int,
    max_appearances:   int,
) -> float:
    """
    Score how frequently a topic appears relative to the most frequent topic.

    WHY RELATIVE NORMALIZATION:
    If the most common topic appeared 8 times and another appeared 4 times,
    the second topic gets 50/100. This keeps scores meaningful regardless
    of the total number of papers uploaded.

    Formula: (appearances / max_appearances) × 100

    Parameters
    ----------
    total_appearances : int — how many questions belong to this cluster
    max_appearances   : int — the maximum across ALL clusters in this subject

    Returns
    -------
    float — score from 0 to 100
    """
    if max_appearances <= 0:
        return 0.0
    return round(min(total_appearances / max_appearances, 1.0) * 100, 2)


def compute_year_coverage_score(
    cluster_years:  List[int],
    all_years:      List[int],
) -> float:
    """
    Score based on how many distinct years this topic appeared in.

    Formula: (distinct years with topic / total distinct years available) × 100

    Example:
      Topic appeared in: 2021, 2022, 2024   → 3 distinct years
      Papers span:       2021, 2022, 2023, 2024, 2025  → 5 distinct years
      Score: (3/5) × 100 = 60

    Parameters
    ----------
    cluster_years : List[int] — years this cluster appeared in
    all_years     : List[int] — all years for which we have papers

    Returns
    -------
    float — score from 0 to 100
    """
    if not all_years:
        return 0.0
    distinct_cluster_years = len(set(cluster_years))
    distinct_total_years   = len(set(all_years))
    return round(min(distinct_cluster_years / distinct_total_years, 1.0) * 100, 2)


def compute_recency_score(
    cluster_years: List[int],
    all_years:     List[int],
    current_year:  Optional[int] = None,
) -> float:
    """
    Score based on how recently this topic appeared.

    WHY RECENCY MATTERS (with caveats):
    A topic that appeared last year is somewhat more noteworthy than one
    that last appeared 6 years ago. However, we weight this at only 20%
    because recent appearance does NOT imply it will repeat.

    FORMULA:
    We use an exponential decay based on years since last appearance:
        recency = exp(−decay × years_since_last)  × 100
    where decay = 0.5 gives:
        0 years ago → 100
        1 year ago  →  61
        2 years ago →  37
        3 years ago →  22
        5 years ago →   8

    Parameters
    ----------
    cluster_years : List[int] — years this cluster appeared in
    all_years     : List[int] — all years for which we have papers
    current_year  : int       — the "present" year (defaults to max of all_years)

    Returns
    -------
    float — score from 0 to 100
    """
    if not cluster_years or not all_years:
        return 0.0

    ref_year = current_year or max(all_years)
    last_year = max(cluster_years)
    years_since = max(0, ref_year - last_year)

    # Exponential decay: topic that appeared this year scores 100,
    # topic that appeared 5+ years ago scores ~8
    decay = 0.5
    score = math.exp(-decay * years_since) * 100
    return round(min(score, 100.0), 2)


def compute_consistency_score(
    cluster_years: List[int],
    all_years:     List[int],
) -> float:
    """
    Score based on how consistently (without gaps) this topic appeared.

    WHY CONSISTENCY MATTERS:
    A topic that appeared in 2021, 2022, 2023, 2024 is more "established"
    than one that appeared in 2019 and then again in 2024 — a 5-year gap
    suggests it may be sporadic rather than consistently examined.

    FORMULA:
    We measure the ratio of consecutive-year pairs to total possible pairs
    in the years this topic was active (from first to last appearance).

    consecutive_pairs = how many adjacent year pairs (y, y+1) both appear
    possible_pairs    = (last_year - first_year)   ← total gaps possible

    If possible_pairs = 0 (only one year or same year): score = 100
    Score = (consecutive_pairs / possible_pairs) × 100

    Example:
      Years: [2021, 2022, 2024]
      Span: 2021–2024 = 3 possible pairs: (21,22), (22,23), (23,24)
      Consecutive pairs present: (21,22) ✓  (22,23) ✗  (23,24) ✓  → 2
      Score = (2/3) × 100 = 67

    Parameters
    ----------
    cluster_years : List[int]
    all_years     : List[int]

    Returns
    -------
    float — score from 0 to 100
    """
    if not cluster_years:
        return 0.0

    unique_years = sorted(set(cluster_years))
    if len(unique_years) == 1:
        return 100.0   # appeared in only one year — no gaps possible

    first_year = unique_years[0]
    last_year  = unique_years[-1]
    span = last_year - first_year   # total number of possible year-to-year transitions

    if span == 0:
        return 100.0

    year_set = set(unique_years)
    consecutive = sum(
        1 for y in range(first_year, last_year)
        if y in year_set and (y + 1) in year_set
    )

    return round((consecutive / span) * 100, 2)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_priority_score(
    cluster_years:     List[int],
    total_appearances: int,
    all_years:         List[int],
    max_appearances:   int,
    weights:           Optional[dict] = None,
    current_year:      Optional[int]  = None,
) -> ScoreBreakdown:
    """
    Compute the full Historical Priority Score for one cluster.

    This calls each component scorer and combines them using weights.

    Parameters
    ----------
    cluster_years     : List[int] — years this topic appeared
    total_appearances : int       — total questions in this cluster
    all_years         : List[int] — all years in the paper bank
    max_appearances   : int       — most appearances of any single cluster
    weights           : dict      — component weights (defaults to DEFAULT_WEIGHTS)
    current_year      : int       — the reference year for recency

    Returns
    -------
    ScoreBreakdown — all components + final weighted score
    """
    w = weights or DEFAULT_WEIGHTS

    freq   = compute_frequency_score(total_appearances, max_appearances)
    cov    = compute_year_coverage_score(cluster_years, all_years)
    rec    = compute_recency_score(cluster_years, all_years, current_year)
    cons   = compute_consistency_score(cluster_years, all_years)

    # Weighted combination
    priority = (
        w.get("frequency",     0.40) * freq  +
        w.get("year_coverage", 0.25) * cov   +
        w.get("recency",       0.20) * rec   +
        w.get("consistency",   0.15) * cons
    )
    priority = round(min(priority, 100.0), 1)

    return ScoreBreakdown(
        frequency_score=freq,
        year_coverage_score=cov,
        recency_score=rec,
        consistency_score=cons,
        priority_score=priority,
        weights_used=w,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_all_clusters(
    clusters:     list,   # List[QuestionCluster] from clustering.py
    questions:    list,   # List[dict] from database
    weights:      Optional[dict] = None,
    current_year: Optional[int]  = None,
) -> List[ScoredCluster]:
    """
    Score every cluster and return a sorted list of ScoredCluster objects.

    This is the main entry point for Phase 6. It takes the output of
    Phase 5 (clusters) and enriches each one with a priority score,
    a score breakdown, and a trend classification.

    Parameters
    ----------
    clusters     : List[QuestionCluster]  — from run_clustering()
    questions    : List[dict]             — original question dicts
    weights      : dict                   — override default scoring weights
    current_year : int                    — override recency reference year

    Returns
    -------
    List[ScoredCluster], sorted by priority_score descending
    """
    from src.trend_analyzer import classify_trend  # avoid circular import

    if not clusters:
        return []

    # Collect all years across every paper in the bank
    all_years = sorted({
        q.get("year") for q in questions if q.get("year")
    })

    # Find the maximum appearances for relative frequency normalization
    max_appearances = max(c.total_appearances for c in clusters)

    ref_year = current_year or (max(all_years) if all_years else CURRENT_YEAR_FALLBACK)

    scored = []
    for cluster in clusters:
        breakdown = compute_priority_score(
            cluster_years=cluster.years,
            total_appearances=cluster.total_appearances,
            all_years=all_years,
            max_appearances=max_appearances,
            weights=weights,
            current_year=ref_year,
        )

        trend = classify_trend(cluster.years, all_years)

        rep_q = questions[cluster.representative_idx]

        sc = ScoredCluster(
            cluster_id=cluster.cluster_id,
            topic_label=cluster.topic_label,
            representative_text=rep_q.get("question_text", ""),
            member_indices=cluster.member_indices,
            years=cluster.years,
            paper_count=cluster.paper_count,
            total_appearances=cluster.total_appearances,
            keywords=cluster.keywords,
            score=breakdown,
            trend=trend,
        )
        scored.append(sc)

    # Sort by priority score descending
    scored.sort(key=lambda s: s.priority_score, reverse=True)
    return scored


def update_cluster_scores_in_db(scored_clusters: List[ScoredCluster]):
    """
    Write priority scores back to the clusters table in the database.

    WHY: The dashboard (Phase 7) can then load pre-scored clusters
    directly from the DB without rerunning the clustering pipeline.
    """
    import sqlite3
    import os

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "qpredict.db"
    )
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Check the table exists before writing
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'"
    )
    if not cursor.fetchone():
        conn.close()
        return

    for sc in scored_clusters:
        cursor.execute(
            "UPDATE clusters SET priority_score = ? WHERE id = ?",
            (sc.priority_score, sc.cluster_id + 1)  # DB IDs are 1-based
        )

    conn.commit()
    conn.close()
