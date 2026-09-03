"""
trend_analyzer.py — Trend classification for QPredict question clusters

WHY THIS FILE EXISTS:
A raw priority score (e.g. 78/100) tells a student "this is important"
but not *why* or *in what pattern* the topic appeared.

A student benefits from knowing:
  "This topic appeared in every paper — consistently asked."
  "This topic just appeared for the first time recently."
  "This topic is declining in frequency."

This module classifies each cluster's year pattern into a human-readable
trend label, and builds the year-timeline visualization data.

TREND LABELS:
  "Frequently Recurring"   — appears in most years, high frequency
  "Consistently Asked"     — appears in consecutive years without gaps
  "Recently Recurring"     — appeared in recent years only
  "Increasing"             — frequency increasing over time
  "Decreasing"             — frequency decreasing over time
  "Sporadic"               — appeared but with significant gaps
  "Rarely Asked"           — appeared very infrequently

DISCLAIMER:
  These labels are descriptive of historical patterns only.
  They do not predict future exam behaviour.
"""

from typing import List, Optional, Dict


# ══════════════════════════════════════════════════════════════════════════════
# TREND CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_trend(
    cluster_years: List[int],
    all_years:     List[int],
) -> str:
    """
    Classify a cluster's historical appearance pattern into a trend label.

    Decision logic (evaluated in priority order):
      1. If appeared in ≥ 80% of all years            → "Frequently Recurring"
      2. If all appearances are consecutive (no gap)   → "Consistently Asked"
      3. If last N years contain all appearances       → "Recently Recurring"
      4. If frequency per year is increasing           → "Increasing"
      5. If frequency per year is decreasing           → "Decreasing"
      6. If appeared but has year gaps                 → "Sporadic"
      7. Otherwise                                     → "Rarely Asked"

    Parameters
    ----------
    cluster_years : List[int] — years this topic appeared (may have duplicates)
    all_years     : List[int] — all years for which we have papers

    Returns
    -------
    str — one of the trend labels above
    """
    if not cluster_years or not all_years:
        return "Rarely Asked"

    unique_cluster = sorted(set(cluster_years))
    unique_all     = sorted(set(all_years))
    n_all          = len(unique_all)

    if n_all == 0:
        return "Rarely Asked"

    coverage_ratio = len(unique_cluster) / n_all

    # ── 1. Frequently Recurring ───────────────────────────────────────────────
    if coverage_ratio >= 0.75:
        return "Frequently Recurring"

    # ── 2. Consistently Asked (consecutive, no gaps) ──────────────────────────
    if len(unique_cluster) >= 2:
        is_consecutive = all(
            unique_cluster[i + 1] - unique_cluster[i] == 1
            for i in range(len(unique_cluster) - 1)
        )
        if is_consecutive and coverage_ratio >= 0.40:
            return "Consistently Asked"

    # ── 3. Recently Recurring ─────────────────────────────────────────────────
    # "Recent" means the last 2 years of available papers
    if len(unique_all) >= 2:
        recent_cutoff = unique_all[-2]   # second-to-last year
        recent_appearances = [y for y in unique_cluster if y >= recent_cutoff]
        if len(recent_appearances) >= 2:
            return "Recently Recurring"
        if len(recent_appearances) >= 1 and unique_cluster[-1] == unique_all[-1]:
            return "Recently Recurring"

    # ── 4 & 5. Increasing / Decreasing ───────────────────────────────────────
    if len(unique_cluster) >= 3 and len(unique_all) >= 3:
        trend_direction = _detect_frequency_trend(unique_cluster, unique_all)
        if trend_direction == "increasing":
            return "Increasing"
        if trend_direction == "decreasing":
            return "Decreasing"

    # ── 6. Sporadic ───────────────────────────────────────────────────────────
    if len(unique_cluster) >= 2:
        max_gap = max(
            unique_cluster[i + 1] - unique_cluster[i]
            for i in range(len(unique_cluster) - 1)
        )
        if max_gap >= 2:
            return "Sporadic"

    # ── 7. Rarely Asked ───────────────────────────────────────────────────────
    return "Rarely Asked"


def _detect_frequency_trend(
    cluster_years: List[int],
    all_years:     List[int],
) -> str:
    """
    Detect whether topic frequency is increasing or decreasing over time.

    We split the available years into two halves and compare how many
    topic appearances fall in each half.

    Returns "increasing", "decreasing", or "stable".
    """
    unique_all = sorted(set(all_years))
    mid_idx = len(unique_all) // 2
    first_half = set(unique_all[:mid_idx])
    second_half = set(unique_all[mid_idx:])

    first_count  = sum(1 for y in cluster_years if y in first_half)
    second_count = sum(1 for y in cluster_years if y in second_half)

    if second_count > first_count:
        return "increasing"
    if second_count < first_count:
        return "decreasing"
    return "stable"


# ══════════════════════════════════════════════════════════════════════════════
# YEAR TIMELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_year_timeline(
    cluster_years: List[int],
    all_years:     List[int],
) -> List[dict]:
    """
    Build a year-by-year presence/absence timeline for a cluster.

    Returns a list of dicts, one per year, showing whether the topic
    appeared that year. Used by the dashboard to render:
        2021 ✓    2022 ✓    2023 ✗    2024 ✓    2025 ✓

    Parameters
    ----------
    cluster_years : List[int] — years this cluster appeared
    all_years     : List[int] — all years for which we have papers

    Returns
    -------
    List[dict] with keys: year, appeared (bool), count (int)
    """
    year_counts: Dict[int, int] = {}
    for y in cluster_years:
        year_counts[y] = year_counts.get(y, 0) + 1

    timeline = []
    for year in sorted(set(all_years)):
        count = year_counts.get(year, 0)
        timeline.append({
            "year":    year,
            "appeared": count > 0,
            "count":   count,
            "symbol":  "✓" if count > 0 else "✗",
        })
    return timeline


def timeline_as_string(timeline: List[dict]) -> str:
    """
    Convert a timeline list into a compact display string.

    Example output: "2021 ✓  2022 ✓  2023 ✗  2024 ✓  2025 ✓"
    """
    parts = [f"{t['year']} {t['symbol']}" for t in timeline]
    return "    ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SUBJECT-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_subject_statistics(
    scored_clusters: list,   # List[ScoredCluster]
    questions:       list,   # List[dict]
) -> dict:
    """
    Compute aggregate statistics for the entire subject.

    Used by the dashboard Overview section.

    Returns a dict with:
      total_clusters        : int
      total_questions       : int
      total_papers          : int
      years_covered         : List[int]
      high_priority_topics  : int   — score ≥ 70
      medium_priority_topics: int   — score 40–69
      low_priority_topics   : int   — score < 40
      top_cluster           : ScoredCluster  — highest scoring
      trend_counts          : dict  — count per trend label
    """
    from collections import Counter

    all_years = sorted({q.get("year") for q in questions if q.get("year")})
    paper_ids = {q.get("paper_id") for q in questions if q.get("paper_id")}
    trend_counts = Counter(sc.trend for sc in scored_clusters)

    high   = [sc for sc in scored_clusters if sc.priority_score >= 70]
    medium = [sc for sc in scored_clusters if 40 <= sc.priority_score < 70]
    low    = [sc for sc in scored_clusters if sc.priority_score < 40]

    return {
        "total_clusters":         len(scored_clusters),
        "total_questions":        len(questions),
        "total_papers":           len(paper_ids),
        "years_covered":          all_years,
        "high_priority_topics":   len(high),
        "medium_priority_topics": len(medium),
        "low_priority_topics":    len(low),
        "top_cluster":            scored_clusters[0] if scored_clusters else None,
        "trend_counts":           dict(trend_counts),
    }
