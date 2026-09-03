"""
test_scoring.py — Tests for scoring.py and trend_analyzer.py

These tests verify:
  1. Each score component produces values in [0, 100]
  2. Component formulas are correct for known inputs
  3. compute_priority_score() combines components correctly
  4. score_all_clusters() returns sorted ScoredCluster objects
  5. classify_trend() returns expected labels
  6. build_year_timeline() returns correct presence/absence data
  7. Edge cases: no years, single year, all years, missing data

HOW TO RUN:
    cd qpredict
    python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scoring import (
    compute_frequency_score,
    compute_year_coverage_score,
    compute_recency_score,
    compute_consistency_score,
    compute_priority_score,
    score_all_clusters,
    ScoreBreakdown,
    ScoredCluster,
    DEFAULT_WEIGHTS,
)
from src.trend_analyzer import (
    classify_trend,
    build_year_timeline,
    timeline_as_string,
    compute_subject_statistics,
)


# ══════════════════════════════════════════════════════════════════════════════
# compute_frequency_score()
# ══════════════════════════════════════════════════════════════════════════════

def test_frequency_score_max():
    """The most frequent topic should score 100."""
    assert compute_frequency_score(10, 10) == 100.0

def test_frequency_score_half():
    """Half the max appearances → score ~50."""
    assert abs(compute_frequency_score(5, 10) - 50.0) < 0.1

def test_frequency_score_zero_appearances():
    assert compute_frequency_score(0, 10) == 0.0

def test_frequency_score_zero_max():
    """Zero max_appearances — should not raise, return 0."""
    assert compute_frequency_score(5, 0) == 0.0

def test_frequency_score_in_range():
    for appearances in range(1, 12):
        score = compute_frequency_score(appearances, 10)
        assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_year_coverage_score()
# ══════════════════════════════════════════════════════════════════════════════

def test_year_coverage_all_years():
    """Topic appears in every year → 100."""
    all_years = [2021, 2022, 2023, 2024, 2025]
    assert compute_year_coverage_score(all_years, all_years) == 100.0

def test_year_coverage_half_years():
    """Topic appears in half the years → ~60% (3 of 5)."""
    score = compute_year_coverage_score([2021, 2022, 2023], [2021, 2022, 2023, 2024, 2025])
    assert abs(score - 60.0) < 0.1

def test_year_coverage_one_year():
    score = compute_year_coverage_score([2021], [2021, 2022, 2023, 2024, 2025])
    assert abs(score - 20.0) < 0.1

def test_year_coverage_empty_all_years():
    assert compute_year_coverage_score([2021], []) == 0.0

def test_year_coverage_empty_cluster():
    assert compute_year_coverage_score([], [2021, 2022]) == 0.0

def test_year_coverage_in_range():
    score = compute_year_coverage_score([2021, 2023], [2021, 2022, 2023, 2024])
    assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_recency_score()
# ══════════════════════════════════════════════════════════════════════════════

def test_recency_appeared_this_year():
    """Topic appeared in the most recent year → score ≈ 100."""
    score = compute_recency_score([2025], [2021, 2022, 2023, 2024, 2025])
    assert score >= 95.0

def test_recency_appeared_one_year_ago():
    """One year ago → exp(-0.5*1) ≈ 60.6."""
    score = compute_recency_score([2024], [2021, 2022, 2023, 2024, 2025], current_year=2025)
    assert 55.0 <= score <= 65.0

def test_recency_appeared_five_years_ago():
    """Five years ago → exp(-0.5*5) ≈ 8.2."""
    score = compute_recency_score([2020], [2020, 2021, 2022, 2023, 2024, 2025], current_year=2025)
    assert score < 15.0

def test_recency_empty_cluster():
    assert compute_recency_score([], [2021, 2022]) == 0.0

def test_recency_in_range():
    score = compute_recency_score([2022], [2021, 2022, 2023, 2024, 2025])
    assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_consistency_score()
# ══════════════════════════════════════════════════════════════════════════════

def test_consistency_all_consecutive():
    """2021,2022,2023,2024 → perfectly consecutive → 100."""
    score = compute_consistency_score([2021, 2022, 2023, 2024], [2021, 2022, 2023, 2024])
    assert score == 100.0

def test_consistency_one_gap():
    """2021,2022,2024 → span 3, only (21,22) consecutive → 1/3 ≈ 33.3."""
    score = compute_consistency_score([2021, 2022, 2024], [2021, 2022, 2023, 2024])
    assert abs(score - (1/3 * 100)) < 1.0

def test_consistency_single_year():
    """Only one year → no gaps possible → 100."""
    score = compute_consistency_score([2022], [2021, 2022, 2023])
    assert score == 100.0

def test_consistency_all_gaps():
    """2021 and 2025 → only gap years → 0."""
    score = compute_consistency_score([2021, 2025], [2021, 2022, 2023, 2024, 2025])
    assert score == 0.0

def test_consistency_empty():
    assert compute_consistency_score([], [2021, 2022]) == 0.0

def test_consistency_in_range():
    score = compute_consistency_score([2021, 2023, 2025], [2021, 2022, 2023, 2024, 2025])
    assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_priority_score()
# ══════════════════════════════════════════════════════════════════════════════

def test_priority_score_range():
    breakdown = compute_priority_score(
        cluster_years=[2021, 2022, 2024, 2025],
        total_appearances=6,
        all_years=[2021, 2022, 2023, 2024, 2025],
        max_appearances=6,
    )
    assert 0.0 <= breakdown.priority_score <= 100.0

def test_priority_score_returns_breakdown():
    bd = compute_priority_score(
        cluster_years=[2022, 2023],
        total_appearances=3,
        all_years=[2021, 2022, 2023, 2024],
        max_appearances=5,
    )
    assert isinstance(bd, ScoreBreakdown)
    assert bd.frequency_score >= 0
    assert bd.year_coverage_score >= 0
    assert bd.recency_score >= 0
    assert bd.consistency_score >= 0

def test_priority_score_high_topic():
    """A topic that appeared in all years at max frequency should score near 100."""
    bd = compute_priority_score(
        cluster_years=[2021, 2022, 2023, 2024, 2025],
        total_appearances=10,
        all_years=[2021, 2022, 2023, 2024, 2025],
        max_appearances=10,
        current_year=2025,
    )
    assert bd.priority_score >= 80.0

def test_priority_score_low_topic():
    """A topic that appeared once, long ago, should score low."""
    bd = compute_priority_score(
        cluster_years=[2019],
        total_appearances=1,
        all_years=[2019, 2020, 2021, 2022, 2023, 2024, 2025],
        max_appearances=10,
        current_year=2025,
    )
    assert bd.priority_score < 40.0

def test_priority_custom_weights():
    """Custom weights should be reflected in weights_used."""
    w = {"frequency": 1.0, "year_coverage": 0.0, "recency": 0.0, "consistency": 0.0}
    bd = compute_priority_score(
        cluster_years=[2024], total_appearances=5,
        all_years=[2021, 2022, 2023, 2024], max_appearances=5,
        weights=w,
    )
    assert bd.weights_used == w
    # Score should equal frequency score (all weight on frequency)
    assert abs(bd.priority_score - bd.frequency_score) < 1.0

def test_score_breakdown_explanation():
    bd = ScoreBreakdown(
        frequency_score=80, year_coverage_score=75,
        recency_score=90, consistency_score=70,
        priority_score=80,
    )
    lines = bd.explanation_lines()
    assert isinstance(lines, list)
    assert len(lines) == 4   # one line per component


# ══════════════════════════════════════════════════════════════════════════════
# classify_trend()
# ══════════════════════════════════════════════════════════════════════════════

ALL_5_YEARS = [2021, 2022, 2023, 2024, 2025]

def test_trend_frequently_recurring():
    # Appeared in 4 of 5 years (80%)
    t = classify_trend([2021, 2022, 2023, 2025], ALL_5_YEARS)
    assert t == "Frequently Recurring"

def test_trend_consistently_asked():
    # Appeared in 3 consecutive years 2022–2024 (60%)
    t = classify_trend([2022, 2023, 2024], ALL_5_YEARS)
    assert t == "Consistently Asked"

def test_trend_sporadic():
    # Appeared in 2 years with gap
    t = classify_trend([2021, 2024], ALL_5_YEARS)
    assert t in ("Sporadic", "Rarely Asked")

def test_trend_rarely_asked():
    t = classify_trend([2021], ALL_5_YEARS)
    assert t == "Rarely Asked"

def test_trend_empty_cluster():
    t = classify_trend([], ALL_5_YEARS)
    assert t == "Rarely Asked"

def test_trend_empty_all_years():
    t = classify_trend([2022], [])
    assert t == "Rarely Asked"

def test_trend_returns_string():
    t = classify_trend([2023, 2024], ALL_5_YEARS)
    assert isinstance(t, str)
    assert len(t) > 0


# ══════════════════════════════════════════════════════════════════════════════
# build_year_timeline()
# ══════════════════════════════════════════════════════════════════════════════

def test_timeline_length():
    tl = build_year_timeline([2021, 2023], [2021, 2022, 2023, 2024])
    assert len(tl) == 4   # one entry per year in all_years

def test_timeline_appeared_flag():
    tl = build_year_timeline([2021, 2023], [2021, 2022, 2023, 2024])
    by_year = {t["year"]: t for t in tl}
    assert by_year[2021]["appeared"] is True
    assert by_year[2022]["appeared"] is False
    assert by_year[2023]["appeared"] is True
    assert by_year[2024]["appeared"] is False

def test_timeline_symbols():
    tl = build_year_timeline([2022], [2021, 2022, 2023])
    by_year = {t["year"]: t for t in tl}
    assert by_year[2022]["symbol"] == "✓"
    assert by_year[2021]["symbol"] == "✗"

def test_timeline_count():
    """Count should reflect how many times a topic appeared in a year."""
    tl = build_year_timeline([2022, 2022, 2022], [2021, 2022, 2023])
    by_year = {t["year"]: t for t in tl}
    assert by_year[2022]["count"] == 3

def test_timeline_as_string_format():
    tl = build_year_timeline([2021, 2023], [2021, 2022, 2023])
    s = timeline_as_string(tl)
    assert "2021" in s
    assert "✓" in s
    assert "✗" in s


# ══════════════════════════════════════════════════════════════════════════════
# score_all_clusters() — integration
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_clusters_and_questions():
    """Build minimal mock objects for integration tests."""
    from src.clustering import QuestionCluster

    questions = [
        {"id": 1, "question_text": "Explain OSI model layers in detail.",       "year": 2021, "paper_id": 1, "marks": 10},
        {"id": 2, "question_text": "Describe the seven OSI architecture layers.", "year": 2022, "paper_id": 2, "marks": 10},
        {"id": 3, "question_text": "What are the OSI model seven layers?",        "year": 2023, "paper_id": 3, "marks": 10},
        {"id": 4, "question_text": "Explain Dijkstra routing algorithm.",         "year": 2021, "paper_id": 1, "marks": 10},
        {"id": 5, "question_text": "Describe Bellman-Ford routing algorithms.",   "year": 2022, "paper_id": 2, "marks": 10},
    ]

    clusters = [
        QuestionCluster(
            cluster_id=0, topic_label="OSI Model",
            member_indices=[0, 1, 2], representative_idx=0,
            years=[2021, 2022, 2023], paper_count=3,
            total_appearances=3, keywords=["osi", "layers"],
        ),
        QuestionCluster(
            cluster_id=1, topic_label="Routing",
            member_indices=[3, 4], representative_idx=3,
            years=[2021, 2022], paper_count=2,
            total_appearances=2, keywords=["routing", "dijkstra"],
        ),
    ]
    return clusters, questions

def test_score_all_clusters_returns_list():
    clusters, questions = _make_mock_clusters_and_questions()
    scored = score_all_clusters(clusters, questions)
    assert isinstance(scored, list)
    assert len(scored) == len(clusters)

def test_score_all_clusters_type():
    clusters, questions = _make_mock_clusters_and_questions()
    scored = score_all_clusters(clusters, questions)
    for sc in scored:
        assert isinstance(sc, ScoredCluster)

def test_score_all_clusters_sorted():
    """Scores must be in descending order."""
    clusters, questions = _make_mock_clusters_and_questions()
    scored = score_all_clusters(clusters, questions)
    for i in range(len(scored) - 1):
        assert scored[i].priority_score >= scored[i + 1].priority_score

def test_score_all_clusters_scores_in_range():
    clusters, questions = _make_mock_clusters_and_questions()
    scored = score_all_clusters(clusters, questions)
    for sc in scored:
        assert 0.0 <= sc.priority_score <= 100.0

def test_score_all_clusters_has_trend():
    clusters, questions = _make_mock_clusters_and_questions()
    scored = score_all_clusters(clusters, questions)
    for sc in scored:
        assert isinstance(sc.trend, str)
        assert len(sc.trend) > 0

def test_score_all_clusters_empty():
    assert score_all_clusters([], []) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
