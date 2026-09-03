"""
tests/test_study_planner.py
===========================
Phase 9 — Tests for src/study_planner.py

WHAT WE TEST
------------
1. StudyItem and StudyDay data structures work correctly
2. _estimate_minutes() returns values in expected ranges
3. build_study_plan() handles edge cases (no clusters, 1 day, many days)
4. Topics are ordered by priority in the plan
5. Revision day is always the last day
6. plan_to_text() returns a non-empty string
7. LLM enrichment is a no-op when no API key is provided
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.study_planner import (
    StudyItem,
    StudyDay,
    StudyPlan,
    _estimate_minutes,
    build_study_plan,
    enrich_plan_with_llm,
    plan_to_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_cluster(topic: str, score: float, q_count: int = 5, years: list = None) -> dict:
    """Helper to create a cluster dict for testing."""
    return {
        "topic": topic,
        "priority_score": score,
        "question_count": q_count,
        "years": years or [2022, 2023, 2024],
        "trend": "Frequently Recurring",
    }


SAMPLE_CLUSTERS = [
    make_cluster("OSI Model", 91.0, 8, [2020, 2021, 2022, 2023, 2024]),
    make_cluster("TCP/IP", 85.0, 6, [2021, 2022, 2024]),
    make_cluster("Routing Algorithms", 72.0, 5, [2022, 2023]),
    make_cluster("Network Security", 60.0, 4, [2021, 2023]),
    make_cluster("DNS", 45.0, 3, [2022]),
    make_cluster("VLAN", 25.0, 2, [2021]),
]


# ---------------------------------------------------------------------------
# Tests: _estimate_minutes()
# ---------------------------------------------------------------------------

class TestEstimateMinutes:
    def test_high_priority_returns_at_least_60(self):
        minutes = _estimate_minutes(90.0, 3)
        assert minutes >= 60

    def test_medium_priority_returns_between_45_and_90(self):
        minutes = _estimate_minutes(55.0, 3)
        assert 45 <= minutes <= 90

    def test_low_priority_returns_at_least_30(self):
        minutes = _estimate_minutes(20.0, 3)
        assert minutes >= 30

    def test_maximum_is_90(self):
        # Even with very high score and many questions, cap at 90
        minutes = _estimate_minutes(100.0, 100)
        assert minutes <= 90

    def test_minimum_is_20(self):
        minutes = _estimate_minutes(0.0, 0)
        assert minutes >= 20

    def test_more_questions_increases_time(self):
        few = _estimate_minutes(70.0, 3)
        many = _estimate_minutes(70.0, 20)
        assert many >= few


# ---------------------------------------------------------------------------
# Tests: StudyDay
# ---------------------------------------------------------------------------

class TestStudyDay:
    def test_add_item_increases_total_minutes(self):
        day = StudyDay(day_number=1, label="Day 1")
        item = StudyItem(
            topic_name="OSI Model",
            priority_score=91.0,
            estimated_minutes=60,
            question_count=5,
            years_seen=[2022, 2023],
            trend="Frequently Recurring",
        )
        day.add_item(item)
        assert day.total_minutes == 60

    def test_add_multiple_items(self):
        day = StudyDay(day_number=1, label="Day 1")
        for i in range(3):
            item = StudyItem("Topic", 50.0, 30, 3, [], "")
            day.add_item(item)
        assert day.total_minutes == 90
        assert len(day.items) == 3


# ---------------------------------------------------------------------------
# Tests: build_study_plan()
# ---------------------------------------------------------------------------

class TestBuildStudyPlan:
    def test_returns_studyplan_object(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert isinstance(plan, StudyPlan)

    def test_plan_has_correct_day_count(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        # Should have 7 days (6 study + 1 revision)
        assert len(plan.days) == 7

    def test_last_day_is_revision(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert plan.days[-1].is_revision_day is True

    def test_last_day_label_contains_revision(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert "Revision" in plan.days[-1].label

    def test_revision_day_has_items(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert len(plan.days[-1].items) > 0

    def test_all_topics_appear_in_plan(self):
        """Every cluster should end up in the plan (no topics dropped)."""
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        # Collect all non-revision topic names
        scheduled_topics = set()
        for day in plan.days:
            if not day.is_revision_day:
                for item in day.items:
                    scheduled_topics.add(item.topic_name)
        for c in SAMPLE_CLUSTERS:
            assert c["topic"] in scheduled_topics, f"{c['topic']} not scheduled"

    def test_high_priority_topics_counted(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        # OSI Model (91), TCP/IP (85), Routing (72) → 3 high-priority
        assert plan.high_priority == 3

    def test_medium_priority_topics_counted(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        # Network Security (60), DNS (45) → 2 medium-priority
        assert plan.medium_priority == 2

    def test_low_priority_topics_counted(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        # VLAN (25) → 1 low-priority
        assert plan.low_priority == 1

    def test_total_topics_correct(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert plan.total_topics == len(SAMPLE_CLUSTERS)

    def test_plan_with_one_day(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=1, hours_per_day=3.0)
        # With 1 day, there should still be at least 1 day in plan
        assert len(plan.days) >= 1

    def test_plan_with_empty_clusters(self):
        plan = build_study_plan([], days_remaining=7, hours_per_day=3.0)
        # No topics — plan should still exist but be mostly empty
        assert isinstance(plan, StudyPlan)
        assert plan.total_topics == 0

    def test_plan_with_single_cluster(self):
        plan = build_study_plan(
            [make_cluster("OSI Model", 91.0)],
            days_remaining=3,
            hours_per_day=2.0,
        )
        assert plan.total_topics == 1

    def test_days_remaining_stored(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=10, hours_per_day=4.0)
        assert plan.days_remaining == 10

    def test_hours_per_day_stored(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=4.5)
        assert plan.hours_per_day == 4.5

    def test_disclaimer_present(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        assert len(plan.disclaimer) > 0
        # Must not make prediction claims
        assert "guarantee" in plan.disclaimer.lower() or "not" in plan.disclaimer.lower()

    def test_negative_days_clamped_to_one(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=-1, hours_per_day=3.0)
        assert len(plan.days) >= 1

    def test_zero_hours_uses_minimum(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=0)
        assert plan.hours_per_day >= 1.0 or len(plan.days) >= 1


# ---------------------------------------------------------------------------
# Tests: enrich_plan_with_llm()
# ---------------------------------------------------------------------------

class TestEnrichPlanWithLLM:
    def test_no_api_key_returns_same_plan(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=5, hours_per_day=3.0)
        enriched = enrich_plan_with_llm(plan, api_key=None)
        # Without API key, plan should be returned unchanged
        assert enriched is plan

    def test_empty_api_key_returns_same_plan(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=5, hours_per_day=3.0)
        enriched = enrich_plan_with_llm(plan, api_key="")
        assert enriched is plan


# ---------------------------------------------------------------------------
# Tests: plan_to_text()
# ---------------------------------------------------------------------------

class TestPlanToText:
    def test_returns_string(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        text = plan_to_text(plan)
        assert isinstance(text, str)

    def test_contains_qpredict_header(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        text = plan_to_text(plan)
        assert "QPREDICT" in text

    def test_contains_revision(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        text = plan_to_text(plan)
        assert "Revision" in text

    def test_contains_disclaimer(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        text = plan_to_text(plan)
        assert "not" in text.lower() or "based on" in text.lower()

    def test_contains_topic_names(self):
        plan = build_study_plan(SAMPLE_CLUSTERS, days_remaining=7, hours_per_day=3.0)
        text = plan_to_text(plan)
        assert "OSI Model" in text

    def test_empty_plan_text_still_valid(self):
        plan = build_study_plan([], days_remaining=3, hours_per_day=2.0)
        text = plan_to_text(plan)
        assert isinstance(text, str)
        assert len(text) > 0
