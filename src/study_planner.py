"""
study_planner.py
================
Phase 9 — Personalized Study Plan Generator

WHAT THIS MODULE DOES
---------------------
Takes the student's time constraints (days remaining, hours/day) and the
Historical Priority Scores from Phase 6, then distributes topics across
the available days in a logical, priority-weighted schedule.

KEY DESIGN DECISIONS
--------------------
1. High-priority topics appear on early days (when the student is freshest).
2. The last day is always a Revision day.
3. Each topic is assigned an estimated study time (30–90 min) based on
   its priority score and number of related questions.
4. The planner fills each day's time budget before moving to the next day.
5. ALL topics are included — even low-priority ones — with a note that
   they still matter.
6. An optional LLM can enrich the plan with study tips per topic.
   If no API key is provided, the offline rule-based plan is used.

NO PREDICTION CLAIMS
--------------------
The study plan is clearly labeled as "based on historical patterns."
It never says "this topic WILL appear."
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StudyItem:
    """
    One topic to study on a given day.

    Attributes
    ----------
    topic_name       : Human-readable topic label (e.g. "OSI Model")
    priority_score   : 0–100 Historical Priority Score from Phase 6
    estimated_minutes: How long the student should spend on this topic
    question_count   : Total related questions found in uploaded papers
    years_seen       : List of years where this topic appeared
    trend            : Trend classification string (e.g. "Frequently Recurring")
    notes            : Optional study tip or note
    """
    topic_name: str
    priority_score: float
    estimated_minutes: int
    question_count: int
    years_seen: list[int]
    trend: str
    notes: str = ""


@dataclass
class StudyDay:
    """
    One day's worth of study.

    Attributes
    ----------
    day_number     : 1-based day index
    label          : e.g. "Day 1" or "Day 7 — Revision"
    items          : List of StudyItem objects for this day
    total_minutes  : Sum of all item estimated_minutes
    is_revision_day: True only for the final day
    """
    day_number: int
    label: str
    items: list[StudyItem] = field(default_factory=list)
    total_minutes: int = 0
    is_revision_day: bool = False

    def add_item(self, item: StudyItem) -> None:
        self.items.append(item)
        self.total_minutes += item.estimated_minutes


@dataclass
class StudyPlan:
    """
    The complete personalized study plan.

    Attributes
    ----------
    days             : Ordered list of StudyDay objects
    days_remaining   : Total days the student has
    hours_per_day    : Hours available per day
    total_topics     : How many topic clusters were scheduled
    high_priority    : Count of topics with score >= 70
    medium_priority  : Count of topics with score 40–69
    low_priority     : Count of topics with score < 40
    disclaimer       : Always-present disclaimer text
    """
    days: list[StudyDay] = field(default_factory=list)
    days_remaining: int = 7
    hours_per_day: float = 3.0
    total_topics: int = 0
    high_priority: int = 0
    medium_priority: int = 0
    low_priority: int = 0
    disclaimer: str = (
        "This study plan is based on historical patterns found in the "
        "uploaded examination papers. It does not guarantee or predict "
        "which questions will appear in a future examination. All topics "
        "in your syllabus remain important."
    )


# ---------------------------------------------------------------------------
# Helper: estimate study time for one topic
# ---------------------------------------------------------------------------

def _estimate_minutes(priority_score: float, question_count: int) -> int:
    """
    Estimate how many minutes a topic deserves.

    Logic
    -----
    - High priority (score >= 70)  → 60–90 minutes
    - Medium priority (40–69)      → 45–60 minutes
    - Low priority (< 40)          → 30–45 minutes

    The question_count adds a small bonus: more questions = more content.

    Returns an integer number of minutes (always between 20 and 90).
    """
    if priority_score >= 70:
        base = 60
    elif priority_score >= 40:
        base = 45
    else:
        base = 30

    # Each extra 5 questions beyond 3 adds 5 minutes, capped at +30
    bonus = min(30, max(0, (question_count - 3) // 5) * 5)
    return min(90, base + bonus)


# ---------------------------------------------------------------------------
# Core planner: offline rule-based
# ---------------------------------------------------------------------------

def build_study_plan(
    clusters: list[dict],
    days_remaining: int,
    hours_per_day: float,
) -> StudyPlan:
    """
    Build a personalized study plan from scored topic clusters.

    Parameters
    ----------
    clusters : list of dicts, each containing at minimum:
        - "topic"          : str   — topic label
        - "priority_score" : float — 0–100 score from scoring.py
        - "question_count" : int   — number of questions in the cluster
        - "years"          : list[int] — years where topic appeared
        - "trend"          : str   — trend classification string
    days_remaining : int  — total days the student has to study
    hours_per_day  : float — hours available per day

    Returns
    -------
    StudyPlan — fully populated plan ready to display
    """
    if days_remaining < 1:
        days_remaining = 1
    if hours_per_day <= 0:
        hours_per_day = 1.0

    # --- Sort clusters by priority score, highest first ---
    sorted_clusters = sorted(
        clusters, key=lambda c: c.get("priority_score", 0), reverse=True
    )

    # --- Convert clusters → StudyItem objects ---
    all_items: list[StudyItem] = []
    high = medium = low = 0

    for c in sorted_clusters:
        score = float(c.get("priority_score", 0))
        q_count = int(c.get("question_count", 1))
        minutes = _estimate_minutes(score, q_count)
        years = sorted(c.get("years", []))
        trend = c.get("trend", "Unknown")

        if score >= 70:
            high += 1
        elif score >= 40:
            medium += 1
        else:
            low += 1

        item = StudyItem(
            topic_name=c.get("topic", "Unknown Topic"),
            priority_score=score,
            estimated_minutes=minutes,
            question_count=q_count,
            years_seen=years,
            trend=trend,
        )
        all_items.append(item)

    plan = StudyPlan(
        days_remaining=days_remaining,
        hours_per_day=hours_per_day,
        total_topics=len(all_items),
        high_priority=high,
        medium_priority=medium,
        low_priority=low,
    )

    if not all_items:
        return plan

    # --- Reserve the last day as a revision day ---
    # If only 1 day, that day becomes revision (all topics reviewed briefly)
    study_days = max(1, days_remaining - 1)
    minutes_per_day = int(hours_per_day * 60)

    # --- Distribute items across study days ---
    # Use a greedy bin-packing approach: fill each day until full
    day_buckets: list[list[StudyItem]] = [[] for _ in range(study_days)]
    day_used_minutes: list[int] = [0] * study_days

    for item in all_items:
        # Find the day with the most remaining capacity that still has room
        best_day = 0
        best_remaining = -1
        for i in range(study_days):
            remaining = minutes_per_day - day_used_minutes[i]
            if remaining >= item.estimated_minutes and remaining > best_remaining:
                best_day = i
                best_remaining = remaining

        # If no day has room, add to the day with the most remaining space
        # (overflow — better than dropping the topic)
        if best_remaining < 0:
            best_day = day_used_minutes.index(min(day_used_minutes))

        day_buckets[best_day].append(item)
        day_used_minutes[best_day] += item.estimated_minutes

    # --- Build StudyDay objects ---
    for i, bucket in enumerate(day_buckets):
        day_num = i + 1
        day = StudyDay(
            day_number=day_num,
            label=f"Day {day_num}",
            is_revision_day=False,
        )
        for item in bucket:
            day.add_item(item)
        plan.days.append(day)

    # --- Add revision day ---
    revision_day = StudyDay(
        day_number=days_remaining,
        label=f"Day {days_remaining} — Revision",
        is_revision_day=True,
    )
    # Revision day: add top-priority items as a quick review checklist
    for item in all_items[:min(10, len(all_items))]:
        review_item = StudyItem(
            topic_name=item.topic_name,
            priority_score=item.priority_score,
            estimated_minutes=15,  # Short review
            question_count=item.question_count,
            years_seen=item.years_seen,
            trend=item.trend,
            notes="Quick revision — focus on representative question",
        )
        revision_day.add_item(review_item)
    plan.days.append(revision_day)

    return plan


# ---------------------------------------------------------------------------
# Optional LLM enrichment
# ---------------------------------------------------------------------------

def enrich_plan_with_llm(
    plan: StudyPlan,
    api_key: Optional[str] = None,
    provider: str = "groq",
) -> StudyPlan:
    """
    Optionally enrich each StudyItem with a study tip from an LLM.

    This function is a no-op if no API key is provided or if the
    required library is not installed.

    IMPORTANT
    ---------
    The LLM only adds a study tip string to each item's `notes` field.
    It does NOT affect scoring, ordering, or any analytical result.
    All analytical logic remains in the rule-based code above.

    Parameters
    ----------
    plan     : A StudyPlan already built by build_study_plan()
    api_key  : Optional API key (Groq, OpenAI, etc.)
    provider : "groq" (default) or "openai"

    Returns
    -------
    The same StudyPlan, with notes fields potentially filled in.
    """
    if not api_key:
        return plan  # Nothing to do — return unchanged

    try:
        if provider == "groq":
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            model = "llama3-8b-8192"
        elif provider == "openai":
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            model = "gpt-3.5-turbo"
        else:
            return plan

    except ImportError:
        # Library not installed — silently skip LLM enrichment
        return plan

    for day in plan.days:
        if day.is_revision_day:
            continue  # Skip revision day — tips not needed here
        for item in day.items:
            if item.notes:
                continue  # Already has a note
            try:
                prompt = (
                    f"You are a study coach. Give a student ONE concise study tip "
                    f"(max 2 sentences) for studying the topic: '{item.topic_name}'. "
                    f"Focus on exam preparation. Be specific and actionable."
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80,
                    temperature=0.7,
                )
                item.notes = response.choices[0].message.content.strip()
            except Exception:
                # Never crash the app because of LLM failure
                item.notes = ""

    return plan


# ---------------------------------------------------------------------------
# Utility: summarize plan as plain text (for display / export)
# ---------------------------------------------------------------------------

def plan_to_text(plan: StudyPlan) -> str:
    """
    Convert a StudyPlan to a readable plain-text string.

    Useful for copying to clipboard or displaying in a text area.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("QPREDICT — PERSONALIZED STUDY PLAN")
    lines.append("=" * 60)
    lines.append(
        f"Duration: {plan.days_remaining} days | "
        f"{plan.hours_per_day:.1f} hours/day"
    )
    lines.append(
        f"Topics: {plan.total_topics} total | "
        f"{plan.high_priority} high | "
        f"{plan.medium_priority} medium | "
        f"{plan.low_priority} low priority"
    )
    lines.append("")
    lines.append(f"⚠  {plan.disclaimer}")
    lines.append("")

    for day in plan.days:
        lines.append("-" * 60)
        lines.append(f"  {day.label}  ({day.total_minutes} min)")
        lines.append("-" * 60)
        if day.is_revision_day:
            lines.append("  Quick revision of top topics:")
        for item in day.items:
            star = "★" if item.priority_score >= 70 else "◇"
            lines.append(
                f"  {star} {item.topic_name} "
                f"[Score: {item.priority_score:.0f}/100] "
                f"~{item.estimated_minutes} min"
            )
            if item.notes:
                lines.append(f"      💡 {item.notes}")
        lines.append("")

    return "\n".join(lines)
