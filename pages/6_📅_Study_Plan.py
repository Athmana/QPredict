"""
pages/6_📅_Study_Plan.py
========================
Phase 9 — Personalized Study Plan Page

This page:
1. Asks the student for time constraints
2. Loads scored clusters from the database
3. Calls study_planner.build_study_plan()
4. Optionally calls study_planner.enrich_plan_with_llm() if API key provided
5. Displays the day-by-day plan
6. Lets the student copy the plan as plain text
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from database.database import initialize_database, get_all_subjects, get_questions_for_subject
from src.study_planner import build_study_plan, enrich_plan_with_llm, plan_to_text

st.set_page_config(page_title="Study Plan — QPredict", page_icon="📅", layout="wide")
initialize_database()

# ---------------------------------------------------------------------------
# Helper: build topic list from uploaded questions (no clusters table needed)
# ---------------------------------------------------------------------------

def load_topics_from_questions(subject: str) -> list[dict]:
    """
    Derive study topics directly from uploaded questions for a subject.
    Groups questions by (section, topic keyword) and assigns a simple
    priority score based on frequency and year spread.
    """
    questions = get_questions_for_subject(subject)
    if not questions:
        return []

    # Group by year to count frequency
    from collections import Counter
    year_counts: Counter = Counter()
    for q in questions:
        if q.get("year"):
            year_counts[q["year"]] += 1

    all_years = sorted(year_counts.keys())
    total_q   = len(questions)

    # Use section as a coarse topic grouping; fall back to "General"
    from collections import defaultdict
    by_section: dict = defaultdict(list)
    for q in questions:
        key = q.get("section") or "General"
        by_section[key].append(q)

    result = []
    for section, qs in by_section.items():
        years = sorted({q["year"] for q in qs if q.get("year")})
        q_count = len(qs)
        # Simple score: (frequency share × 50) + (year spread share × 50)
        freq_share  = q_count / total_q if total_q else 0
        span_share  = len(years) / len(all_years) if all_years else 0
        score = round((freq_share * 50) + (span_share * 50), 1)

        if score >= 40:
            trend = "Frequently Recurring"
        elif score >= 20:
            trend = "Moderately Recurring"
        else:
            trend = "Sporadic"

        label = f"Part {section}" if len(section) <= 5 else section
        result.append({
            "topic":          label,
            "priority_score": score,
            "question_count": q_count,
            "years":          years,
            "trend":          trend,
        })

    result.sort(key=lambda x: x["priority_score"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Page UI
# ---------------------------------------------------------------------------

st.title("📅 Personalized Study Plan")
st.markdown(
    "QPredict uses Historical Priority Scores to help you distribute your "
    "study time intelligently. Topics with stronger historical patterns get "
    "earlier placement in your schedule."
)

# Disclaimer — always shown prominently
st.info(
    "⚠️  **Disclaimer:** This plan is based on patterns found in uploaded "
    "examination papers. It does not predict or guarantee future examination "
    "content. All topics in your syllabus remain important."
)

st.divider()

# --- Subject selector ---
subjects = get_all_subjects()
if not subjects:
    st.warning("No papers uploaded yet. Go to **📤 Upload** to add papers first.")
    st.page_link("pages/1_📤_Upload.py", label="Go to Upload →")
    st.stop()

subject  = st.selectbox("Subject", subjects)
clusters = load_topics_from_questions(subject)

if not clusters:
    st.warning(f"No questions found for **{subject}**. Upload and process papers first.")
    st.stop()

st.success(f"✅ {len(clusters)} topic sections loaded for **{subject}**.")

# ---------------------------------------------------------------------------
# Student inputs
# ---------------------------------------------------------------------------

st.subheader("⚙️ Your Study Constraints")

col1, col2, col3 = st.columns(3)

with col1:
    days_remaining = st.number_input(
        "Days remaining until exam",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
        help="How many days do you have before your examination?",
    )

with col2:
    hours_per_day = st.number_input(
        "Hours available per day",
        min_value=0.5,
        max_value=16.0,
        value=3.0,
        step=0.5,
        help="How many hours can you study each day?",
    )

with col3:
    st.metric("Total study time", f"{int(days_remaining * hours_per_day * 60)} min")

# Optional LLM enrichment
with st.expander("🤖 Optional: Add AI Study Tips (requires API key)"):
    st.markdown(
        "If you have a Groq or OpenAI API key, QPredict can add a personalized "
        "study tip for each topic. This is optional — the plan works without it."
    )
    provider = st.selectbox("LLM Provider", ["groq", "openai"], index=0)
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here (optional)",
    )

# Priority filter
min_score = st.slider(
    "Minimum Priority Score to include",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Set to 0 to include all topics. Increase to focus only on higher-priority topics.",
)

# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------

if st.button("📅 Generate Study Plan", type="primary", use_container_width=True):
    filtered_clusters = [c for c in clusters if c["priority_score"] >= min_score]

    if not filtered_clusters:
        st.warning("No topics match the selected minimum score. Lower the filter.")
        st.stop()

    with st.spinner("Building your personalized study plan..."):
        plan = build_study_plan(filtered_clusters, int(days_remaining), float(hours_per_day))

        if api_key:
            with st.spinner("Adding AI study tips (this may take a moment)..."):
                plan = enrich_plan_with_llm(plan, api_key=api_key, provider=provider)

    st.success("✅ Study plan generated!")

    # --- Overview metrics ---
    st.subheader("📊 Plan Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Topics", plan.total_topics)
    m2.metric("🔴 High Priority", plan.high_priority)
    m3.metric("🟡 Medium Priority", plan.medium_priority)
    m4.metric("🔵 Low Priority", plan.low_priority)

    st.divider()

    # --- Day-by-day display ---
    st.subheader("📆 Your Study Schedule")

    for day in plan.days:
        if day.is_revision_day:
            with st.expander(f"🔄 {day.label}  ·  {day.total_minutes} min total", expanded=False):
                st.markdown("**Quick revision of top-priority topics:**")
                for item in day.items:
                    st.markdown(f"- **{item.topic_name}** — ~15 min quick review")
        else:
            expanded = day.day_number <= 3  # Auto-expand first 3 days
            with st.expander(
                f"📖 {day.label}  ·  {day.total_minutes} min total", expanded=expanded
            ):
                if not day.items:
                    st.caption("No topics scheduled for this day.")
                    continue

                for item in day.items:
                    # Priority badge
                    if item.priority_score >= 70:
                        badge = "🔴 High"
                        color = "red"
                    elif item.priority_score >= 40:
                        badge = "🟡 Medium"
                        color = "orange"
                    else:
                        badge = "🔵 Low"
                        color = "blue"

                    col_a, col_b, col_c = st.columns([4, 1, 1])
                    with col_a:
                        st.markdown(f"**{item.topic_name}**")
                        if item.notes:
                            st.caption(f"💡 {item.notes}")
                        if item.years_seen:
                            years_str = ", ".join(str(y) for y in item.years_seen)
                            st.caption(f"📅 Appeared in: {years_str}")
                    with col_b:
                        st.markdown(f":{color}[{badge}]")
                        st.caption(f"Score: {item.priority_score:.0f}/100")
                    with col_c:
                        st.markdown(f"⏱ **{item.estimated_minutes} min**")
                        st.caption(item.trend)

                    st.divider()

    # --- Plain text export ---
    st.subheader("📋 Export Plan")
    plan_text = plan_to_text(plan)
    st.text_area(
        "Copy this plan to clipboard:",
        value=plan_text,
        height=300,
        label_visibility="collapsed",
    )
    st.download_button(
        label="⬇️ Download as .txt",
        data=plan_text,
        file_name="qpredict_study_plan.txt",
        mime="text/plain",
    )

    # Store plan in session state so user can revisit without regenerating
    st.session_state["last_study_plan"] = plan

else:
    st.markdown("*Configure your constraints above and click **Generate Study Plan**.*")
