"""
3_📊_Dashboard.py — Historical Priority Dashboard (Phase 7 centerpiece)

This is the main student-facing page. It runs the full analysis pipeline
and presents results as priority-ranked topic cards with:
  - Plotly charts (priority bar, year heatmap, trend pie)
  - Year timelines with coloured chips
  - Score breakdown radars
  - CSV export
  - Expandable topic cards with "Why this score?" explanations
"""

import streamlit as st
import sys, os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database.database import initialize_database, get_all_subjects, get_questions_for_subject
from src.embeddings import get_or_compute_embeddings, is_available as embeddings_available
from src.clustering import run_clustering
from src.scoring import score_all_clusters, DEFAULT_WEIGHTS
from src.trend_analyzer import compute_subject_statistics
from src.ui_helpers import (
    render_disclaimer, render_topic_card, render_year_timeline,
    priority_bar_chart, year_heatmap, trend_pie_chart, export_priority_csv,
    TREND_META,
)

st.set_page_config(page_title="Dashboard — QPredict", page_icon="📊", layout="wide")
initialize_database()

# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Historical Priority Dashboard")
render_disclaimer()

subjects = get_all_subjects()
if not subjects:
    st.info("No papers uploaded yet. Go to **📤 Upload** to add your first paper.")
    st.stop()

# ── Settings sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Analysis settings")
    subject = st.selectbox("Subject", subjects)
    dist_threshold = st.slider("Cluster threshold", 0.10, 0.70, 0.35, 0.05,
                               help="Distance = 1 − similarity. Lower = stricter.")
    st.markdown("**Score weights**")
    w_freq = st.slider("Frequency",     0.0, 1.0, DEFAULT_WEIGHTS["frequency"],     0.05)
    w_cov  = st.slider("Year coverage", 0.0, 1.0, DEFAULT_WEIGHTS["year_coverage"], 0.05)
    w_rec  = st.slider("Recency",       0.0, 1.0, DEFAULT_WEIGHTS["recency"],       0.05)
    w_cons = st.slider("Consistency",   0.0, 1.0, DEFAULT_WEIGHTS["consistency"],   0.05)
    weights = {"frequency": w_freq, "year_coverage": w_cov,
               "recency": w_rec,   "consistency":  w_cons}
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 0.05:
        st.warning(f"Weights sum to {total_w:.2f}. Ideally they should sum to 1.0.")

    st.divider()
    if embeddings_available():
        st.success("🟢 Sentence Transformers ready")
    else:
        st.warning("🟡 TF-IDF fallback active")

# ── Load questions ─────────────────────────────────────────────────────────────
questions = get_questions_for_subject(subject)
if len(questions) < 3:
    st.info(f"Only {len(questions)} question(s) for '{subject}'. Upload more papers.")
    st.stop()

st.write(f"**{len(questions)} questions** across all years for **{subject}**.")
run_btn = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

progress = st.progress(0, text="Step 1/3 — Loading embeddings…")
embeddings, emb_method = get_or_compute_embeddings(questions)
progress.progress(33, text="Step 2/3 — Clustering questions…")

cluster_result = run_clustering(questions, embeddings,
                                algorithm="agglomerative",
                                distance_threshold=dist_threshold)
progress.progress(66, text="Step 3/3 — Computing priority scores…")

if not cluster_result["clusters"]:
    progress.empty()
    st.warning("No clusters found. Try increasing the cluster threshold.")
    st.stop()

scored = score_all_clusters(cluster_result["clusters"], questions, weights=weights)
progress.progress(100, text="Analysis complete!")
progress.empty()

stats = compute_subject_statistics(scored, questions)
all_years = stats["years_covered"]

st.success(
    f"✅ Found **{len(scored)} topic clusters** from {len(questions)} questions "
    f"across {len(all_years)} year(s) — {emb_method.replace('_', ' ')}."
)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW METRICS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📋 Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers analyzed",    stats["total_papers"])
c2.metric("Questions analyzed", stats["total_questions"])
c3.metric("Topic clusters",     stats["total_clusters"])
c4.metric("High priority (≥70)", stats["high_priority_topics"])
c5.metric("Years covered",      len(all_years))

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
tab_bar, tab_heat, tab_pie = st.tabs(["Priority Rankings", "Year Heatmap", "Trend Distribution"])

with tab_bar:
    fig_bar = priority_bar_chart(scored, top_n=min(15, len(scored)))
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_heat:
    if len(all_years) >= 2:
        fig_heat = year_heatmap(scored, all_years, top_n=min(15, len(scored)))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need at least 2 years of papers to show heatmap.")

with tab_pie:
    if stats["trend_counts"]:
        fig_pie = trend_pie_chart(stats["trend_counts"])
        st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RANKING TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("🏆 Priority Rankings")

from src.trend_analyzer import build_year_timeline, timeline_as_string
table_rows = []
for rank, sc in enumerate(scored, start=1):
    tl = build_year_timeline(sc.years, all_years)
    meta = TREND_META.get(sc.trend, {"icon": "⚪"})
    table_rows.append({
        "Rank":      rank,
        "Topic":     sc.topic_label,
        "Score":     f"{sc.priority_score:.0f}/100",
        "Trend":     f"{meta['icon']} {sc.trend}",
        "Timeline":  timeline_as_string(tl),
        "Papers":    sc.paper_count,
        "Questions": sc.total_appearances,
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# Export CSV
csv_data = export_priority_csv(scored, all_years)
st.download_button(
    label="⬇ Download Priority List as CSV",
    data=csv_data,
    file_name=f"qpredict_{subject.lower().replace(' ', '_')}_priorities.csv",
    mime="text/csv",
)

# ══════════════════════════════════════════════════════════════════════════════
# TOPIC CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📚 Topic Cards")
st.caption("Click a card to expand it and see representative questions, score breakdown, and year coverage.")

# Filter controls
col_filter, col_sort = st.columns([2, 1])
with col_filter:
    trend_filter = st.multiselect(
        "Filter by trend",
        options=list(TREND_META.keys()),
        default=[],
        help="Leave empty to show all trends.",
    )
with col_sort:
    min_score = st.slider("Minimum score", 0, 100, 0, 5)

display_clusters = [
    sc for sc in scored
    if sc.priority_score >= min_score
    and (not trend_filter or sc.trend in trend_filter)
]

if not display_clusters:
    st.info("No topics match the current filters.")
else:
    st.caption(f"Showing {len(display_clusters)} of {len(scored)} topics.")
    for i, sc in enumerate(display_clusters):
        render_topic_card(sc, questions, all_years, expanded=(i == 0))
