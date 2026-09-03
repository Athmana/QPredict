"""
ui_helpers.py — Shared Streamlit UI components for QPredict

WHY THIS FILE EXISTS:
Several pages show the same types of content — priority score badges,
year timelines, topic cards, score breakdown tables. Rather than
repeating the same Streamlit code in every page, we centralise
reusable components here.

This follows the DRY principle:
  Don't Repeat Yourself — define once, use everywhere.

It also makes the UI consistent: the same topic card looks identical
on the Dashboard page, the Clusters page, and any future page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# SCORE BADGE
# ══════════════════════════════════════════════════════════════════════════════

def score_color(score: float) -> str:
    """
    Return a CSS colour string for a priority score.

    Bands:
      ≥ 75  →  red    (high priority)
      ≥ 50  →  orange (medium priority)
      ≥ 25  →  blue   (low-medium priority)
      < 25  →  grey   (low priority)
    """
    if score >= 75:
        return "#d32f2f"
    elif score >= 50:
        return "#e65100"
    elif score >= 25:
        return "#1565c0"
    return "#757575"


def render_score_badge(score: float, label: str = "Priority Score"):
    """
    Render a large coloured score number with a label beneath it.
    """
    color = score_color(score)
    st.markdown(
        f"""
        <div style="text-align:center; padding:8px 0;">
            <span style="font-size:2.4rem; font-weight:700; color:{color};">
                {score:.0f}
            </span>
            <span style="font-size:1.1rem; color:{color};">/100</span>
            <div style="font-size:0.75rem; color:#57606a; margin-top:2px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TREND BADGE
# ══════════════════════════════════════════════════════════════════════════════

TREND_META = {
    "Frequently Recurring": {"icon": "🔴", "color": "#d32f2f"},
    "Consistently Asked":   {"icon": "🟠", "color": "#e65100"},
    "Recently Recurring":   {"icon": "🟡", "color": "#f9a825"},
    "Increasing":           {"icon": "🟢", "color": "#2e7d32"},
    "Decreasing":           {"icon": "🔵", "color": "#1565c0"},
    "Sporadic":             {"icon": "⚪", "color": "#757575"},
    "Rarely Asked":         {"icon": "⚫", "color": "#424242"},
}


def render_trend_badge(trend: str):
    meta = TREND_META.get(trend, {"icon": "⚪", "color": "#757575"})
    st.markdown(
        f'<span style="color:{meta["color"]}; font-weight:600;">'
        f'{meta["icon"]} {trend}</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER BANNER
# ══════════════════════════════════════════════════════════════════════════════

def render_disclaimer():
    """
    Display the standard QPredict historical disclaimer.
    Must be shown on every page that shows Priority Scores.
    """
    st.info(
        "📌 **Historical Priority Score** is based on patterns found in uploaded "
        "examination papers. It is **not** a prediction or guarantee of future "
        "examination questions. Use it as a study guide grounded in evidence.",
        icon=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# YEAR TIMELINE DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def render_year_timeline(timeline: list):
    """
    Render a horizontal year timeline as coloured chips.

      2021 ✓   2022 ✓   2023 ✗   2024 ✓   2025 ✓

    Parameters
    ----------
    timeline : list of dicts with keys year, appeared, symbol, count
    """
    chips = []
    for t in timeline:
        if t["appeared"]:
            chip = (
                f'<span style="background:#e8f5e9; color:#2e7d32; '
                f'padding:3px 8px; border-radius:12px; margin:2px; '
                f'font-weight:600; font-size:0.85rem;">'
                f'{t["year"]} ✓</span>'
            )
        else:
            chip = (
                f'<span style="background:#fafafa; color:#bdbdbd; '
                f'padding:3px 8px; border-radius:12px; margin:2px; '
                f'font-size:0.85rem;">'
                f'{t["year"]} ✗</span>'
            )
        chips.append(chip)
    st.markdown("&nbsp;".join(chips), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def priority_bar_chart(scored_clusters: list, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of the top N topics by priority score.

    WHY PLOTLY:
    Plotly charts are interactive — the student can hover to see exact
    scores, zoom in, and export as PNG. Matplotlib charts are static.

    Parameters
    ----------
    scored_clusters : List[ScoredCluster]
    top_n           : int  — how many topics to show

    Returns
    -------
    plotly Figure
    """
    top = scored_clusters[:top_n]
    labels = [sc.topic_label[:30] for sc in reversed(top)]
    scores = [sc.priority_score for sc in reversed(top)]
    trends = [sc.trend for sc in reversed(top)]
    colors = [TREND_META.get(t, {}).get("color", "#1565c0") for t in trends]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}/100<extra></extra>",
    ))
    fig.update_layout(
        title="Historical Priority Rankings",
        xaxis_title="Priority Score (0–100)",
        xaxis_range=[0, 110],
        height=max(300, 40 * len(top)),
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=False)
    return fig


def year_heatmap(scored_clusters: list, all_years: list, top_n: int = 15) -> go.Figure:
    """
    Heatmap showing which topics appeared in which years.

    Rows  = top N topics (by priority score)
    Cols  = years
    Color = number of questions that year

    WHY THIS IS USEFUL:
    The heatmap lets a student see at a glance which topics have been
    consistently tested and which have gaps. It's more information-dense
    than reading individual topic cards.
    """
    top = scored_clusters[:top_n]
    years = sorted(all_years)

    # Build a matrix: rows=topics, cols=years, values=question count
    matrix = []
    y_labels = []
    for sc in top:
        row = []
        for yr in years:
            count = sc.years.count(yr)
            row.append(count)
        matrix.append(row)
        y_labels.append(sc.topic_label[:25])

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[str(y) for y in years],
        y=y_labels,
        colorscale=[[0, "#f5f5f5"], [0.01, "#bbdefb"], [1, "#1565c0"]],
        showscale=True,
        colorbar=dict(title="Questions", thickness=12),
        hovertemplate="Topic: %{y}<br>Year: %{x}<br>Questions: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Topic × Year Coverage Heatmap",
        xaxis_title="Year",
        height=max(350, 35 * len(top)),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )
    return fig


def trend_pie_chart(trend_counts: dict) -> go.Figure:
    """
    Pie chart showing distribution of trend labels across all clusters.
    """
    labels = list(trend_counts.keys())
    values = list(trend_counts.values())
    colors = [TREND_META.get(l, {}).get("color", "#999") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        hovertemplate="%{label}: %{value} topics (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="Trend Distribution",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        font=dict(size=12),
        legend=dict(orientation="v", x=1.0, y=0.5),
        showlegend=True,
    )
    return fig


def score_components_radar(score_breakdown) -> go.Figure:
    """
    Radar chart showing the four score components for one topic.

    WHY: A radar (spider) chart makes it immediately clear which
    components are strong and which are weak for a given topic.
    "High frequency, low recency" is instantly visible.
    """
    categories = ["Frequency", "Year Coverage", "Recency", "Consistency"]
    values = [
        score_breakdown.frequency_score,
        score_breakdown.year_coverage_score,
        score_breakdown.recency_score,
        score_breakdown.consistency_score,
    ]
    # Close the radar by repeating first point
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(21, 101, 192, 0.15)",
        line_color="#1565c0",
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
        ),
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="white",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_priority_csv(scored_clusters: list, all_years: list) -> str:
    """
    Generate a CSV string of the full priority analysis for download.

    WHY: Students may want to take this data into Excel, share it, or
    print it as a study reference.
    """
    from src.trend_analyzer import build_year_timeline, timeline_as_string

    rows = []
    for rank, sc in enumerate(scored_clusters, start=1):
        timeline = build_year_timeline(sc.years, all_years)
        rows.append({
            "Rank":                rank,
            "Topic":               sc.topic_label,
            "Priority Score":      f"{sc.priority_score:.0f}/100",
            "Trend":               sc.trend,
            "Year Timeline":       timeline_as_string(timeline),
            "Total Appearances":   sc.total_appearances,
            "Papers":              sc.paper_count,
            "Frequency Score":     f"{sc.score.frequency_score:.0f}",
            "Year Coverage Score": f"{sc.score.year_coverage_score:.0f}",
            "Recency Score":       f"{sc.score.recency_score:.0f}",
            "Consistency Score":   f"{sc.score.consistency_score:.0f}",
            "Keywords":            ", ".join(sc.keywords[:5]),
            "Representative Question": sc.representative_text,
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC CARD (reusable)
# ══════════════════════════════════════════════════════════════════════════════

def render_topic_card(sc, questions: list, all_years: list, expanded: bool = False):
    """
    Render a full topic card for one ScoredCluster.

    Used by both the Dashboard page and the Priority Dashboard page.

    Parameters
    ----------
    sc         : ScoredCluster
    questions  : List[dict]   — full question bank (for member lookup)
    all_years  : List[int]    — all years in the subject
    expanded   : bool         — whether expander starts open
    """
    from src.trend_analyzer import build_year_timeline

    meta  = TREND_META.get(sc.trend, {"icon": "⚪", "color": "#757575"})
    timeline = build_year_timeline(sc.years, all_years)

    header = (
        f"{meta['icon']} **{sc.topic_label}**"
        f"  ·  **{sc.priority_score:.0f}/100**"
        f"  ·  {sc.total_appearances} questions"
        f"  ·  {sc.trend}"
    )

    with st.expander(header, expanded=expanded):
        col_main, col_score = st.columns([3, 1])

        with col_main:
            st.markdown("**Representative question:**")
            st.info(f'"{sc.representative_text}"')
            st.markdown("**Year coverage:**")
            render_year_timeline(timeline)

        with col_score:
            render_score_badge(sc.priority_score)
            st.plotly_chart(
                score_components_radar(sc.score),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Why this score?
        with st.expander("💡 Why this score?", expanded=False):
            for line in sc.score.explanation_lines():
                st.markdown(line)
            st.caption(
                "Weights: Frequency 40% · Year Coverage 25% · "
                "Recency 20% · Consistency 15%"
            )

        # All member questions
        with st.expander(f"📋 All {sc.total_appearances} related questions", expanded=False):
            for idx in sc.member_indices:
                q = questions[idx]
                yr    = q.get("year", "?")
                marks = f"  *{q.get('marks')} marks*" if q.get("marks") else ""
                st.markdown(f"- **{yr}**{marks} — {q['question_text']}")
