"""
5_📖_Syllabus.py — Syllabus Intelligence page (Phase 8)

Upload a syllabus PDF (or paste unit names manually), and QPredict
will map your question clusters onto syllabus units to show:
  - Which units the exam tests most heavily
  - Which syllabus topics have appeared in past papers
  - Which topics have NEVER appeared (possible gaps)
  - Unit-wise priority ranking
"""

import streamlit as st
import sys, os
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database.database import initialize_database, get_all_subjects, get_questions_for_subject
from src.embeddings import get_or_compute_embeddings, is_available as embeddings_available
from src.clustering import run_clustering
from src.scoring import score_all_clusters, DEFAULT_WEIGHTS
from src.trend_analyzer import compute_subject_statistics
from src.syllabus_mapper import (
    run_syllabus_analysis, extract_syllabus_units, units_from_manual_input
)
from src.pdf_parser import extract_text_from_pdf, is_scanned_pdf
from src.ui_helpers import render_disclaimer, render_year_timeline, TREND_META

st.set_page_config(page_title="Syllabus — QPredict", page_icon="📖", layout="wide")
initialize_database()

# ══════════════════════════════════════════════════════════════════════════════
st.title("📖 Syllabus Intelligence")
st.write(
    "Upload your syllabus and QPredict will map every question cluster "
    "to its syllabus unit, showing which units are historically most important."
)
render_disclaimer()

subjects = get_all_subjects()
if not subjects:
    st.info("No papers uploaded yet. Go to **📤 Upload** first.")
    st.stop()

# ── Settings ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")
    subject        = st.selectbox("Subject", subjects)
    dist_threshold = st.slider("Cluster threshold", 0.10, 0.70, 0.35, 0.05)
    if embeddings_available():
        st.success("🟢 Sentence Transformers ready")
    else:
        st.warning("🟡 TF-IDF fallback active")

questions = get_questions_for_subject(subject)
if len(questions) < 3:
    st.info(f"Only {len(questions)} questions for '{subject}'. Upload more papers.")
    st.stop()

st.write(f"**{len(questions)} questions** loaded for **{subject}**.")

# ══════════════════════════════════════════════════════════════════════════════
# SYLLABUS INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Step 1 — Provide Syllabus")
tab_pdf, tab_manual = st.tabs(["📄 Upload Syllabus PDF", "✏️ Enter Units Manually"])

syllabus_text   = ""
manual_text     = ""
syllabus_source = "none"

with tab_pdf:
    syl_file = st.file_uploader(
        "Upload your syllabus as a PDF",
        type=["pdf"],
        help="The PDF should contain unit names and topic lists.",
    )
    if syl_file:
        syl_save = os.path.join(PROJECT_ROOT, "data", "uploads", syl_file.name)
        with open(syl_save, "wb") as f:
            f.write(syl_file.getbuffer())
        with st.spinner("Extracting syllabus text…"):
            parsed_syl = extract_text_from_pdf(syl_save)
        if parsed_syl.error:
            st.error(f"Could not read syllabus: {parsed_syl.error}")
        elif is_scanned_pdf(parsed_syl):
            st.warning("⚠️ Syllabus appears to be a scanned image — text may be incomplete.")
            syllabus_text = parsed_syl.full_text
        else:
            syllabus_text = parsed_syl.full_text
            syllabus_source = "pdf"
            st.success(f"✅ Extracted {len(syllabus_text):,} characters from {parsed_syl.total_pages} pages.")

            # Show preview of detected units
            preview_units = extract_syllabus_units(syllabus_text)
            if preview_units:
                st.info(f"🔍 Auto-detected **{len(preview_units)} units** from the syllabus.")
                for u in preview_units:
                    st.caption(f"  Unit {u.unit_number}: {u.unit_name} ({len(u.topics)} topics)")
            else:
                st.warning(
                    "Could not auto-detect unit structure from this PDF. "
                    "Try the **Enter Units Manually** tab."
                )

with tab_manual:
    st.write("Enter your syllabus unit structure. Each unit on its own line, topics indented below.")
    st.caption("Example format:")
    st.code(
        "Unit 1: Introduction to Networks\n"
        "  OSI Model\n"
        "  TCP/IP\n"
        "Unit 2: Data Link Layer\n"
        "  Framing\n"
        "  Error Detection\n"
        "  Flow Control",
        language="text"
    )
    manual_text = st.text_area(
        "Paste your syllabus units here",
        height=250,
        placeholder="Unit 1: Introduction\n  OSI Model\n  TCP/IP\nUnit 2: Data Link Layer\n  ...",
    )
    if manual_text.strip():
        preview = units_from_manual_input(manual_text)
        if preview:
            syllabus_source = "manual"
            st.success(f"✅ Parsed **{len(preview)} units** from your input.")
            for u in preview:
                st.caption(f"  Unit {u.unit_number}: {u.unit_name} ({len(u.topics)} topics)")
        else:
            st.warning("Could not parse units. Check the format above.")

# ── Run button ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Step 2 — Run Syllabus Analysis")

if syllabus_source == "none":
    st.info("Upload a syllabus PDF or enter units manually above, then click Run.")

run_btn = st.button(
    "▶ Run Syllabus Analysis",
    type="primary",
    use_container_width=True,
    disabled=(syllabus_source == "none"),
)

if not run_btn:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

progress = st.progress(0, "Step 1/4 — Loading embeddings…")
embeddings, emb_method = get_or_compute_embeddings(questions)

progress.progress(25, "Step 2/4 — Clustering questions…")
cluster_result = run_clustering(questions, embeddings, distance_threshold=dist_threshold)

if not cluster_result["clusters"]:
    progress.empty()
    st.warning("No question clusters found. Try increasing the cluster threshold.")
    st.stop()

progress.progress(50, "Step 3/4 — Scoring clusters…")
scored = score_all_clusters(cluster_result["clusters"], questions)

progress.progress(75, "Step 4/4 — Mapping to syllabus units…")
syl_result = run_syllabus_analysis(
    syllabus_text=syllabus_text,
    scored_clusters=scored,
    questions=questions,
    cluster_embeddings=embeddings,
    manual_units_text=manual_text,
)
progress.progress(100, "Done!")
progress.empty()

if syl_result["total_units"] == 0:
    st.error(
        "No syllabus units were found. "
        "Please check your input format or try the manual entry tab."
    )
    st.stop()

unit_analyses = syl_result["unit_analyses"]
mappings      = syl_result["mappings"]
units         = syl_result["units"]
stats         = compute_subject_statistics(scored, questions)
all_years     = stats["years_covered"]

st.success(
    f"✅ Mapped **{len(scored)} topic clusters** across "
    f"**{syl_result['total_units']} syllabus units** "
    f"(parse method: {syl_result['parse_method']})."
)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW METRICS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📋 Unit Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Syllabus units",    syl_result["total_units"])
c2.metric("Topic clusters",    len(scored))
c3.metric("Questions analyzed", len(questions))
c4.metric("Years covered",     len(all_years))

# ── Unit coverage bar chart ────────────────────────────────────────────────────
st.subheader("📊 Questions per Unit (historical exam coverage)")

unit_bar_data = [
    {
        "Unit":        ua.unit.full_label[:40],
        "Questions":   ua.total_questions,
        "Avg Score":   ua.avg_priority_score,
        "Clusters":    ua.cluster_count,
    }
    for ua in sorted(unit_analyses, key=lambda a: a.total_questions, reverse=True)
]

if unit_bar_data:
    df_bar = pd.DataFrame(unit_bar_data)
    fig = go.Figure(go.Bar(
        x=df_bar["Questions"],
        y=df_bar["Unit"],
        orientation="h",
        marker_color=[
            "#d32f2f" if q >= df_bar["Questions"].max() * 0.7
            else "#e65100" if q >= df_bar["Questions"].max() * 0.4
            else "#1565c0"
            for q in df_bar["Questions"]
        ],
        text=df_bar["Questions"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Questions: %{x}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Total exam questions from this unit",
        height=max(280, 40 * len(unit_bar_data)),
        margin=dict(l=10, r=40, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Priority ranking table ─────────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Unit Priority Rankings")
st.caption("Units sorted by average historical priority score of their question clusters.")

rows = []
for rank, ua in enumerate(unit_analyses, start=1):
    rows.append({
        "Rank":          rank,
        "Unit":          ua.unit.full_label,
        "Topics":        len(ua.unit.topics),
        "Clusters":      ua.cluster_count,
        "Questions":     ua.total_questions,
        "Avg Score":     f"{ua.avg_priority_score:.0f}/100",
        "Max Score":     f"{ua.max_priority_score:.0f}/100",
        "Years":         ", ".join(str(y) for y in ua.years_covered) if ua.years_covered else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# CSV export
csv = pd.DataFrame(rows).to_csv(index=False)
st.download_button(
    "⬇ Download Unit Analysis as CSV",
    data=csv,
    file_name=f"qpredict_{subject.lower().replace(' ', '_')}_units.csv",
    mime="text/csv",
)

# ══════════════════════════════════════════════════════════════════════════════
# UNIT DETAIL CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📚 Unit Detail Cards")
st.caption("Expand a unit to see all its question clusters, syllabus topics, and coverage.")

# Map clusters back for display: cluster_id → mapping
mapping_by_cluster = {m.cluster_id: m for m in mappings}

for ua in unit_analyses:
    has_clusters = ua.cluster_count > 0
    coverage_pct = (ua.cluster_count / len(scored) * 100) if scored else 0

    icon  = "🔴" if ua.avg_priority_score >= 70 else "🟠" if ua.avg_priority_score >= 40 else "🔵"
    label = (
        f"{icon} **{ua.unit.full_label}**"
        f"  ·  {ua.cluster_count} clusters"
        f"  ·  {ua.total_questions} questions"
        f"  ·  Avg: {ua.avg_priority_score:.0f}/100"
    )

    with st.expander(label, expanded=False):
        col_l, col_r = st.columns([3, 2])

        with col_l:
            if has_clusters:
                st.success(
                    f"✅ This unit has been tested in **{len(ua.years_covered)} year(s)**: "
                    f"{', '.join(str(y) for y in ua.years_covered)}"
                )
            else:
                st.info("ℹ️ No question clusters were mapped to this unit.")

            # Syllabus topics list
            st.markdown("**Syllabus topics in this unit:**")
            if ua.unit.topics:
                for t in ua.unit.topics[:10]:
                    # Check if any cluster was mapped here with high confidence
                    matched = any(
                        sc.topic_label.lower() in t.lower() or t.lower() in sc.topic_label.lower()
                        for sc in ua.mapped_clusters
                    )
                    marker = "✅" if matched else "⬜"
                    st.markdown(f"  {marker} {t}")
                if len(ua.unit.topics) > 10:
                    st.caption(f"  ... and {len(ua.unit.topics) - 10} more topics.")
            else:
                st.caption("No individual topics extracted from this unit.")

        with col_r:
            st.metric("Clusters mapped",  ua.cluster_count)
            st.metric("Total questions",  ua.total_questions)
            st.metric("Avg priority",     f"{ua.avg_priority_score:.0f}/100")
            st.metric("Best cluster",     f"{ua.max_priority_score:.0f}/100")

        # Cluster cards within this unit
        if has_clusters:
            st.markdown("**Question clusters in this unit:**")
            for sc in sorted(ua.mapped_clusters, key=lambda x: x.priority_score, reverse=True):
                m      = mapping_by_cluster.get(sc.cluster_id)
                conf   = f"  *(mapping confidence: {m.similarity_score:.0%})*" if m else ""
                trend_icon = TREND_META.get(sc.trend, {}).get("icon", "⚪")
                st.markdown(
                    f"- {trend_icon} **{sc.topic_label}** — "
                    f"Score: **{sc.priority_score:.0f}/100** · "
                    f"{sc.total_appearances} questions · "
                    f"{', '.join(str(y) for y in sc.years)}"
                    f"{conf}"
                )
                st.caption(f"  *\"{sc.representative_text[:90]}{'…' if len(sc.representative_text) > 90 else ''}\"*")

# ══════════════════════════════════════════════════════════════════════════════
# UNMAPPED / UNTESTED TOPICS
# ══════════════════════════════════════════════════════════════════════════════

untested_topics = []
for ua in unit_analyses:
    if ua.cluster_count == 0:
        for t in ua.unit.topics[:5]:
            untested_topics.append(f"{ua.unit.full_label}: {t}")

if untested_topics:
    st.divider()
    with st.expander(f"📭 Syllabus topics with no matching exam questions ({len(untested_topics)} found)", expanded=False):
        st.info(
            "These topics appear in the syllabus but no matching questions were found "
            "in the uploaded papers. They may be new additions or less frequently tested."
        )
        for t in untested_topics[:20]:
            st.markdown(f"- {t}")
