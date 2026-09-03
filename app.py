"""
app.py — QPredict main entry point (Phase 7 refactor)

HOW STREAMLIT MULTI-PAGE APPS WORK:
Streamlit automatically turns any Python files inside a `pages/` folder
into navigation pages. The file naming controls the order and display name:
  pages/1_📤_Upload.py          → "📤 Upload"
  pages/2_📚_My_Papers.py       → "📚 My Papers"
  pages/3_📊_Dashboard.py       → "📊 Dashboard"

This app.py is the HOME page — it shows a project overview and
quick-start instructions, then directs the student to the right page.

WHY REFACTOR TO MULTI-PAGE?
app.py was growing very large (1300+ lines). Splitting into pages:
  - Each file has one job → easier to read and maintain
  - Students can navigate directly to the feature they need
  - Streamlit's built-in sidebar handles navigation automatically
"""

import streamlit as st
import sys
import os

# ── Ensure project modules are importable from all pages ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from database.database import initialize_database, get_all_papers, get_all_subjects, get_all_questions

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="QPredict",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize DB on every startup ───────────────────────────────────────────
initialize_database()

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.title("📚 QPredict")
st.subheader("Turn past papers into smarter preparation.")

st.markdown("""
QPredict analyzes previous-year examination question papers and answers:
- Which questions have appeared repeatedly?
- Which topics are historically most important?
- Which topics appear consistently, which are sporadic?
- What should you focus on first?
""")

st.info(
    "📌 **Reminder:** QPredict shows historical patterns — not future predictions. "
    "Every score is based on evidence from uploaded papers, not speculation.",
    icon=None,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# LIVE STATS
# ══════════════════════════════════════════════════════════════════════════════

papers    = get_all_papers()
subjects  = get_all_subjects()
questions = get_all_questions()
years     = sorted({p["year"] for p in papers if p.get("year")})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Papers uploaded",    len(papers))
col2.metric("Subjects",           len(subjects))
col3.metric("Questions extracted", len(questions))
col4.metric("Years covered",
            f"{min(years)}–{max(years)}" if len(years) >= 2 else str(years[0]) if years else "—")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# QUICK START GUIDE
# ══════════════════════════════════════════════════════════════════════════════

if not papers:
    st.subheader("🚀 Getting started")
    st.markdown("""
**Step 1** — Go to **📤 Upload Papers** in the sidebar.  
Upload your previous-year question papers as PDF files.
Enter the subject name and examination year for each paper.

**Step 2** — Go to **📊 Dashboard** after uploading.  
Run the full analysis to see priority-ranked topics with year timelines.

**Step 3** — Click any topic card to see all related questions,
a score breakdown, and a "Why this score?" explanation.
    """)
else:
    st.subheader("📋 Your subjects")

    for subject in subjects:
        subject_papers    = [p for p in papers if p.get("subject") == subject]
        subject_questions = [q for q in questions if q.get("subject") == subject]
        paper_years       = sorted({p["year"] for p in subject_papers if p.get("year")})

        with st.expander(
            f"**{subject}**  ·  {len(subject_papers)} papers  ·  {len(subject_questions)} questions",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Papers", len(subject_papers))
            c2.metric("Questions", len(subject_questions))
            c3.metric("Years",
                      f"{min(paper_years)}–{max(paper_years)}"
                      if len(paper_years) >= 2 else str(paper_years[0]) if paper_years else "—")

            if paper_years:
                st.caption("Years: " + " · ".join(str(y) for y in paper_years))

            col_a, col_b = st.columns(2)
            col_a.page_link("pages/3_📊_Dashboard.py",    label="→ Open Dashboard",     icon="📊")
            col_b.page_link("pages/1_📤_Upload.py",       label="→ Upload more papers", icon="📤")

st.divider()
st.caption("QPredict · AI-powered examination intelligence · Built phase by phase.")
