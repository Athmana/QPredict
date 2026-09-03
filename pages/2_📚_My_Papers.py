"""
2_📚_My_Papers.py — Uploaded papers library

Shows all papers in the database with status, question counts, and delete controls.
"""

import streamlit as st
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database.database import (
    initialize_database, get_all_papers, get_questions_for_paper,
    get_all_subjects, delete_paper,
)

st.set_page_config(page_title="My Papers — QPredict", page_icon="📚", layout="wide")
initialize_database()

st.header("📚 My Papers")
st.write("All uploaded examination papers and their extraction status.")

papers = get_all_papers()

if not papers:
    st.info("No papers uploaded yet. Go to **📤 Upload** to add your first paper.")
    st.stop()

subjects = get_all_subjects()
years    = sorted({p["year"] for p in papers if p.get("year")}, reverse=True)
total_q  = sum(len(get_questions_for_paper(p["id"])) for p in papers)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total papers",    len(papers))
c2.metric("Subjects",        len(subjects))
c3.metric("Total questions", total_q)
c4.metric("Years",           f"{min(years)}–{max(years)}" if len(years) >= 2 else str(years[0]) if years else "—")

st.divider()

# ── Delete All ────────────────────────────────────────────────────────────────
with st.expander("⚠️ Danger zone"):
    st.warning("Deleting papers also removes all their extracted questions.")
    if st.button("🗑️ Delete ALL papers and questions", type="primary"):
        for p in papers:
            delete_paper(p["id"])
        st.success("All papers deleted.")
        st.rerun()

st.divider()

# ── Per-paper list ────────────────────────────────────────────────────────────
STATUS_ICONS = {
    "uploaded":   "🟡 uploaded",
    "extracting": "🔵 extracting",
    "extracted":  "🟢 extracted",
    "error":      "🔴 error",
    "pending":    "⚪ pending",
}

papers_by_subject = {}
for p in papers:
    s = p["subject"] or "Unknown Subject"
    papers_by_subject.setdefault(s, []).append(p)

for subj, subj_papers in sorted(papers_by_subject.items()):
    with st.expander(f"📚 **{subj}**  ·  {len(subj_papers)} papers", expanded=True):
        # Column headers
        hc = st.columns([3, 1, 1, 2, 2, 1])
        hc[0].markdown("**File**")
        hc[1].markdown("**Year**")
        hc[2].markdown("**Questions**")
        hc[3].markdown("**Exam type**")
        hc[4].markdown("**Status**")
        hc[5].markdown("**Delete**")
        st.divider()

        for p in sorted(subj_papers, key=lambda x: x["year"] or 0, reverse=True):
            qc = len(get_questions_for_paper(p["id"]))
            rc = st.columns([3, 1, 1, 2, 2, 1])
            rc[0].write(f"📄 {p['filename']}")
            rc[1].write(f"**{p['year'] or '—'}**")
            rc[2].write(str(qc) if qc else "—")
            rc[3].write(p.get("exam_type") or "—")
            rc[4].write(STATUS_ICONS.get(p["processing_status"], p["processing_status"]))
            if rc[5].button("🗑️", key=f"del_{p['id']}", help=f"Delete {p['filename']}"):
                delete_paper(p["id"])
                st.success(f"Deleted **{p['filename']}**")
                st.rerun()
