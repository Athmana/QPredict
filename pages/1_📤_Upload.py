"""
1_📤_Upload.py — Upload examination papers page

Handles PDF upload, metadata entry, text extraction, and question parsing.
All Phase 1 + Phase 2 logic lives here.
"""

import streamlit as st
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.pdf_parser import extract_text_from_pdf, is_scanned_pdf, save_extracted_text
from src.question_extractor import extract_questions, extraction_summary
from database.database import (
    initialize_database, insert_paper, update_paper_status,
    save_questions,
)

st.set_page_config(page_title="Upload — QPredict", page_icon="📤", layout="wide")
initialize_database()

UPLOADS_DIR  = os.path.join(PROJECT_ROOT, "data", "uploads")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(UPLOADS_DIR,  exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
st.header("📤 Upload Examination Papers")
st.write("Upload previous-year question papers. QPredict will extract text and identify questions automatically.")

with st.form("upload_form"):
    st.subheader("Step 1 — Paper Details")
    c1, c2 = st.columns(2)
    with c1:
        subject   = st.text_input("Subject *", placeholder="e.g. Computer Networks")
        year      = st.number_input("Examination Year *", 2000, 2100, 2024, step=1)
    with c2:
        exam_type = st.selectbox("Examination Type",
                                 ["", "Semester End Exam", "Mid-Term", "Internal Assessment", "Other"])
        semester  = st.text_input("Semester / Term", placeholder="e.g. Semester 3")

    st.subheader("Step 2 — Upload PDF Files")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    submitted = st.form_submit_button("Extract Text & Identify Questions", type="primary", use_container_width=True)

if submitted:
    if not subject.strip():
        st.error("Please enter a subject name.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        st.stop()

    st.divider()
    st.subheader("Step 3 — Extraction Results")

    for uf in uploaded_files:
        if not uf.name.lower().endswith(".pdf"):
            st.error(f"⚠️ {uf.name} is not a PDF file. Skipping.")
            continue

        with st.expander(f"📄 {uf.name}", expanded=True):
            save_path = os.path.join(UPLOADS_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())

            with st.spinner("Extracting text…"):
                parsed = extract_text_from_pdf(save_path)

            if parsed.error:
                st.error(f"❌ {parsed.error}")
                continue

            if is_scanned_pdf(parsed):
                st.warning("⚠️ This PDF may be scanned — text extraction may be incomplete.")

            paper_id = insert_paper(
                filename=uf.name, subject=subject.strip(),
                year=int(year), exam_type=exam_type or None, semester=semester or None
            )

            with st.spinner("Identifying questions…"):
                questions = extract_questions(parsed.full_text, paper_id=paper_id, year=int(year))
                save_questions(questions)

            update_paper_status(paper_id, "extracted")
            save_extracted_text(parsed, PROCESSED_DIR)
            summary = extraction_summary(questions)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages",            parsed.total_pages)
            c2.metric("Questions found",  summary["total_questions"])
            c3.metric("Sections",         len(summary["sections_found"]) or "—")
            c4.metric("Paper ID",         f"#{paper_id}")

            st.success(
                f"✅ Extracted **{summary['total_questions']} questions** "
                f"from {parsed.total_pages} pages."
            )

            if summary["total_questions"] == 0:
                st.warning("No questions auto-detected. Check raw text below.")

            tab_q, tab_raw = st.tabs([
                f"Questions ({summary['total_questions']})", "Raw Text"
            ])

            with tab_q:
                if not questions:
                    st.info("No questions detected.")
                else:
                    sections_seen = []
                    seen_s = set()
                    for q in questions:
                        k = q.section or "(No Section)"
                        if k not in seen_s:
                            sections_seen.append(k)
                            seen_s.add(k)
                    for sec in sections_seen:
                        qs = [q for q in questions if (q.section or "(No Section)") == sec]
                        st.subheader(f"Section {sec}" if sec != "(No Section)" else "Questions")
                        for q in qs:
                            badge = f"  `{q.marks} marks`" if q.marks else ""
                            with st.expander(
                                f"**{q.question_number or '–'}**  "
                                f"{q.question_text[:80]}{'…' if len(q.question_text) > 80 else ''}"
                                f"{badge}", expanded=False
                            ):
                                st.write(q.question_text)

            with tab_raw:
                st.text_area("Full extracted text", parsed.full_text, height=400,
                             label_visibility="collapsed")
