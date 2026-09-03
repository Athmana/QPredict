"""
pages/7_❓_Quiz.py
==================
Phase 9 — Practice Quiz Generator Page

Generates practice MCQs and short-answer questions from uploaded paper questions.
Works without a clusters DB table — loads questions directly per subject.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from database.database import initialize_database, get_all_subjects, get_questions_for_subject
from src.quiz_generator import generate_quiz

st.set_page_config(page_title="Practice Quiz — QPredict", page_icon="❓", layout="wide")
initialize_database()

st.title("❓ Practice Quiz Generator")
st.markdown(
    "Generate practice questions based on topics from your uploaded examination papers. "
    "Use this to test your understanding and identify gaps."
)
st.info(
    "⚠️  **Note:** Questions are drawn from your uploaded papers. "
    "They are for practice only and do not represent predicted exam content."
)

st.divider()

# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------

subjects = get_all_subjects()

if not subjects:
    st.warning("No papers uploaded yet. Go to **📤 Upload** to add papers first.")
    st.page_link("pages/1_📤_Upload.py", label="Go to Upload →")
    st.stop()

# ---------------------------------------------------------------------------
# Quiz configuration
# ---------------------------------------------------------------------------

st.subheader("⚙️ Configure Your Quiz")

col1, col2 = st.columns(2)

with col1:
    subject = st.selectbox(
        "Subject",
        options=subjects,
        help="All questions will be drawn from this subject's uploaded papers.",
    )

    difficulty = st.selectbox(
        "Difficulty level",
        options=["easy", "medium", "hard"],
        index=1,
        help=(
            "Easy: straightforward recall | "
            "Medium: understanding and application | "
            "Hard: analysis and synthesis"
        ),
    )

with col2:
    num_mcq   = st.slider("Number of MCQ questions",          min_value=1, max_value=8, value=3)
    num_short = st.slider("Number of short-answer questions", min_value=0, max_value=5, value=3)

# Optional LLM
with st.expander("🤖 Optional: Use AI for better questions (requires API key)"):
    st.markdown(
        "With an API key, QPredict can generate high-quality MCQs with detailed "
        "answer explanations. Without one, questions are drawn directly from your papers."
    )
    q_provider = st.selectbox("LLM Provider", ["groq", "openai"], key="quiz_provider")
    q_api_key  = st.text_input(
        "API Key",
        type="password",
        key="quiz_api_key",
        placeholder="Paste your API key here (optional)",
    )

# ---------------------------------------------------------------------------
# Generate quiz
# ---------------------------------------------------------------------------

generate_btn = st.button("🎯 Generate Quiz", type="primary", use_container_width=True)

if generate_btn or st.session_state.get("quiz_generated"):
    if generate_btn:
        st.session_state.pop("current_quiz",  None)
        st.session_state.pop("user_answers",  None)
        st.session_state.pop("submitted",     None)
        st.session_state.pop("quiz_subject",  None)

        questions = get_questions_for_subject(subject)
        if not questions:
            st.warning(f"No questions found for **{subject}**. Upload and process papers first.")
            st.session_state["quiz_generated"] = False
            st.stop()

        # Pass questions as {question_text, year} dicts
        cluster_questions = [
            {"question_text": q["question_text"], "year": q.get("year")}
            for q in questions
        ]

        with st.spinner(f"Generating quiz for '{subject}'…"):
            quiz = generate_quiz(
                topic_name=subject,
                cluster_questions=cluster_questions,
                num_mcq=num_mcq,
                num_short=num_short,
                difficulty=difficulty,
                api_key=q_api_key if q_api_key else None,
                provider=q_provider,
            )

        st.session_state["current_quiz"]  = quiz
        st.session_state["quiz_generated"] = True
        st.session_state["submitted"]      = False
        st.session_state["user_answers"]   = {}
        st.session_state["quiz_subject"]   = subject

    quiz = st.session_state.get("current_quiz")
    if not quiz or not quiz.questions:
        st.warning("Could not generate questions. Try a different subject.")
        st.session_state["quiz_generated"] = False
        st.stop()

    source_badge = "🤖 AI-Generated" if quiz.source == "llm" else "📄 Paper-Based (Offline)"
    st.success(f"Quiz ready! {source_badge}  ·  Subject: **{st.session_state.get('quiz_subject','')}**")
    st.divider()

    # -----------------------------------------------------------------------
    # Display quiz questions
    # -----------------------------------------------------------------------

    mcq_questions   = [q for q in quiz.questions if q.question_type == "mcq"]
    short_questions = [q for q in quiz.questions if q.question_type == "short"]
    submitted    = st.session_state.get("submitted", False)
    user_answers = st.session_state.get("user_answers", {})

    with st.form("quiz_form"):
        # MCQ section
        if mcq_questions:
            st.subheader("📝 Multiple Choice Questions")
            for i, q in enumerate(mcq_questions):
                q_key = f"mcq_{i}"
                st.markdown(f"**Q{i + 1}.** {q.question_text}")
                if q.source_year:
                    st.caption(f"*(Based on {q.source_year} paper)*")
                option_labels = [f"{opt.label}. {opt.text}" for opt in q.options]
                selected = st.radio(
                    f"Select answer for Q{i + 1}",
                    options=option_labels,
                    key=q_key,
                    label_visibility="collapsed",
                    disabled=submitted,
                )
                user_answers[q_key] = selected
                st.markdown("")

        # Short-answer section
        if short_questions:
            st.subheader("✍️ Short Answer Questions")
            for i, q in enumerate(short_questions):
                q_key = f"short_{i}"
                st.markdown(f"**SA{i + 1}.** {q.question_text}")
                if q.source_year:
                    st.caption(f"*(From {q.source_year} paper)*")
                answer_text = st.text_area(
                    f"Your answer for SA{i + 1}",
                    key=q_key,
                    height=100,
                    placeholder="Type your answer here…",
                    disabled=submitted,
                    label_visibility="collapsed",
                )
                user_answers[q_key] = answer_text
                st.markdown("")

        submit_btn = st.form_submit_button(
            "✅ Submit & See Results",
            disabled=submitted,
            use_container_width=True,
        )
        if submit_btn:
            st.session_state["submitted"]    = True
            st.session_state["user_answers"] = user_answers
            st.rerun()

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------

    if st.session_state.get("submitted"):
        st.divider()
        st.subheader("📊 Results")

        correct_count = 0
        total_mcq = len(mcq_questions)

        if mcq_questions:
            for i, q in enumerate(mcq_questions):
                q_key = f"mcq_{i}"
                selected_raw   = user_answers.get(q_key, "")
                selected_label = selected_raw[0] if selected_raw else ""
                is_correct     = (selected_label == q.correct_answer)

                if is_correct:
                    correct_count += 1
                    icon = "✅"
                else:
                    icon = "❌"

                with st.expander(
                    f"{icon} Q{i + 1}: {q.question_text[:80]}{'...' if len(q.question_text) > 80 else ''}",
                    expanded=not is_correct,
                ):
                    for opt in q.options:
                        if opt.label == q.correct_answer:
                            st.markdown(f"**✓ {opt.label}. {opt.text}** ← Correct")
                        elif opt.label == selected_label:
                            st.markdown(f"~~{opt.label}. {opt.text}~~ ← Your answer")
                        else:
                            st.markdown(f"{opt.label}. {opt.text}")
                    if q.explanation:
                        st.info(f"💡 **Explanation:** {q.explanation}")

            if total_mcq > 0:
                pct = int(correct_count / total_mcq * 100)
                st.metric("MCQ Score", f"{correct_count} / {total_mcq}  ({pct}%)")

        if short_questions:
            st.subheader("✍️ Short Answer — Model Answers")
            for i, q in enumerate(short_questions):
                with st.expander(f"SA{i + 1}: {q.question_text[:80]}..."):
                    user_text = user_answers.get(f"short_{i}", "")
                    if user_text:
                        st.markdown("**Your answer:**")
                        st.write(user_text)
                        st.divider()
                    if q.correct_answer:
                        st.markdown("**Model answer:**")
                        st.success(q.correct_answer)
                    if q.explanation:
                        st.info(f"💡 **Key points:** {q.explanation}")

        if st.button("🔁 Try Again / New Quiz", use_container_width=True):
            for key in ["current_quiz", "quiz_generated", "submitted", "user_answers", "quiz_subject"]:
                st.session_state.pop(key, None)
            st.rerun()
