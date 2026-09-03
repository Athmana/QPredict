"""
4_🔍_Deep_Analysis.py — Similarity, semantic, and cluster explorer

Advanced analysis tools for students who want to understand the
underlying AI analysis — TF-IDF vs embeddings, nearest neighbours,
raw cluster browser.
"""

import streamlit as st
import sys, os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database.database import initialize_database, get_all_subjects, get_questions_for_subject
from src.similarity import run_tfidf_similarity_analysis, DEFAULT_THRESHOLD
from src.embeddings import (
    run_embedding_similarity_analysis, get_or_compute_embeddings,
    is_available as embeddings_available, DEFAULT_MODEL,
)
from src.clustering import run_clustering

st.set_page_config(page_title="Deep Analysis — QPredict", page_icon="🔍", layout="wide")
initialize_database()

st.header("🔍 Deep Analysis")
st.write("Explore the raw similarity scores, semantic matches, and clustering details behind the dashboard.")

subjects = get_all_subjects()
if not subjects:
    st.info("No papers uploaded yet.")
    st.stop()

subject = st.selectbox("Select subject", subjects)
questions = get_questions_for_subject(subject)

if len(questions) < 2:
    st.info(f"Only {len(questions)} question(s) for '{subject}'. Upload more papers.")
    st.stop()

st.write(f"**{len(questions)} questions** loaded.")

tab_sim, tab_sem, tab_clust = st.tabs([
    "📊 TF-IDF Similarity",
    "🧠 Semantic Similarity",
    "📦 Cluster Explorer",
])

# ── TF-IDF tab ────────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("TF-IDF + Cosine Similarity (Phase 3)")
    threshold = st.slider("Threshold", 0.30, 0.99, DEFAULT_THRESHOLD, 0.05, key="tfidf_thresh")
    if st.button("Run TF-IDF Analysis", key="run_tfidf"):
        with st.spinner("Computing TF-IDF similarity…"):
            results = run_tfidf_similarity_analysis(questions, threshold=threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pairs found",  results["total_pairs"])
        c2.metric("Groups found", results["total_groups"])
        c3.metric("Threshold",    f"{threshold:.0%}")

        if results["total_pairs"] == 0:
            st.info("No pairs above this threshold. Try lowering it.")
        else:
            pairs = results["pairs"]
            cross  = [p for p in pairs if p.is_cross_year()]
            st.success(f"Found {len(cross)} cross-year pairs.")

            rows = []
            for p in pairs[:100]:
                rows.append({
                    "Year A":     p.year_a or "?",
                    "Question A": p.question_a[:70] + ("…" if len(p.question_a) > 70 else ""),
                    "Year B":     p.year_b or "?",
                    "Question B": p.question_b[:70] + ("…" if len(p.question_b) > 70 else ""),
                    "Similarity": f"{p.similarity:.0%}",
                    "Cross-year": "✓" if p.is_cross_year() else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Semantic tab ──────────────────────────────────────────────────────────────
with tab_sem:
    st.subheader("Sentence Embeddings (Phase 4)")
    if embeddings_available():
        st.success(f"🟢 Using model `{DEFAULT_MODEL}`")
    else:
        st.warning("🟡 sentence-transformers not installed — using TF-IDF fallback")

    sem_thresh = st.slider("Semantic threshold", 0.40, 0.99, 0.70, 0.05, key="sem_thresh")

    col_run, col_cmp = st.columns(2)
    run_sem = col_run.button("Run Semantic Analysis", key="run_sem")
    run_cmp = col_cmp.button("Compare: TF-IDF vs Embeddings", key="run_cmp")

    if run_sem or run_cmp:
        with st.spinner("Generating embeddings…"):
            sem_results = run_embedding_similarity_analysis(questions, threshold=sem_thresh)

        method = sem_results["method"]
        if "embedding" in method:
            st.info(f"✅ Used sentence embeddings (`{sem_results['model_name']}`)")
        else:
            st.warning("⚠️ Used TF-IDF fallback")

        c1, c2 = st.columns(2)
        c1.metric("Semantic pairs", sem_results["total_pairs"])
        c2.metric("Groups",         sem_results["total_groups"])

        if run_cmp:
            with st.spinner("Running TF-IDF for comparison…"):
                tfidf_r = run_tfidf_similarity_analysis(questions, threshold=sem_thresh)

            st.subheader("Comparison")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**TF-IDF**")
                st.metric("Pairs", tfidf_r["total_pairs"])
                st.metric("Groups", tfidf_r["total_groups"])
            with cc2:
                st.markdown("**Embeddings**")
                st.metric("Pairs", sem_results["total_pairs"])
                st.metric("Groups", sem_results["total_groups"])

            emb_keys   = {(min(p.idx_a, p.idx_b), max(p.idx_a, p.idx_b)) for p in sem_results["pairs"]}
            tfidf_keys = {(min(p.idx_a, p.idx_b), max(p.idx_a, p.idx_b)) for p in tfidf_r["pairs"]}
            only_emb   = emb_keys - tfidf_keys
            st.metric("Pairs only embeddings found", len(only_emb),
                      help="Questions with different words but same meaning — missed by TF-IDF.")

        # Nearest neighbours
        st.divider()
        st.subheader("🔍 Nearest Neighbours")
        q_opts = {f"[{q.get('year','?')}] {q['question_text'][:70]}": i for i, q in enumerate(questions)}
        chosen = st.selectbox("Select a question", list(q_opts.keys()), key="nn_select")
        cidx   = q_opts[chosen]
        sim_m  = sem_results["sim_matrix"]
        if sim_m.size > 0:
            scores = sorted([(j, float(sim_m[cidx][j])) for j in range(len(questions)) if j != cidx],
                            key=lambda x: x[1], reverse=True)[:5]
            for rank, (j, score) in enumerate(scores, 1):
                q = questions[j]
                st.markdown(f"**{rank}.** `{score:.0%}` — [{q.get('year','?')}] {q['question_text']}")

# ── Cluster explorer tab ──────────────────────────────────────────────────────
with tab_clust:
    st.subheader("Cluster Explorer (Phase 5)")
    c_algo   = st.selectbox("Algorithm", ["agglomerative", "dbscan"], key="c_algo")
    c_thresh = st.slider("Distance threshold", 0.10, 0.70, 0.35, 0.05, key="c_thresh")

    if st.button("Run Clustering", key="run_clust"):
        with st.spinner("Generating embeddings…"):
            embeddings, emb_m = get_or_compute_embeddings(questions)
        with st.spinner("Clustering…"):
            cr = run_clustering(questions, embeddings, algorithm=c_algo, distance_threshold=c_thresh)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Clusters",   cr["total_clusters"])
        cc2.metric("Grouped Qs", cr["clustered"])
        cc3.metric("Unique Qs",  cr["unclustered"])

        sim_m = cr["sim_matrix"]
        for cluster in cr["clusters"]:
            rep = questions[cluster.representative_idx]
            years = sorted({questions[i].get("year") for i in cluster.member_indices if questions[i].get("year")})
            with st.expander(
                f"**{cluster.topic_label}**  ·  {cluster.total_appearances} questions  ·  {', '.join(str(y) for y in years)}",
                expanded=False
            ):
                st.info(f'Representative: "{rep["question_text"]}"')
                st.caption(f"Keywords: {', '.join(cluster.keywords[:4])}")
                for idx in cluster.member_indices:
                    q  = questions[idx]
                    yr = f"**{q.get('year','?')}**"
                    sim = float(sim_m[idx][cluster.representative_idx]) if sim_m.size > 0 else 0.0
                    badge = "`rep`" if idx == cluster.representative_idx else f"`{sim:.0%}`"
                    st.markdown(f"- {yr} — {q['question_text']}  {badge}")
