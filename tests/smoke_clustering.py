"""Quick clustering smoke test — run directly: python tests/smoke_clustering.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clustering import (
    embeddings_to_distance_matrix, cluster_agglomerative,
    build_clusters, run_clustering, extract_topic_label,
)
from src.embeddings import embed_questions, cosine_similarity_matrix
import numpy as np

QUESTIONS = [
    {"question_text": "Explain the seven layers of the OSI reference model.", "year": 2021, "id": 201, "paper_id": 1},
    {"question_text": "Describe the OSI architecture and each layer function.", "year": 2022, "id": 202, "paper_id": 2},
    {"question_text": "What are the seven layers of the OSI model?",           "year": 2023, "id": 203, "paper_id": 3},
    {"question_text": "Explain Dijkstra algorithm for finding shortest path.", "year": 2021, "id": 204, "paper_id": 1},
    {"question_text": "Describe Bellman-Ford routing algorithms.",             "year": 2022, "id": 205, "paper_id": 2},
]
texts = [q["question_text"] for q in QUESTIONS]

embs, method = embed_questions(texts)
assert embs.shape[0] == len(texts), "wrong row count"
print(f"[OK] embed_questions: shape={embs.shape}, method={method}")

dist = embeddings_to_distance_matrix(embs)
assert dist.shape == (len(texts), len(texts))
assert all(abs(dist[i][i]) < 0.01 for i in range(len(texts)))
print("[OK] distance matrix")

labels = cluster_agglomerative(embs, distance_threshold=0.60)
assert len(labels) == len(texts)
print(f"[OK] agglomerative labels: {labels.tolist()}")

sim = cosine_similarity_matrix(embs)
clusters = build_clusters(QUESTIONS, embs, labels, sim)
print(f"[OK] build_clusters: {len(clusters)} clusters")
for c in clusters:
    assert c.representative_idx in c.member_indices
    print(f"     '{c.topic_label}': members={c.member_indices}, years={c.years}")

result = run_clustering(QUESTIONS, embs, distance_threshold=0.70)
assert result["clustered"] + result["unclustered"] == result["total_questions"]
print(f"[OK] run_clustering: {result['total_clusters']} clusters, {result['clustered']} grouped")

label, kws = extract_topic_label(texts[:3])
assert len(label) > 0
print(f"[OK] extract_topic_label: '{label}', keywords={kws}")

print("\nAll clustering smoke tests PASSED.")
