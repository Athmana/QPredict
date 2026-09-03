"""
test_syllabus.py — Tests for syllabus_mapper.py

Tests verify:
  1. extract_syllabus_units() correctly parses unit headers and topics
  2. units_from_manual_input() parses the manual entry format
  3. embed_syllabus_units() returns correct shape
  4. map_clusters_to_units() assigns each cluster a unit
  5. build_unit_analyses() aggregates correctly
  6. run_syllabus_analysis() full pipeline
  7. Edge cases: empty text, no units, unmatched clusters

HOW TO RUN:
    cd qpredict
    python -m pytest tests/test_syllabus.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.syllabus_mapper import (
    SyllabusUnit,
    UnitMapping,
    UnitAnalysis,
    extract_syllabus_units,
    units_from_manual_input,
    embed_syllabus_units,
    map_clusters_to_units,
    build_unit_analyses,
    run_syllabus_analysis,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_SYLLABUS = """
COMPUTER NETWORKS
Semester 5

UNIT I — Introduction to Computer Networks
  What is a Network
  Network Topologies
  OSI Reference Model
  TCP/IP Protocol Suite

UNIT II — Data Link Layer
  Framing
  Error Detection and Correction
  Flow Control
  Sliding Window Protocols

UNIT III — Network Layer
  Routing Algorithms
  Dijkstra's Algorithm
  Bellman-Ford Algorithm
  Congestion Control

UNIT IV — Transport Layer
  TCP and UDP
  Connection Management
  Flow Control
  Error Control

UNIT V — Application Layer
  DNS
  HTTP and HTTPS
  Email Protocols
  FTP
"""

MANUAL_INPUT = """
Unit 1: Introduction
  OSI Model
  TCP/IP
Unit 2: Data Link Layer
  Framing
  Error Detection
Unit 3: Network Layer
  Routing
  Dijkstra
"""


def _make_mock_scored_clusters():
    """Build minimal ScoredCluster-like objects for mapping tests."""
    from src.clustering import QuestionCluster
    from src.scoring import ScoredCluster, ScoreBreakdown

    questions = [
        {"id": 1, "question_text": "Explain the OSI reference model layers.", "year": 2021, "paper_id": 1},
        {"id": 2, "question_text": "Describe the OSI architecture.",           "year": 2022, "paper_id": 2},
        {"id": 3, "question_text": "Explain Dijkstra routing algorithm.",      "year": 2021, "paper_id": 1},
        {"id": 4, "question_text": "Describe Bellman-Ford routing.",           "year": 2022, "paper_id": 2},
        {"id": 5, "question_text": "Explain TCP three-way handshake.",         "year": 2023, "paper_id": 3},
    ]
    scored = [
        ScoredCluster(
            cluster_id=0, topic_label="OSI Model",
            representative_text="Explain the OSI reference model layers.",
            member_indices=[0, 1], years=[2021, 2022], paper_count=2,
            total_appearances=2, keywords=["osi", "model"],
            score=ScoreBreakdown(priority_score=80.0),
        ),
        ScoredCluster(
            cluster_id=1, topic_label="Routing Algorithms",
            representative_text="Explain Dijkstra routing algorithm.",
            member_indices=[2, 3], years=[2021, 2022], paper_count=2,
            total_appearances=2, keywords=["routing", "dijkstra"],
            score=ScoreBreakdown(priority_score=70.0),
        ),
        ScoredCluster(
            cluster_id=2, topic_label="TCP Connection",
            representative_text="Explain TCP three-way handshake.",
            member_indices=[4], years=[2023], paper_count=1,
            total_appearances=1, keywords=["tcp", "connection"],
            score=ScoreBreakdown(priority_score=50.0),
        ),
    ]
    return scored, questions


# ══════════════════════════════════════════════════════════════════════════════
# SyllabusUnit dataclass
# ══════════════════════════════════════════════════════════════════════════════

def test_syllabusunit_full_label():
    u = SyllabusUnit(unit_number="1", unit_name="Introduction")
    assert "1" in u.full_label
    assert "Introduction" in u.full_label

def test_syllabusunit_all_text():
    u = SyllabusUnit(unit_number="1", unit_name="Networks", topics=["OSI", "TCP/IP"])
    assert "Networks" in u.all_text
    assert "OSI"      in u.all_text
    assert "TCP/IP"   in u.all_text

def test_syllabusunit_to_dict():
    u = SyllabusUnit(unit_number="2", unit_name="Data Link", topics=["Framing"])
    d = u.to_dict()
    assert d["unit_number"] == "2"
    assert d["unit_name"]   == "Data Link"
    assert "Framing" in d["topics"]


# ══════════════════════════════════════════════════════════════════════════════
# extract_syllabus_units()
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_finds_units():
    units = extract_syllabus_units(SAMPLE_SYLLABUS)
    assert len(units) >= 3

def test_extract_unit_numbers_present():
    units = extract_syllabus_units(SAMPLE_SYLLABUS)
    numbers = {u.unit_number.upper() for u in units}
    assert "I" in numbers or "1" in numbers or "II" in numbers

def test_extract_topics_collected():
    units = extract_syllabus_units(SAMPLE_SYLLABUS)
    all_topics = [t for u in units for t in u.topics]
    assert len(all_topics) > 0

def test_extract_osi_in_topics():
    units = extract_syllabus_units(SAMPLE_SYLLABUS)
    all_topics = " ".join(t.lower() for u in units for t in u.topics)
    assert "osi" in all_topics or "network" in all_topics

def test_extract_empty_text():
    units = extract_syllabus_units("")
    assert units == []

def test_extract_no_units_text():
    text = "This is just a paragraph without any unit headers."
    units = extract_syllabus_units(text)
    assert isinstance(units, list)  # should return list, not raise


# ══════════════════════════════════════════════════════════════════════════════
# units_from_manual_input()
# ══════════════════════════════════════════════════════════════════════════════

def test_manual_parses_units():
    units = units_from_manual_input(MANUAL_INPUT)
    assert len(units) == 3

def test_manual_unit_numbers():
    units = units_from_manual_input(MANUAL_INPUT)
    numbers = {u.unit_number for u in units}
    assert "1" in numbers
    assert "2" in numbers
    assert "3" in numbers

def test_manual_unit_names():
    units = units_from_manual_input(MANUAL_INPUT)
    names = [u.unit_name for u in units]
    assert any("introduction" in n.lower() for n in names)

def test_manual_topics_parsed():
    units = units_from_manual_input(MANUAL_INPUT)
    u1 = next(u for u in units if u.unit_number == "1")
    assert len(u1.topics) >= 2
    topics_lower = [t.lower() for t in u1.topics]
    assert any("osi" in t for t in topics_lower)

def test_manual_empty_input():
    units = units_from_manual_input("")
    assert units == []


# ══════════════════════════════════════════════════════════════════════════════
# embed_syllabus_units()
# ══════════════════════════════════════════════════════════════════════════════

def test_embed_units_shape():
    units = [
        SyllabusUnit("1", "Introduction", ["OSI Model", "TCP/IP"]),
        SyllabusUnit("2", "Data Link",    ["Framing", "Error Detection"]),
    ]
    embs = embed_syllabus_units(units)
    assert embs.shape[0] == 2
    assert embs.shape[1] > 0

def test_embed_units_empty():
    embs = embed_syllabus_units([])
    assert embs.size == 0

def test_embed_units_dtype():
    units = [SyllabusUnit("1", "Introduction", ["OSI"])]
    embs = embed_syllabus_units(units)
    assert embs.dtype in (np.float32, np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# map_clusters_to_units()
# ══════════════════════════════════════════════════════════════════════════════

def test_map_returns_one_per_cluster():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)
    assert len(mappings) == len(scored)

def test_map_valid_unit_indices():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)
    for m in mappings:
        assert 0 <= m.assigned_unit_idx < len(units)

def test_map_similarity_in_range():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)
    for m in mappings:
        assert -1.0 <= m.similarity_score <= 1.0

def test_map_empty_units():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)
    mappings = map_clusters_to_units(scored, questions, [], np.array([]), cluster_embs)
    assert mappings == []


# ══════════════════════════════════════════════════════════════════════════════
# build_unit_analyses()
# ══════════════════════════════════════════════════════════════════════════════

def test_build_unit_analyses_count():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)
    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)

    analyses = build_unit_analyses(units, scored, mappings)
    assert len(analyses) == len(units)

def test_build_unit_analyses_sorted():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)
    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)

    analyses = build_unit_analyses(units, scored, mappings)
    for i in range(len(analyses) - 1):
        assert analyses[i].avg_priority_score >= analyses[i + 1].avg_priority_score

def test_build_all_clusters_assigned():
    """Total clusters across all units must equal total clusters."""
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    units = units_from_manual_input(MANUAL_INPUT)
    unit_embs = embed_syllabus_units(units)
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)
    mappings = map_clusters_to_units(scored, questions, units, unit_embs, cluster_embs)

    analyses = build_unit_analyses(units, scored, mappings)
    total_mapped = sum(a.cluster_count for a in analyses)
    assert total_mapped == len(scored)


# ══════════════════════════════════════════════════════════════════════════════
# run_syllabus_analysis() — full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_with_manual():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    result = run_syllabus_analysis(
        syllabus_text="",
        scored_clusters=scored,
        questions=questions,
        cluster_embeddings=cluster_embs,
        manual_units_text=MANUAL_INPUT,
    )
    assert result["total_units"] == 3
    assert result["parse_method"] == "manual"
    assert len(result["unit_analyses"]) == 3
    assert len(result["mappings"]) == len(scored)

def test_full_pipeline_with_pdf_text():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    result = run_syllabus_analysis(
        syllabus_text=SAMPLE_SYLLABUS,
        scored_clusters=scored,
        questions=questions,
        cluster_embeddings=cluster_embs,
    )
    assert result["total_units"] >= 3
    assert result["parse_method"] in ("pdf", "manual")
    assert len(result["mappings"]) == len(scored)

def test_full_pipeline_no_input():
    from src.embeddings import embed_questions
    scored, questions = _make_mock_scored_clusters()
    texts = [q["question_text"] for q in questions]
    cluster_embs, _ = embed_questions(texts)

    result = run_syllabus_analysis(
        syllabus_text="",
        scored_clusters=scored,
        questions=questions,
        cluster_embeddings=cluster_embs,
        manual_units_text="",
    )
    assert result["total_units"] == 0
    assert result["parse_method"] == "none"

def test_full_pipeline_empty_clusters():
    result = run_syllabus_analysis(
        syllabus_text=SAMPLE_SYLLABUS,
        scored_clusters=[],
        questions=[],
        cluster_embeddings=np.array([]),
        manual_units_text="",
    )
    assert len(result["mappings"]) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
