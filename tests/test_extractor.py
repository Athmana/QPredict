"""
test_extractor.py — Unit tests for question_extractor.py

These tests verify that:
  1. Individual helper functions (detect_section, detect_question_start,
     extract_marks, clean_question_text) work correctly.
  2. The full extract_questions() pipeline handles realistic exam text.

HOW TO RUN:
    cd qpredict
    python -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.question_extractor import (
    detect_section,
    detect_question_start,
    extract_marks,
    clean_question_text,
    is_noise_line,
    extract_questions,
    extraction_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# detect_section()
# ══════════════════════════════════════════════════════════════════════════════

def test_section_part_A():
    assert detect_section("PART A — Answer all questions") == "A"

def test_section_part_b_lowercase():
    assert detect_section("Part b") == "B"

def test_section_roman():
    assert detect_section("SECTION II") == "II"

def test_section_not_a_section():
    assert detect_section("Q1. Explain TCP/IP.") is None

def test_section_not_in_middle():
    """A section keyword buried in a question text should NOT trigger."""
    assert detect_section("Explain the different parts of the CPU.") is None


# ══════════════════════════════════════════════════════════════════════════════
# detect_question_start()
# ══════════════════════════════════════════════════════════════════════════════

def test_detect_numbered_dot():
    assert detect_question_start("1. What is TCP?") == "1"

def test_detect_numbered_paren():
    assert detect_question_start("2) Define bandwidth.") == "2"

def test_detect_Q_prefix():
    result = detect_question_start("Q3. Explain the OSI model.")
    assert result == "3"

def test_detect_sub_question():
    result = detect_question_start("5(a). Describe the three-way handshake.")
    assert result == "5(a)"

def test_detect_sub_question_space():
    result = detect_question_start("5 (a). Describe subnetting.")
    # spaces should be removed from the number
    assert result == "5(a)"

def test_detect_letter_sub():
    result = detect_question_start("(b) Explain CSMA/CD.")
    assert result == "(b)"

def test_detect_no_match():
    assert detect_question_start("Answer any FIVE questions.") is None

def test_detect_no_match_instruction():
    assert detect_question_start("Time allowed: 3 hours") is None


# ══════════════════════════════════════════════════════════════════════════════
# extract_marks()
# ══════════════════════════════════════════════════════════════════════════════

def test_marks_square_bracket():
    assert extract_marks("Explain the OSI model. [10 Marks]") == 10

def test_marks_round_bracket():
    assert extract_marks("Define TCP. (5 marks)") == 5

def test_marks_plain():
    assert extract_marks("Explain routing algorithms. 10 Marks") == 10

def test_marks_none():
    assert extract_marks("What is a protocol?") is None

def test_marks_ignores_unrealistic():
    """Values outside 1-100 should be rejected."""
    assert extract_marks("Question worth 0 marks") is None


# ══════════════════════════════════════════════════════════════════════════════
# clean_question_text()
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_removes_number_prefix():
    result = clean_question_text("1. Explain the OSI model.", "1")
    assert "Explain the OSI model" in result
    assert not result.startswith("1.")

def test_clean_removes_Q_prefix():
    result = clean_question_text("Q3. Define bandwidth.", "3")
    assert "Define bandwidth" in result
    assert "Q3" not in result

def test_clean_removes_marks():
    result = clean_question_text("Explain TCP. [10 Marks]", "1")
    assert "[10 Marks]" not in result
    assert "Explain TCP" in result

def test_clean_preserves_technical_terms():
    result = clean_question_text("2. Compare TCP and UDP protocols.", "2")
    assert "TCP" in result
    assert "UDP" in result


# ══════════════════════════════════════════════════════════════════════════════
# is_noise_line()
# ══════════════════════════════════════════════════════════════════════════════

def test_noise_time_allowed():
    assert is_noise_line("Time Allowed: 3 Hours") is True

def test_noise_max_marks():
    assert is_noise_line("Maximum Marks: 100") is True

def test_noise_answer_instruction():
    assert is_noise_line("Answer any five questions.") is True

def test_noise_not_a_question():
    assert is_noise_line("Explain the OSI model.") is False


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: extract_questions()
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_PAPER = """
UNIVERSITY EXAMINATION — 2024
SUBJECT: COMPUTER NETWORKS
Time Allowed: 3 Hours                      Maximum Marks: 100

Answer any FIVE questions.

PART A

1. What is the difference between TCP and UDP?
2. Define the term bandwidth.
3. What is a MAC address?

PART B

4. Explain the OSI reference model with a neat diagram. [10 Marks]
5 (a). Describe the TCP three-way handshake. [5 Marks]
5 (b). Explain flow control mechanisms in TCP. [5 Marks]
6. Compare circuit switching and packet switching. [10 Marks]
"""

def test_extract_finds_questions():
    questions = extract_questions(SAMPLE_PAPER, paper_id=1, year=2024)
    assert len(questions) >= 6

def test_extract_sections_detected():
    questions = extract_questions(SAMPLE_PAPER, paper_id=1, year=2024)
    sections = {q.section for q in questions}
    assert "A" in sections
    assert "B" in sections

def test_extract_marks_captured():
    questions = extract_questions(SAMPLE_PAPER, paper_id=1, year=2024)
    marked = [q for q in questions if q.marks is not None]
    assert len(marked) >= 3

def test_extract_year_set():
    questions = extract_questions(SAMPLE_PAPER, paper_id=1, year=2024)
    for q in questions:
        assert q.year == 2024

def test_extract_paper_id_set():
    questions = extract_questions(SAMPLE_PAPER, paper_id=99, year=2024)
    for q in questions:
        assert q.paper_id == 99

def test_extract_no_noise_in_questions():
    """Instruction lines must not appear as questions."""
    questions = extract_questions(SAMPLE_PAPER)
    texts = [q.question_text.lower() for q in questions]
    assert not any("time allowed" in t for t in texts)
    assert not any("maximum marks" in t for t in texts)
    assert not any("answer any five" in t for t in texts)

def test_extraction_summary():
    questions = extract_questions(SAMPLE_PAPER, paper_id=1, year=2024)
    summary = extraction_summary(questions)
    assert summary["total_questions"] == len(questions)
    assert "A" in summary["sections_found"]
    assert "B" in summary["sections_found"]

def test_empty_text_returns_empty_list():
    assert extract_questions("") == []

def test_no_questions_in_header_only():
    header = "UNIVERSITY OF TECHNOLOGY\nTime Allowed: 3 Hours\nMaximum Marks: 100\n"
    questions = extract_questions(header)
    assert len(questions) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
