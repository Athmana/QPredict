"""
test_parser.py — Unit tests for pdf_parser.py

HOW TO RUN:
    cd qpredict
    python -m pytest tests/ -v

WHY TESTS?
Writing tests forces you to think about edge cases before they become
bugs in production. It also gives you a way to verify that your code
still works correctly after making changes.

In this file we test:
  1. clean_page_text() — our text cleaning function
  2. is_scanned_pdf()  — our scanned-PDF detector
  3. ParsedPDF         — our data structure
"""

import sys
import os

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pdf_parser import clean_page_text, is_scanned_pdf, ParsedPDF, PageContent


# ──────────────────────────────────────────────
# Tests for clean_page_text()
# ──────────────────────────────────────────────

def test_clean_removes_standalone_page_number():
    """A line that is just '1' or '42' should be removed."""
    raw = "Some question text\n42\nMore question text"
    result = clean_page_text(raw)
    assert "42" not in result.split("\n") or "42" in result.replace("\n42\n", "")
    # The clean text should not have a standalone '42' line
    lines = [l.strip() for l in result.split("\n")]
    assert "42" not in lines


def test_clean_removes_page_marker():
    """'Page 3 of 10' style markers should be removed."""
    raw = "PART A\nPage 3 of 10\nQ1. Explain TCP/IP"
    result = clean_page_text(raw)
    assert "Page 3 of 10" not in result
    assert "Explain TCP/IP" in result


def test_clean_preserves_question_text():
    """Important question content must not be removed."""
    raw = "Q1. Explain the OSI reference model. [10 Marks]"
    result = clean_page_text(raw)
    assert "OSI reference model" in result
    assert "10 Marks" in result


def test_clean_collapses_blank_lines():
    """Three or more consecutive blank lines should become at most two."""
    raw = "Question 1\n\n\n\n\nQuestion 2"
    result = clean_page_text(raw)
    # There should not be 3 consecutive newlines in the result
    assert "\n\n\n" not in result


def test_clean_empty_input():
    """Empty string input should return empty string."""
    assert clean_page_text("") == ""


def test_clean_preserves_technical_terms():
    """Technical terms like TCP, UDP, OSI must not be removed."""
    raw = "Explain the differences between TCP and UDP protocols."
    result = clean_page_text(raw)
    assert "TCP" in result
    assert "UDP" in result


# ──────────────────────────────────────────────
# Tests for is_scanned_pdf()
# ──────────────────────────────────────────────

def test_scanned_detection_empty_pages():
    """A PDF with empty pages should be detected as scanned."""
    pdf = ParsedPDF(filename="test.pdf", total_pages=3)
    pdf.pages = [
        PageContent(page_number=1, raw_text="", cleaned_text=""),
        PageContent(page_number=2, raw_text="", cleaned_text=""),
        PageContent(page_number=3, raw_text="", cleaned_text=""),
    ]
    assert is_scanned_pdf(pdf) is True


def test_scanned_detection_real_text():
    """A PDF with real question text should not be detected as scanned."""
    pdf = ParsedPDF(filename="test.pdf", total_pages=1)
    long_text = "Q1. Explain the seven layers of the OSI model in detail. " * 10
    pdf.pages = [
        PageContent(page_number=1, raw_text=long_text, cleaned_text=long_text)
    ]
    assert is_scanned_pdf(pdf) is False


def test_no_pages():
    """A ParsedPDF with no pages should be treated as scanned."""
    pdf = ParsedPDF(filename="empty.pdf")
    assert is_scanned_pdf(pdf) is True


# ──────────────────────────────────────────────
# Simple smoke test
# ──────────────────────────────────────────────

def test_parsed_pdf_defaults():
    """ParsedPDF should have sensible defaults."""
    pdf = ParsedPDF(filename="test.pdf")
    assert pdf.total_pages == 0
    assert pdf.pages == []
    assert pdf.full_text == ""
    assert pdf.error is None


if __name__ == "__main__":
    # Allow running this file directly: python tests/test_parser.py
    import pytest
    pytest.main([__file__, "-v"])
