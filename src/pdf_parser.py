"""
pdf_parser.py — PDF text extraction for QPredict

WHY THIS FILE EXISTS:
Uploaded examination papers arrive as PDF files. PDFs are not plain
text — they're a structured binary format. Before QPredict can analyze
any questions, it must first convert PDFs into readable text.

This file contains all logic for:
  1. Receiving a PDF file path
  2. Opening it with PyMuPDF
  3. Extracting text page by page
  4. Doing light cleaning (removing obvious noise like page numbers)
  5. Returning structured output

WHY PyMuPDF?
PyMuPDF (imported as `fitz`) is the fastest and most reliable Python
library for PDF text extraction. It handles:
  - Multi-column layouts
  - Mixed font sizes
  - Tables (partially)
  - Standard text-based PDFs

LIMITATION: This module handles text-based PDFs only.
Scanned PDFs (images of paper) require OCR — that's a future module.

HOW TO THINK ABOUT PAGES:
A PDF is a sequence of pages. Each page has a visual layout.
PyMuPDF extracts text from each page in reading order (top to bottom,
left to right for most Western exam papers).
"""

import fitz  # PyMuPDF — 'fitz' is the internal module name
import os
import re
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class PageContent:
    """
    Represents the extracted content of a single PDF page.

    WHY A DATACLASS:
    Python dataclasses are a clean way to define structured data.
    Instead of returning a plain tuple like (1, "text"), we return an
    object where the meaning of each field is explicit.
    """
    page_number: int        # 1-based (humans count from 1, not 0)
    raw_text: str           # Text exactly as extracted from the PDF
    cleaned_text: str       # Text after removing obvious noise


@dataclass
class ParsedPDF:
    """
    Represents the complete extraction result for one PDF file.

    Attributes
    ----------
    filename      : original PDF filename
    total_pages   : how many pages were found
    pages         : list of PageContent objects, one per page
    full_text     : all pages joined into one string (for quick access)
    error         : if something went wrong, the error message lives here
    """
    filename: str
    total_pages: int = 0
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""
    error: str = None


# ──────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> ParsedPDF:
    """
    Open a PDF file and extract all text from it.

    This is the primary public function of this module. Call it with a
    path to a PDF and receive a ParsedPDF object.

    Parameters
    ----------
    pdf_path : str — absolute or relative path to the PDF file

    Returns
    -------
    ParsedPDF — structured extraction result

    HOW IT WORKS:
      1. fitz.open() loads the PDF into memory
      2. We loop over each page
      3. page.get_text("text") extracts text in reading order
      4. We clean each page's text
      5. We assemble everything into a ParsedPDF object

    WHAT CAN GO WRONG:
      - File doesn't exist → FileNotFoundError
      - File is corrupted → fitz raises an exception
      - File is a scanned image PDF → text will be empty or minimal
      - Password-protected PDF → fitz raises an exception
    We catch all of these and store the error in ParsedPDF.error so
    the application can show a user-friendly message.
    """
    result = ParsedPDF(filename=os.path.basename(pdf_path))

    if not os.path.exists(pdf_path):
        result.error = f"File not found: {pdf_path}"
        return result

    try:
        # fitz.open() is the PyMuPDF way to open a PDF
        doc = fitz.open(pdf_path)
        result.total_pages = len(doc)

        all_text_parts = []

        for page_index in range(len(doc)):
            page = doc[page_index]

            # get_text("text") extracts plain text in reading order.
            # Other modes exist: "blocks", "words", "html", "dict"
            # We use "text" for simplicity at this stage.
            raw_text = page.get_text("text")

            # Clean the raw text for this page
            cleaned = clean_page_text(raw_text)

            page_content = PageContent(
                page_number=page_index + 1,  # convert 0-based to 1-based
                raw_text=raw_text,
                cleaned_text=cleaned
            )
            result.pages.append(page_content)
            all_text_parts.append(cleaned)

        # Join all pages into one string, separating pages with a marker
        result.full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text_parts)
        doc.close()

    except fitz.FileDataError:
        result.error = "This PDF appears to be corrupted or password-protected."
    except Exception as e:
        result.error = f"Unexpected error while reading PDF: {str(e)}"

    return result


# ──────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────

def clean_page_text(raw_text: str) -> str:
    """
    Apply light cleaning to raw extracted text from one page.

    WHY CLEANING IS NEEDED:
    PDFs for examination papers typically contain:
      - University name headers (repeated on every page)
      - Exam date/time instructions (same on all pages)
      - Page numbers like "Page 1 of 4" or just "1"
      - Horizontal rules like "──────────────────"
      - Blank lines or excessive whitespace

    These are noise — they don't contain questions and they can confuse
    the question parser we build in Phase 2.

    WHAT WE CLEAN HERE (Phase 1 — conservative):
      - Excessive blank lines (3+ → 2)
      - Leading/trailing whitespace per line
      - Standalone single-digit page numbers
      - "Page X of Y" style markers

    WHAT WE DO NOT CLEAN HERE:
    We avoid aggressive cleaning at this stage because we might
    accidentally remove important question text. The question extractor
    in Phase 2 will do more targeted cleaning.

    Parameters
    ----------
    raw_text : str — text as extracted directly from one page

    Returns
    -------
    str — lightly cleaned text
    """
    if not raw_text:
        return ""

    lines = raw_text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Strip leading and trailing whitespace from each line
        stripped = line.strip()

        # Skip lines that are just standalone page numbers
        # Pattern: line contains only a number (possibly with whitespace)
        if re.match(r"^\d{1,3}$", stripped):
            continue

        # Skip "Page X of Y" or "Page X" lines
        if re.match(r"^[Pp]age\s+\d+(\s+of\s+\d+)?$", stripped):
            continue

        cleaned_lines.append(stripped)

    # Re-join the lines
    cleaned_text = "\n".join(cleaned_lines)

    # Collapse runs of 3 or more blank lines down to 2 blank lines
    # re.sub pattern: \n{3,} means "three or more newlines in a row"
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def is_scanned_pdf(parsed_pdf: ParsedPDF, min_chars_per_page: int = 50) -> bool:
    """
    Detect whether a PDF is likely a scanned image rather than text-based.

    WHY THIS MATTERS:
    If a PDF is a scanned image, PyMuPDF will return very little or no
    text. We need to detect this so we can show the user an informative
    message rather than displaying empty results.

    HOW WE DETECT IT:
    If the average number of characters per page is below a threshold,
    it's very likely the PDF is scanned. A real text page typically has
    hundreds of characters.

    Parameters
    ----------
    parsed_pdf        : ParsedPDF — the extraction result to check
    min_chars_per_page: int       — threshold (default 50)

    Returns
    -------
    bool — True if the PDF appears to be scanned
    """
    if not parsed_pdf.pages:
        return True

    total_chars = sum(len(p.cleaned_text) for p in parsed_pdf.pages)
    avg_chars = total_chars / len(parsed_pdf.pages)
    return avg_chars < min_chars_per_page


def save_extracted_text(parsed_pdf: ParsedPDF, output_dir: str) -> str:
    """
    Save the full extracted text to a .txt file in output_dir.

    WHY: Having the extracted text as a separate file is useful for:
      - Debugging (you can open the .txt file and inspect it)
      - Reprocessing without re-parsing the PDF
      - Quick text searches

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(parsed_pdf.filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_extracted.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== Extracted from: {parsed_pdf.filename} ===\n")
        f.write(f"Total pages: {parsed_pdf.total_pages}\n\n")
        f.write(parsed_pdf.full_text)

    return output_path
