"""
question_extractor.py — Parse raw PDF text into structured Question objects

Supports the KTU (APJ Abdul Kalam Technological University) exam paper format:

  PART A — 10 plain question lines, no numbering prefix.
            Marks appear as "(3)" on the same line (at end) or on the next line.

  PART B — Module sub-sections. Question numbers (11, 12 …) and sub-part
            labels (a), b)) appear as standalone lines; the actual question
            text follows on the next non-empty line.

            Two layouts appear in practice:

            Layout 1 — number and sub-part fused on one line before the text:
              "11 a) Explain transposition cipher..."

            Layout 2 — number / sub-part on its own line, text on next line:
              "11"          ← standalone number line
              "a)"          ← standalone sub-part line
              "Explain..."  ← question text
              "(7)"         ← marks (separate line)

Also handles more traditional formats (Q1., 1., 1) …) so the extractor
works on non-KTU papers too.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Question:
    """One extracted question from an examination paper."""
    question_number: str = ""
    section: str = ""
    question_text: str = ""
    marks: Optional[int] = None
    raw_text: str = ""
    paper_id: Optional[int] = None
    year: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "question_number": self.question_number,
            "section":         self.section,
            "question_text":   self.question_text,
            "marks":           self.marks,
            "raw_text":        self.raw_text,
            "paper_id":        self.paper_id,
            "year":            self.year,
        }


# ══════════════════════════════════════════════════════════════════════════════
# COMPILED REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# Section headers — "PART A", "Part B", "SECTION I", "Module I", "Module II" …
SECTION_PATTERNS = [
    re.compile(r"^\s*PART\s+([A-Z])\b",           re.IGNORECASE),
    re.compile(r"^\s*SECTION\s+([IVXivx\d]+)\b",  re.IGNORECASE),
    re.compile(r"^\s*Module\s+([IVXivx\d]+)\b",   re.IGNORECASE),
]

# Traditional numbered question starts (non-KTU style)
# Each tuple: (pattern, group_index_for_number)
TRADITIONAL_Q_PATTERNS = [
    (re.compile(r"^\s*[Qq][\.\s]?\s*(\d+)\s*[\.\):]"),         1),  # Q1. Q1) Q.1
    (re.compile(r"^\s*(\d+\s*\([a-zA-Z]\))\s*[\.\)]?\s"),      1),  # 5(a). 5(a)
    (re.compile(r"^\s*(\d{1,2})\.\s"),                          1),  # 1.  2.
    (re.compile(r"^\s*(\d{1,2})\)\s"),                          1),  # 1)  2)
]

# KTU Layout 1 — number+subpart fused, followed by text on the same line
# e.g. "11 a) Explain transposition cipher…"
# e.g. "a) Describe subnetting…"
KTU_FUSED_PATTERN = re.compile(
    r"^\s*(\d{1,2}\s*[a-d]\)|[a-d]\))\s+(.+)", re.IGNORECASE
)

# KTU standalone sub-part label on its own line: "a)", "b)", "c)", "d)"
# (possibly preceded by a question number: "12 a)" but nothing after)
KTU_STANDALONE_SUBPART = re.compile(
    r"^\s*(\d{1,2}\s+)?([a-d])\)\s*$", re.IGNORECASE
)

# KTU standalone question number on its own line: "11", "12", … "20"
# (two-digit numbers that are plausible KTU question numbers 11-30)
KTU_STANDALONE_NUMBER = re.compile(r"^\s*([1-2]\d)\s*$")

# Marks patterns — find marks value anywhere in a text block
MARKS_PATTERNS = [
    re.compile(r"\[\s*(\d+)\s*[Mm]arks?\s*\]"),   # [10 Marks]
    re.compile(r"\(\s*(\d+)\s*[Mm]arks?\s*\)"),   # (10 Marks)
    re.compile(r"(\d+)\s*[Mm]arks?\b"),            # 10 Marks
    re.compile(r"\[\s*(\d+)\s*[Mm]\s*\]"),         # [10M]
    re.compile(r"^\s*\((\d{1,2})\)\s*$"),          # (7) or (14) alone on a line
    re.compile(r"\((\d{1,2})\)\s*$"),              # text ending with (7)
]

# Noise lines to skip
NOISE_PATTERNS = [
    re.compile(r"^\s*---\s*PAGE BREAK\s*---\s*$",  re.IGNORECASE),
    re.compile(r"time\s*allowed",                   re.IGNORECASE),
    re.compile(r"maximum\s*marks",                  re.IGNORECASE),
    re.compile(r"max\.\s*marks",                    re.IGNORECASE),
    re.compile(r"answer\s*(all|any)\s*(of\s*)?(the\s*)?\w*\s*question", re.IGNORECASE),
    re.compile(r"use\s*of\s*(calculator|tables)",   re.IGNORECASE),
    re.compile(r"roll\s*no",                        re.IGNORECASE),
    re.compile(r"registration\s*no",                re.IGNORECASE),
    re.compile(r"reg\s*no\s*[:\.]",                 re.IGNORECASE),
    re.compile(r"^\s*\*+\s*$"),
    re.compile(r"^\s*[-─═]+\s*$"),
    re.compile(r"^\s*invigilator",                  re.IGNORECASE),
    re.compile(r"^\s*examiner",                     re.IGNORECASE),
    re.compile(r"downloaded\s*from",                re.IGNORECASE),
    re.compile(r"ktunotes",                         re.IGNORECASE),
    re.compile(r"^\s*duration\s*:",                 re.IGNORECASE),
    re.compile(r"^\s*course\s*(code|name)\s*:",     re.IGNORECASE),
    re.compile(r"^\s*marks\s*$",                    re.IGNORECASE),
    re.compile(r"^\s*OR\s*$",                       re.IGNORECASE),
    # KTU barcode / exam hall ticket numbers (e.g. "1000CST433122204")
    re.compile(r"^\s*\d{4}[A-Z]{2,}[\dA-Z]+\s*$"),
    # Standalone Roman numeral lines that are module labels bleeding through
    re.compile(r"^\s*[IVXivx]{1,4}\s*$"),
    # Single uppercase letter line (stray OCR artefact)
    re.compile(r"^\s*[A-Z]\s*$"),
    # Standalone single-digit lines not caught by page-number filter
    re.compile(r"^\s*[a-zA-Z]?\d{1,2}\s*$"),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_questions(full_text: str, paper_id: int = None, year: int = None) -> List[Question]:
    """
    Parse raw PDF text and return a list of Question objects.

    Strategy
    --------
    Pass 1 — split text into logical line tokens, detect section headers and
             OR separators, group lines that belong to the same question.
    Pass 2 — walk the grouped lines, detect KTU and traditional question starts,
             flush completed questions.
    """
    lines = full_text.split("\n")
    questions: List[Question] = []
    current_section: str = ""

    # State for building current question
    current_q_number: str = ""
    current_q_lines: List[str] = []

    # Lookahead buffer — used to detect KTU Layout 2 (standalone number/subpart
    # followed by text on the next line)
    pending_number: str = ""   # a standalone number like "11" waiting for subpart/text
    pending_subpart: str = ""  # a standalone subpart like "a)" waiting for text

    def flush():
        nonlocal current_q_number, current_q_lines
        if not current_q_number and not current_q_lines:
            return
        raw = " ".join(l for l in current_q_lines if l).strip()
        if not raw:
            current_q_number = ""
            current_q_lines = []
            return
        marks = extract_marks_from_block(raw, current_q_lines)
        clean_text = clean_question_text(raw, current_q_number)
        if len(clean_text) >= 10:
            questions.append(Question(
                question_number=current_q_number,
                section=current_section,
                question_text=clean_text,
                marks=marks,
                raw_text=raw,
                paper_id=paper_id,
                year=year,
            ))
        current_q_number = ""
        current_q_lines = []

    def start_question(number: str, first_line: str):
        flush()
        nonlocal current_q_number, current_q_lines
        current_q_number = number
        current_q_lines = [first_line]

    # ── Part A counter (KTU: plain question lines in PART A) ──────────────────
    # We count question lines in Part A since they have no numeric prefix.
    in_part_a = False
    part_a_count = 0

    for line in lines:
        stripped = line.strip()

        # ── Empty lines ────────────────────────────────────────────────────────
        if not stripped:
            if current_q_lines:
                current_q_lines.append("")
            continue

        # ── Noise ──────────────────────────────────────────────────────────────
        if _is_noise(stripped):
            continue

        # ── Standalone marks line e.g. "(3)" or "(14)" ─────────────────────────
        # Attach to current question's lines so extract_marks_from_block finds it
        if re.match(r"^\s*\(\d{1,2}\)\s*$", stripped):
            if current_q_lines:
                current_q_lines.append(stripped)
            continue

        # ── Section header ─────────────────────────────────────────────────────
        section = _detect_section(stripped)
        if section:
            flush()
            # Reset Part A state
            in_part_a = (section.upper() == "A")
            part_a_count = 0
            pending_number = ""
            pending_subpart = ""
            current_section = section
            continue

        # ── "OR" separator — treat as section boundary (flush current question) ─
        if re.match(r"^\s*'?\"?OR\"?'?\s*$", stripped, re.IGNORECASE):
            flush()
            pending_number = ""
            pending_subpart = ""
            continue

        # ══════════════════════════════════════════════════════════════════════
        # PART B / non-Part-A  detection
        # ══════════════════════════════════════════════════════════════════════

        # ── KTU Layout 1: fused "11 a) text…" or "a) text…" ──────────────────
        fused = KTU_FUSED_PATTERN.match(stripped)
        if fused and not in_part_a:
            num_label = re.sub(r"\s+", "", fused.group(1))  # "11a)" or "a)"
            text_part  = fused.group(2).strip()
            # Absorb any pending standalone number into the label
            if pending_number and not num_label[0].isdigit():
                num_label = pending_number + num_label
            pending_number = ""
            pending_subpart = ""
            start_question(num_label, text_part)
            continue

        # ── KTU Layout 2 step 1: standalone sub-part "a)" ────────────────────
        standalone_sub = KTU_STANDALONE_SUBPART.match(stripped)
        if standalone_sub and not in_part_a:
            sub_letter = standalone_sub.group(2).lower()
            if pending_number:
                pending_subpart = pending_number + sub_letter + ")"
            else:
                pending_subpart = sub_letter + ")"
            pending_number = ""
            continue

        # ── KTU Layout 2 step 1 (alt): standalone question number "11" ───────
        standalone_num = KTU_STANDALONE_NUMBER.match(stripped)
        if standalone_num and not in_part_a:
            # Flush previous question, record pending number
            flush()
            pending_number = standalone_num.group(1)
            pending_subpart = ""
            continue

        # ── Traditional numbered question (non-KTU) ───────────────────────────
        trad_number = _detect_traditional_q(stripped)
        if trad_number:
            pending_number = ""
            pending_subpart = ""
            start_question(trad_number, stripped)
            continue

        # ══════════════════════════════════════════════════════════════════════
        # At this point: "stripped" is plain content — either:
        #   (a) A Part A question line  (in_part_a == True)
        #   (b) The text line that follows a KTU Layout 2 number/subpart
        #   (c) A continuation of the current question
        # ══════════════════════════════════════════════════════════════════════

        if pending_subpart:
            # This is the text line for a pending subpart
            start_question(pending_subpart, stripped)
            pending_subpart = ""
            pending_number = ""
            continue

        if pending_number:
            # A standalone number was seen but no subpart followed — treat the
            # number as a question number and this line as its text
            start_question(pending_number, stripped)
            pending_number = ""
            continue

        if in_part_a:
            # Plain line in Part A — each line is its own question
            # (skip very short lines that are probably labels/noise)
            if len(stripped) >= 15:
                part_a_count += 1
                start_question(str(part_a_count), stripped)
            continue

        # Continuation line — append to current question
        if current_q_number:
            current_q_lines.append(stripped)

    flush()
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_section(line: str) -> Optional[str]:
    for pat in SECTION_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group(1).upper()
    return None


def _detect_traditional_q(line: str) -> Optional[str]:
    for pat, grp in TRADITIONAL_Q_PATTERNS:
        m = pat.match(line)
        if m:
            number = m.group(grp).strip()
            return re.sub(r"\s+", "", number)
    return None


def _is_noise(line: str) -> bool:
    for pat in NOISE_PATTERNS:
        if pat.search(line):
            return True
    return False


def extract_marks_from_block(raw_text: str, lines: List[str]) -> Optional[int]:
    """
    Try to find a marks value.  Prefer the standalone "(N)" line pattern first
    (very common in KTU papers) then fall back to inline patterns.
    """
    # Look for a standalone "(N)" in the lines list
    for l in lines:
        m = re.match(r"^\s*\((\d{1,2})\)\s*$", l.strip())
        if m:
            v = int(m.group(1))
            if 1 <= v <= 100:
                return v

    # Inline marks patterns
    for pat in MARKS_PATTERNS:
        m = pat.search(raw_text)
        if m:
            try:
                v = int(m.group(1))
                if 1 <= v <= 100:
                    return v
            except (ValueError, IndexError):
                pass
    return None


def clean_question_text(raw_text: str, question_number: str) -> str:
    """Remove question-number prefix and marks annotations, normalize whitespace."""
    text = raw_text.strip()

    # Remove numeric/subpart prefix
    prefix_patterns = [
        re.compile(r"^\s*[Qq][\.\s]?\s*\d+\s*[\.\):\s]+"),
        re.compile(r"^\s*\d+\s*\([a-zA-Z]\)\s*[\.\)]?\s*"),
        re.compile(r"^\s*\([a-zA-Z]+\)\s*"),
        re.compile(r"^\s*\d{1,2}[\.\)]\s+"),
        re.compile(r"^\s*\d{1,2}\s{2,}"),
        re.compile(r"^\s*[a-d]\)\s+", re.IGNORECASE),
        re.compile(r"^\s*\d{1,2}[a-d]\)\s*", re.IGNORECASE),
    ]
    for pat in prefix_patterns:
        text = pat.sub("", text, count=1)

    # Remove marks annotations
    marks_cleanup = [
        re.compile(r"\[\s*\d+\s*[Mm]arks?\s*\]"),
        re.compile(r"\(\s*\d+\s*[Mm]arks?\s*\)"),
        re.compile(r"\[\s*\d+\s*[Mm]\s*\]"),
        re.compile(r"\s*\(\d{1,2}\)\s*$"),   # trailing (7) or (14)
    ]
    for pat in marks_cleanup:
        text = pat.sub("", text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.rstrip(".-,;:")
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Expose these so upload page can call detect_section / detect_question_start
# with the same names it used before.
detect_section = _detect_section
detect_question_start = _detect_traditional_q
is_noise_line = _is_noise
extract_marks = lambda text: extract_marks_from_block(text, [])


def extraction_summary(questions: List[Question]) -> dict:
    sections = list({q.section for q in questions if q.section})
    with_marks = [q for q in questions if q.marks is not None]
    total_marks = sum(q.marks for q in with_marks)
    return {
        "total_questions":      len(questions),
        "sections_found":       sorted(sections),
        "questions_with_marks": len(with_marks),
        "total_marks_on_paper": total_marks,
        "avg_marks":            round(total_marks / len(with_marks), 1) if with_marks else 0,
    }
