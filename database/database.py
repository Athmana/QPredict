"""
database.py — QPredict database layer

WHY THIS FILE EXISTS:
Every piece of data QPredict works with — papers, questions, clusters,
scores — needs to be stored somewhere reliable. This file creates and
manages a SQLite database.

SQLite is a file-based database. The entire database lives in one file:
  data/qpredict.db

You do not need to install or configure any database server.
Python includes SQLite support out of the box via the `sqlite3` module.

WHAT THIS FILE DOES:
1. Creates the database file if it doesn't exist.
2. Creates all required tables if they don't exist.
3. Provides simple functions to insert and query data.
"""

import sqlite3
import os
from datetime import datetime

# ──────────────────────────────────────────────
# DATABASE LOCATION
# ──────────────────────────────────────────────

# Build a path to the database file relative to this file's location.
# os.path.dirname(__file__)  →  the directory this file is in  (database/)
# os.path.join(..., "..")    →  one level up                   (qpredict/)
# os.path.join(..., "data")  →  the data folder                (qpredict/data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "qpredict.db")

# Ensure the data directory exists (needed on Streamlit Cloud where it isn't committed)
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)


def get_connection():
    """
    Open and return a connection to the SQLite database.

    WHY: Every database operation needs a connection. Rather than
    repeating the connection code everywhere, we centralise it here.

    The check_same_thread=False flag is needed because Streamlit can
    call functions from multiple threads. It's safe here because we
    open and close connections per-operation.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Return rows as dict-like objects so we can access columns by name
    # instead of by index. e.g. row["subject"] instead of row[2]
    conn.row_factory = sqlite3.Row
    return conn


# Alias so pages can do: from database.database import get_db_connection
get_db_connection = get_connection


def initialize_database():
    """
    Create all tables if they do not already exist.

    WHY: The first time QPredict runs, the database file doesn't exist.
    This function sets up the complete schema. Running it again is safe
    because of the 'IF NOT EXISTS' clause — existing data is never lost.

    TABLES CREATED IN THIS PHASE:
      papers      — one row per uploaded PDF
      questions   — one row per extracted question (Phase 2+)

    More tables (embeddings, clusters) will be added in later phases.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── papers table ──────────────────────────────────────────────────
    # Stores metadata about each uploaded PDF.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT    NOT NULL,
            subject          TEXT,
            year             INTEGER,
            exam_type        TEXT,
            semester         TEXT,
            upload_date      TEXT    NOT NULL,
            processing_status TEXT   NOT NULL DEFAULT 'pending'
        )
    """)

    # ── questions table ───────────────────────────────────────────────
    # Stores individual questions extracted from papers.
    # This table will be populated in Phase 2.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id         INTEGER NOT NULL,
            question_number  TEXT,
            section          TEXT,
            question_text    TEXT    NOT NULL,
            normalized_text  TEXT,
            marks            INTEGER,
            unit             TEXT,
            topic            TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at: {os.path.abspath(DB_PATH)}")


def insert_paper(filename, subject, year, exam_type=None, semester=None):
    """
    Save a new paper record and return its generated ID.

    WHY: When a user uploads a PDF we immediately create a database
    record for it. The returned ID is then used to associate extracted
    questions with this specific paper.

    Parameters
    ----------
    filename  : str   — original PDF filename
    subject   : str   — e.g. "Computer Networks"
    year      : int   — examination year, e.g. 2024
    exam_type : str   — e.g. "semester", "midterm"
    semester  : str   — e.g. "Semester 3"

    Returns
    -------
    int — the auto-generated paper ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO papers (filename, subject, year, exam_type, semester,
                            upload_date, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        subject,
        year,
        exam_type,
        semester,
        datetime.now().isoformat(),
        "uploaded"
    ))

    # lastrowid gives us the auto-generated primary key for this insert
    paper_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return paper_id


def get_all_papers():
    """
    Return all paper records as a list of dicts.

    WHY: The dashboard needs to display all uploaded papers. This
    function fetches every row from the papers table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY year DESC, upload_date DESC")
    rows = cursor.fetchall()
    conn.close()
    # Convert sqlite3.Row objects to plain dicts for easy use in the UI
    return [dict(row) for row in rows]


def update_paper_status(paper_id, status):
    """
    Update the processing_status of a paper.

    WHY: Processing a paper goes through stages:
      uploaded → extracting → extracted → error

    Tracking this lets the UI show progress and lets us resume
    partially-processed papers.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE papers SET processing_status = ? WHERE id = ?",
        (status, paper_id)
    )
    conn.commit()
    conn.close()


def save_questions(questions: list) -> int:
    """
    Insert a list of Question objects into the questions table.

    WHY: After the question extractor runs, we persist every question to
    the database so we can query them later (for similarity, clustering,
    scoring, etc.) without re-parsing the PDF each time.

    Parameters
    ----------
    questions : list of Question dataclass instances

    Returns
    -------
    int — the number of questions successfully inserted
    """
    if not questions:
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    from src.text_cleaner import normalize_question  # Phase 3: normalize on save

    count = 0
    for q in questions:
        normalized = normalize_question(q.question_text)
        cursor.execute("""
            INSERT INTO questions
                (paper_id, question_number, section, question_text,
                 normalized_text, marks, unit, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            q.paper_id,
            q.question_number,
            q.section,
            q.question_text,
            normalized,
            q.marks,
            q.unit if hasattr(q, "unit") else None,
            q.topic if hasattr(q, "topic") else None,
        ))
        count += 1

    conn.commit()
    conn.close()
    return count


def get_questions_for_paper(paper_id: int) -> list:
    """
    Return all questions belonging to a specific paper.

    Parameters
    ----------
    paper_id : int — the paper's database ID

    Returns
    -------
    list of dicts, one per question
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM questions WHERE paper_id = ? ORDER BY id",
        (paper_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_questions() -> list:
    """
    Return every question in the database, joined with its paper's year.

    WHY: The analysis modules (Phase 3+) need all questions at once to
    compute similarity and clusters across the entire question bank.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.*, p.year, p.subject
        FROM questions q
        JOIN papers p ON q.paper_id = p.id
        ORDER BY p.year, q.id
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_questions_for_subject(subject: str) -> list:
    """
    Return all questions for a specific subject, joined with paper year.

    WHY: The similarity analysis page lets the student pick a subject
    and analyze only those papers. This query supports that workflow.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.*, p.year, p.subject, p.filename
        FROM questions q
        JOIN papers p ON q.paper_id = p.id
        WHERE p.subject = ?
        ORDER BY p.year, q.id
    """, (subject,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_subjects() -> list:
    """Return a sorted list of distinct subject names in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT subject FROM papers WHERE subject IS NOT NULL ORDER BY subject")
    rows = cursor.fetchall()
    conn.close()
    return [row["subject"] for row in rows]


def delete_paper(paper_id: int) -> None:
    """Delete a paper and all its questions from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM questions WHERE paper_id = ?", (paper_id,))
    cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()
