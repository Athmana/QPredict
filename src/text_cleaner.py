"""
text_cleaner.py — Normalize question text before similarity comparison

WHY THIS FILE EXISTS:
Two questions can express the same concept in completely different words:

  "Explain the OSI reference model."
  "Describe the seven layers of the OSI architecture."

A machine comparing these character-by-character would say they're very
different. But if we normalize both first:

  "explain osi reference model"
  "describe seven layers osi architecture"

Now a TF-IDF model can detect that "OSI" appears in both and give them
a meaningful similarity score.

WHAT NORMALIZATION DOES:
  1. Lowercase — "TCP" and "tcp" are the same thing
  2. Expand common abbreviations — optional, helps match variants
  3. Remove punctuation — "model." and "model" are the same
  4. Remove stopwords — "the", "a", "of", "is" carry no meaning
  5. Collapse whitespace — tidy up

WHAT NORMALIZATION MUST NOT DO:
  - Remove technical terms like TCP, UDP, OSI, HTTP, SQL, CNN, RNN
  - Stem words aggressively (e.g. "routing" → "rout" loses meaning)
  - Remove numbers that are part of technical names (IPv4, 802.11)

WHY NOT USE NLTK/spaCy HERE?
For the Phase 3 baseline we deliberately keep dependencies minimal.
We use only Python's standard library + a small hand-curated stopword
list. spaCy will be an option in Phase 4 when we introduce embeddings.
"""

import re
import string
from typing import List


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD LIST
# ══════════════════════════════════════════════════════════════════════════════
# WHY A CUSTOM LIST:
# Generic NLP stopword lists (e.g. NLTK's 179-word list) would remove
# words like "not", "between", "difference" — which matter in exam
# questions. We use a conservative list of the most common function
# words that carry zero domain meaning in an exam context.

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "as", "if",
    "also", "any", "all", "each", "both", "more", "very",
    # Common exam instruction words that sometimes leak into question text
    "briefly", "clearly", "neatly", "suitable", "appropriate",
    "following", "given", "using", "write", "draw", "list",
    "note", "marks", "mark",
}

# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL TERM PROTECTION
# ══════════════════════════════════════════════════════════════════════════════
# These are uppercase technical acronyms that must survive normalization.
# We lowercase everything, but we preserve these by treating them as
# important tokens. The TF-IDF vectorizer will still see them as distinct
# terms from their lowercase variants.
#
# This list is illustrative — TF-IDF naturally handles technical terms
# well because they appear in few documents (high IDF weight).
PROTECTED_TERMS = {
    "tcp", "udp", "ip", "http", "https", "ftp", "smtp", "dns",
    "dhcp", "osi", "mac", "lan", "wan", "arp", "icmp", "ssl", "tls",
    "sql", "nosql", "html", "css", "xml", "json", "api", "rest",
    "cpu", "ram", "rom", "io", "os", "gui", "cli",
    "cnn", "rnn", "lstm", "gru", "gan", "bert", "gpt",
    "ipv4", "ipv6", "ieee",
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLEANING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_question(text: str, remove_stopwords: bool = True) -> str:
    """
    Normalize a question string for TF-IDF comparison.

    This is the primary public function of this module.

    Steps applied (in order):
      1. Lowercase
      2. Remove punctuation (keep hyphens in compound terms)
      3. Tokenize on whitespace
      4. Remove stopwords (if enabled)
      5. Remove tokens that are pure punctuation or empty
      6. Re-join into a single string

    Parameters
    ----------
    text              : str  — raw or cleaned question text
    remove_stopwords  : bool — set False to keep all words (useful for
                               debugging or when text is very short)

    Returns
    -------
    str — normalized text ready for TF-IDF

    Examples
    --------
    >>> normalize_question("Explain the OSI reference model. [10 Marks]")
    'explain osi reference model'

    >>> normalize_question("What is the difference between TCP and UDP?")
    'difference between tcp udp'
    """
    if not text or not text.strip():
        return ""

    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove marks annotations that may have survived extraction
    text = re.sub(r"\[\s*\d+\s*marks?\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*marks?\s*\)", " ", text, flags=re.IGNORECASE)

    # Step 3: replace hyphens/underscores between words with a space
    # so "three-way" becomes "three way" and is treated as two tokens
    text = text.replace("-", " ").replace("_", " ")

    # Step 4: remove all remaining punctuation except apostrophes
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    punct_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punct_to_remove))

    # Step 5: tokenize (split on whitespace)
    tokens = text.split()

    # Step 6: filter tokens
    clean_tokens = []
    for token in tokens:
        # Skip empty tokens
        if not token:
            continue
        # Skip pure-digit tokens (e.g. "10", "3") unless part of a
        # technical term — already lowercased so "ipv4" stays
        if token.isdigit():
            continue
        # Skip stopwords (but keep protected technical terms)
        if remove_stopwords and token in STOPWORDS and token not in PROTECTED_TERMS:
            continue
        # Skip very short tokens (single letters) unless they're meaningful
        # Keep "i/o", single-letter technical acronyms if needed
        if len(token) == 1:
            continue
        clean_tokens.append(token)

    return " ".join(clean_tokens)


def normalize_questions(texts: List[str], remove_stopwords: bool = True) -> List[str]:
    """
    Normalize a list of question strings.

    Convenience wrapper around normalize_question() for batch processing.

    Parameters
    ----------
    texts            : List[str] — list of raw question strings
    remove_stopwords : bool

    Returns
    -------
    List[str] — list of normalized strings, same order as input
    """
    return [normalize_question(t, remove_stopwords) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def is_too_short(normalized_text: str, min_tokens: int = 3) -> bool:
    """
    Return True if a normalized question has too few tokens to be meaningful.

    WHY: Very short questions (1–2 words after normalization) produce
    unreliable similarity scores. We flag them so the UI can warn the
    student rather than silently giving bad results.

    Example:
        "Define." → normalized → "" → too short → True
        "What is TCP?" → normalized → "tcp" → 1 token → too short → True
        "Explain OSI model layers." → 3 tokens → False
    """
    return len(normalized_text.split()) < min_tokens
