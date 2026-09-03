"""
rag_retriever.py
================
Phase 10 — FAISS Vector Index + Retriever

WHAT THIS MODULE DOES
---------------------
1. Takes a list of text chunks (from chunker.py)
2. Converts each chunk into an embedding vector
3. Stores all vectors in a FAISS index (an in-memory vector database)
4. At query time: converts the query into a vector, then finds the
   top-K most similar chunks using cosine similarity

WHY FAISS?
----------
FAISS (Facebook AI Similarity Search) is a library designed to search
for similar vectors extremely fast — even across millions of vectors.

For QPredict's use case (a few hundred to a few thousand chunks from
uploaded PDFs), FAISS is much faster than a simple loop over all vectors,
and it requires no database server.

FAISS INDEX TYPE USED
---------------------
We use IndexFlatIP (Inner Product) with L2-normalised vectors.

Why?
  Cosine similarity(A, B) = dot_product(A, B)
  when both A and B are unit vectors (L2-normalised)

So we normalise all vectors to unit length, then use inner product
search, which gives us cosine similarity.

OFFLINE FALLBACK
----------------
If FAISS is not installed, the retriever falls back to a pure NumPy
cosine similarity search. Slower, but always works.

TF-IDF FALLBACK
---------------
If sentence-transformers is not installed, the retriever uses TF-IDF
vectors to embed the chunks. Less semantic power, but always works.

PERSISTENCE
-----------
The FAISS index can be saved to disk and reloaded, so we don't have
to re-embed all chunks every time the app restarts.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Embedding helper (reuses src/embeddings.py logic)
# ---------------------------------------------------------------------------

def _get_sentence_transformer():
    """
    Return a SentenceTransformer model, or None if not installed.
    Cached so the model is loaded only once per process.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def _get_embeddings(texts: list[str], tfidf_vectorizer=None):
    """
    Convert a list of texts into embedding vectors.

    Parameters
    ----------
    texts            : List of strings to embed
    tfidf_vectorizer : If provided (and sentence-transformers is not available),
                       use this pre-fitted TF-IDF vectorizer for transform-only
                       (no re-fitting). If None and TF-IDF fallback is needed,
                       a new vectorizer is fitted on `texts`.

    Returns a tuple: (np.ndarray of shape (n, dim), vectorizer_or_None)
    The vectorizer is returned so the caller can store it for later reuse.
    """
    st_model = _get_sentence_transformer()
    if st_model is not None:
        vecs = st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32), None   # no vectorizer needed

    # TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    if tfidf_vectorizer is not None:
        # Reuse the pre-fitted vocabulary — transform only, no re-fit
        matrix = tfidf_vectorizer.transform(texts)
    else:
        # First call — fit on these texts
        tfidf_vectorizer = TfidfVectorizer(max_features=512, sublinear_tf=True)
        matrix = tfidf_vectorizer.fit_transform(texts)

    return matrix.toarray().astype(np.float32), tfidf_vectorizer


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
    return vecs / norms


# ---------------------------------------------------------------------------
# RAGIndex class
# ---------------------------------------------------------------------------

class RAGIndex:
    """
    A vector index over document chunks.

    Usage
    -----
    # Build index from chunks
    index = RAGIndex()
    index.build(chunks)           # chunks from chunker.py

    # Search
    results = index.search("What is the OSI model?", top_k=4)
    for r in results:
        print(r["text"], r["score"], r["source_name"])

    # Save / load
    index.save("models/rag_index.pkl")
    index2 = RAGIndex.load("models/rag_index.pkl")
    """

    def __init__(self) -> None:
        self._chunks: list[dict] = []         # original chunk dicts
        self._vectors: Optional[np.ndarray] = None  # normalised embeddings
        self._faiss_index = None              # FAISS index object (or None)
        self._use_faiss: bool = False
        self._tfidf_vectorizer = None         # stored when TF-IDF fallback is used

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and build the search index.

        Parameters
        ----------
        chunks : List of chunk dicts from chunker.chunk_text()
        """
        if not chunks:
            self._chunks = []
            self._vectors = None
            return

        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # Embed all chunks; store the vectorizer so search() can reuse it
        raw_vecs, self._tfidf_vectorizer = _get_embeddings(texts)
        self._vectors = _normalise(raw_vecs)

        # Try to build FAISS index
        try:
            import faiss  # type: ignore
            dim = self._vectors.shape[1]
            idx = faiss.IndexFlatIP(dim)   # Inner Product (= cosine after normalisation)
            idx.add(self._vectors)
            self._faiss_index = idx
            self._use_faiss = True
        except ImportError:
            # FAISS not installed — will use NumPy fallback in search()
            self._faiss_index = None
            self._use_faiss = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top-K most relevant chunks for a query string.

        Parameters
        ----------
        query  : The student's question
        top_k  : How many chunks to return

        Returns
        -------
        List of chunk dicts (same format as input), each with an added
        "score" field (float, 0–1, higher = more relevant).
        Sorted by score descending.
        """
        if not self._chunks or self._vectors is None:
            return []

        # Embed the query using the same vectorizer that was used at build time
        q_vec, _ = _get_embeddings([query], tfidf_vectorizer=self._tfidf_vectorizer)
        q_vec = _normalise(q_vec)  # shape (1, dim)

        top_k = min(top_k, len(self._chunks))

        if self._use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec, top_k)
            scores = scores[0]     # shape (top_k,)
            indices = indices[0]   # shape (top_k,)
        else:
            # NumPy fallback: dot product (= cosine since normalised)
            scores_all = (self._vectors @ q_vec.T).flatten()
            indices = np.argsort(scores_all)[::-1][:top_k]
            scores = scores_all[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:        # FAISS returns -1 for padded results
                continue
            chunk = dict(self._chunks[idx])   # copy so we don't mutate
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the index to disk.

        Saves chunks + normalised vectors (+ FAISS index if available).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "chunks"          : self._chunks,
            "vectors"         : self._vectors,
            "use_faiss"       : self._use_faiss,
            "tfidf_vectorizer": self._tfidf_vectorizer,
        }
        if self._use_faiss and self._faiss_index is not None:
            try:
                import faiss
                import tempfile, io
                # Serialise FAISS index to bytes
                buf = io.BytesIO()
                faiss.write_index(self._faiss_index,
                                  faiss.PyCallbackIOWriter(buf.write))
                payload["faiss_bytes"] = buf.getvalue()
            except Exception:
                payload["faiss_bytes"] = None
        else:
            payload["faiss_bytes"] = None

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "RAGIndex":
        """
        Load a previously saved index from disk.

        Returns a new RAGIndex ready to search().
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        idx = cls()
        idx._chunks            = payload["chunks"]
        idx._vectors           = payload["vectors"]
        idx._use_faiss         = payload.get("use_faiss", False)
        idx._tfidf_vectorizer  = payload.get("tfidf_vectorizer", None)

        faiss_bytes = payload.get("faiss_bytes")
        if idx._use_faiss and faiss_bytes:
            try:
                import faiss, io
                buf = io.BytesIO(faiss_bytes)
                idx._faiss_index = faiss.read_index(
                    faiss.PyCallbackIOReader(buf.read)
                )
            except Exception:
                idx._use_faiss = False
                idx._faiss_index = None

        return idx

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        return self._vectors is not None and len(self._chunks) > 0

    @property
    def backend(self) -> str:
        """Returns "faiss" or "numpy" — useful for UI display."""
        return "faiss" if self._use_faiss else "numpy"
