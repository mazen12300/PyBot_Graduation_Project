"""
retriever.py
-------------
Loads the RAG index and returns the top-k most relevant knowledge-base
chunks for a given student question.

Retrieval strategy (auto-selected at import time):
  1. Dense embeddings (model/kb_embeddings.npy) if present — produced by
     colab_generate_embeddings.ipynb using a real sentence-embedding model.
     This gives better semantic matching (e.g. "how does a BST work" will
     match "binary search tree" even with no shared keywords).
  2. TF-IDF + cosine similarity (model/kb_tfidf_*.pkl) otherwise — works
     immediately with no GPU/internet, produced by build_retriever_index.py.
"""

import os

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

_kb_chunks = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_dense_embeddings = None
_mode = None


def _load():
    global _kb_chunks, _tfidf_vectorizer, _tfidf_matrix, _dense_embeddings, _mode

    kb_path = os.path.join(MODEL_DIR, "kb_chunks.pkl")
    if not os.path.exists(kb_path):
        _mode = "unavailable"
        return

    _kb_chunks = joblib.load(kb_path)

    embeddings_path = os.path.join(MODEL_DIR, "kb_embeddings.npy")
    if os.path.exists(embeddings_path):
        _dense_embeddings = np.load(embeddings_path)
        _mode = "dense"
        return

    _tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "kb_tfidf_vectorizer.pkl"))
    _tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "kb_tfidf_matrix.pkl"))
    _mode = "tfidf"


_load()


def retriever_mode() -> str:
    return _mode or "unavailable"


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05):
    """Returns a list of {title, subject, text, score} dicts, best first."""
    if _mode in (None, "unavailable"):
        return []

    if _mode == "dense":
        # Expects an embed_query() function to be wired in by the caller
        # (see app.py) since generating a live embedding needs whichever
        # model produced kb_embeddings.npy.
        raise RuntimeError(
            "Dense mode requires calling retrieve_with_embedding() instead."
        )

    query_vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        if sims[idx] < min_score:
            continue
        chunk = _kb_chunks[idx]
        results.append({
            "title": chunk["title"],
            "subject": chunk["subject"],
            "text": chunk["text"],
            "score": float(sims[idx]),
        })
    return results


def retrieve_with_embedding(query_embedding, top_k: int = 3, min_score: float = 0.3):
    """Use this variant when kb_embeddings.npy is present (dense mode)."""
    if _mode != "dense":
        raise RuntimeError("Dense embeddings not loaded — call retrieve() instead.")

    sims = cosine_similarity([query_embedding], _dense_embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        if sims[idx] < min_score:
            continue
        chunk = _kb_chunks[idx]
        results.append({
            "title": chunk["title"],
            "subject": chunk["subject"],
            "text": chunk["text"],
            "score": float(sims[idx]),
        })
    return results
