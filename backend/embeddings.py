"""Semantic embeddings + cosine similarity.

This is the whole ML core of ProjectGuard: no keyword matching, no TF-IDF.
Text goes in, a 384-dim sentence embedding comes out, and duplicate detection
is entirely cosine similarity between those vectors.
"""

from functools import lru_cache

import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model():
    # Load the Sentence Transformer locally.
    # The model is loaded only when embeddings are actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """Convert text into a normalized semantic embedding."""

    text = (text or "").strip()

    if not text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vec = get_model().encode(
        text,
        normalize_embeddings=True
    )

    return vec.astype(np.float32)


def bytes_to_vec(b: bytes | None) -> np.ndarray:
    if not b:
        return np.zeros(
            EMBEDDING_DIM,
            dtype=np.float32
        )

    return np.frombuffer(
        b,
        dtype=np.float32
    )


def vec_to_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0.0 or nb == 0.0:
        return 0.0

    sim = float(
        np.dot(a, b) / (na * nb)
    )

    return max(
        0.0,
        min(1.0, sim)
    )


def similarity_to_percent(sim: float) -> float:
    """Convert cosine similarity into a displayed percentage."""

    return round(sim * 100, 1)


def category_for(pct: float) -> str:

    if pct >= 80:
        return "Very High"

    if pct >= 60:
        return "High"

    if pct >= 30:
        return "Moderate"

    return "Low"