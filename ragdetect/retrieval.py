"""Finding similar texts and letting them vote (lessons 3 and 4)."""

from __future__ import annotations

import faiss
import numpy as np


def _unit_float32(vectors) -> np.ndarray:
    """A float32 copy of `vectors` with every row scaled to length 1."""
    vectors = np.array(vectors, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(vectors)  # works in place, hence the copy above
    return vectors


class NeighbourIndex:
    """Exact nearest-neighbour search by cosine similarity, using FAISS.

    For unit-length vectors the dot product *is* the cosine similarity, so an
    inner-product index (IndexFlatIP) finds the most similar texts.
    """

    def __init__(self, embeddings):
        vectors = _unit_float32(embeddings)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries, k: int):
        """Return (similarities, positions), both of shape (n_queries, k), closest first."""
        k = min(k, len(self))
        return self.index.search(_unit_float32(queries), k)


def neighbour_vote(similarities, neighbour_labels, temperature: float | None = None) -> np.ndarray:
    """Turn the labels of each query's neighbours into P(machine).

    temperature=None  every neighbour gets one equal vote.
    temperature=t     neighbour weights are softmax(similarity / t). The smaller
                      t is, the more the closest neighbours dominate. t=1 is
                      what the original experiment used.
    """
    labels = np.asarray(neighbour_labels, dtype=np.float64)
    if temperature is None:
        return labels.mean(axis=1)
    scaled = np.asarray(similarities, dtype=np.float64) / temperature
    weights = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return (weights * labels).sum(axis=1) / weights.sum(axis=1)


class RetrievalDetector:
    """A detector that knows nothing except a pool of labelled examples.

    To judge a new text it finds the k most similar texts in the pool and
    returns their (weighted) share of machine-generated labels.
    """

    def __init__(self, k: int = 7, temperature: float | None = 1.0):
        self.k = k
        self.temperature = temperature

    def fit(self, embeddings, labels) -> "RetrievalDetector":
        self.index = NeighbourIndex(embeddings)
        self.labels = np.asarray(labels, dtype=np.int64)
        return self

    def neighbours(self, query_embeddings):
        """(similarities, positions) of each query's k nearest pool texts."""
        return self.index.search(query_embeddings, self.k)

    def predict_proba(self, query_embeddings) -> np.ndarray:
        similarities, positions = self.neighbours(query_embeddings)
        return neighbour_vote(similarities, self.labels[positions], self.temperature)
