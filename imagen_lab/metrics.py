from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .embeddings import EmbeddingCache


def _stack_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for emb in embeddings:
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        vectors.append(arr)
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0)


def intra_batch_distance(embeddings: Iterable[np.ndarray]) -> Dict[str, object]:
    matrix = _stack_embeddings(embeddings)
    n = matrix.shape[0]
    if n == 0:
        return {
            "available": False,
            "pairwise_min": None,
            "pairwise_mean": None,
            "per_item_min": [],
        }
    if n == 1:
        return {
            "available": True,
            "pairwise_min": 0.0,
            "pairwise_mean": 0.0,
            "per_item_min": [0.0],
        }

    sims = matrix @ matrix.T
    dists = 1.0 - np.clip(sims, -1.0, 1.0)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise = dists[mask]
    per_item = dists + np.eye(n, dtype=np.float32) * 2.0
    per_item_min = per_item.min(axis=1)

    return {
        "available": True,
        "pairwise_min": float(pairwise.min(initial=0.0)),
        "pairwise_mean": float(pairwise.mean() if pairwise.size else 0.0),
        "per_item_min": per_item_min.astype(np.float32).tolist(),
    }


def history_distance(
    batch_embeddings: Iterable[np.ndarray],
    cache: EmbeddingCache | None,
) -> Dict[str, object]:
    matrix = _stack_embeddings(batch_embeddings)
    n = matrix.shape[0]
    if n == 0 or cache is None or not cache.enabled:
        return {
            "available": False,
            "min_distance": None,
            "mean_distance": None,
            "per_item_min": [],
        }

    history = cache.as_array()
    if history.size == 0:
        return {
            "available": False,
            "min_distance": None,
            "mean_distance": None,
            "per_item_min": [],
        }

    sims = matrix @ history.T
    dists = 1.0 - np.clip(sims, -1.0, 1.0)
    per_item_min = dists.min(axis=1)

    return {
        "available": True,
        "min_distance": float(per_item_min.min(initial=1.0)),
        "mean_distance": float(per_item_min.mean() if per_item_min.size else 0.0),
        "per_item_min": per_item_min.astype(np.float32).tolist(),
    }
