from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class EmbeddingHistoryConfig:
    """Configuration for embedding history cache."""

    enabled: bool = True
    max_embeddings: int = 512


class EmbeddingCache:
    """A ring-buffer style cache for storing the last *N* embeddings."""

    def __init__(self, config: EmbeddingHistoryConfig | None = None) -> None:
        cfg = config or EmbeddingHistoryConfig()
        self.enabled: bool = bool(cfg.enabled)
        self.max_embeddings: int = int(max(0, cfg.max_embeddings))
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.max_embeddings or None)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    def add(self, embedding: Optional[np.ndarray]) -> None:
        if not self.enabled or self.max_embeddings <= 0 or embedding is None:
            return
        self._buffer.append(self._normalize(embedding))

    def extend(self, embeddings: Iterable[np.ndarray]) -> None:
        for emb in embeddings:
            self.add(emb)

    def iter(self) -> Iterator[np.ndarray]:
        return iter(self._buffer)

    def as_array(self) -> np.ndarray:
        if not self._buffer:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(list(self._buffer), axis=0)

    def to_list(self) -> List[List[float]]:
        return [emb.tolist() for emb in self._buffer]
