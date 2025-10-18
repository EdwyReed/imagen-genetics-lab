from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp01(value: float) -> float:
    return clamp(float(value), 0.0, 1.0)


def maybe(probability: float) -> bool:
    return random.random() < probability


def pick_one(seq: Sequence[T]) -> T:
    if not seq:
        raise ValueError("pick_one() requires a non-empty sequence")
    return random.choice(seq)


@dataclass
class WeightedSelector:
    sfw_level: float
    temperature: float

    def _nsfw_value(self, item, default: float = 0.5) -> float:
        if isinstance(item, dict) and "nsfw" in item:
            try:
                return float(item["nsfw"])
            except Exception:
                return default
        return default

    def _weights(self, seq: Sequence[T]) -> List[float]:
        k = self._bias_strength()
        target = clamp01(self.sfw_level)
        weights: List[float] = []
        scale = 3.0
        any_nsfw = any(isinstance(x, dict) and "nsfw" in x for x in seq)
        for item in seq:
            if any_nsfw:
                diff = abs(self._nsfw_value(item) - target)
                weight = math.exp(-k * (diff * scale) ** 2)
            else:
                weight = 1.0
            weights.append(max(weight, 1e-12))
        return weights

    def _bias_strength(self) -> float:
        t = clamp(float(self.temperature), 0.05, 1.5)
        return 0.75 + (1.0 - clamp01(t)) * 5.25

    def pick(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("WeightedSelector.pick() received an empty sequence")
        weights = self._weights(seq)
        return random.choices(seq, weights=weights, k=1)[0]

    def sample(self, seq: Sequence[T], k: int) -> List[T]:
        k = min(int(k), len(seq))
        if k <= 0:
            return []
        pool = list(seq)
        out: List[T] = []
        for _ in range(k):
            if not pool:
                break
            choice = self.pick(pool)
            out.append(choice)
            pool.remove(choice)
        return out


def pick_from_ids(seq: Iterable[dict], item_id: str | None):
    if item_id is None:
        return None
    for item in seq:
        if item.get("id") == item_id:
            return item
    return None
