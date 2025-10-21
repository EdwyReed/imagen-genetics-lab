from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class DeterministicRNG:
    seed: Optional[int]
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def choice(self, seq):
        if not seq:
            raise ValueError("choice on empty sequence")
        return self._random.choice(seq)

    def choices(self, population, weights=None, k: int = 1):
        return self._random.choices(population, weights=weights, k=k)

    def randint(self, a: int, b: int) -> int:
        return self._random.randint(a, b)

    def shuffle(self, seq) -> None:
        self._random.shuffle(seq)

    def random(self) -> float:
        return self._random.random()

    def getstate(self):  # pragma: no cover - passthrough
        return self._random.getstate()

    def setstate(self, state):  # pragma: no cover - passthrough
        self._random.setstate(state)
