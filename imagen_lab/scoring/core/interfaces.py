from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ScoringRequest:
    response: Any
    prompt: str
    scene: Any
    session_id: str
    meta: Mapping[str, Any]
    weights: Mapping[str, float]
    ga_context: Mapping[str, Optional[int]]


@dataclass(frozen=True)
class ScoringResult:
    batch: Any


class ScoringEngineProtocol(Protocol):
    def score(self, request: ScoringRequest) -> ScoringResult:
        """Persist artifacts and compute metrics."""
