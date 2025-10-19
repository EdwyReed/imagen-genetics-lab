from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from imagen_lab.image.imagen.interfaces import ImagenResult

from .metrics import ScoreReport


@dataclass(frozen=True)
class ScoringRequest:
    imagen: ImagenResult
    prompt: str
    caption: str
    scene: Any
    session_id: str
    meta: Mapping[str, Any]
    weights: Mapping[str, float]
    ga_context: Mapping[str, Any]
    timestamp: int | None = None


@dataclass(frozen=True)
class ScoredVariant:
    prompt_path: str
    report: ScoreReport
    metadata: Mapping[str, Any]
    fitness_style: float
    fitness_coverage: float
    fitness_visual: float
    composite_fitness: float

    def weighted_fitness(self, w_style: float, w_nsfw: float) -> float:
        return (w_style * self.fitness_style) + (w_nsfw * (1.0 - self.fitness_coverage))


@dataclass(frozen=True)
class ScoringResult:
    variants: Sequence[ScoredVariant]

    def is_empty(self) -> bool:
        return not self.variants

    def best(self, w_style: float, w_nsfw: float) -> ScoredVariant | None:
        best_variant: ScoredVariant | None = None
        best_value = float("-inf")
        for variant in self.variants:
            value = variant.weighted_fitness(w_style, w_nsfw)
            if value > best_value:
                best_value = value
                best_variant = variant
        return best_variant


class ScoringEngineProtocol(Protocol):
    def score(self, request: ScoringRequest) -> ScoringResult:
        """Persist artifacts and compute metrics."""
