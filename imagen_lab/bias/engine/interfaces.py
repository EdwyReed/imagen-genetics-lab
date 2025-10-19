from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class BiasContext:
    profile_id: str | None
    macro_snapshot: Mapping[str, Any]
    meso_snapshot: Mapping[str, Any]
    sfw_level: float
    temperature: float
    gene_fitness: Mapping[str, float] | None = None
    penalties: Mapping[str, float] | None = None


class BiasEngineProtocol(Protocol):
    def compute_bias(self, context: BiasContext) -> Mapping[str, Any]:
        """Return gene probabilities and applied rules."""
