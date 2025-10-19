"""Minimal genetic search abstraction operating on macro/meso aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from ..scene.builder import PrePrompt
from ..scoring.core import ScoreResult

__all__ = ["GenePlan", "GAEngine"]


@dataclass(frozen=True)
class GenePlan:
    prompt_id: str
    genes: Mapping[str, str]
    macro: Mapping[str, float | str]
    meso: Mapping[str, float]
    fitness: float


class GAEngine:
    """Applies simple selection pressure using meso aggregates."""

    def select_top(self, results: Sequence[tuple[PrePrompt, ScoreResult]]) -> List[GenePlan]:
        ranked = sorted(
            (
                GenePlan(
                    prompt_id=f"plan_{idx}",
                    genes=pre_prompt.selected_genes,
                    macro=pre_prompt.macro_controls,
                    meso=pre_prompt.meso_signals,
                    fitness=score.fitness.get("fitness_visual", 0.0),
                )
                for idx, (pre_prompt, score) in enumerate(results)
            ),
            key=lambda plan: plan.fitness,
            reverse=True,
        )
        return ranked[:3]

    def mutate(self, plan: GenePlan) -> GenePlan:
        mutated_macro = dict(plan.macro)
        if "novelty_preference" in mutated_macro:
            mutated_macro["novelty_preference"] = min(
                1.0, float(mutated_macro["novelty_preference"]) + 0.05
            )
        return GenePlan(
            prompt_id=f"{plan.prompt_id}_mut", genes=plan.genes, macro=mutated_macro, meso=plan.meso, fitness=plan.fitness
        )
