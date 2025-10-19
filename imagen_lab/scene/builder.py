"""Scene builder responsible for transforming JSON assets into pre-prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ..io.json_loader import BiasRule, CharacterProfile, GeneOption, StyleProfile

__all__ = [
    "PrePrompt",
    "SceneBuilder",
]


@dataclass(frozen=True)
class PrePrompt:
    """Structured pre-prompt object consumed by downstream services."""

    style_id: str
    character_id: str | None
    scene_summary: str
    macro_controls: Mapping[str, float | str]
    meso_signals: Mapping[str, float]
    selected_genes: Mapping[str, str]
    visual_axes: Mapping[str, str]


class SceneBuilder:
    """Constructs pre-prompts while applying weight-driven biases."""

    def __init__(
        self,
        style: StyleProfile,
        *,
        character: Optional[CharacterProfile] = None,
        gene_biases: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        self.style = style
        self.character = character
        self._gene_biases: Mapping[str, Mapping[str, float]] = gene_biases or {}

    def build(
        self,
        macro_weights: Mapping[str, float | str],
        meso_weights: Mapping[str, float],
    ) -> PrePrompt:
        """Assemble a pre-prompt using weight-biased gene selection."""

        applied_macro = dict(self.style.macro_defaults)
        applied_macro.update(macro_weights)
        applied_meso = dict(self.style.meso_defaults)
        applied_meso.update(meso_weights)
        if self.character:
            applied_macro.update(self.character.macro_overrides)

        gene_choices = _merge_gene_pools(self.style, self.character)
        biases = _compute_biases(self.style.bias_rules, applied_macro)
        meso_biases = _compute_meso_biases(applied_meso)
        selected = {
            category: _select_option(
                options,
                biases.get(category, 0.0) + meso_biases.get(category, 0.0),
                self._gene_biases.get(category, {}),
            )
            for category, options in gene_choices.items()
        }
        visual_axes = {
            "pose": selected.get("pose", ""),
            "lighting": selected.get("lighting", ""),
            "palette": selected.get("palette", ""),
            "wardrobe": selected.get("wardrobe", ""),
        }
        summary_tokens = [self.style.description]
        if self.character:
            summary_tokens.append(self.character.summary)
        summary_tokens.append(
            f"focus mode {applied_macro.get('focus_mode', 'style_first')} with sfw {applied_macro.get('sfw_level', 0.5)}"
        )
        return PrePrompt(
            style_id=self.style.id,
            character_id=self.character.id if self.character else None,
            scene_summary="; ".join(token for token in summary_tokens if token),
            macro_controls=applied_macro,
            meso_signals=applied_meso,
            selected_genes=selected,
            visual_axes=visual_axes,
        )


def _merge_gene_pools(style: StyleProfile, character: Optional[CharacterProfile]) -> Dict[str, Dict[str, GeneOption]]:
    pools: Dict[str, Dict[str, GeneOption]] = {
        category: {option.id: option for option in options}
        for category, options in style.gene_pools.items()
    }
    if character:
        for category, options in character.gene_overrides.items():
            pools.setdefault(category, {})
            for option in options:
                pools[category][option.id] = option
    return pools


def _compute_biases(rules: Mapping[int, BiasRule] | list[BiasRule], macro: Mapping[str, float | str]) -> Dict[str, float]:
    applied: Dict[str, float] = {}
    iterable = rules.values() if isinstance(rules, Mapping) else rules
    for rule in iterable:
        if rule.applies(macro):
            for category, delta in rule.adjust.items():
                applied[category] = applied.get(category, 0.0) + float(delta)
    return applied


def _select_option(
    options: Mapping[str, GeneOption],
    bias: float,
    history: Mapping[str, float],
) -> str:
    if not options:
        return ""
    weighted = {
        option_id: option.weight + bias + _history_bias(history.get(option_id))
        for option_id, option in options.items()
    }
    best_id = max(sorted(weighted.keys()), key=lambda option_id: weighted[option_id])
    return best_id


_MESO_CATEGORY_MAP: Mapping[str, tuple[str, ...]] = {
    "fitness_body_focus": ("pose", "wardrobe"),
    "fitness_lighting": ("lighting",),
    "fitness_color": ("palette",),
    "fitness_coverage": ("wardrobe",),
    "fitness_composition": ("pose", "lighting"),
    "fitness_style": ("pose", "palette"),
    "fitness_alignment": ("pose", "lighting"),
}


def _compute_meso_biases(meso: Mapping[str, float]) -> Dict[str, float]:
    adjustments: Dict[str, float] = {}
    for metric, value in meso.items():
        categories = _MESO_CATEGORY_MAP.get(metric)
        if not categories:
            continue
        normalized = (float(value) - 50.0) / 100.0
        for category in categories:
            adjustments[category] = adjustments.get(category, 0.0) + normalized
    return adjustments


def _history_bias(average_fitness: Optional[float]) -> float:
    if average_fitness is None:
        return 0.0
    # Convert 0..100 averages into a modest 0..0.5 bonus range.
    return (float(average_fitness) - 50.0) / 200.0
