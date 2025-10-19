"""Scoring utilities that convert generated artefacts into analytics metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from ..scene.builder import PrePrompt

__all__ = ["Aggregator", "ScoreResult", "ScoringEngine"]


@dataclass(frozen=True)
class Aggregator:
    components: Mapping[str, float]
    target: float | None = None

    def compute(self, micro: Mapping[str, float]) -> float:
        score = 0.0
        for metric, weight in self.components.items():
            score += micro.get(metric, 0.0) * weight
        if self.target is not None:
            return max(0.0, 100.0 - abs(score - self.target))
        return score


@dataclass(frozen=True)
class ScoreResult:
    micro_metrics: Mapping[str, float]
    meso_aggregates: Mapping[str, float]
    fitness: Mapping[str, float]


class ScoringEngine:
    """Computes micro metrics and higher-level aggregates."""

    def __init__(self, aggregators: Mapping[str, Aggregator]) -> None:
        self.aggregators = aggregators

    def score(self, pre_prompt: PrePrompt, caption: str) -> ScoreResult:
        micro = _micro_metrics(pre_prompt, caption)
        meso = {
            name: aggregator.compute(micro)
            for name, aggregator in self.aggregators.items()
        }
        fitness = {
            "fitness_visual": meso.get("fitness_style", 0.0) * 0.4
            + meso.get("fitness_lighting", 0.0) * 0.2
            + meso.get("fitness_color", 0.0) * 0.2
            + meso.get("fitness_composition", 0.0) * 0.2,
            "fitness_alignment": micro.get("clip_prompt_alignment", 0.0),
            "fitness_sfw_balance": meso.get("fitness_coverage", micro.get("coverage_ratio", 0.0)),
            "fitness_era_match": (micro.get("era_authenticity", 0.0) + micro.get("era_80s", 0.0)) / 2.0,
            "fitness_novelty": meso.get("fitness_novelty", micro.get("novelty_score", 0.0)),
            "fitness_cleanliness": meso.get(
                "fitness_cleanliness",
                100.0 - micro.get("visual_noise_level", 0.0) - micro.get("ai_artifacts", 0.0) / 2.0,
            ),
        }
        return ScoreResult(micro_metrics=micro, meso_aggregates=meso, fitness=fitness)


def _micro_metrics(pre_prompt: PrePrompt, caption: str) -> Dict[str, float]:
    macro = pre_prompt.macro_controls
    coverage = float(macro.get("coverage_target", 0.5))
    gloss = float(macro.get("gloss_priority", 0.5))
    lighting = float(macro.get("lighting_softness", 0.5))
    retro = float(macro.get("retro_authenticity", 0.5))
    novelty = float(macro.get("novelty_preference", 0.5))
    illustration = float(macro.get("illustration_strength", 0.5))
    sfw = float(macro.get("sfw_level", 0.5))
    focus_mode = str(macro.get("focus_mode", "style_first"))
    selected = pre_prompt.selected_genes

    def scale(value: float, factor: float = 100.0) -> float:
        return max(0.0, min(1.0, value)) * factor

    caption_length = max(1, len(caption.split()))
    intimacy = 0.5 if focus_mode == "style_first" else 0.7
    if focus_mode == "body_first":
        intimacy = 0.85

    hip_bias = 0.6 if "hip" in "".join(selected.values()) else 0.45
    retro_bonus = 1.0 if str(macro.get("era_target", "")) == "80s" else 0.7

    micro: Dict[str, float] = {
        "style_core": illustration * 90.0 + gloss * 10.0,
        "era_80s": scale(retro * retro_bonus),
        "medium_watercolor": scale(illustration),
        "gloss_intensity": scale(gloss),
        "softness_blur": scale(lighting),
        "paper_texture": illustration * 50.0,
        "contrast_balance": 55.0 + (0.5 - coverage) * 20.0,
        "color_harmony": illustration * 70.0 + retro * 30.0,
        "palette_vividness": novelty * 80.0,
        "poster_layout": retro * 60.0,
        "medium_authenticity": illustration * 85.0,
        "coverage_ratio": scale(coverage),
        "skin_exposure": scale(1.0 - coverage),
        "lingerie_focus": (1.0 - coverage) * 70.0,
        "thigh_focus": (1.0 - coverage) * 65.0,
        "chest_focus": (1.0 - coverage) * 68.0,
        "hip_emphasis": hip_bias * 100.0,
        "clothing_tightness": (1.0 - coverage) * 75.0,
        "fabric_sheer": gloss * 80.0,
        "accessory_count": novelty * 40.0,
        "footwear_type": 55.0,
        "clothing_layers": coverage * 85.0,
        "outfit_coherence": 60.0 + illustration * 35.0,
        "pose_suggestiveness": (1.0 - coverage) * 85.0,
        "pose_balance": 65.0 + lighting * 20.0,
        "curve_accentuation": (1.0 - coverage) * 90.0,
        "camera_intimacy": intimacy * 100.0,
        "frame_tightness": 55.0 + (1.0 - coverage) * 25.0,
        "body_alignment": 60.0 + lighting * 20.0,
        "gaze_directness": 50.0 + novelty * 20.0,
        "motion_static": 40.0 + coverage * 20.0,
        "focus_centering": 70.0 + (1.0 - coverage) * 10.0,
        "highlight_distribution": gloss * 80.0 + lighting * 20.0,
        "highlight_on_skin_ratio": gloss * 75.0,
        "lighting_directionality": 40.0 + (1.0 - lighting) * 30.0,
        "ambient_softness": scale(lighting),
        "warmth_temperature": 50.0 + retro * 20.0,
        "shadow_depth": (1.0 - lighting) * 70.0,
        "background_cleanliness": sfw * 60.0 + lighting * 40.0,
        "negative_space_ratio": 35.0 + lighting * 20.0,
        "paper_grain_visibility": illustration * 55.0,
        "mood_playfulness": novelty * 90.0,
        "mood_confidence": (1.0 - coverage / 2.0) * 80.0,
        "mood_innocence": sfw * 80.0,
        "tone_glamour": gloss * 85.0,
        "genre_alignment": retro * 90.0,
        "era_authenticity": retro * 95.0,
        "storytelling_level": min(100.0, 30.0 + caption_length * 1.5),
        "expression_tone": 55.0 + novelty * 25.0,
        "rule_of_thirds_alignment": 60.0 + retro * 15.0,
        "visual_balance": 62.0 + illustration * 20.0,
        "foreground_focus": 55.0 + (1.0 - coverage) * 15.0,
        "depth_of_field_effect": lighting * 50.0,
        "subject_size_ratio": 60.0 + (1.0 - coverage) * 20.0,
        "clip_prompt_alignment": 72.0 + novelty * 12.0,
        "identity_consistency": 78.0,
        "novelty_score": scale(novelty),
        "aesthetic_coherence": 65.0 + illustration * 30.0,
        "medium_realism": 40.0 + (1.0 - illustration) * 30.0,
        "ai_artifacts": max(0.0, 25.0 - illustration * 12.0),
        "visual_noise_level": max(0.0, 35.0 - illustration * 15.0 - lighting * 10.0),
    }

    micro.update(
        {
            "fitness_visual": micro["style_core"] * 0.4
            + micro["highlight_distribution"] * 0.2
            + micro["color_harmony"] * 0.2
            + micro["visual_balance"] * 0.2,
            "fitness_body_focus": micro["chest_focus"] * 0.4
            + micro["thigh_focus"] * 0.4
            + micro["pose_suggestiveness"] * 0.2,
            "fitness_coverage": micro["coverage_ratio"],
            "fitness_lighting": micro["ambient_softness"] * 0.6
            + micro["highlight_distribution"] * 0.4,
            "fitness_color": micro["color_harmony"] * 0.6
            + micro["poster_layout"] * 0.4,
            "fitness_composition": micro["visual_balance"] * 0.5
            + micro["background_cleanliness"] * 0.5,
            "fitness_novelty": micro["novelty_score"],
            "fitness_alignment": micro["clip_prompt_alignment"],
            "fitness_cleanliness": 100.0 - micro["visual_noise_level"] - micro["ai_artifacts"] / 2.0,
        }
    )
    return micro
