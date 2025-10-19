"""Weight management utilities for blending config, presets, and feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ..io.json_loader import CharacterProfile, StyleProfile
from ..scoring.core import ScoreResult

__all__ = ["ProfileVector", "WeightManager", "Conflict"]


@dataclass
class ProfileVector:
    macro: Dict[str, float | str]
    meso: Dict[str, float]


@dataclass(frozen=True)
class Conflict:
    field: str
    message: str
    applied_value: float


class WeightManager:
    """Blends macro and meso weights and manages profile evolution."""

    def __init__(
        self,
        macro_config: Mapping[str, float | str],
        meso_config: Mapping[str, float],
        *,
        profile: Optional[ProfileVector] = None,
    ) -> None:
        self.macro_config = dict(macro_config)
        self.meso_config = dict(meso_config)
        self.profile = profile or ProfileVector(macro={}, meso={})

    def merge_profile(self, profile: ProfileVector) -> None:
        """Merge an externally loaded profile into the active state."""

        self.profile.macro.update(profile.macro)
        for key, value in profile.meso.items():
            self.profile.meso[key] = float(value)

    def blend(
        self,
        style: StyleProfile,
        character: Optional[CharacterProfile],
    ) -> Tuple[Dict[str, float | str], Dict[str, float], Tuple[Conflict, ...]]:
        macro = dict(style.macro_defaults)
        if character:
            macro.update(character.macro_overrides)
        macro.update(self.macro_config)
        macro.update(self.profile.macro)

        meso = dict(style.meso_defaults)
        meso.update(self.meso_config)
        meso.update(self.profile.meso)

        conflicts = list(_apply_conflict_resolution(macro))
        return macro, meso, tuple(conflicts)

    def update_from_scores(self, score: ScoreResult) -> None:
        learning_rate = 0.3
        for key, value in score.meso_aggregates.items():
            current = float(self.profile.meso.get(key, value))
            self.profile.meso[key] = (1 - learning_rate) * current + learning_rate * float(value)

        sfw_balance = score.fitness.get("fitness_sfw_balance")
        if sfw_balance is not None:
            current = float(self.profile.macro.get("coverage_target", self.macro_config.get("coverage_target", 0.5)))
            target = min(1.0, max(0.0, sfw_balance / 100.0))
            self.profile.macro["coverage_target"] = (1 - learning_rate) * current + learning_rate * target

        macro_targets = {
            "gloss_priority": score.micro_metrics.get("gloss_intensity", 50.0) / 100.0,
            "lighting_softness": score.micro_metrics.get("ambient_softness", 50.0) / 100.0,
            "retro_authenticity": score.micro_metrics.get("era_authenticity", 50.0) / 100.0,
            "novelty_preference": score.micro_metrics.get("novelty_score", 50.0) / 100.0,
            "illustration_strength": score.micro_metrics.get("medium_watercolor", 50.0) / 100.0,
        }
        for key, target in macro_targets.items():
            current = float(self.profile.macro.get(key, self.macro_config.get(key, target)))
            self.profile.macro[key] = (1 - learning_rate) * current + learning_rate * float(target)


def _apply_conflict_resolution(macro: Dict[str, float | str]) -> Iterable[Conflict]:
    sfw = float(macro.get("sfw_level", 0.5) or 0.0)
    coverage = float(macro.get("coverage_target", 0.5) or 0.0)
    if sfw >= 0.8 and coverage < 0.35:
        macro["coverage_target"] = 0.35
        yield Conflict(
            field="coverage_target",
            message="Increased to maintain SFW expectations for high sfw_level",
            applied_value=0.35,
        )
