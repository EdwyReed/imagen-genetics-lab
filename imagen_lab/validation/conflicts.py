"""Conflict detection and safe correction for macro and weight parameters.

The rules implemented here are derived from the governance guidelines captured
in ``wiki/refactor_plan.md`` (section 12). They aim to keep user-provided macro
regulators, meso aggregates, and fitness weights within safe envelopes before a
run starts. Each rule documents the conflict it resolves and the automatic
corrections that are applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping


COVERAGE_SFW_THRESHOLD = 0.7
COVERAGE_MIN = 0.45
BODY_FOCUS_MIN = 0.35
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort coercion of numerical user inputs."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ValidationContext:
    sfw_level: float
    macro_snapshot: Mapping[str, Any] | None = None
    meso_snapshot: Mapping[str, Any] | None = None
    weights: Mapping[str, float] | None = None


@dataclass
class ValidationConflict:
    rule: str
    message: str
    severity: str = "warning"
    corrections: list[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload = {"rule": self.rule, "message": self.message, "severity": self.severity}
        if self.corrections:
            payload["corrections"] = list(self.corrections)
        return payload


@dataclass
class ValidationResult:
    sfw_level: float
    macro_snapshot: Dict[str, Any]
    meso_snapshot: Dict[str, Any]
    weights: Dict[str, float]
    conflicts: list[ValidationConflict]
    notifications: list[str]

    def conflicts_payload(self) -> list[Dict[str, Any]]:
        return [conflict.as_dict() for conflict in self.conflicts]


def _ensure_mapping(source: Mapping[str, Any] | None) -> Dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, MutableMapping):
        return dict(source)
    return {str(key): value for key, value in source.items()}


def validate_run_parameters(context: ValidationContext) -> ValidationResult:
    """Validate macro parameters and fitness weights before a run.

    The validation enforces the following documented rules:

    ``coverage_vs_sfw``
        ``sfw_level > 0.7`` conflicts with ``coverage_target < 0.3``.
        The validator raises ``coverage_target`` to at least ``0.45`` and gently
        boosts ``fitness_body_focus`` so that the wardrobe sampler prefers safer
        options. This mirrors the compromise described in the architecture plan.

    ``weight_bounds``
        Fitness weights are clamped to ``[0.0, 1.0]`` and re-normalised if their
        sum exceeds ``1.0``. When both weights collapse to zero, the defaults of
        ``style=0.7`` and ``nsfw=0.3`` are restored.

    ``sfw_bounds``
        The requested ``sfw_level`` is clamped to ``[0.0, 1.0]`` to prevent
        accidental misconfiguration.
    """

    macro_snapshot = _ensure_mapping(context.macro_snapshot)
    meso_snapshot = _ensure_mapping(context.meso_snapshot)
    weights = _ensure_mapping(context.weights)

    conflicts: list[ValidationConflict] = []
    notifications: list[str] = []

    requested_sfw = _coerce_float(context.sfw_level, 0.6) or 0.6
    sfw_level = max(0.0, min(1.0, requested_sfw))
    if abs(sfw_level - requested_sfw) > 1e-6:
        conflict = ValidationConflict(
            rule="macro.sfw_bounds",
            message=f"sfw_level {requested_sfw:.2f} out of bounds; clamped to {sfw_level:.2f}",
            severity="info",
            corrections=[
                {"field": "macro.sfw_level", "previous": requested_sfw, "corrected": sfw_level}
            ],
        )
        conflicts.append(conflict)
        notifications.append(conflict.message)
    macro_snapshot["sfw_level"] = sfw_level

    coverage_target = _coerce_float(macro_snapshot.get("coverage_target"))
    if (
        coverage_target is not None
        and sfw_level > COVERAGE_SFW_THRESHOLD
        and coverage_target < 0.3
    ):
        conflict = ValidationConflict(
            rule="macro.coverage_vs_sfw",
            message=(
                "High sfw_level requires a safer coverage_target; raised from "
                f"{coverage_target:.2f} to {COVERAGE_MIN:.2f} and boosted fitness_body_focus."
            ),
            severity="warning",
            corrections=[
                {
                    "field": "macro.coverage_target",
                    "previous": coverage_target,
                    "corrected": COVERAGE_MIN,
                    "policy": "raise_minimum",
                }
            ],
        )
        macro_snapshot["coverage_target"] = COVERAGE_MIN

        body_focus = _coerce_float(meso_snapshot.get("fitness_body_focus"), 0.3) or 0.3
        corrected_body_focus = max(body_focus, BODY_FOCUS_MIN)
        if corrected_body_focus != body_focus:
            meso_snapshot["fitness_body_focus"] = corrected_body_focus
            conflict.corrections.append(
                {
                    "field": "meso.fitness_body_focus",
                    "previous": body_focus,
                    "corrected": corrected_body_focus,
                    "policy": "boost_floor",
                }
            )
        else:
            meso_snapshot["fitness_body_focus"] = body_focus
        conflicts.append(conflict)
        notifications.append(conflict.message)
    elif coverage_target is not None:
        macro_snapshot["coverage_target"] = coverage_target

    style_weight = _coerce_float(weights.get("style"), 0.7)
    nsfw_weight = _coerce_float(weights.get("nsfw"), 0.3)
    if style_weight is None:
        style_weight = 0.7
    if nsfw_weight is None:
        nsfw_weight = 0.3

    clamped_style = max(WEIGHT_MIN, min(WEIGHT_MAX, style_weight))
    clamped_nsfw = max(WEIGHT_MIN, min(WEIGHT_MAX, nsfw_weight))
    weight_conflict: ValidationConflict | None = None

    if abs(clamped_style - style_weight) > 1e-6 or abs(clamped_nsfw - nsfw_weight) > 1e-6:
        weight_conflict = ValidationConflict(
            rule="weights.bounds",
            message="Fitness weights were clamped to the [0, 1] range.",
            severity="info",
            corrections=[],
        )
        if abs(clamped_style - style_weight) > 1e-6:
            weight_conflict.corrections.append(
                {
                    "field": "weights.style",
                    "previous": style_weight,
                    "corrected": clamped_style,
                    "policy": "clamp",
                }
            )
        if abs(clamped_nsfw - nsfw_weight) > 1e-6:
            weight_conflict.corrections.append(
                {
                    "field": "weights.nsfw",
                    "previous": nsfw_weight,
                    "corrected": clamped_nsfw,
                    "policy": "clamp",
                }
            )
        style_weight = clamped_style
        nsfw_weight = clamped_nsfw

    total = style_weight + nsfw_weight
    if total > 1.0:
        normalised_style = style_weight / total
        normalised_nsfw = nsfw_weight / total
        if weight_conflict is None:
            weight_conflict = ValidationConflict(
                rule="weights.normalise",
                message="Fitness weights exceeded a total of 1.0 and were normalised.",
                severity="info",
            )
        else:
            weight_conflict.rule = "weights.bounds+normalise"
            weight_conflict.message = (
                "Fitness weights were clamped to [0, 1] and normalised to sum to 1.0."
            )
        weight_conflict.corrections.append(
            {
                "field": "weights.style",
                "previous": style_weight,
                "corrected": normalised_style,
                "policy": "normalise",
            }
        )
        weight_conflict.corrections.append(
            {
                "field": "weights.nsfw",
                "previous": nsfw_weight,
                "corrected": normalised_nsfw,
                "policy": "normalise",
            }
        )
        style_weight = normalised_style
        nsfw_weight = normalised_nsfw

    if total <= 0.0:
        default_style, default_nsfw = 0.7, 0.3
        if weight_conflict is None:
            weight_conflict = ValidationConflict(
                rule="weights.defaults",
                message="Fitness weights were unset; restored defaults style=0.7, nsfw=0.3.",
                severity="info",
            )
        weight_conflict.corrections.append(
            {
                "field": "weights.style",
                "previous": style_weight,
                "corrected": default_style,
                "policy": "restore_default",
            }
        )
        weight_conflict.corrections.append(
            {
                "field": "weights.nsfw",
                "previous": nsfw_weight,
                "corrected": default_nsfw,
                "policy": "restore_default",
            }
        )
        style_weight, nsfw_weight = default_style, default_nsfw

    if weight_conflict is not None:
        conflicts.append(weight_conflict)
        notifications.append(weight_conflict.message)

    weights["style"] = style_weight
    weights["nsfw"] = nsfw_weight

    return ValidationResult(
        sfw_level=sfw_level,
        macro_snapshot=macro_snapshot,
        meso_snapshot=meso_snapshot,
        weights=weights,
        conflicts=conflicts,
        notifications=notifications,
    )
