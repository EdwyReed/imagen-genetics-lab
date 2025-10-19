"""Regulator catalog and bias mixing utilities.

The module defines macro/meso regulators that map higher level feedback
signals to concrete genes in the evolutionary genome.  It also includes a
mini-DSL for describing regulator rules, together with helpers that project
observed gene values to the desired state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


class RegulatorLevel(str, Enum):
    """Logical grouping of regulators."""

    MACRO = "macro"
    MESO = "meso"


@dataclass(frozen=True)
class RegulatorSpec:
    """Metadata describing a single regulator."""

    id: str
    level: RegulatorLevel
    description: str
    gene_links: Mapping[str, float]
    baseline: float = 1.0
    min_value: float = 0.0
    max_value: float = 2.0

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


DEFAULT_REGULATORS: Mapping[str, RegulatorSpec] = {
    spec.id: spec
    for spec in (
        RegulatorSpec(
            id="subject_focus",
            level=RegulatorLevel.MACRO,
            description="Controls the balance between the hero character and the environment.",
            gene_links={"hero_focus": 0.6, "environment_story": -0.4},
            baseline=1.0,
            min_value=0.4,
            max_value=1.8,
        ),
        RegulatorSpec(
            id="composition_symmetry",
            level=RegulatorLevel.MACRO,
            description="Encourages symmetrical framing versus dynamic asymmetric layouts.",
            gene_links={"symmetry_bias": 0.5, "dynamic_angle": -0.3},
            baseline=1.0,
            min_value=0.5,
            max_value=1.7,
        ),
        RegulatorSpec(
            id="lighting_drama",
            level=RegulatorLevel.MACRO,
            description="Amplifies cinematic high-contrast lighting cues.",
            gene_links={"key_light_ratio": 0.7, "fill_light_level": -0.4},
            baseline=1.1,
            min_value=0.3,
            max_value=1.9,
        ),
        RegulatorSpec(
            id="color_vibrancy",
            level=RegulatorLevel.MACRO,
            description="Balances muted palettes against vibrant colors.",
            gene_links={"palette_saturation": 0.8, "palette_muted": -0.5},
            baseline=1.0,
            min_value=0.4,
            max_value=1.8,
        ),
        RegulatorSpec(
            id="nsfw_pressure",
            level=RegulatorLevel.MACRO,
            description="Suppresses risky content in production workflows.",
            gene_links={"nsfw_gate": -0.8, "safe_pose": 0.5},
            baseline=0.8,
            min_value=0.0,
            max_value=1.2,
        ),
        RegulatorSpec(
            id="camera_distance",
            level=RegulatorLevel.MACRO,
            description="Biases between close-ups and wide establishing shots.",
            gene_links={"focal_length": -0.6, "shot_scale": 0.7},
            baseline=1.0,
            min_value=0.5,
            max_value=1.6,
        ),
        RegulatorSpec(
            id="pose_dynamics",
            level=RegulatorLevel.MESO,
            description="Encourages motion-rich posing instead of static stances.",
            gene_links={"pose_energy": 0.9, "pose_grounded": -0.5},
            baseline=1.0,
            min_value=0.4,
            max_value=1.9,
        ),
        RegulatorSpec(
            id="prop_density",
            level=RegulatorLevel.MESO,
            description="Adjusts the amount of supporting props in the scene.",
            gene_links={"prop_frequency": 0.8, "clean_background": -0.6},
            baseline=1.0,
            min_value=0.3,
            max_value=1.7,
        ),
        RegulatorSpec(
            id="background_complexity",
            level=RegulatorLevel.MESO,
            description="Switches between minimal and cluttered backgrounds.",
            gene_links={"background_detail": 0.9, "depth_simplicity": -0.6},
            baseline=1.0,
            min_value=0.3,
            max_value=1.8,
        ),
        RegulatorSpec(
            id="emotion_intensity",
            level=RegulatorLevel.MESO,
            description="Controls how expressive the subject emotions should be.",
            gene_links={"emotion_energy": 0.7, "calm_expression": -0.5},
            baseline=1.0,
            min_value=0.3,
            max_value=1.8,
        ),
        RegulatorSpec(
            id="texture_detail",
            level=RegulatorLevel.MESO,
            description="Balances painterly smoothness versus crisp photoreal detail.",
            gene_links={"micro_detail": 0.8, "stylized_blend": -0.4},
            baseline=1.0,
            min_value=0.4,
            max_value=1.8,
        ),
        RegulatorSpec(
            id="shot_angle",
            level=RegulatorLevel.MESO,
            description="Encourages heroic low angles or relaxed eye-level framing.",
            gene_links={"low_angle_bias": 0.7, "eye_level_bias": -0.4},
            baseline=1.0,
            min_value=0.5,
            max_value=1.6,
        ),
    )
}


@dataclass(frozen=True)
class RegulatorProfile:
    """Named preset with preferred regulator intensities."""

    name: str
    regulators: Mapping[str, float]
    description: Optional[str] = None
    source_path: Optional[Path] = None

    def resolve(self, catalog: Mapping[str, RegulatorSpec]) -> Dict[str, float]:
        resolved: Dict[str, float] = {}
        for reg_id, spec in catalog.items():
            resolved[reg_id] = spec.clamp(self.regulators.get(reg_id, spec.baseline))
        return resolved

    def bias_snapshot(self, catalog: Mapping[str, RegulatorSpec]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return macro and meso regulator maps compatible with the bias engine."""

        resolved = self.resolve(catalog)
        macro: Dict[str, float] = {}
        meso: Dict[str, float] = {}
        for reg_id, spec in catalog.items():
            if spec.level is RegulatorLevel.MACRO:
                macro[reg_id] = resolved[reg_id]
            else:
                meso[reg_id] = resolved[reg_id]
        return macro, meso

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
        *,
        source: str = "<memory>",
        catalog: Optional[Mapping[str, RegulatorSpec]] = None,
        min_regulators: int = 10,
        max_regulators: int = 12,
    ) -> "RegulatorProfile":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source}: profile payload must be a mapping")

        schema_version = payload.get("schema_version", 1)
        if not isinstance(schema_version, int) or schema_version != 1:
            raise ValueError(f"{source}: unsupported profile schema_version {schema_version!r}")

        identifier = payload.get("id") or payload.get("profile_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{source}: profile id must be a non-empty string")
        name = identifier.strip()

        description = payload.get("description")
        if isinstance(description, str):
            description = description.strip() or None
        else:
            description = None

        regulators: Dict[str, float] = {}
        combined = {}
        for key in ("regulators", "macro", "macro_weights", "meso", "meso_aggregates"):
            block = payload.get(key)
            if block is None:
                continue
            if not isinstance(block, Mapping):
                raise ValueError(f"{source}: section '{key}' must be an object")
            combined.update(block)

        if not combined:
            raise ValueError(f"{source}: profile must declare at least one regulator")

        for reg_id, value in combined.items():
            try:
                regulators[str(reg_id)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}: regulator '{reg_id}' must be numeric") from exc

        count = len(regulators)
        if count < min_regulators or count > max_regulators:
            raise ValueError(
                f"{source}: expected between {min_regulators} and {max_regulators} regulators, got {count}"
            )

        if catalog:
            resolved = {}
            for reg_id, spec in catalog.items():
                value = regulators.get(reg_id, spec.baseline)
                resolved[reg_id] = spec.clamp(value)
        else:
            resolved = dict(regulators)

        path_hint = None
        if isinstance(source, str) and source not in ("", "<memory>"):
            try:
                path_hint = Path(source)
            except TypeError:  # pragma: no cover - defensive
                path_hint = None

        return cls(
            name=name,
            regulators=resolved,
            description=description,
            source_path=path_hint,
        )

    @classmethod
    def load(cls, path: Path, *, catalog: Optional[Mapping[str, RegulatorSpec]] = None) -> "RegulatorProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(data, source=str(path), catalog=catalog)


# --- Mini DSL for regulator rules -------------------------------------------------


class RuleParseError(ValueError):
    """Raised when the DSL line cannot be parsed."""


@dataclass(frozen=True)
class RegulatorRule:
    regulator_id: str

    def apply(self, value: float, spec: RegulatorSpec) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class MultiplyRule(RegulatorRule):
    factor: float

    def apply(self, value: float, spec: RegulatorSpec) -> float:
        return spec.clamp(value * self.factor)


@dataclass(frozen=True)
class ClampRule(RegulatorRule):
    lower: float
    upper: float

    def apply(self, value: float, spec: RegulatorSpec) -> float:
        lower = max(spec.min_value, min(spec.max_value, self.lower))
        upper = max(lower, min(spec.max_value, self.upper))
        return max(lower, min(upper, value))


@dataclass(frozen=True)
class ForbidRule(RegulatorRule):
    def apply(self, value: float, spec: RegulatorSpec) -> float:
        return spec.min_value


def parse_rule(line: str) -> RegulatorRule:
    """Parse a single DSL line."""

    stripped = line.strip()
    if not stripped:
        raise RuleParseError("empty rule")

    if stripped.startswith("!"):
        regulator_id = stripped[1:].strip()
        if not regulator_id:
            raise RuleParseError("forbid rule is missing regulator id")
        return ForbidRule(regulator_id=regulator_id)

    if "*=" in stripped:
        left, right = stripped.split("*=", 1)
        regulator_id = left.strip()
        try:
            factor = float(right.strip())
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise RuleParseError(f"invalid multiplier in '{line}'") from exc
        return MultiplyRule(regulator_id=regulator_id, factor=factor)

    if "clamp" in stripped:
        parts = stripped.split()
        if len(parts) != 3 or parts[1] != "clamp":
            raise RuleParseError(f"invalid clamp syntax in '{line}'")
        regulator_id = parts[0]
        bounds = parts[2]
        if ".." not in bounds:
            raise RuleParseError(f"invalid clamp bounds in '{line}'")
        lower_text, upper_text = bounds.split("..", 1)
        try:
            lower = float(lower_text)
            upper = float(upper_text)
        except ValueError as exc:
            raise RuleParseError(f"invalid clamp numbers in '{line}'") from exc
        return ClampRule(regulator_id=regulator_id, lower=lower, upper=upper)

    raise RuleParseError(f"unknown rule syntax: '{line}'")


def parse_rules(lines: Iterable[str]) -> Tuple[RegulatorRule, ...]:
    return tuple(parse_rule(line) for line in lines if line.strip())


# --- Observed → Desired projection ------------------------------------------------


@dataclass
class RegulationState:
    values: Dict[str, float]

    @classmethod
    def from_genes(
        cls,
        genes: Mapping[str, float],
        catalog: Mapping[str, RegulatorSpec],
    ) -> "RegulationState":
        values: Dict[str, float] = {}
        for reg_id, spec in catalog.items():
            total = spec.baseline
            for gene, coeff in spec.gene_links.items():
                total += coeff * genes.get(gene, 0.0)
            values[reg_id] = spec.clamp(total)
        return cls(values=values)


class RegulationProjector:
    """Projects observed gene values onto desired gene adjustments."""

    def __init__(self, catalog: Mapping[str, RegulatorSpec]):
        self._catalog = catalog

    def apply_rules(
        self,
        observed: RegulationState,
        rules: Sequence[RegulatorRule],
    ) -> RegulationState:
        updated = dict(observed.values)
        for rule in rules:
            spec = self._catalog.get(rule.regulator_id)
            if spec is None:
                raise KeyError(f"unknown regulator '{rule.regulator_id}'")
            current = updated.get(spec.id, spec.baseline)
            updated[spec.id] = rule.apply(current, spec)
        return RegulationState(values=updated)

    def to_genes(
        self,
        desired: RegulationState,
    ) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        for reg_id, target in desired.values.items():
            spec = self._catalog[reg_id]
            delta = target - spec.baseline
            coeff_sum = sum(abs(coeff) for coeff in spec.gene_links.values()) or 1.0
            for gene, coeff in spec.gene_links.items():
                share = coeff / coeff_sum
                adjustments[gene] = adjustments.get(gene, 0.0) + delta * share
        return adjustments

    def project(
        self,
        observed_genes: Mapping[str, float],
        rules: Sequence[RegulatorRule],
    ) -> Dict[str, float]:
        observed_state = RegulationState.from_genes(observed_genes, self._catalog)
        desired_state = self.apply_rules(observed_state, rules)
        return self.to_genes(desired_state)


# --- Bias blending ----------------------------------------------------------------


@dataclass(frozen=True)
class BiasSource:
    name: str
    regulators: Mapping[str, float]
    weight: float = 1.0


class BiasMixer:
    def __init__(
        self,
        catalog: Mapping[str, RegulatorSpec],
        *,
        floor: float = 0.3,
        ceiling: float = 1.9,
        temperature: float = 1.0,
        ema_decay: float = 0.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if floor > ceiling:
            raise ValueError("floor cannot be greater than ceiling")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be within [0, 1)")
        self._catalog = catalog
        self._floor = floor
        self._ceiling = ceiling
        self._temperature = temperature
        self._ema_decay = ema_decay

    def _clamp(self, value: float) -> float:
        return max(self._floor, min(self._ceiling, value))

    def _smooth(self, value: float) -> float:
        if value <= 0:
            return self._floor
        if self._temperature == 1.0:
            return value
        return self._clamp(pow(value, 1.0 / self._temperature))

    def mix(
        self,
        sources: Sequence[BiasSource],
        *,
        ema_state: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        totals: Dict[str, float] = {reg_id: 0.0 for reg_id in self._catalog}
        weights: Dict[str, float] = {reg_id: 0.0 for reg_id in self._catalog}

        for source in sources:
            if source.weight <= 0:
                continue
            for reg_id, spec in self._catalog.items():
                base = spec.baseline
                value = source.regulators.get(reg_id, base)
                value = spec.clamp(value)
                totals[reg_id] += source.weight * self._smooth(value)
                weights[reg_id] += source.weight

        mixed: Dict[str, float] = {}
        for reg_id, spec in self._catalog.items():
            if weights[reg_id] == 0.0:
                value = spec.baseline
            else:
                value = totals[reg_id] / weights[reg_id]
            mixed[reg_id] = self._clamp(value)

        if ema_state and self._ema_decay > 0.0:
            for reg_id, spec in self._catalog.items():
                ema_value = spec.clamp(ema_state.get(reg_id, spec.baseline))
                mixed_value = mixed[reg_id]
                mixed[reg_id] = self._clamp(
                    (1.0 - self._ema_decay) * mixed_value + self._ema_decay * ema_value
                )

        return mixed


__all__ = [
    "BiasMixer",
    "BiasSource",
    "ClampRule",
    "DEFAULT_REGULATORS",
    "ForbidRule",
    "MultiplyRule",
    "RegulatorLevel",
    "RegulationProjector",
    "RegulationState",
    "RegulatorProfile",
    "RegulatorRule",
    "RegulatorSpec",
    "RuleParseError",
    "parse_rule",
    "parse_rules",
]
