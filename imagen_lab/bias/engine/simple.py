"""Simple bias engine for scene option sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .interfaces import BiasContext, BiasEngineProtocol


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float conversion used by the bias engine."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class SimpleBiasEngine(BiasEngineProtocol):
    """Heuristic bias engine that projects macro signals onto scene slots.

    The engine implements the behavioural guidance outlined in ``wiki/refactor_plan``
    by clamping SFW coverage conflicts and tracking the rules that were
    applied while preparing sampling probabilities for the scene builder.
    """

    coverage_floor: float = 0.45
    novelty_threshold: float = 0.7

    def compute_bias(self, context: BiasContext) -> Mapping[str, Any]:
        macro: Dict[str, Any] = dict(context.macro_snapshot or {})
        meso: Dict[str, Any] = dict(context.meso_snapshot or {})

        applied_rules: list[str] = []
        conflicts: list[Dict[str, Any]] = []

        requested_sfw = _coerce_float(macro.get("sfw_level"), context.sfw_level)
        if requested_sfw is None:
            sfw_level = context.sfw_level
            applied_rules.append(f"default_sfw_level={sfw_level:.2f}")
        else:
            sfw_level = requested_sfw

        clamped_sfw = max(0.0, min(1.0, sfw_level))
        if abs(clamped_sfw - sfw_level) > 1e-6:
            applied_rules.append(f"clamp_sfw_level:{sfw_level:.2f}->{clamped_sfw:.2f}")
        sfw_level = clamped_sfw

        coverage_target = _coerce_float(macro.get("coverage_target"))
        if coverage_target is not None and sfw_level > 0.7 and coverage_target < self.coverage_floor:
            conflicts.append(
                {
                    "type": "coverage_vs_sfw",
                    "requested": coverage_target,
                    "sfw_level": sfw_level,
                    "enforced_min": self.coverage_floor,
                }
            )
            coverage_target = self.coverage_floor
            applied_rules.append("coverage_floor_enforced")

        slot_targets: Dict[str, Dict[str, Any]] = {}

        # Lower SFW levels permit higher NSFW content. We bias the maximum NSFW
        # threshold for the more sensitive slots accordingly.
        max_sensitive = min(0.9, 0.25 + (1.0 - sfw_level) * 0.6)
        max_environmental = min(0.9, 0.35 + (1.0 - sfw_level) * 0.4)

        slot_targets["pose"] = {"max_nsfw": max_sensitive}
        slot_targets["wardrobe"] = {"max_nsfw": max_sensitive}
        slot_targets["mood"] = {"max_nsfw": max_sensitive}
        slot_targets["background"] = {"max_nsfw": max_environmental}
        slot_targets["lighting"] = {"max_nsfw": min(0.9, 0.35 + (1.0 - sfw_level) * 0.5)}
        slot_targets["palette"] = {"max_nsfw": 1.0}

        if coverage_target is not None:
            slot_targets["wardrobe"]["min_coverage"] = coverage_target

        novelty = _coerce_float(macro.get("novelty_preference"))
        if novelty is None:
            novelty = _coerce_float(meso.get("fitness_novelty"))
        if novelty is not None and novelty > self.novelty_threshold:
            boost = min(2.0, 1.0 + (novelty - self.novelty_threshold) * 2.0)
            slot_targets.setdefault("palette", {})["temperature_boost"] = boost
            applied_rules.append(f"novelty_palette_boost={boost:.2f}")

        gene_bias: Dict[str, float] = {}
        fitness_map = context.gene_fitness or {}
        if fitness_map:
            values = [
                float(value)
                for value in fitness_map.values()
                if isinstance(value, (int, float))
            ]
            if values:
                mean_val = sum(values) / len(values)
                spread = max(1e-6, max(values) - min(values))
                penalties = context.penalties or {}
                for gene_id, value in fitness_map.items():
                    try:
                        fitness_val = float(value)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue
                    normalized = 0.5 + 0.5 * ((fitness_val - mean_val) / spread)
                    multiplier = 0.6 + max(0.0, min(1.0, normalized)) * 0.8
                    penalty = penalties.get(gene_id)
                    if penalty is not None:
                        try:
                            penalty_val = float(penalty)
                        except (TypeError, ValueError):  # pragma: no cover
                            penalty_val = 0.0
                        multiplier *= max(0.1, 1.0 - max(0.0, min(0.9, penalty_val)))
                    gene_bias[gene_id] = max(0.05, min(2.0, multiplier))

        return {
            "slot_targets": slot_targets,
            "applied_rules": applied_rules,
            "conflicts": conflicts,
            "sfw_level": sfw_level,
            "gene_bias": gene_bias,
        }
