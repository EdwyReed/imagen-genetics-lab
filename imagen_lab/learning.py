from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Mapping, Optional, Sequence

from .config import FeedbackConfig
from .scoring import DEFAULT_STYLE_WEIGHTS, STYLE_COMPONENT_KEYS, normalize_weights

_EPS = 1e-4


class StyleFeedback:
    """Stateful helper that turns scorer metrics into prompt guidance."""

    def __init__(self, cfg: FeedbackConfig) -> None:
        self._component_alpha = max(0.0, min(1.0, float(cfg.component_alpha)))
        self._gene_alpha = max(0.0, min(1.0, float(cfg.gene_alpha)))
        self._bias_floor = max(0.0, float(cfg.bias_floor))
        self._bias_ceiling = max(self._bias_floor, float(cfg.bias_ceiling))
        self._component_margin = max(0.0, float(cfg.component_margin))
        self._top_k = max(1, int(cfg.top_k))
        history_size = max(1, int(cfg.history))

        max_weight = max(DEFAULT_STYLE_WEIGHTS.values()) if DEFAULT_STYLE_WEIGHTS else 1.0
        base_levels = {
            key: (DEFAULT_STYLE_WEIGHTS.get(key, 0.0) / max_weight if max_weight else 0.0)
            for key in STYLE_COMPONENT_KEYS
        }
        self._weights: Dict[str, float] = dict(DEFAULT_STYLE_WEIGHTS)
        self._component_levels: Dict[str, float] = dict(base_levels)
        self._component_trend: Dict[str, float] = {key: 0.0 for key in STYLE_COMPONENT_KEYS}
        self._mean_contributions: Dict[str, float] = {
            key: DEFAULT_STYLE_WEIGHTS.get(key, 0.0) for key in STYLE_COMPONENT_KEYS
        }
        self._composition_targets: Dict[str, float] = {
            "cropping_tightness": 0.55,
            "thirds_alignment": 0.55,
            "negative_space": 0.35,
        }
        self._gene_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._recent_highlights: deque[Dict[str, Any]] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        gene_ids: Mapping[str, Optional[str]],
        template_id: Optional[str],
        summary: str,
        batch_metrics: Mapping[str, Any],
        best_image: Any,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Ingest a freshly-scored scene.

        Parameters
        ----------
        gene_ids:
            Mapping of gene-slot name to selected catalog identifier.
        template_id:
            Template used for the caption scaffold.
        summary:
            Human-readable description of the scene (for logging back into prompt).
        batch_metrics:
            Aggregated metrics returned by :func:`save_and_score`.
        best_image:
            Best individual of the batch (``ScoredImage``-like object).
        weights:
            Explicit style-weight mapping. When omitted the method tries to pull
            the information from ``batch_metrics`` instead.
        """

        style_metrics = batch_metrics.get("style") if isinstance(batch_metrics, Mapping) else {}
        weight_source: Mapping[str, float] = weights or {}
        if not weight_source and isinstance(style_metrics, Mapping):
            raw = style_metrics.get("weights")
            if isinstance(raw, Mapping):
                weight_source = raw
        if weight_source:
            normalized = normalize_weights(weight_source, keys=STYLE_COMPONENT_KEYS, defaults=DEFAULT_STYLE_WEIGHTS)
            self._weights = dict(normalized)

        components = getattr(best_image, "style_components", {}) or {}
        contributions = getattr(best_image, "style_contributions", {}) or {}
        style_score = float(getattr(best_image, "style", 0.0)) / 100.0
        composition = getattr(best_image, "composition_raw", {}) or {}

        # Blend component intensities and trend
        alpha = self._component_alpha
        for key in STYLE_COMPONENT_KEYS:
            value = float(components.get(key, 0.0))
            prev = self._component_levels.get(key, 0.0)
            blended = (1.0 - alpha) * prev + alpha * value
            self._component_levels[key] = blended
            self._component_trend[key] = blended - prev
            contrib_prev = self._mean_contributions.get(key, 0.0)
            contrib_val = float(contributions.get(key, 0.0))
            self._mean_contributions[key] = (1.0 - alpha) * contrib_prev + alpha * contrib_val

        for key in tuple(self._composition_targets.keys()):
            value = float(composition.get(key, self._composition_targets[key]))
            prev = self._composition_targets.get(key, value)
            self._composition_targets[key] = (1.0 - alpha) * prev + alpha * value

        # Update running preferences with a light decay
        decay = 1.0 - self._gene_alpha * 0.5
        for slot_map in self._gene_scores.values():
            for gene in list(slot_map.keys()):
                slot_map[gene] = max(0.0, slot_map[gene] * decay)

        self._update_gene("template", template_id, style_score)
        for slot, gene_id in gene_ids.items():
            if slot == "template":
                continue
            self._update_gene(slot, gene_id, style_score)

        highlight = {
            "scene": summary,
            "style": getattr(best_image, "style", 0),
            "nsfw": getattr(best_image, "nsfw", 0),
            "components": {key: float(components.get(key, 0.0)) for key in STYLE_COMPONENT_KEYS},
        }
        if summary:
            self._recent_highlights.appendleft(highlight)

    def _update_gene(self, slot: str, gene_id: Optional[str], value: float) -> None:
        if not gene_id:
            return
        slot_map = self._gene_scores.setdefault(slot, {})
        prev = slot_map.get(gene_id, value)
        slot_map[gene_id] = (1.0 - self._gene_alpha) * prev + self._gene_alpha * max(0.0, value)

    # ------------------------------------------------------------------
    # Prompt payload helpers
    # ------------------------------------------------------------------
    def apply_bias(self, slot: str, seq: Sequence[Any], weights: Sequence[float]) -> Sequence[float]:
        if not seq:
            return list(weights)
        slot_scores = self._gene_scores.get(slot)
        if not slot_scores:
            return list(weights)
        baseline = sum(slot_scores.values()) / max(len(slot_scores), 1)
        if baseline < _EPS:
            baseline = _EPS
        adjusted = []
        for item, base_w in zip(seq, weights):
            if isinstance(item, Mapping):
                gene_id = item.get("id")
            else:
                gene_id = str(item)
            score = slot_scores.get(gene_id)
            if score is None:
                multiplier = 1.0
            else:
                multiplier = (score + _EPS) / (baseline + _EPS)
            multiplier = max(self._bias_floor, min(self._bias_ceiling, multiplier))
            adjusted.append(max(float(base_w) * multiplier, _EPS))
        return adjusted

    def snapshot(self) -> Dict[str, Any]:
        max_weight = max(self._weights.values()) if self._weights else 1.0
        component_focus = []
        boost: list[str] = []
        cooldown: list[str] = []
        for key in STYLE_COMPONENT_KEYS:
            weight = float(self._weights.get(key, 0.0))
            level = float(self._component_levels.get(key, 0.0))
            trend = float(self._component_trend.get(key, 0.0))
            expected = weight / max_weight if max_weight else 0.0
            if level + self._component_margin < expected:
                boost.append(key)
            elif level > expected + self._component_margin:
                cooldown.append(key)
            component_focus.append(
                {
                    "component": key,
                    "weight": weight,
                    "level": level,
                    "trend": trend,
                    "expected": expected,
                    "contribution": float(self._mean_contributions.get(key, 0.0)),
                }
            )
        component_focus.sort(key=lambda item: item["weight"], reverse=True)

        preferences: Dict[str, list[Dict[str, float]]] = {}
        for slot, scores in self._gene_scores.items():
            if not scores:
                continue
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
            preferences[slot] = [
                {"id": gene_id, "score": float(score)}
                for gene_id, score in top
            ]

        notes: list[str] = []
        if boost:
            notes.append("Boost components: " + ", ".join(boost))
        if cooldown:
            notes.append("Ease off components: " + ", ".join(cooldown))
        if preferences:
            pref_bits = []
            for slot, items in preferences.items():
                if items:
                    pref_bits.append(f"{slot}→{items[0]['id']}")
            if pref_bits:
                notes.append("Preferred genes: " + ", ".join(pref_bits))
        if not self._recent_highlights:
            notes.append("No feedback history yet; follow base style weights.")

        return {
            "weights": dict(self._weights),
            "component_levels": {key: float(self._component_levels.get(key, 0.0)) for key in STYLE_COMPONENT_KEYS},
            "component_trend": {key: float(self._component_trend.get(key, 0.0)) for key in STYLE_COMPONENT_KEYS},
            "component_focus": component_focus,
            "boost_components": boost,
            "cooldown_components": cooldown,
            "preferences": preferences,
            "composition_targets": dict(self._composition_targets),
            "mean_contributions": {key: float(self._mean_contributions.get(key, 0.0)) for key in STYLE_COMPONENT_KEYS},
            "recent_highlights": list(self._recent_highlights),
            "notes": notes,
        }

    @staticmethod
    def baseline_snapshot() -> Dict[str, Any]:
        weights = dict(DEFAULT_STYLE_WEIGHTS)
        max_weight = max(weights.values()) if weights else 1.0
        component_levels = {
            key: (weights.get(key, 0.0) / max_weight if max_weight else 0.0)
            for key in STYLE_COMPONENT_KEYS
        }
        focus = [
            {
                "component": key,
                "weight": float(weights.get(key, 0.0)),
                "level": float(component_levels.get(key, 0.0)),
                "trend": 0.0,
                "expected": float(component_levels.get(key, 0.0)),
                "contribution": float(weights.get(key, 0.0)),
            }
            for key in STYLE_COMPONENT_KEYS
        ]
        focus.sort(key=lambda item: item["weight"], reverse=True)
        return {
            "weights": weights,
            "component_levels": component_levels,
            "component_trend": {key: 0.0 for key in STYLE_COMPONENT_KEYS},
            "component_focus": focus,
            "boost_components": [],
            "cooldown_components": [],
            "preferences": {},
            "composition_targets": {
                "cropping_tightness": 0.55,
                "thirds_alignment": 0.55,
                "negative_space": 0.35,
            },
            "mean_contributions": {key: float(weights.get(key, 0.0)) for key in STYLE_COMPONENT_KEYS},
            "recent_highlights": [],
            "notes": ["No feedback history yet; follow base style weights."],
        }


__all__ = ["StyleFeedback"]
