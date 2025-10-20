from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .preferences import BiasConfig


@dataclass
class SelectionMeta:
  weight: float
  mean_preference: float
  contributions: Dict[str, float]


class WeightedSelector:
  """Temperature-driven weighted sampling with bias logging."""

  def __init__(self, bias: BiasConfig, sfw_level: float, temperature: float):
    self.bias = bias
    self.sfw_level = max(0.0, min(1.0, float(sfw_level)))
    self.temperature = max(0.05, float(temperature))

  # ------------------------------------------------------------------
  def pick_one(self, group: str, options: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not options:
      raise ValueError(f"No options available for group '{group}'")
    scored = [self._score_option(group, option) for option in options]
    weights = [entry.weight for entry in scored]
    chosen = random.choices(scored, weights=weights, k=1)[0]
    result = dict(chosen.item)
    result.setdefault("selection_meta", chosen.meta.__dict__)
    return result

  # ------------------------------------------------------------------
  def pick_many(self, group: str, options: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    pool = list(options)
    random.shuffle(pool)
    taken: List[Dict[str, Any]] = []
    while pool and len(taken) < count:
      item = self.pick_one(group, pool)
      taken.append(item)
      pool = [opt for opt in pool if opt.get("id") != item.get("id")]
    return taken

  # ------------------------------------------------------------------
  def _score_option(self, group: str, option: Dict[str, Any]):
    attributes: Dict[str, Any] = option.get("attributes", {})
    preferences = self._preferences_for(group)

    contributions: Dict[str, float] = {}
    matches: List[float] = []

    for key, target in preferences.items():
      if key == "era" and isinstance(target, dict):
        score = self._score_era(attributes.get("era", {}), target)
        if score is not None:
          contributions[key] = score
          matches.append(score)
      else:
        val = attributes.get(key)
        if val is None:
          continue
        try:
          target_val = float(target)
        except (TypeError, ValueError):
          continue
        score = 1.0 - abs(float(val) - target_val)
        score = max(0.0, min(1.0, score))
        contributions[key] = score
        matches.append(score)

    if matches:
      mean_pref = sum(matches) / len(matches)
    else:
      mean_pref = 0.5

    bias = (mean_pref - 0.5) * 2.0  # [-1, 1]
    slope = 1.0 / max(self.temperature, 0.05)
    base_weight = float(option.get("weight", 1.0))
    weight = math.exp(bias * slope) * max(base_weight, 1e-6)

    meta = SelectionMeta(weight=weight, mean_preference=mean_pref, contributions=contributions)
    return _ScoredOption(item=option, weight=weight, meta=meta)

  # ------------------------------------------------------------------
  def _score_era(self, option_map: Dict[str, Any], pref_map: Dict[str, Any]) -> float | None:
    if not option_map or not pref_map:
      return None
    total_weight = 0.0
    score = 0.0
    for era, pref in pref_map.items():
      try:
        pref_val = float(pref)
      except (TypeError, ValueError):
        continue
      opt_val = float(option_map.get(era, 0.0))
      score += pref_val * opt_val
      total_weight += pref_val
    if total_weight <= 0:
      return None
    normalized = score / total_weight
    return max(0.0, min(1.0, normalized))

  # ------------------------------------------------------------------
  def _preferences_for(self, group: str) -> Dict[str, Any]:
    group_key = group.split(":", 1)[0]
    base = self.bias.preferences_for(group_key)
    auto: Dict[str, Any] = {}
    if group_key in {"poses", "actions", "wardrobe", "wardrobe_sets", "models", "moods"}:
      auto.setdefault("sexuality", self.sfw_level)
      auto.setdefault("nudity", self.sfw_level)
    elif group_key in {"palettes", "lighting", "lighting_presets", "backgrounds"}:
      auto.setdefault("artistic", 0.6 + 0.3 * self.sfw_level)
    return {**auto, **base}


@dataclass
class _ScoredOption:
  item: Dict[str, Any]
  weight: float
  meta: SelectionMeta
