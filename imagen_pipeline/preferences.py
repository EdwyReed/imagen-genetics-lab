from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable


def _merge_nested(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
  result = dict(base)
  for key, value in incoming.items():
    if isinstance(value, dict) and isinstance(result.get(key), dict):
      result[key] = _merge_nested(result[key], value)
    else:
      result[key] = value
  return result


@dataclass
class BiasConfig:
  """Stores user preferences for option weighting."""

  weights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
  overrides: Dict[str, Any] = field(default_factory=dict)

  # ------------------------------------------------------------------
  @classmethod
  def from_file(cls, path: Path | str | None) -> "BiasConfig":
    if not path:
      return cls()
    file_path = Path(path)
    if not file_path.exists():
      raise FileNotFoundError(f"Bias configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
      payload = json.load(fh)
    weights = payload.get("weights", {}) if isinstance(payload, dict) else {}
    return cls(weights=weights)

  # ------------------------------------------------------------------
  def apply_overrides(self, overrides: Iterable[str]) -> None:
    for item in overrides:
      if not item:
        continue
      if "=" not in item:
        continue
      key_path, value_str = item.split("=", 1)
      try:
        value: Any = float(value_str)
      except ValueError:
        value = value_str
      parts = key_path.split(".")
      if not parts:
        continue
      group = parts[0]
      if len(parts) == 1:
        attr = "value"
        self._set_override(group, attr, value)
      elif len(parts) == 2:
        attr = parts[1]
        self._set_override(group, attr, value)
      else:
        attr = parts[1]
        subkey = parts[2]
        current = self.overrides.setdefault(group, {}).get(attr, {})
        if not isinstance(current, dict):
          current = {}
        current[subkey] = value
        self.overrides.setdefault(group, {})[attr] = current

  # ------------------------------------------------------------------
  def _set_override(self, group: str, attr: str, value: Any) -> None:
    group_entry = self.overrides.setdefault(group, {})
    group_entry[attr] = value

  # ------------------------------------------------------------------
  def preferences_for(self, group: str) -> Dict[str, Any]:
    base = self.weights.get("global", {})
    group_weights = self.weights.get(group, {})
    override_weights = self.overrides.get("global", {})
    group_override = self.overrides.get(group, {})

    merged = _merge_nested(base, group_weights)
    merged = _merge_nested(merged, override_weights)
    merged = _merge_nested(merged, group_override)
    return merged

  # ------------------------------------------------------------------
  def to_dict(self) -> Dict[str, Any]:
    return {
      "weights": self.weights,
      "overrides": self.overrides,
    }
