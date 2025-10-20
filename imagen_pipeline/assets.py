from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class Option:
  """Normalized option entry with typed metadata."""

  id: str
  desc: str
  data: Dict[str, Any]

  @property
  def attributes(self) -> Dict[str, Any]:
    return self.data.get("attributes", {})

  def as_dict(self) -> Dict[str, Any]:
    base = {"id": self.id, "desc": self.desc}
    base.update({k: v for k, v in self.data.items() if k not in {"id", "desc"}})
    return base


class PromptAssets:
  """Loads prompt building blocks from dedicated JSON files."""

  def __init__(self, root: Path | str):
    self.root = Path(root)
    if not self.root.exists():
      raise FileNotFoundError(f"Prompt assets directory not found: {self.root}")
    self._cache: Dict[str, Any] = {}
    self._load_all()

  # ------------------------------------------------------------------
  def _read_json(self, name: str, default: Any) -> Any:
    path = self.root / name
    if not path.exists():
      return default
    with path.open("r", encoding="utf-8") as fh:
      return json.load(fh)

  def _normalize_list(self, payload: Iterable[dict]) -> List[Option]:
    options = []
    for item in payload:
      if not isinstance(item, dict):
        continue
      iid = item.get("id") or item.get("name")
      desc = item.get("desc") or item.get("description") or item.get("label") or str(iid)
      options.append(Option(id=str(iid), desc=str(desc), data=dict(item)))
    return options

  def _load_all(self) -> None:
    self._cache["poses"] = self._normalize_list(self._read_json("poses.json", []))
    self._cache["actions"] = self._normalize_list(self._read_json("actions.json", []))
    self._cache["palettes"] = self._normalize_list(self._read_json("palettes.json", []))
    self._cache["lighting_presets"] = self._normalize_list(self._read_json("lighting.json", []))
    self._cache["backgrounds"] = self._normalize_list(self._read_json("backgrounds.json", []))
    self._cache["props"] = self._normalize_list(self._read_json("props.json", []))
    self._cache["moods"] = self._normalize_list(self._read_json("moods.json", []))
    self._cache["models"] = self._normalize_list(self._read_json("models.json", []))

    camera = self._read_json("camera.json", {})
    self._cache["camera"] = {
      "angles": self._normalize_list(camera.get("angles", [])),
      "framing": self._normalize_list(camera.get("framing", [])),
      "lenses": self._normalize_list(camera.get("lenses", [])),
      "depth": self._normalize_list(camera.get("depth", [])),
    }

    self._cache["wardrobe"] = {
      key: self._normalize_list(value)
      for key, value in self._read_json("wardrobe.json", {}).items()
    }
    self._cache["wardrobe_sets"] = self._normalize_list(self._read_json("wardrobe_sets.json", []))

    self._cache["style_controller"] = self._read_json("style.json", {})
    rules = self._read_json("rules.json", {})
    self._cache["rules"] = {
      "caption_length": rules.get("caption_length", {"min_words": 18, "max_words": 48}),
      "required_terms": rules.get("required_terms", []),
    }

  # ------------------------------------------------------------------
  def as_dict(self) -> Dict[str, Any]:
    """Return a merged dictionary compatible with the legacy pipeline."""
    merged: Dict[str, Any] = {
      "poses": [opt.as_dict() for opt in self._cache["poses"]],
      "actions": [opt.as_dict() for opt in self._cache["actions"]],
      "palettes": [opt.as_dict() for opt in self._cache["palettes"]],
      "lighting_presets": [opt.as_dict() for opt in self._cache["lighting_presets"]],
      "backgrounds": [opt.as_dict() for opt in self._cache["backgrounds"]],
      "props": [opt.as_dict() for opt in self._cache["props"]],
      "moods": [opt.as_dict() for opt in self._cache["moods"]],
      "model_descriptions": [opt.as_dict() for opt in self._cache["models"]],
      "camera": {
        key: [opt.as_dict() for opt in group]
        for key, group in self._cache["camera"].items()
      },
      "wardrobe": {
        key: [opt.as_dict() for opt in group]
        for key, group in self._cache["wardrobe"].items()
      },
      "wardrobe_sets": [opt.as_dict() for opt in self._cache["wardrobe_sets"]],
      "style_controller": self._cache["style_controller"],
      "rules": self._cache["rules"],
    }
    return merged

  # ------------------------------------------------------------------
  def required_terms(self) -> List[str]:
    return list(self._cache.get("rules", {}).get("required_terms", []))

  # ------------------------------------------------------------------
  def group(self, name: str) -> List[Option]:
    if name == "models":
      return list(self._cache["models"])
    if name == "moods":
      return list(self._cache["moods"])
    if name in {"poses", "actions", "palettes", "lighting_presets", "backgrounds", "props"}:
      return list(self._cache[name])
    if name.startswith("camera:"):
      key = name.split(":", 1)[1]
      return list(self._cache["camera"].get(key, []))
    if name.startswith("wardrobe:"):
      key = name.split(":", 1)[1]
      return list(self._cache["wardrobe"].get(key, []))
    if name == "wardrobe_sets":
      return list(self._cache["wardrobe_sets"])
    return []

  # ------------------------------------------------------------------
  def debug_snapshot(self) -> Dict[str, Any]:
    """Return the raw cached payload for logging or debugging."""
    snap: Dict[str, Any] = {}
    for key, value in self._cache.items():
      if isinstance(value, list):
        snap[key] = [opt.as_dict() if isinstance(opt, Option) else opt for opt in value]
      elif isinstance(value, dict):
        new_val: Dict[str, Any] = {}
        for subk, subval in value.items():
          new_val[subk] = [opt.as_dict() if isinstance(opt, Option) else opt for opt in subval]
        snap[key] = new_val
      else:
        snap[key] = value
    return snap
