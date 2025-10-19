from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, MutableMapping):
        return value
    return {}


def _iter_dicts(values: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                yield item
    elif isinstance(values, tuple):
        for item in values:
            if isinstance(item, dict):
                yield item


@dataclass(frozen=True)
class Catalog:
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, Mapping):
            raise ValueError(f"catalog file {path} must contain a JSON object")
        return cls(raw=data)

    # ---------------------------------------------------------------------
    # section helpers
    # ---------------------------------------------------------------------
    def section(self, key: str, default: Optional[Iterable[dict]] = None) -> List[dict]:
        values = list(_iter_dicts(self.raw.get(key)))
        if values:
            return values
        return list(default or [])

    def nested_section(self, *keys: str, default: Optional[Iterable[dict]] = None) -> List[dict]:
        data: Any = self.raw
        for key in keys:
            data = _as_mapping(data).get(key, {})
        values = list(_iter_dicts(data))
        if values:
            return values
        return list(default or [])

    # ------------------------------------------------------------------
    # style metadata helpers
    # ------------------------------------------------------------------
    def _brand_profile(self) -> Mapping[str, Any]:
        profiles = self.raw.get("brand_profiles")
        for profile in _as_mapping(profiles).values():
            if isinstance(profile, Mapping):
                return profile
        return {}

    def rules(self) -> Dict[str, Any]:
        rules = _as_mapping(self.raw.get("rules"))
        if rules:
            return dict(rules)
        profile_rules = _as_mapping(self._brand_profile().get("rules"))
        return dict(profile_rules)

    def style_controller(self) -> Dict[str, Any]:
        style = _as_mapping(self.raw.get("style_controller"))
        if style:
            return dict(style)
        profile_style = _as_mapping(self._brand_profile().get("style_controller"))
        return dict(profile_style)

    # ------------------------------------------------------------------
    # wardrobe helpers
    # ------------------------------------------------------------------
    def wardrobe_groups(self) -> Dict[str, List[dict]]:
        groups: Dict[str, List[dict]] = {}
        for group, items in _as_mapping(self.raw.get("wardrobe")).items():
            if not isinstance(group, str):
                continue
            groups[group] = list(_iter_dicts(items))
        return groups

    def wardrobe_sets(self) -> List[dict]:
        return list(_iter_dicts(self.raw.get("wardrobe_sets")))

    def find_description(self, item_id: str) -> Optional[str]:
        for items in self.wardrobe_groups().values():
            for item in items:
                if item.get("id") == item_id:
                    desc = item.get("desc")
                    return str(desc) if isinstance(desc, str) else None
        for prop in self.section("props", []):
            if prop.get("id") == item_id:
                desc = prop.get("desc")
                return str(desc) if isinstance(desc, str) else None
        return None

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)
