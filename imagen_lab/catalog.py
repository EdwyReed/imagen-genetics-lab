from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Catalog:
    raw: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(raw=data)

    def section(self, key: str, default: Optional[List[dict]] = None) -> List[dict]:
        value = self.raw.get(key, default or [])
        if not isinstance(value, list):
            return list(default or [])
        return value

    def nested_section(self, *keys: str, default: Optional[List[dict]] = None) -> List[dict]:
        data: Any = self.raw
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                data = {}
        if isinstance(data, list):
            return data
        return list(default or [])

    def rules(self) -> Dict[str, Any]:
        rules = self.raw.get("rules", {})
        return rules if isinstance(rules, dict) else {}

    def style_controller(self) -> Dict[str, Any]:
        style = self.raw.get("style_controller", {})
        return style if isinstance(style, dict) else {}

    def find_description(self, item_id: str) -> Optional[str]:
        wardrobe = self.raw.get("wardrobe", {})
        if isinstance(wardrobe, dict):
            for group_items in wardrobe.values():
                for item in group_items:
                    if item.get("id") == item_id:
                        return item.get("desc")
        for prop in self.section("props", []):
            if prop.get("id") == item_id:
                return prop.get("desc")
        return None

    def wardrobe_groups(self) -> Dict[str, List[dict]]:
        wardrobe = self.raw.get("wardrobe", {})
        if isinstance(wardrobe, dict):
            return wardrobe
        return {}

    def wardrobe_sets(self) -> List[dict]:
        sets = self.raw.get("wardrobe_sets", [])
        if isinstance(sets, list):
            return sets
        return []

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []
