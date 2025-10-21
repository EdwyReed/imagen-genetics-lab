from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .rng import DeterministicRNG
from .schema import SelectionConfig


DICT_FILES = [
    ("character", "characters.json"),
    ("pose", "poses.json"),
    ("action", "actions.json"),
    ("style", "styles.json"),
    ("clothes", "clothes.json"),
]


def _load_json(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Dictionary {path} must contain a list of objects")
    return data


def load_dictionaries(root: Path) -> Dict[str, List[Dict[str, object]]]:
    dictionaries: Dict[str, List[Dict[str, object]]] = {}
    for key, filename in DICT_FILES:
        path = root / filename
        dictionaries[key] = _load_json(path)
    return dictionaries


def _weights(seq: List[Dict[str, object]]) -> List[float]:
    weights = []
    any_weight = False
    for item in seq:
        weight = item.get("weight") or item.get("bias") or item.get("probability")
        if weight is not None:
            try:
                weight = float(weight)
                any_weight = True
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid weight in dictionary entry: {item}") from exc
        else:
            weight = 1.0
        weights.append(weight)
    if not any_weight:
        return []
    return weights


class OptionCollector:
    def __init__(
        self,
        dictionaries: Dict[str, List[Dict[str, object]]],
        selection: SelectionConfig,
        rng: DeterministicRNG,
    ) -> None:
        self._dicts = dictionaries
        self._selection = selection
        self._rng = rng
        self._counters: Dict[str, int] = {
            "character": 0,
            "pose": 0,
            "action": 0,
            "style": 0,
            "clothes": 0,
        }

    def _choose_with_ids(
        self,
        key: str,
        entries: List[Dict[str, object]],
        id_list: List[str],
    ) -> Dict[str, object]:
        if not id_list:
            raise ValueError("id_list must not be empty")
        index = self._counters[key] % len(id_list)
        self._counters[key] += 1
        target_id = id_list[index]
        for entry in entries:
            if entry.get("id") == target_id:
                return entry
        raise KeyError(f"Dictionary {key} missing id={target_id}")

    def _choose_random(self, entries: List[Dict[str, object]]) -> Dict[str, object]:
        weights = _weights(entries)
        if weights:
            return self._rng.choices(entries, weights=weights, k=1)[0]
        return self._rng.choice(entries)

    def _select(self, key: str, entries: List[Dict[str, object]]) -> Dict[str, object]:
        id_list = getattr(self._selection, f"{key}_ids")
        if id_list:
            return self._choose_with_ids(key, entries, id_list)
        return self._choose_random(entries)

    def collect(self) -> Dict[str, Dict[str, object]]:
        output: Dict[str, Dict[str, object]] = {}
        for key, _filename in DICT_FILES:
            entries = self._dicts[key]
            if not entries:
                raise ValueError(f"Dictionary {key} is empty")
            output[key] = self._select(key, entries)
        return output

    @staticmethod
    def summarize(options: Dict[str, Dict[str, object]]) -> str:
        parts = []
        for key in ("character", "pose", "action", "style", "clothes"):
            entry = options.get(key, {})
            summary = entry.get("summary") or entry.get("description") or entry.get("name")
            parts.append(f"{key}={summary}")
        return ", ".join(parts)


__all__ = ["load_dictionaries", "OptionCollector", "DICT_FILES"]
