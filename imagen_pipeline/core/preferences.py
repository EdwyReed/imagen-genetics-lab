"""Bias and lock configuration utilities for Imagen pipeline v0.9."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping

from .schema import validate


@dataclass
class LockSet:
    """Allow/deny lists for a single asset group."""

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    def normalized(self) -> "LockSet":
        return LockSet(sorted(set(self.allow)), sorted(set(self.deny)))

    def is_pinned(self) -> bool:
        return len(self.allow) == 1 and not self.deny


class BiasConfig:
    """Bias weights and locks loaded from bias.json."""

    def __init__(self, *, weights: Mapping[str, float] | None = None, locks: Mapping[str, LockSet] | None = None):
        self.weights: Dict[str, float] = dict(weights or {})
        self.locks: Dict[str, LockSet] = {group: locks[group] for group in locks} if locks else {}

    @classmethod
    def from_file(cls, path: Path | str, *, schema_dir: Path | str | None = None) -> "BiasConfig":
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Bias file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        schema_dir = Path(schema_dir or Path("schemas"))
        locks_schema_path = schema_dir / "locks.schema.json"
        if locks_schema_path.is_file() and "locks" in payload:
            schema = json.loads(locks_schema_path.read_text(encoding="utf-8"))
            validate(payload["locks"], schema)
        weights = payload.get("weights", {})
        locks = {
            group: LockSet(
                allow=list(config.get("allow", [])),
                deny=list(config.get("deny", [])),
            ).normalized()
            for group, config in payload.get("locks", {}).items()
        }
        return cls(weights=weights, locks=locks)

    def weight_for(self, group: str, asset_id: str) -> float:
        return float(self.weights.get(f"{group}:{asset_id}", 1.0))

    def locks_for_group(self, group: str) -> LockSet:
        return self.locks.get(group, LockSet())

    def merge_locks(self, override: Mapping[str, Mapping[str, Iterable[str]]]) -> Dict[str, LockSet]:
        merged: Dict[str, LockSet] = {group: LockSet(list(lock.allow), list(lock.deny)) for group, lock in self.locks.items()}
        for group, config in override.items():
            allow = set(config.get("allow", []))
            deny = set(config.get("deny", []))
            base = merged.setdefault(group, LockSet())
            if allow:
                base.allow = list(allow)
            base.deny = list(set(base.deny) | deny)
        for group in list(merged.keys()):
            merged[group] = merged[group].normalized()
        return merged


__all__ = ["BiasConfig", "LockSet"]
