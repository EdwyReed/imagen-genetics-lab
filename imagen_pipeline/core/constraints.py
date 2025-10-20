"""Constraint resolution utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple

from .preferences import LockSet

AssetRecord = Mapping[str, object]
SelectedAssets = Mapping[str, Mapping[str, AssetRecord]]


class ConstraintError(RuntimeError):
    """Raised when assets cannot satisfy constraints."""


@dataclass
class ScenarioContext:
    """Context for constraint evaluation."""

    locks: Mapping[str, LockSet] = field(default_factory=dict)
    required_terms: Iterable[str] = field(default_factory=list)
    inject_rules: Iterable[str] = field(default_factory=list)

    def required_term_set(self) -> set[str]:
        return set(self.required_terms)

    def inject_rule_set(self) -> set[str]:
        return set(self.inject_rules)


def parse_reference(ref: str) -> Tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"Malformed asset reference '{ref}'. Expected group:id format.")
    group, asset_id = ref.split(":", 1)
    return group, asset_id


def is_compatible(
    candidate: AssetRecord,
    selected: SelectedAssets,
    scenario_context: ScenarioContext | None = None,
) -> bool:
    """Return True if *candidate* fits within current *selected* set."""

    scenario_context = scenario_context or ScenarioContext()
    required = candidate.get("requires", []) or []
    excludes = candidate.get("excludes", []) or []
    for ref in required:
        group, asset_id = parse_reference(str(ref))
        group_assets = selected.get(group, {})
        if asset_id not in group_assets:
            return False
    for ref in excludes:
        group, asset_id = parse_reference(str(ref))
        group_assets = selected.get(group, {})
        if asset_id in group_assets:
            return False
    return True


__all__ = [
    "ConstraintError",
    "ScenarioContext",
    "is_compatible",
    "parse_reference",
]
