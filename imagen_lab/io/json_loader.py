"""Utilities for loading and validating style and character JSON definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

__all__ = [
    "BiasRule",
    "GeneOption",
    "StyleProfile",
    "CharacterProfile",
    "JsonLoader",
]


@dataclass(frozen=True)
class BiasRule:
    """A rule that adjusts gene selection weights when a condition is met."""

    when: Mapping[str, str]
    adjust: Mapping[str, float]

    def applies(self, values: Mapping[str, Any]) -> bool:
        """Return True when the condition matches the provided values."""

        for key, expression in self.when.items():
            if key not in values:
                return False
            value = values[key]
            if not _evaluate_expression(value, expression):
                return False
        return True


@dataclass(frozen=True)
class GeneOption:
    """Represents a selectable option within a gene category."""

    id: str
    weight: float = 1.0
    description: str | None = None


@dataclass
class StyleProfile:
    """A structured representation of a style JSON definition."""

    id: str
    name: str
    description: str
    macro_defaults: Dict[str, Any]
    meso_defaults: Dict[str, float]
    gene_pools: Dict[str, List[GeneOption]]
    bias_rules: List[BiasRule] = field(default_factory=list)


@dataclass
class CharacterProfile:
    """A structured representation of a character JSON definition."""

    id: str
    name: str
    summary: str
    macro_overrides: Dict[str, Any]
    traits: Dict[str, Any]
    gene_overrides: Dict[str, List[GeneOption]] = field(default_factory=dict)


class JsonLoader:
    """Loads style and character definitions from JSON files."""

    def __init__(self, styles_path: Path, characters_path: Optional[Path] = None) -> None:
        self.styles_path = styles_path
        self.characters_path = characters_path

    def load_styles(self) -> Dict[str, StyleProfile]:
        return {profile.id: profile for profile in self._load_profiles(self.styles_path, _style_from_payload)}

    def load_characters(self) -> Dict[str, CharacterProfile]:
        if not self.characters_path:
            return {}
        return {
            profile.id: profile
            for profile in self._load_profiles(self.characters_path, _character_from_payload)
        }

    def _load_profiles(
        self,
        path: Path,
        factory: Callable[[Mapping[str, Any]], _T],
    ) -> Iterator[_T]:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            yield factory(_read_json(path))
            return
        for json_path in sorted(path.glob("*.json")):
            yield factory(_read_json(json_path))


_T = Any


def _style_from_payload(payload: Mapping[str, Any]) -> StyleProfile:
    _require_keys(payload, {"id", "name", "description", "genes"})
    macro_defaults = dict(payload.get("macro_defaults", {}))
    _validate_macro_defaults(macro_defaults)
    meso_defaults = {
        key: float(value)
        for key, value in payload.get("meso_defaults", {}).items()
    }
    gene_pools = {
        key: [
            _gene_option(option)
            for option in _ensure_iterable(options, f"genes[{key}]")
        ]
        for key, options in payload.get("genes", {}).items()
    }
    bias_rules = [
        BiasRule(rule["when"], rule["adjust"])
        for rule in payload.get("bias_rules", [])
    ]
    return StyleProfile(
        id=str(payload["id"]),
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        macro_defaults=macro_defaults,
        meso_defaults=meso_defaults,
        gene_pools=gene_pools,
        bias_rules=bias_rules,
    )


def _character_from_payload(payload: Mapping[str, Any]) -> CharacterProfile:
    _require_keys(payload, {"id", "name", "summary"})
    macro_overrides = dict(payload.get("macro_overrides", {}))
    _validate_macro_defaults(macro_overrides, allow_strings=True)
    gene_overrides = {
        key: [
            _gene_option(option)
            for option in _ensure_iterable(options, f"gene_overrides[{key}]")
        ]
        for key, options in payload.get("gene_overrides", {}).items()
    }
    return CharacterProfile(
        id=str(payload["id"]),
        name=str(payload["name"]),
        summary=str(payload.get("summary", "")),
        macro_overrides=macro_overrides,
        traits=dict(payload.get("traits", {})),
        gene_overrides=gene_overrides,
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _gene_option(payload: Any) -> GeneOption:
    if isinstance(payload, str):
        return GeneOption(id=payload)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping or string for gene option, received: {payload!r}")
    option_id = str(payload.get("id") or payload.get("name"))
    if not option_id:
        raise ValueError(f"Gene option missing identifier: {payload!r}")
    return GeneOption(
        id=option_id,
        weight=float(payload.get("weight", 1.0)),
        description=payload.get("description"),
    )


def _ensure_iterable(value: Any, context: str) -> Iterable[Any]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return value
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError(f"Expected iterable for {context}, received: {value!r}")


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing keys {missing} in payload: {mapping!r}")


_NUMERIC_MACRO_KEYS = {
    "sfw_level",
    "gloss_priority",
    "coverage_target",
    "illustration_strength",
    "novelty_preference",
    "lighting_softness",
    "retro_authenticity",
}


def _validate_macro_defaults(mapping: Mapping[str, Any], *, allow_strings: bool = False) -> None:
    for key, value in mapping.items():
        if key in {"era_target", "focus_mode"}:
            if not allow_strings and not isinstance(value, str):
                raise TypeError(f"Macro value for {key} must be a string, received {value!r}")
            continue
        if key in _NUMERIC_MACRO_KEYS:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError(f"Macro value for {key} must be numeric, received {value!r}") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"Macro value for {key} must be within [0, 1], received {numeric}")
        elif not allow_strings:
            raise KeyError(f"Unexpected macro key {key} in definition")
def _evaluate_expression(value: Any, expression: str) -> bool:
    try:
        target = float(value)
    except (TypeError, ValueError):
        target = value
    expression = expression.strip()
    if expression.startswith(("<=", ">=", "==", "!=")):
        op = expression[:2]
        rhs = expression[2:].strip()
    elif expression and expression[0] in "<>":
        op = expression[0]
        rhs = expression[1:].strip()
    else:
        op = "=="
        rhs = expression
    comparison = _coerce(rhs, target)
    if op == "<=":
        return float(target) <= float(comparison)
    if op == ">=":
        return float(target) >= float(comparison)
    if op == "<":
        return float(target) < float(comparison)
    if op == ">":
        return float(target) > float(comparison)
    if op == "!=":
        return target != comparison
    return target == comparison


def _coerce(raw: str, reference: Any) -> Any:
    raw = raw.strip()
    if isinstance(reference, (float, int)):
        try:
            return float(raw)
        except ValueError:
            return reference
    return raw
