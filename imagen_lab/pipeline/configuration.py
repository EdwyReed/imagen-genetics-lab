"""Configuration loader for the refactored pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    yaml = None  # type: ignore

from ..scoring.core import Aggregator
from ..weights.manager import ProfileVector

__all__ = ["PipelineConfig", "load_aggregators", "load_profile"]


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run_no_scoring: bool = False
    resume_best: bool = False
    population: int = 6
    generations: int = 2
    ollama_model: str = "ollama-lite"


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path
    artifacts_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    styles_path: Path
    characters_path: Optional[Path]
    style_preset: str
    character_preset: Optional[str]
    macro_weights: Mapping[str, float | str]
    meso_overrides: Mapping[str, float]
    aggregators: Mapping[str, Aggregator]
    runtime: RuntimeConfig
    storage: StorageConfig
    profile_id: str
    profile_defaults: Optional[ProfileVector]

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        text = path.read_text(encoding="utf-8")
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            payload = _parse_yaml_like(text)
        base_dir = path.parent
        styles_path = base_dir / payload.get("styles_path", "catalogs/styles")
        characters_path = payload.get("characters_path")
        if characters_path is not None:
            characters_path = base_dir / characters_path
        storage = StorageConfig(
            database_path=base_dir / payload.get("storage", {}).get("database", "runs.sqlite"),
            artifacts_path=base_dir / payload.get("storage", {}).get("artifacts", "artifacts"),
        )
        runtime_payload = payload.get("runtime", {})
        runtime = RuntimeConfig(
            dry_run_no_scoring=bool(runtime_payload.get("dry_run_no_scoring", False)),
            resume_best=bool(runtime_payload.get("resume_best", False)),
            population=int(runtime_payload.get("population", 6)),
            generations=int(runtime_payload.get("generations", 2)),
            ollama_model=str(runtime_payload.get("ollama_model", "ollama-lite")),
        )
        aggregators = load_aggregators(payload.get("meso_aggregates", {}))
        profile_id = str(payload.get("profile", {}).get("active", "default"))
        profile_defaults = load_profile(payload.get("profile", {}).get("definitions", {}).get(profile_id))
        return cls(
            styles_path=styles_path,
            characters_path=characters_path,
            style_preset=str(payload.get("style_preset")),
            character_preset=payload.get("character_preset"),
            macro_weights=payload.get("macro_weights", {}),
            meso_overrides=payload.get("meso_overrides", {}),
            aggregators=aggregators,
            runtime=runtime,
            storage=storage,
            profile_id=profile_id,
            profile_defaults=profile_defaults,
        )


def load_aggregators(payload: Mapping[str, Any]) -> Dict[str, Aggregator]:
    aggregators: Dict[str, Aggregator] = {}
    for name, definition in payload.items():
        components = {
            metric: float(weight)
            for metric, weight in definition.get("components", {}).items()
        }
        target = definition.get("target")
        aggregators[name] = Aggregator(components=components, target=float(target) if target is not None else None)
    return aggregators


def load_profile(payload: Optional[Mapping[str, Any]]) -> Optional[ProfileVector]:
    if not payload:
        return None
    return ProfileVector(
        macro=dict(payload.get("macro", {})),
        meso={key: float(value) for key, value in payload.get("meso", {}).items()},
    )


def _parse_yaml_like(text: str) -> Dict[str, Any]:
    entries: list[Tuple[int, str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.lstrip().partition(":")
        entries.append((indent, key.strip(), value.strip()))

    root: Dict[str, Any] = {}
    stack: list[Tuple[Dict[str, Any], int]] = [(root, -1)]
    for idx, (indent, key, value) in enumerate(entries):
        next_indent = entries[idx + 1][0] if idx + 1 < len(entries) else -1
        while stack and indent <= stack[-1][1]:
            stack.pop()
        current = stack[-1][0] if stack else root
        if not value:
            if next_indent > indent:
                target: Dict[str, Any] = {}
                current[key] = target
                stack.append((target, indent))
            else:
                current[key] = None
        else:
            current[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    if value.startswith("{") and value.endswith("}"):
        return {}
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
