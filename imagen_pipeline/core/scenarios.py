"""Scenario loading and validation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Mapping, Optional

from .preferences import LockSet
from .schema import validate

LOGGER = logging.getLogger("imagen.scenarios")


@dataclass
class Stage:
    """Single scenario stage."""

    stage_id: str
    cycles: int
    temperature: float
    style_profile: Optional[str]
    locks: Mapping[str, LockSet] = field(default_factory=dict)
    required_terms: List[str] = field(default_factory=list)
    inject_rules: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    """Validated scenario definition."""

    scenario_id: str
    stages: List[Stage]

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)


class ScenarioLoader:
    """Load scenarios from JSON or YAML files."""

    def __init__(self, *, schema_dir: Path | str | None = None) -> None:
        self.schema_dir = Path(schema_dir or Path("schemas")).resolve()
        self._schema = self._load_schema("scenario.schema.json")

    def _load_schema(self, name: str) -> Mapping[str, object]:
        path = self.schema_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Scenario schema not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, path: Path | str) -> Scenario:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        payload = self._read_payload(path)
        validate(payload, self._schema)
        scenario_id = str(payload.get("id", path.stem))
        stages = [self._build_stage(data, index) for index, data in enumerate(payload.get("stages", []))]
        LOGGER.info("scenario '%s' loaded with %d stages", scenario_id, len(stages))
        return Scenario(scenario_id, stages)

    def _read_payload(self, path: Path) -> Mapping[str, object]:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore

                return yaml.safe_load(text)
            except Exception:  # pragma: no cover - optional dependency path
                LOGGER.debug("PyYAML unavailable, falling back to JSON loader for %s", path)
                return json.loads(text)
        return json.loads(text)

    def _build_stage(self, data: Mapping[str, object], index: int) -> Stage:
        stage_id = str(data.get("stage_id") or f"stage_{index}")
        cycles = int(data.get("cycles", 1))
        temperature = float(data.get("temperature", 0.0))
        style_profile = data.get("style_profile")
        locks_config = data.get("locks", {})
        locks: dict[str, LockSet] = {}
        for group, config in locks_config.items():
            locks[group] = LockSet(
                allow=list(config.get("allow", [])),
                deny=list(config.get("deny", [])),
            ).normalized()
        required_terms = [str(term) for term in data.get("required_terms", [])]
        inject_rules = [str(rule) for rule in data.get("inject_rules", [])]
        return Stage(
            stage_id=stage_id,
            cycles=cycles,
            temperature=temperature,
            style_profile=str(style_profile) if style_profile else None,
            locks=locks,
            required_terms=required_terms,
            inject_rules=inject_rules,
        )


__all__ = ["Scenario", "ScenarioLoader", "Stage"]
