"""Core assets module for Imagen pipeline v0.9."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .schema import SchemaValidationError, validate

ASSET_GROUPS: tuple[str, ...] = (
    "actions",
    "backgrounds",
    "camera",
    "characters",
    "lighting",
    "models",
    "moods",
    "palettes",
    "poses",
    "props",
    "rules",
    "style",
    "wardrobe",
    "wardrobe_sets",
)

LOGGER = logging.getLogger("imagen.assets")


@dataclass(frozen=True)
class AssetSource:
    """Metadata describing where an asset originated from."""

    root: Path
    group: str
    path: Path


class AssetConflictError(RuntimeError):
    """Raised when conflicting asset ids are encountered in strict mode."""

    def __init__(self, group: str, asset_id: str, first: Path, second: Path):
        message = (
            f"Asset conflict for {group}:{asset_id} between "
            f"{first.as_posix()} and {second.as_posix()}"
        )
        super().__init__(message)
        self.group = group
        self.asset_id = asset_id
        self.first = first
        self.second = second


class AssetLibrary:
    """Load and provide access to asset definitions."""

    _validation_cache: Dict[Path, float] = {}

    def __init__(
        self,
        asset_roots: Iterable[Path | str],
        *,
        schema_dir: Path | str | None = None,
        fail_on_conflict: bool = False,
    ) -> None:
        self.roots = [Path(root).resolve() for root in asset_roots]
        self.fail_on_conflict = fail_on_conflict
        self.schema_dir = Path(schema_dir or Path("schemas")).resolve()
        self._asset_schema = self._load_schema("asset.schema.json")
        self._style_schema = self._load_schema("style.schema.json")
        self._catalog: Dict[str, Dict[str, Dict[str, Any]]] = {group: {} for group in ASSET_GROUPS}
        self._sources: Dict[str, Dict[str, AssetSource]] = {group: {} for group in ASSET_GROUPS}
        LOGGER.info("loading asset roots in order: %s", [r.as_posix() for r in self.roots])
        self._load_assets()

    # ------------------------------------------------------------------
    # Schema helpers
    def _load_schema(self, name: str) -> Dict[str, Any]:
        path = self.schema_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    def _load_assets(self) -> None:
        for root in self.roots:
            for group in ASSET_GROUPS:
                group_dir = root / group
                if not group_dir.exists():
                    continue
                self._load_group_dir(root, group, group_dir)

    def _load_group_dir(self, root: Path, group: str, group_dir: Path) -> None:
        try:
            entries = sorted(p for p in group_dir.iterdir() if p.suffix.lower() == ".json")
        except FileNotFoundError:
            return
        for entry in entries:
            if not entry.is_file():
                continue
            asset = self._read_asset(entry)
            self._register_asset(root, group, entry, asset)

    def _read_asset(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        schema = self._asset_schema
        if data.get("group"):
            LOGGER.warning("Field 'group' is ignored for %s", path)
        if data.get("weight") is None:
            data["weight"] = 1.0
        if "tags" not in data:
            data["tags"] = []
        if "requires" not in data:
            data["requires"] = []
        if "excludes" not in data:
            data["excludes"] = []
        if "meta" not in data:
            data["meta"] = {}
        # Validate base schema
        self._validate_with_cache(path, data, schema)
        # Additional validation for style group
        if data.get("kind"):
            # When kind is specified, treat as style schema (profile or token)
            validate(data, self._style_schema)
        return data

    def _validate_with_cache(self, path: Path, data: Mapping[str, Any], schema: Mapping[str, Any]):
        stamp = path.stat().st_mtime_ns
        cached = self._validation_cache.get(path)
        if cached == stamp:
            return
        validate(data, schema)
        self._validation_cache[path] = stamp

    def _register_asset(self, root: Path, group: str, path: Path, asset: Dict[str, Any]) -> None:
        asset_id = asset.get("id")
        if not asset_id:
            raise SchemaValidationError("missing id", path=[group, path.name])
        existing = self._catalog[group].get(asset_id)
        source = AssetSource(root=root, group=group, path=path)
        if existing is not None:
            prev_source = self._sources[group][asset_id]
            conflict_msg = (
                f"Asset id {group}:{asset_id} replaced "
                f"{prev_source.path.as_posix()} -> {path.as_posix()}"
            )
            if self.fail_on_conflict:
                raise AssetConflictError(group, asset_id, prev_source.path, path)
            LOGGER.warning(conflict_msg)
        self._catalog[group][asset_id] = asset
        self._sources[group][asset_id] = source

    # ------------------------------------------------------------------
    def groups(self) -> Iterable[str]:
        return ASSET_GROUPS

    def assets_for_group(self, group: str) -> Iterable[Dict[str, Any]]:
        return self._catalog[group].values()

    def get(self, group: str, asset_id: str) -> Dict[str, Any]:
        return self._catalog[group][asset_id]

    def source_of(self, group: str, asset_id: str) -> AssetSource:
        return self._sources[group][asset_id]

    @property
    def catalog(self) -> Mapping[str, Mapping[str, Dict[str, Any]]]:
        return self._catalog

    @property
    def active_roots(self) -> list[str]:
        return [str(r) for r in self.roots]


__all__ = [
    "ASSET_GROUPS",
    "AssetConflictError",
    "AssetLibrary",
    "AssetSource",
]
