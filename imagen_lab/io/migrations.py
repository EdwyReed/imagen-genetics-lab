"""Utilities for migrating legacy catalogues and SQLite history.

The helpers in this module upgrade the JSON catalogues living in
``catalogs/`` as well as the historical ``scores.sqlite`` database into the
schema-aware structures described in ``wiki/refactor_plan.md``.  The
functions are intentionally free of side effects other than writing the
converted files so that they can be used from tests or ad-hoc scripts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from imagen_lab.db.repo.interfaces import PromptRecord, RunRecord, ScoreRecord
from imagen_lab.db.repo.sqlite import SQLiteRepository
from imagen_lab.io.json_documents import (
    CatalogDocument,
    CharacterDocument,
    CharacterEntry,
    StyleDocument,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _style_document_payload(document: StyleDocument) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "schema_version": document.header.schema_version,
        "id_namespace": document.header.id_namespace,
        "extends": document.header.extends,
        "merge": dict(document.header.merge),
        "brand": document.brand,
        "purpose": document.purpose,
        "style_controller": dict(document.style_controller),
        "macro_bias": dict(document.macro_bias),
        "meso_templates": {key: list(values) for key, values in document.meso_templates.items()},
        "bias_rules": list(document.bias_rules),
        "constraints": dict(document.constraints),
        "scene_notes": list(document.scene_notes),
    }
    if document.extras:
        payload.update(document.extras)
    return payload


def _character_entry_payload(entry: CharacterEntry) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": entry.id,
        "name": entry.name,
        "summary": entry.summary,
        "style_variants": list(entry.style_variants),
    }
    if entry.prompt_hint:
        payload["prompt_hint"] = entry.prompt_hint
    if entry.tags:
        payload["tags"] = list(entry.tags)
    if entry.visual_traits:
        payload["visual_traits"] = list(entry.visual_traits)
    if entry.signature_props:
        payload["signature_props"] = list(entry.signature_props)
    if entry.personality:
        payload["personality"] = entry.personality
    if entry.backstory:
        payload["backstory"] = entry.backstory
    if entry.extra:
        payload.update(dict(entry.extra))
    return payload


def _character_document_payload(document: CharacterDocument) -> Dict[str, object]:
    return {
        "schema_version": document.header.schema_version,
        "id_namespace": document.header.id_namespace,
        "extends": document.header.extends,
        "merge": dict(document.header.merge),
        "characters": [_character_entry_payload(entry) for entry in document.characters],
        **({} if not document.extras else dict(document.extras)),
    }


def _catalog_document_payload(document: CatalogDocument) -> Dict[str, object]:
    return {
        "schema_version": document.header.schema_version,
        "id_namespace": document.header.id_namespace,
        "extends": document.header.extends,
        "merge": dict(document.header.merge),
        "entries": [dict(entry.payload) for entry in document.entries],
        **({} if not document.extras else dict(document.extras)),
    }


@dataclass(frozen=True)
class CatalogMigrationSummary:
    """Summary of the converted catalogue files."""

    styles: Sequence[Path]
    characters: Sequence[Path]
    primitive_catalogs: Sequence[Path]


def migrate_catalogs(legacy_root: Path, destination: Path) -> CatalogMigrationSummary:
    """Convert legacy style and character catalogues into schema-aware JSON."""

    legacy_root = Path(legacy_root)
    destination = Path(destination)

    style_targets: List[Path] = []
    for path in sorted(legacy_root.glob("*.json")):
        document = StyleDocument.from_path(path)
        target = destination / "styles" / path.name
        style_targets.append(_write_json(target, _style_document_payload(document)))

    character_targets: List[Path] = []
    character_dir = legacy_root / "characters"
    if character_dir.exists():
        for path in sorted(character_dir.glob("*.json")):
            document = CharacterDocument.from_path(path)
            target = destination / "characters" / path.name
            character_targets.append(_write_json(target, _character_document_payload(document)))

    primitive_targets: List[Path] = []
    primitive_dir = legacy_root / "primitive"
    if primitive_dir.exists():
        for path in sorted(primitive_dir.glob("*.json")):
            document = CatalogDocument.from_path(path)
            target = destination / "primitive" / path.name
            primitive_targets.append(_write_json(target, _catalog_document_payload(document)))

    return CatalogMigrationSummary(
        styles=tuple(style_targets),
        characters=tuple(character_targets),
        primitive_catalogs=tuple(primitive_targets),
    )


@dataclass(frozen=True)
class SQLiteMigrationSummary:
    """Summary of the migrated SQLite history."""

    session_id: str
    prompts: int
    scores: int


def migrate_sqlite_history(
    legacy_db: Path,
    new_db: Path,
    *,
    session_id: str | None = None,
) -> SQLiteMigrationSummary:
    """Upgrade the legacy ``scores`` table into the normalised repository schema."""

    legacy_db = Path(legacy_db)
    new_db = Path(new_db)
    if not legacy_db.exists():
        raise FileNotFoundError(legacy_db)

    repo = SQLiteRepository(new_db)
    timestamp = int(time.time())
    session = session_id or f"legacy-import-{timestamp}"

    repo.log_run(
        RunRecord(
            session_id=session,
            mode="import",
            payload={"source": str(legacy_db)},
            created_at=timestamp,
        )
    )

    with sqlite3.connect(legacy_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scores'"
        )
        if cursor.fetchone() is None:
            return SQLiteMigrationSummary(session_id=session, prompts=0, scores=0)

        columns = [row[1] for row in conn.execute("PRAGMA table_info('scores')").fetchall()]
        select_sql = "SELECT " + ",".join(columns) + " FROM scores"
        rows = [dict(zip(columns, row)) for row in conn.execute(select_sql)]

    prompt_count = 0
    for index, row in enumerate(rows, start=1):
        path = str(row.get("path") or f"{session}-legacy-{index:04d}.jpg")
        created_at = int(row.get("ts") or timestamp)
        schema_version = int(row.get("schema_version") or 1)

        metrics: Dict[str, float] = {}
        for key in ("nsfw", "style", "clip_style", "specular", "illu_bias"):
            value = row.get(key)
            if value is not None:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue

        caption_text = row.get("notes") if isinstance(row.get("notes"), str) else None

        prompt_record = PromptRecord(
            path=path,
            session_id=session,
            prompt=caption_text or "[legacy prompt unavailable]",
            params={
                "legacy_source": str(legacy_db),
                "legacy_metrics": metrics,
                "schema_version": schema_version,
            },
            gene_choices={"legacy": True, "path": path},
            option_probabilities=None,
            caption=caption_text,
            imagen_version="legacy",
            fitness=metrics.get("style", 0.0),
            created_at=created_at,
            status="legacy",
        )

        micro = dict(metrics)
        micro["schema_version"] = schema_version
        score_record = ScoreRecord(
            prompt_path=path,
            micro_metrics=micro,
            meso_metrics={},
            fitness_visual=metrics.get("style"),
            fitness_body_focus=None,
            fitness_alignment=None,
            fitness_cleanliness=None,
            fitness_era_match=None,
            fitness_novelty=None,
            clip_alignment=metrics.get("clip_style"),
            ai_artifacts=metrics.get("specular"),
            created_at=created_at,
        )

        repo.record_cycle(prompt=prompt_record, score=score_record)
        prompt_count += 1

    return SQLiteMigrationSummary(
        session_id=session,
        prompts=prompt_count,
        scores=prompt_count,
    )


__all__ = [
    "CatalogMigrationSummary",
    "SQLiteMigrationSummary",
    "migrate_catalogs",
    "migrate_sqlite_history",
]

