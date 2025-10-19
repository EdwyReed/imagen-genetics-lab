import json
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.characters import CharacterLibrary
from imagen_lab.db.repo.sqlite import SQLiteRepository
from imagen_lab.io.migrations import migrate_catalogs, migrate_sqlite_history
from imagen_lab.styles import StyleLibrary


def _create_legacy_catalog(tmp_path: Path) -> Path:
    catalog_dir = tmp_path / "legacy"
    catalog_dir.mkdir()

    legacy_style = {
        "style_name": "Legacy Jelly",
        "purpose": "Testing migration",
        "style_controller": {"aesthetic": "Gloss watercolor"},
        "bias_rules": ["legacy rule"],
        "scene_notes": ["studio"],
    }
    (catalog_dir / "jelly.json").write_text(json.dumps(legacy_style), encoding="utf-8")

    legacy_character = {
        "id": "legacy_muse",
        "name": "Legacy Muse",
        "summary": "Muse from legacy payload",
        "style_variants": ["Legacy Jelly"],
    }
    character_dir = catalog_dir / "characters"
    character_dir.mkdir()
    (character_dir / "legacy.json").write_text(json.dumps(legacy_character), encoding="utf-8")

    return catalog_dir


def _create_legacy_scores_db(tmp_path: Path) -> Path:
    legacy_db = tmp_path / "scores.sqlite"
    conn = sqlite3.connect(legacy_db)
    try:
        conn.execute(
            """
            CREATE TABLE scores(
                path TEXT PRIMARY KEY,
                ts INTEGER,
                nsfw REAL,
                style REAL,
                clip_style REAL,
                specular REAL,
                illu_bias REAL,
                notes TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO scores(path, ts, nsfw, style, clip_style, specular, illu_bias, notes) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                "legacy-001.jpg",
                1_700_000_000,
                12.0,
                54.0,
                32.0,
                21.0,
                11.0,
                "Legacy caption",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return legacy_db


def test_migrate_catalogs_and_sqlite(tmp_path: Path) -> None:
    legacy_dir = _create_legacy_catalog(tmp_path)
    destination = tmp_path / "export"

    summary = migrate_catalogs(legacy_dir, destination)
    assert summary.styles and summary.characters

    style_library = StyleLibrary.load(summary.styles[0].parent)
    character_library = CharacterLibrary.load(summary.characters[0])

    assert any(style.brand == "Legacy Jelly" for style in style_library.all())
    assert character_library.find("legacy_muse") is not None

    legacy_db = _create_legacy_scores_db(tmp_path)
    new_db = tmp_path / "repo.sqlite"
    sqlite_summary = migrate_sqlite_history(legacy_db, new_db, session_id="legacy-session")

    repo = SQLiteRepository(new_db)
    prompt = repo.get_prompt("legacy-001.jpg")
    score = repo.get_score("legacy-001.jpg")

    assert sqlite_summary.prompts == 1
    assert prompt is not None and prompt["status"] == "legacy"
    assert score is not None
    assert score["micro"]["schema_version"] == 1
