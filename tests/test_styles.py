import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.io.json_documents import STYLE_SCHEMA_VERSION
from imagen_lab.styles import StyleLibrary


def test_style_library_loads_directory(tmp_path: Path) -> None:
    styles_dir = tmp_path / "styles"
    styles_dir.mkdir()

    canonical = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:test@v1",
        "extends": None,
        "merge": {},
        "brand": "Test Style",
        "purpose": "Testing style loader",
        "style_controller": {"aesthetic": "Test aesthetic"},
        "macro_bias": {"gloss_priority": 0.5},
        "meso_templates": {"palette": ["palette:test"]},
        "bias_rules": ["rule one"],
        "constraints": {"coverage_min": 0.4},
        "scene_notes": ["notes"],
    }

    legacy = {
        "version": "2.6",
        "style_name": "Legacy Style",
        "purpose": "Legacy purpose",
        "style_controller": {"aesthetic": "Legacy aesthetic"},
        "bias_rules": ["legacy rule"],
        "scene_notes": "studio",
    }

    (styles_dir / "canonical.json").write_text(json.dumps(canonical), encoding="utf-8")
    (styles_dir / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")

    library = StyleLibrary.load(styles_dir)

    assert library.documents
    assert library.documents[0].header.schema_version == STYLE_SCHEMA_VERSION

    canonical_profile = library.find("style:test@v1")
    assert canonical_profile is not None
    assert canonical_profile.brand == "Test Style"
    assert canonical_profile.scene_notes == ("notes",)

    legacy_profile = next(
        (profile for profile in library.all() if profile.brand == "Legacy Style"),
        None,
    )
    assert legacy_profile is not None
    assert legacy_profile.header.schema_version == STYLE_SCHEMA_VERSION
    assert legacy_profile.bias_rules == ("legacy rule",)


def test_style_library_validates_schema(tmp_path: Path) -> None:
    payload = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:invalid@v1",
        "extends": None,
        "merge": {"scene_notes": "invalid"},
        "brand": "Invalid",
        "purpose": "",
        "style_controller": {},
        "macro_bias": {},
        "meso_templates": {},
        "bias_rules": [],
        "constraints": {},
        "scene_notes": [],
    }

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        StyleLibrary.load(path)

    assert "merge['scene_notes']" in str(exc.value)
