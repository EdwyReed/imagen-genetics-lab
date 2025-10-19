import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.io.json_documents import (
    CATALOG_SCHEMA_VERSION,
    CatalogDocument,
    CatalogRegistry,
    STYLE_SCHEMA_VERSION,
)
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


def test_style_library_resolves_extends_merge(tmp_path: Path) -> None:
    styles_dir = tmp_path / "styles"
    styles_dir.mkdir()

    catalog_doc = CatalogDocument.from_raw(
        {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "id_namespace": "catalog:palettes@2024-05-01",
            "extends": None,
            "merge": {},
            "entries": [
                {"id": "palette:pastel_dawn"},
                {"id": "palette:neon_spark"},
            ],
        },
        source="<memory>",
    )
    registry = CatalogRegistry([catalog_doc])

    base = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:base@v1",
        "extends": None,
        "merge": {
            "scene_notes": "append_unique",
            "bias_rules": "append_unique",
        },
        "brand": "Base Style",
        "purpose": "Base purpose",
        "style_controller": {"aesthetic": "Base"},
        "macro_bias": {"gloss_priority": 0.3, "novelty_preference": 0.6},
        "meso_templates": {"palette": ["palette:pastel_dawn"]},
        "bias_rules": ["keep", "remove"],
        "constraints": {"coverage_min": 0.2},
        "scene_notes": ["base"],
    }

    child = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:child@v1",
        "extends": "style:base@v1",
        "merge": {
            "scene_notes": "append_unique",
            "bias_rules": "remove",
            "meso_templates": "append_unique",
            "macro_bias": "remove",
        },
        "brand": "Child Style",
        "purpose": "Child purpose",
        "style_controller": {"aesthetic": "Child", "detail": "extra"},
        "macro_bias": {"novelty_preference": 0.0},
        "meso_templates": {
            "palette": ["palette:pastel_dawn", "palette:neon_spark"],
        },
        "bias_rules": ["remove"],
        "constraints": {"coverage_min": 0.3},
        "scene_notes": ["child"],
    }

    (styles_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (styles_dir / "child.json").write_text(json.dumps(child), encoding="utf-8")

    library = StyleLibrary.load(styles_dir, catalogs=registry)

    child_profile = library.find("style:child@v1")
    assert child_profile is not None
    assert child_profile.brand == "Child Style"
    assert child_profile.scene_notes == ("base", "child")
    assert child_profile.bias_rules == ("keep",)
    assert child_profile.meso_templates["palette"] == (
        "palette:pastel_dawn",
        "palette:neon_spark",
    )
    assert "novelty_preference" not in child_profile.macro_bias
    assert child_profile.constraints["coverage_min"] == 0.3

    base_profile = library.find("style:base@v1")
    assert base_profile is not None
    assert base_profile.bias_rules == ("keep", "remove")
