import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.io.json_documents import (
    CATALOG_SCHEMA_VERSION,
    CHARACTER_SCHEMA_VERSION,
    STYLE_SCHEMA_VERSION,
    CatalogDocument,
    CatalogRegistry,
    CharacterDocument,
    SchemaValidationError,
    StyleDocument,
)


def test_character_document_upgrades_legacy_payload() -> None:
    legacy = {
        "id": "legacy_hero",
        "name": "Legacy Hero",
        "summary": "Hero from legacy payload",
        "style_variants": ["Retro"],
    }

    document = CharacterDocument.from_raw(legacy, source="<memory>")

    assert document.header.schema_version == CHARACTER_SCHEMA_VERSION
    assert document.characters[0].id == "legacy_hero"
    assert document.characters[0].style_variants == ("Retro",)
    assert document.header.id_namespace.startswith("characters:")


def test_style_document_upgrades_legacy_payload() -> None:
    legacy = {
        "version": "2.6",
        "style_name": "Jelly Pin-Up",
        "purpose": "Legacy style description",
        "style_controller": {"aesthetic": "Gloss watercolor"},
        "bias_rules": ["rule one"],
        "scene_notes": "studio lighting",
    }

    document = StyleDocument.from_raw(legacy, source="<memory>")

    assert document.header.schema_version == STYLE_SCHEMA_VERSION
    assert document.brand == "Jelly Pin-Up"
    assert document.purpose == "Legacy style description"
    assert document.bias_rules == ("rule one",)
    assert document.header.id_namespace.startswith("style:document@2.6")


def test_style_document_validates_merge_modes() -> None:
    payload = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:test@v1",
        "extends": None,
        "merge": {"scene_notes": "invalid"},
        "brand": "Test",
        "purpose": "",
        "style_controller": {},
        "macro_bias": {},
        "meso_templates": {},
        "bias_rules": [],
        "constraints": {},
        "scene_notes": [],
    }

    with pytest.raises(SchemaValidationError) as exc:
        StyleDocument.from_raw(payload, source="<memory>")

    assert "merge['scene_notes']" in str(exc.value)


def test_catalog_document_parses_entries() -> None:
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "id_namespace": "catalog:palettes@2024-05-01",
        "extends": None,
        "merge": {},
        "entries": [
            {"id": "palette:pastel_dawn", "label": "Pastel Dawn"},
            {"id": "palette:neon_twilight", "label": "Neon Twilight"},
        ],
    }

    document = CatalogDocument.from_raw(payload, source="<memory>")

    assert document.header.schema_version == CATALOG_SCHEMA_VERSION
    assert document.catalog_version == "2024-05-01"
    assert [entry.id for entry in document.entries] == [
        "palette:pastel_dawn",
        "palette:neon_twilight",
    ]


def test_style_document_validates_catalog_references() -> None:
    catalog_payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "id_namespace": "catalog:palettes@2024-05-01",
        "extends": None,
        "merge": {},
        "entries": [
            {"id": "palette:pastel_dawn"},
        ],
    }
    registry = CatalogRegistry([CatalogDocument.from_raw(catalog_payload, source="<memory>")])

    valid = {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": "style:catalog-test@v1",
        "extends": None,
        "merge": {},
        "brand": "Catalog Valid",
        "purpose": "Check references",
        "style_controller": {},
        "macro_bias": {},
        "meso_templates": {"palette": ["palette:pastel_dawn"]},
        "bias_rules": [],
        "constraints": {},
        "scene_notes": [],
    }

    StyleDocument.from_raw(valid, source="<memory>", catalogs=registry)

    invalid = dict(valid)
    invalid["meso_templates"] = {"palette": ["palette:missing"]}

    with pytest.raises(SchemaValidationError) as exc:
        StyleDocument.from_raw(invalid, source="<memory>", catalogs=registry)

    assert "unknown catalog entry 'palette:missing'" in str(exc.value)
