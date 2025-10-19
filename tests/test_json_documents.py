import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.io.json_documents import (
    CHARACTER_SCHEMA_VERSION,
    STYLE_SCHEMA_VERSION,
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
