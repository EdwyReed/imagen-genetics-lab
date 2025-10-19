import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.characters import CharacterLibrary, CharacterProfile
from imagen_lab.io.json_documents import CHARACTER_SCHEMA_VERSION


def test_character_library_returns_variant_specific_profile(monkeypatch) -> None:
    library = CharacterLibrary(
        [
            CharacterProfile(
                id="variant_a",
                name="Variant A",
                summary="Variant A summary",
                style_variants=("Variant A",),
            ),
            CharacterProfile(
                id="variant_b",
                name="Variant B",
                summary="Variant B summary",
                style_variants=("Variant B",),
            ),
        ]
    )

    chosen = library.choose("Variant A")

    assert chosen.id == "variant_a"


def test_character_library_uses_default_when_variant_missing() -> None:
    library = CharacterLibrary(
        [
            CharacterProfile(
                id="default_pick",
                name="Default",
                summary="Default summary",
                style_variants=("Variant A",),
            ),
            CharacterProfile(
                id="variant_b",
                name="Variant B",
                summary="Variant B summary",
                style_variants=("Variant B",),
            ),
        ]
    )

    chosen = library.choose("Unknown", default_id="default_pick")

    assert chosen.id == "default_pick"


def test_character_library_loads_directory(tmp_path: Path) -> None:
    characters_dir = tmp_path / "characters"
    characters_dir.mkdir()

    first = {
        "id": "leaf_guardian",
        "name": "Leaf Guardian",
        "summary": "Guardian with leafy crown",
        "style_variants": ["Forest"],
        "visual_traits": ["leaf crown", "emerald eyes"],
        "signature_props": ["vine staff"],
    }
    second = {
        "characters": [
            {
                "id": "mist_sprite",
                "name": "Mist Sprite",
                "summary": "Misty friend",
                "style_variants": ["Forest"],
                "tags": ["mist"],
                "personality": "Whispers in fog",
            }
        ]
    }

    (characters_dir / "leaf.json").write_text(json.dumps(first), encoding="utf-8")
    (characters_dir / "mist.json").write_text(json.dumps(second), encoding="utf-8")

    library = CharacterLibrary.load(characters_dir)

    ids = {profile.id for profile in library.candidates_for("Forest")}
    assert ids == {"leaf_guardian", "mist_sprite"}

    guardian = library.find("leaf_guardian")
    assert guardian is not None
    assert "leaf crown" in guardian.visual_traits

    assert library.documents
    assert library.documents[0].header.schema_version == CHARACTER_SCHEMA_VERSION


def test_character_library_validates_schema(tmp_path: Path) -> None:
    payload = {
        "schema_version": CHARACTER_SCHEMA_VERSION,
        "id_namespace": "characters:test@v1",
        "extends": None,
        "merge": {},
        "characters": [
            {
                "name": "Nameless",
                "summary": "Missing id",
                "style_variants": ["Test"],
            }
        ],
    }

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        CharacterLibrary.load(path)

    assert "characters[0].id must be a non-empty string" in str(exc.value)
