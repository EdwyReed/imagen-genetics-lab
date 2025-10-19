"""Utilities for validating and migrating versioned JSON documents.

The loader works with lightweight JSON catalogues (styles, characters) that
ship with the wiki documents in this repository.  Each file carries a small
schema header with the fields described in :mod:`wiki/refactor_plan.md`:

``schema_version``
    Integer revision of the document schema.

``id_namespace``
    Fully-qualified identifier for the document (for example
    ``"characters:sugar_trouble@v1"``).

``extends`` and ``merge``
    Hints for inheritance/override semantics between catalogues.

The helper classes below provide a strict validator with human readable error
messages as well as a tiny migration framework for upgrading legacy payloads
into the latest schema revision.  The implementation intentionally avoids
third-party dependencies (``pydantic``/``jsonschema``) so that the tests can
run in the hermetic environment used by the exercises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


MERGE_MODES: frozenset[str] = frozenset({"override", "append_unique", "remove"})


class SchemaError(ValueError):
    """Base class for schema validation issues."""

    def __init__(self, source: str, message: str):
        super().__init__(f"{source}: {message}")
        self.source = source


class SchemaValidationError(SchemaError):
    """Raised when a document fails structural validation."""

    def __init__(self, source: str, errors: Iterable[str]):
        joined = "; ".join(errors)
        super().__init__(source, joined)
        self.errors = tuple(errors)


class SchemaMigrationError(SchemaError):
    """Raised when a document cannot be migrated to the target version."""


@dataclass(frozen=True)
class SchemaHeader:
    """Common header shared by the style and character documents."""

    schema_version: int
    id_namespace: str
    extends: Optional[str]
    merge: Mapping[str, str]


def _normalise_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_list_of_strings(value: object) -> List[str]:
    if isinstance(value, (list, tuple)):
        cleaned = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)
        return cleaned
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return [stripped]
    return []


def _ensure_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, MutableMapping):
        return value
    return {}


def _default_namespace(entries: Sequence[Mapping[str, object]], source: str, prefix: str) -> str:
    for entry in entries:
        identifier = entry.get("id")
        if isinstance(identifier, str) and identifier.strip():
            slug = identifier.strip().replace(" ", "_")
            return f"{prefix}:{slug}@legacy"
    stem = Path(source).stem if source not in {"<memory>", "<string>"} else "document"
    slug = stem.replace(" ", "_") or "catalog"
    return f"{prefix}:{slug}@legacy"


# ---------------------------------------------------------------------------
# Character documents
# ---------------------------------------------------------------------------


CHARACTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CharacterEntry:
    """Validated character payload."""

    id: str
    name: str
    summary: str
    style_variants: Tuple[str, ...]
    prompt_hint: str
    tags: Tuple[str, ...]
    visual_traits: Tuple[str, ...]
    signature_props: Tuple[str, ...]
    personality: str
    backstory: str
    extra: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, source: str, index: int) -> "CharacterEntry":
        errors: List[str] = []

        identifier = _normalise_string(raw.get("id")) or _normalise_string(raw.get("slug"))
        if not identifier:
            errors.append(f"characters[{index}].id must be a non-empty string")

        name = _normalise_string(raw.get("name"))
        summary = _normalise_string(raw.get("summary"))
        prompt_hint = _normalise_string(raw.get("prompt_hint"))
        personality = _normalise_string(raw.get("personality"))
        backstory = _normalise_string(raw.get("backstory"))

        style_variants = tuple(_as_list_of_strings(raw.get("style_variants")))
        if not style_variants:
            errors.append(f"characters[{index}].style_variants must contain at least one string")

        tags = tuple(_as_list_of_strings(raw.get("tags")))
        visual_traits = tuple(_as_list_of_strings(raw.get("visual_traits")))
        signature_props = tuple(_as_list_of_strings(raw.get("signature_props")))

        known = {
            "id",
            "slug",
            "name",
            "summary",
            "style_variants",
            "prompt_hint",
            "tags",
            "visual_traits",
            "signature_props",
            "personality",
            "backstory",
        }
        extras: Dict[str, object] = {}
        for key, value in raw.items():
            if key not in known:
                extras[key] = value

        if errors:
            raise SchemaValidationError(source, errors)

        clean_name = name or identifier
        clean_summary = summary or clean_name

        return cls(
            id=identifier,
            name=clean_name,
            summary=clean_summary,
            style_variants=style_variants,
            prompt_hint=prompt_hint,
            tags=tags,
            visual_traits=visual_traits,
            signature_props=signature_props,
            personality=personality,
            backstory=backstory,
            extra=extras,
        )


@dataclass(frozen=True)
class CharacterDocument:
    """Versioned character catalogue."""

    header: SchemaHeader
    characters: Tuple[CharacterEntry, ...]
    extras: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, data: object, *, source: str) -> "CharacterDocument":
        canonical = _upgrade_character_payload(data, source)
        return cls._from_canonical(canonical, source=source)

    @classmethod
    def from_path(cls, path: Path) -> "CharacterDocument":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - exercised in tests
            message = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
            raise SchemaError(str(path), message) from exc
        return cls.from_raw(payload, source=str(path))

    @classmethod
    def _from_canonical(cls, data: Mapping[str, object], *, source: str) -> "CharacterDocument":
        errors: List[str] = []

        version = data.get("schema_version")
        if not isinstance(version, int) or version != CHARACTER_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be integer {CHARACTER_SCHEMA_VERSION}, got {version!r}"
            )

        namespace = _normalise_string(data.get("id_namespace"))
        if not namespace:
            errors.append("id_namespace must be a non-empty string")

        extends = data.get("extends")
        if extends is not None and not isinstance(extends, str):
            errors.append("extends must be either a string or null")

        merge_field = data.get("merge", {})
        merge: Dict[str, str] = {}
        if merge_field is None:
            merge_field = {}
        if not isinstance(merge_field, Mapping):
            errors.append("merge must be an object")
        else:
            for key, value in merge_field.items():
                if not isinstance(key, str):
                    errors.append("merge keys must be strings")
                    continue
                if not isinstance(value, str) or value not in MERGE_MODES:
                    errors.append(
                        f"merge['{key}'] must be one of {sorted(MERGE_MODES)}, got {value!r}"
                    )
                    continue
                merge[key] = value

        raw_characters = data.get("characters")
        if not isinstance(raw_characters, list):
            errors.append("characters must be an array")
            raw_characters = []
        elif not raw_characters:
            errors.append("characters must contain at least one entry")

        if errors:
            raise SchemaValidationError(source, errors)

        header = SchemaHeader(
            schema_version=version,
            id_namespace=namespace,
            extends=extends,
            merge=merge,
        )

        entries: List[CharacterEntry] = []
        for idx, item in enumerate(raw_characters):
            if not isinstance(item, Mapping):
                raise SchemaValidationError(
                    source,
                    [f"characters[{idx}] must be a JSON object"],
                )
            entries.append(CharacterEntry.from_mapping(item, source=source, index=idx))

        extras: Dict[str, object] = {}
        for key, value in data.items():
            if key not in {"schema_version", "id_namespace", "extends", "merge", "characters"}:
                extras[key] = value

        return cls(header=header, characters=tuple(entries), extras=extras)

    def to_profiles(self) -> List["CharacterProfile"]:
        from imagen_lab.characters import CharacterProfile  # local import to avoid cycle

        return [
            CharacterProfile(
                id=entry.id,
                name=entry.name,
                summary=entry.summary,
                style_variants=entry.style_variants,
                prompt_hint=entry.prompt_hint,
                tags=entry.tags,
                visual_traits=entry.visual_traits,
                signature_props=entry.signature_props,
                personality=entry.personality,
                backstory=entry.backstory,
                extra=entry.extra,
            )
            for entry in self.characters
        ]


def _upgrade_character_payload(data: object, source: str) -> Mapping[str, object]:
    """Upgrade a raw JSON payload into the canonical character schema."""

    if isinstance(data, Mapping) and isinstance(data.get("schema_version"), int):
        version = data["schema_version"]
        if version == CHARACTER_SCHEMA_VERSION:
            return data
        if version > CHARACTER_SCHEMA_VERSION:
            raise SchemaMigrationError(source, f"unsupported schema version {version}")
    else:
        version = 0

    upgraded = data
    while version < CHARACTER_SCHEMA_VERSION:
        step = version + 1
        converter = _CHARACTER_MIGRATIONS.get((version, step))
        if converter is None:
            raise SchemaMigrationError(source, f"no migration path from version {version}")
        upgraded = converter(upgraded, source=source)
        if not isinstance(upgraded, Mapping):
            raise SchemaMigrationError(source, "migration did not return a JSON object")
        version = upgraded.get("schema_version", version)
        if version != step:
            raise SchemaMigrationError(
                source,
                f"migration step expected schema_version {step}, got {version!r}",
            )

    return upgraded


def _migrate_characters_v0_to_v1(payload: object, *, source: str) -> Mapping[str, object]:
    entries: List[Mapping[str, object]] = []

    if isinstance(payload, Mapping):
        if "characters" in payload:
            raw_list = payload.get("characters")
            if isinstance(raw_list, Iterable) and not isinstance(raw_list, (str, bytes)):
                for item in raw_list:
                    if isinstance(item, Mapping):
                        entries.append(item)
                    else:
                        raise SchemaMigrationError(
                            source, "legacy characters list must contain objects"
                        )
            else:
                raise SchemaMigrationError(source, "legacy characters field must be an array")
        else:
            entries.append(payload)
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        for item in payload:
            if isinstance(item, Mapping):
                entries.append(item)
            else:
                raise SchemaMigrationError(source, "legacy arrays must contain objects")
    else:
        raise SchemaMigrationError(source, "legacy character payload must be an object or array")

    if not entries:
        raise SchemaMigrationError(source, "legacy character payload contains no entries")

    namespace = _default_namespace(entries, source, "characters")

    return {
        "schema_version": CHARACTER_SCHEMA_VERSION,
        "id_namespace": namespace,
        "extends": None,
        "merge": {},
        "characters": [dict(entry) for entry in entries],
    }


_CHARACTER_MIGRATIONS: Dict[Tuple[int, int], Callable[[object], Mapping[str, object]]] = {
    (0, 1): lambda payload, source: _migrate_characters_v0_to_v1(payload, source=source),
}


# ---------------------------------------------------------------------------
# Style documents
# ---------------------------------------------------------------------------


STYLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StyleDocument:
    """Validated style catalogue."""

    header: SchemaHeader
    brand: str
    purpose: str
    style_controller: Mapping[str, object]
    macro_bias: Mapping[str, float]
    meso_templates: Mapping[str, Tuple[str, ...]]
    bias_rules: Tuple[str, ...]
    constraints: Mapping[str, float]
    scene_notes: Tuple[str, ...]
    extras: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, data: object, *, source: str) -> "StyleDocument":
        canonical = _upgrade_style_payload(data, source)
        return cls._from_canonical(canonical, source=source)

    @classmethod
    def from_path(cls, path: Path) -> "StyleDocument":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            message = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
            raise SchemaError(str(path), message) from exc
        return cls.from_raw(payload, source=str(path))

    @classmethod
    def _from_canonical(cls, data: Mapping[str, object], *, source: str) -> "StyleDocument":
        errors: List[str] = []

        version = data.get("schema_version")
        if not isinstance(version, int) or version != STYLE_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be integer {STYLE_SCHEMA_VERSION}, got {version!r}"
            )

        namespace = _normalise_string(data.get("id_namespace"))
        if not namespace:
            errors.append("id_namespace must be a non-empty string")

        extends = data.get("extends")
        if extends is not None and not isinstance(extends, str):
            errors.append("extends must be either a string or null")

        merge_field = data.get("merge", {})
        merge: Dict[str, str] = {}
        if merge_field is None:
            merge_field = {}
        if not isinstance(merge_field, Mapping):
            errors.append("merge must be an object")
        else:
            for key, value in merge_field.items():
                if not isinstance(key, str):
                    errors.append("merge keys must be strings")
                    continue
                if not isinstance(value, str) or value not in MERGE_MODES:
                    errors.append(
                        f"merge['{key}'] must be one of {sorted(MERGE_MODES)}, got {value!r}"
                    )
                    continue
                merge[key] = value

        brand = _normalise_string(data.get("brand"))
        purpose = _normalise_string(data.get("purpose"))

        style_controller = data.get("style_controller")
        if style_controller is None:
            style_controller = {}
        if not isinstance(style_controller, Mapping):
            errors.append("style_controller must be an object")
            style_controller = {}

        macro_bias = data.get("macro_bias", {})
        macro_bias_map: Dict[str, float] = {}
        if macro_bias is None:
            macro_bias = {}
        if not isinstance(macro_bias, Mapping):
            errors.append("macro_bias must be an object")
        else:
            for key, value in macro_bias.items():
                if not isinstance(key, str):
                    errors.append("macro_bias keys must be strings")
                    continue
                if not isinstance(value, (int, float)):
                    errors.append(f"macro_bias['{key}'] must be a number")
                    continue
                macro_bias_map[key] = float(value)

        meso_templates = data.get("meso_templates", {})
        meso_map: Dict[str, Tuple[str, ...]] = {}
        if meso_templates is None:
            meso_templates = {}
        if not isinstance(meso_templates, Mapping):
            errors.append("meso_templates must be an object")
        else:
            for key, value in meso_templates.items():
                if not isinstance(key, str):
                    errors.append("meso_templates keys must be strings")
                    continue
                items = tuple(_as_list_of_strings(value))
                meso_map[key] = items

        bias_rules = tuple(_as_list_of_strings(data.get("bias_rules")))
        constraints_field = data.get("constraints", {})
        constraints: Dict[str, float] = {}
        if constraints_field is None:
            constraints_field = {}
        if not isinstance(constraints_field, Mapping):
            errors.append("constraints must be an object")
        else:
            for key, value in constraints_field.items():
                if not isinstance(key, str):
                    errors.append("constraints keys must be strings")
                    continue
                if not isinstance(value, (int, float)):
                    errors.append(f"constraints['{key}'] must be a number")
                    continue
                constraints[key] = float(value)

        scene_notes = tuple(_as_list_of_strings(data.get("scene_notes")))

        if errors:
            raise SchemaValidationError(source, errors)

        header = SchemaHeader(
            schema_version=version,
            id_namespace=namespace,
            extends=extends,
            merge=merge,
        )

        extras: Dict[str, object] = {}
        for key, value in data.items():
            if key not in {
                "schema_version",
                "id_namespace",
                "extends",
                "merge",
                "brand",
                "purpose",
                "style_controller",
                "macro_bias",
                "meso_templates",
                "bias_rules",
                "constraints",
                "scene_notes",
            }:
                extras[key] = value

        return cls(
            header=header,
            brand=brand,
            purpose=purpose,
            style_controller=dict(style_controller),
            macro_bias=macro_bias_map,
            meso_templates=meso_map,
            bias_rules=bias_rules,
            constraints=constraints,
            scene_notes=scene_notes,
            extras=extras,
        )


def _upgrade_style_payload(data: object, source: str) -> Mapping[str, object]:
    if isinstance(data, Mapping) and isinstance(data.get("schema_version"), int):
        version = data["schema_version"]
        if version == STYLE_SCHEMA_VERSION:
            return data
        if version > STYLE_SCHEMA_VERSION:
            raise SchemaMigrationError(source, f"unsupported schema version {version}")
    else:
        version = 0

    upgraded = data
    while version < STYLE_SCHEMA_VERSION:
        step = version + 1
        converter = _STYLE_MIGRATIONS.get((version, step))
        if converter is None:
            raise SchemaMigrationError(source, f"no migration path from version {version}")
        upgraded = converter(upgraded, source=source)
        if not isinstance(upgraded, Mapping):
            raise SchemaMigrationError(source, "migration did not return a JSON object")
        version = upgraded.get("schema_version", version)
        if version != step:
            raise SchemaMigrationError(
                source,
                f"migration step expected schema_version {step}, got {version!r}",
            )
    return upgraded


def _migrate_style_v0_to_v1(payload: object, *, source: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise SchemaMigrationError(source, "legacy style payload must be an object")

    namespace = _normalise_string(payload.get("id_namespace"))
    if not namespace:
        version_hint = _normalise_string(payload.get("version")) or "legacy"
        default_ns = _default_namespace([payload], source, "style")
        base = default_ns.split(":", 1)[-1].split("@", 1)[0]
        namespace = f"style:{base}@{version_hint}"

    brand = _normalise_string(payload.get("brand"))
    if not brand:
        brand = _normalise_string(payload.get("style_name")) or "Custom Style"

    purpose = _normalise_string(payload.get("purpose"))

    style_controller = _ensure_mapping(payload.get("style_controller"))

    meso_templates: Dict[str, List[str]] = {}
    if "meso_templates" in payload:
        raw_templates = payload.get("meso_templates")
        if isinstance(raw_templates, Mapping):
            for key, value in raw_templates.items():
                meso_templates[str(key)] = _as_list_of_strings(value)

    return {
        "schema_version": STYLE_SCHEMA_VERSION,
        "id_namespace": namespace,
        "extends": payload.get("extends"),
        "merge": payload.get("merge", {}),
        "brand": brand,
        "purpose": purpose,
        "style_controller": dict(style_controller),
        "macro_bias": dict(_ensure_mapping(payload.get("macro_bias"))),
        "meso_templates": meso_templates,
        "bias_rules": list(_as_list_of_strings(payload.get("bias_rules"))),
        "constraints": dict(_ensure_mapping(payload.get("constraints"))),
        "scene_notes": list(_as_list_of_strings(payload.get("scene_notes"))),
        "legacy_payload": dict(payload),
    }


_STYLE_MIGRATIONS: Dict[Tuple[int, int], Callable[[object], Mapping[str, object]]] = {
    (0, 1): lambda payload, source: _migrate_style_v0_to_v1(payload, source=source),
}


__all__ = [
    "SchemaError",
    "SchemaValidationError",
    "SchemaMigrationError",
    "SchemaHeader",
    "CharacterEntry",
    "CharacterDocument",
    "StyleDocument",
    "CHARACTER_SCHEMA_VERSION",
    "STYLE_SCHEMA_VERSION",
    "MERGE_MODES",
]

