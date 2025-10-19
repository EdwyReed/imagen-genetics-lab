from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional, Sequence, Tuple, Mapping

from .io.json_documents import CharacterDocument, SchemaError


EMPTY_MAPPING: Mapping[str, object] = MappingProxyType({})


def _clean(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_list(value: object) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@dataclass(frozen=True)
class CharacterProfile:
    """Simple container describing a catalog character option."""

    id: str
    name: str
    summary: str
    style_variants: tuple[str, ...]
    prompt_hint: str = ""
    tags: tuple[str, ...] = ()
    visual_traits: tuple[str, ...] = ()
    signature_props: tuple[str, ...] = ()
    personality: str = ""
    backstory: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "CharacterProfile":
        identifier = _clean(raw.get("id")) or _clean(raw.get("slug"))
        if not identifier:
            raise ValueError("character entry missing 'id'")
        name = _clean(raw.get("name")) or identifier
        summary = _clean(raw.get("summary")) or name
        prompt_hint = _clean(raw.get("prompt_hint"))
        style_variants = tuple(_as_list(raw.get("style_variants")))
        tags = tuple(_as_list(raw.get("tags")))
        visual_traits = tuple(_as_list(raw.get("visual_traits")))
        signature_props = tuple(_as_list(raw.get("signature_props")))
        personality = _clean(raw.get("personality"))
        backstory = _clean(raw.get("backstory"))
        known_keys = {
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
            if key not in known_keys:
                extras[key] = value
        extra_mapping = MappingProxyType(dict(extras)) if extras else EMPTY_MAPPING
        return cls(
            id=identifier,
            name=name,
            summary=summary,
            style_variants=style_variants,
            prompt_hint=prompt_hint,
            tags=tags,
            visual_traits=visual_traits,
            signature_props=signature_props,
            personality=personality,
            backstory=backstory,
            extra=extra_mapping,
        )

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "style_variants": list(self.style_variants),
            "prompt_hint": self.prompt_hint,
            "tags": list(self.tags),
        }
        if self.visual_traits:
            data["visual_traits"] = list(self.visual_traits)
        if self.signature_props:
            data["signature_props"] = list(self.signature_props)
        if self.personality:
            data["personality"] = self.personality
        if self.backstory:
            data["backstory"] = self.backstory
        if self.extra:
            data["extra"] = dict(self.extra)
        return data


class CharacterLibrary:
    """Lookup helper for characters defined in a separate catalog."""

    def __init__(
        self,
        characters: Sequence[CharacterProfile],
        *,
        documents: Sequence[CharacterDocument] | None = None,
    ):
        self._characters: List[CharacterProfile] = list(characters)
        self._by_id: Dict[str, CharacterProfile] = {c.id: c for c in self._characters}
        self._by_variant: Dict[str, List[CharacterProfile]] = {}
        for profile in self._characters:
            for variant in profile.style_variants:
                self._by_variant.setdefault(variant, []).append(profile)
        self._documents: Tuple[CharacterDocument, ...] = tuple(documents or ())

    @classmethod
    def load(cls, path: Path) -> "CharacterLibrary":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        characters: List[CharacterProfile] = []
        documents: List[CharacterDocument] = []

        def consume_payload(payload: object, *, source: str) -> None:
            try:
                document = CharacterDocument.from_raw(payload, source=source)
            except SchemaError as exc:
                raise ValueError(str(exc)) from exc
            profiles = document.to_profiles()
            if not profiles:
                raise ValueError(f"{source}: character document contains no entries")
            characters.extend(profiles)
            documents.append(document)

        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                if not child.is_file():
                    continue
                try:
                    payload = json.loads(child.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    message = f"{child}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
                    raise ValueError(message) from exc
                consume_payload(payload, source=str(child))
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                message = f"{path}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
                raise ValueError(message) from exc
            consume_payload(payload, source=str(path))

        if not characters:
            raise ValueError("character catalog contains no valid entries")
        return cls(characters, documents=documents)

    def to_dict(self) -> Dict[str, object]:
        return {"characters": [c.to_dict() for c in self._characters]}

    @property
    def documents(self) -> Tuple[CharacterDocument, ...]:
        return self._documents

    def find(self, character_id: str | None) -> Optional[CharacterProfile]:
        if not character_id:
            return None
        return self._by_id.get(character_id)

    def candidates_for(self, variant: str | None) -> List[CharacterProfile]:
        if variant and variant in self._by_variant:
            return list(self._by_variant[variant])
        return list(self._characters)

    def choose(
        self,
        variant: str | None = None,
        *,
        default_id: str | None = None,
    ) -> CharacterProfile:
        if default_id:
            chosen = self.find(default_id)
            if chosen is not None:
                return chosen
        candidates = self.candidates_for(variant)
        if not candidates:
            raise ValueError("no characters available for selection")
        return random.choice(candidates)

