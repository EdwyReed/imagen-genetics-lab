from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .io.json_documents import SchemaError, SchemaHeader, StyleDocument

@dataclass(frozen=True)
class StyleProfile:
    """Structured representation of a style document."""

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
    def from_document(cls, document: StyleDocument) -> "StyleProfile":
        header = SchemaHeader(
            schema_version=document.header.schema_version,
            id_namespace=document.header.id_namespace,
            extends=document.header.extends,
            merge=MappingProxyType(dict(document.header.merge)),
        )

        style_controller = MappingProxyType(dict(document.style_controller))
        macro_bias = MappingProxyType(dict(document.macro_bias))
        meso_templates = MappingProxyType(
            {key: tuple(values) for key, values in document.meso_templates.items()}
        )
        constraints = MappingProxyType(dict(document.constraints))
        extras = MappingProxyType(dict(document.extras)) if document.extras else MappingProxyType({})

        return cls(
            header=header,
            brand=document.brand,
            purpose=document.purpose,
            style_controller=style_controller,
            macro_bias=macro_bias,
            meso_templates=meso_templates,
            bias_rules=document.bias_rules,
            constraints=constraints,
            scene_notes=document.scene_notes,
            extras=extras,
        )

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "schema_version": self.header.schema_version,
            "id_namespace": self.header.id_namespace,
            "extends": self.header.extends,
            "merge": dict(self.header.merge),
            "brand": self.brand,
            "purpose": self.purpose,
            "style_controller": dict(self.style_controller),
            "macro_bias": dict(self.macro_bias),
            "meso_templates": {k: list(v) for k, v in self.meso_templates.items()},
            "bias_rules": list(self.bias_rules),
            "constraints": dict(self.constraints),
            "scene_notes": list(self.scene_notes),
        }
        if self.extras:
            data.update(self.extras)
        return data


class StyleLibrary:
    """Loader for schema-aware style catalogues."""

    def __init__(
        self,
        styles: Sequence[StyleProfile],
        *,
        documents: Sequence[StyleDocument] | None = None,
    ):
        self._styles: Tuple[StyleProfile, ...] = tuple(styles)
        self._by_namespace: Dict[str, StyleProfile] = {
            style.header.id_namespace: style for style in self._styles
        }
        self._documents: Tuple[StyleDocument, ...] = tuple(documents or ())

    @classmethod
    def load(cls, path: Path) -> "StyleLibrary":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        styles: List[StyleProfile] = []
        documents: List[StyleDocument] = []

        def consume(payload: object, *, source: str) -> None:
            try:
                document = StyleDocument.from_raw(payload, source=source)
            except SchemaError as exc:
                raise ValueError(str(exc)) from exc
            profile = StyleProfile.from_document(document)
            styles.append(profile)
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
                consume(payload, source=str(child))
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                message = f"{path}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
                raise ValueError(message) from exc
            consume(payload, source=str(path))

        if not styles:
            raise ValueError("style catalog contains no valid entries")

        return cls(styles, documents=documents)

    def all(self) -> List[StyleProfile]:
        return list(self._styles)

    def find(self, namespace: str | None) -> Optional[StyleProfile]:
        if not namespace:
            return None
        return self._by_namespace.get(namespace)

    def default(self) -> StyleProfile:
        if not self._styles:
            raise ValueError("no styles available")
        return self._styles[0]

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        return {"styles": [style.to_dict() for style in self._styles]}

    @property
    def documents(self) -> Tuple[StyleDocument, ...]:
        return self._documents


__all__ = ["StyleProfile", "StyleLibrary"]

