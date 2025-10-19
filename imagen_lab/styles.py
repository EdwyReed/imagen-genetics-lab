from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .io.json_documents import (
    CatalogRegistry,
    SchemaError,
    SchemaHeader,
    StyleDocument,
)


def _sequence_merge(parent: List[str], child: List[str], mode: str) -> List[str]:
    if mode == "override":
        return list(child)
    if mode == "append_unique":
        result = list(parent)
        for item in child:
            if item not in result:
                result.append(item)
        return result
    if mode == "remove":
        removal = set(child)
        return [item for item in parent if item not in removal]
    raise ValueError(f"unsupported merge mode '{mode}' for sequence")


def _mapping_merge(
    parent: Mapping[str, object],
    child: Mapping[str, object],
    mode: str,
    *,
    deep: bool = False,
) -> Dict[str, object]:
    if mode == "override":
        return deepcopy(child) if deep else dict(child)
    if mode == "append_unique":
        result = deepcopy(parent) if deep else dict(parent)
        if deep:
            for key, value in child.items():
                result[key] = deepcopy(value)
        else:
            result.update(child)
        return result
    if mode == "remove":
        result = deepcopy(parent) if deep else dict(parent)
        for key in child.keys():
            result.pop(key, None)
        return result
    raise ValueError(f"unsupported merge mode '{mode}' for mapping")


def _templates_merge(
    parent: Mapping[str, List[str]],
    child: Mapping[str, List[str]],
    mode: str,
) -> Dict[str, List[str]]:
    if mode == "override":
        return {key: list(values) for key, values in child.items()}
    if mode == "append_unique":
        result = {key: list(values) for key, values in parent.items()}
        for key, values in child.items():
            existing = result.setdefault(key, [])
            for item in values:
                if item not in existing:
                    existing.append(item)
        return result
    if mode == "remove":
        result = {key: list(values) for key, values in parent.items()}
        for key, values in child.items():
            if not values:
                result.pop(key, None)
                continue
            if key not in result:
                continue
            removal = set(values)
            filtered = [item for item in result[key] if item not in removal]
            if filtered:
                result[key] = filtered
            else:
                result.pop(key, None)
        return result
    raise ValueError(f"unsupported merge mode '{mode}' for meso_templates")


def _document_payload(document: StyleDocument) -> Dict[str, object]:
    return {
        "brand": document.brand,
        "purpose": document.purpose,
        "style_controller": dict(document.style_controller),
        "macro_bias": dict(document.macro_bias),
        "meso_templates": {k: list(v) for k, v in document.meso_templates.items()},
        "bias_rules": list(document.bias_rules),
        "constraints": dict(document.constraints),
        "scene_notes": list(document.scene_notes),
        "extras": dict(document.extras),
    }


def _merge_payload(
    base: Mapping[str, object],
    override: Mapping[str, object],
    merge_modes: Mapping[str, str],
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "brand": override.get("brand") or base.get("brand", ""),
        "purpose": override.get("purpose") or base.get("purpose", ""),
        "style_controller": {},
        "macro_bias": {},
        "meso_templates": {},
        "bias_rules": [],
        "constraints": {},
        "scene_notes": [],
        "extras": {},
    }

    style_mode = merge_modes.get("style_controller", "override")
    result["style_controller"] = _mapping_merge(
        base.get("style_controller", {}),
        override.get("style_controller", {}),
        style_mode,
        deep=True,
    )

    macro_mode = merge_modes.get("macro_bias", "override")
    result["macro_bias"] = _mapping_merge(
        base.get("macro_bias", {}),
        override.get("macro_bias", {}),
        macro_mode,
    )

    template_mode = merge_modes.get("meso_templates", "override")
    result["meso_templates"] = _templates_merge(
        base.get("meso_templates", {}),
        override.get("meso_templates", {}),
        template_mode,
    )

    bias_mode = merge_modes.get("bias_rules", "override")
    result["bias_rules"] = _sequence_merge(
        list(base.get("bias_rules", [])),
        list(override.get("bias_rules", [])),
        bias_mode,
    )

    constraints_mode = merge_modes.get("constraints", "override")
    result["constraints"] = _mapping_merge(
        base.get("constraints", {}),
        override.get("constraints", {}),
        constraints_mode,
    )

    notes_mode = merge_modes.get("scene_notes", "override")
    result["scene_notes"] = _sequence_merge(
        list(base.get("scene_notes", [])),
        list(override.get("scene_notes", [])),
        notes_mode,
    )

    extras_mode = merge_modes.get("extras", "append_unique")
    result["extras"] = _mapping_merge(
        base.get("extras", {}),
        override.get("extras", {}),
        extras_mode,
        deep=True,
    )

    return result


def _finalise_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        "brand": payload.get("brand", ""),
        "purpose": payload.get("purpose", ""),
        "style_controller": dict(payload.get("style_controller", {})),
        "macro_bias": dict(payload.get("macro_bias", {})),
        "meso_templates": {
            key: tuple(values)
            for key, values in payload.get("meso_templates", {}).items()
        },
        "bias_rules": tuple(payload.get("bias_rules", [])),
        "constraints": dict(payload.get("constraints", {})),
        "scene_notes": tuple(payload.get("scene_notes", [])),
        "extras": dict(payload.get("extras", {})),
    }


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
        return cls.from_resolved(document, _finalise_payload(_document_payload(document)))

    @classmethod
    def from_resolved(
        cls, document: StyleDocument, resolved: Mapping[str, object]
    ) -> "StyleProfile":
        header = SchemaHeader(
            schema_version=document.header.schema_version,
            id_namespace=document.header.id_namespace,
            extends=document.header.extends,
            merge=MappingProxyType(dict(document.header.merge)),
        )

        style_controller = MappingProxyType(dict(resolved.get("style_controller", {})))
        macro_bias = MappingProxyType(dict(resolved.get("macro_bias", {})))
        meso_templates = MappingProxyType(
            {key: tuple(values) for key, values in resolved.get("meso_templates", {}).items()}
        )
        constraints = MappingProxyType(dict(resolved.get("constraints", {})))
        extras_mapping = resolved.get("extras", {})
        extras = (
            MappingProxyType(dict(extras_mapping)) if extras_mapping else MappingProxyType({})
        )

        return cls(
            header=header,
            brand=str(resolved.get("brand", document.brand)),
            purpose=str(resolved.get("purpose", document.purpose)),
            style_controller=style_controller,
            macro_bias=macro_bias,
            meso_templates=meso_templates,
            bias_rules=tuple(resolved.get("bias_rules", document.bias_rules)),
            constraints=constraints,
            scene_notes=tuple(resolved.get("scene_notes", document.scene_notes)),
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
    def load(
        cls, path: Path, *, catalogs: CatalogRegistry | None = None
    ) -> "StyleLibrary":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        styles: List[StyleProfile] = []
        documents: List[StyleDocument] = []
        sources: Dict[str, str] = {}

        def consume(payload: object, *, source: str) -> None:
            try:
                document = StyleDocument.from_raw(
                    payload, source=source, catalogs=catalogs
                )
            except SchemaError as exc:
                raise ValueError(str(exc)) from exc
            namespace = document.header.id_namespace
            if namespace in sources:
                raise ValueError(
                    f"duplicate style namespace '{namespace}' encountered in {source}"
                )
            sources[namespace] = source
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

        if not documents:
            raise ValueError("style catalog contains no valid entries")

        resolved_payloads: Dict[str, Dict[str, object]] = {}
        visiting: Dict[str, bool] = {}
        documents_by_ns: Dict[str, StyleDocument] = {
            document.header.id_namespace: document for document in documents
        }

        def resolve(namespace: str) -> Dict[str, object]:
            if namespace in resolved_payloads:
                return resolved_payloads[namespace]
            if visiting.get(namespace):
                raise ValueError(f"cycle detected in style inheritance at '{namespace}'")
            visiting[namespace] = True
            document = documents_by_ns.get(namespace)
            if document is None:
                raise ValueError(f"style namespace '{namespace}' not loaded")
            base_payload: Dict[str, object] = {}
            if document.header.extends:
                parent_ns = document.header.extends
                if parent_ns not in sources:
                    raise ValueError(
                        f"style '{namespace}' extends unknown namespace '{parent_ns}'"
                    )
                base_payload = resolve(parent_ns)
            merged = _merge_payload(
                base_payload,
                _document_payload(document),
                document.header.merge,
            )
            visiting.pop(namespace, None)
            finalised = _finalise_payload(merged)
            resolved_payloads[namespace] = finalised
            return finalised

        for document in documents:
            resolve(document.header.id_namespace)

        for document in documents:
            payload = resolved_payloads[document.header.id_namespace]
            profile = StyleProfile.from_resolved(document, payload)
            styles.append(profile)

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

