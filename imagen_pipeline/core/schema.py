"""Minimal JSON schema validator for internal use."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping

class SchemaValidationError(ValueError):
    """Raised when a document does not comply with its schema."""

    def __init__(self, message: str, *, path: Iterable[str] | None = None):
        if path:
            message = f"{'/'.join(path)}: {message}"
        super().__init__(message)
        self.path = list(path or [])


def _type_matches(value: Any, expected: str) -> bool:
    mapping = {
        "object": Mapping,
        "array": (list, tuple),
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    python_type = mapping.get(expected)
    if python_type is None:
        raise ValueError(f"Unsupported schema type: {expected}")
    if expected == "number" and isinstance(value, bool):
        return False
    if expected == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _validate_type(value: Any, schema: Mapping[str, Any], path: list[str]):
    expected_type = schema.get("type")
    if not expected_type:
        return
    if isinstance(expected_type, list):
        for t in expected_type:
            if _type_matches(value, t):
                return
        raise SchemaValidationError(f"expected type {expected_type}, got {type(value).__name__}", path=path)
    if not _type_matches(value, expected_type):
        raise SchemaValidationError(f"expected type {expected_type}, got {type(value).__name__}", path=path)


def _validate_enum(value: Any, schema: Mapping[str, Any], path: list[str]):
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"expected one of {schema['enum']}, got {value}", path=path)


def _validate_const(value: Any, schema: Mapping[str, Any], path: list[str]):
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"expected constant {schema['const']}, got {value}", path=path)


def _validate_properties(value: Mapping[str, Any], schema: Mapping[str, Any], path: list[str]):
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in value:
            raise SchemaValidationError(f"missing required property '{key}'", path=path + [key])
    for key, sub_schema in properties.items():
        if key in value:
            _validate(value[key], sub_schema, path + [key])
    if not schema.get("additionalProperties", True):
        extra = set(value.keys()) - set(properties.keys())
        if extra:
            raise SchemaValidationError(f"unexpected properties: {sorted(extra)}", path=path)


def _validate_array(value: Iterable[Any], schema: Mapping[str, Any], path: list[str]):
    if "minItems" in schema:
        if len(value) < schema["minItems"]:
            raise SchemaValidationError(
                f"expected at least {schema['minItems']} items, got {len(value)}", path=path
            )
    if "maxItems" in schema:
        if len(value) > schema["maxItems"]:
            raise SchemaValidationError(
                f"expected at most {schema['maxItems']} items, got {len(value)}", path=path
            )
    if schema.get("uniqueItems"):
        seen = set()
        for idx, item in enumerate(value):
            key = repr(item)
            if key in seen:
                raise SchemaValidationError("array items must be unique", path=path + [str(idx)])
            seen.add(key)
    items_schema = schema.get("items")
    if isinstance(items_schema, Mapping):
        for idx, item in enumerate(value):
            _validate(item, items_schema, path + [str(idx)])


def _validate_pattern(value: str, schema: Mapping[str, Any], path: list[str]):
    pattern = schema.get("pattern")
    if pattern and not re.fullmatch(pattern, value):
        raise SchemaValidationError(f"value '{value}' does not match pattern '{pattern}'", path=path)


def _validate_one_of(value: Any, schema: Mapping[str, Any], path: list[str]):
    if "oneOf" not in schema:
        return
    errors = []
    for sub_schema in schema["oneOf"]:
        try:
            _validate(value, sub_schema, path)
            return
        except SchemaValidationError as exc:  # pragma: no cover - narrow failure path
            errors.append(str(exc))
    raise SchemaValidationError("value did not match any allowed schema", path=path)


def _validate_all_of(value: Any, schema: Mapping[str, Any], path: list[str]):
    for sub_schema in schema.get("allOf", []):
        _validate(value, sub_schema, path)


def _validate(value: Any, schema: Mapping[str, Any], path: list[str]):
    _validate_type(value, schema, path)
    _validate_enum(value, schema, path)
    _validate_const(value, schema, path)
    if isinstance(value, Mapping):
        _validate_properties(value, schema, path)
    if isinstance(value, (list, tuple)):
        _validate_array(value, schema, path)
    if isinstance(value, str):
        _validate_pattern(value, schema, path)
    _validate_all_of(value, schema, path)
    _validate_one_of(value, schema, path)


def validate(value: Any, schema: Mapping[str, Any]) -> None:
    """Validate *value* against the supplied JSON *schema*."""

    _validate(value, schema, [])


__all__ = ["SchemaValidationError", "validate"]
