"""Caption payload helpers shared across Ollama integrations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from imagen_lab.scene.builder.interfaces import SceneDescription


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _style_section(scene: "SceneDescription") -> dict[str, Any]:
    payload = _as_mapping(scene.payload)
    section: dict[str, Any] = {}

    style = _as_mapping(payload.get("style"))
    if style:
        section["style"] = style

    profile = _as_mapping(payload.get("style_profile"))
    if profile:
        section["profile"] = profile

    notes = payload.get("feedback_notes")
    if isinstance(notes, Iterable) and not isinstance(notes, (str, bytes)):
        section["feedback_notes"] = [str(item) for item in notes]

    raw = scene.raw
    profile_id = getattr(raw, "profile_id", None)
    if profile_id is not None:
        section["profile_id"] = profile_id

    applied_rules = getattr(raw, "applied_rules", None)
    if applied_rules:
        section["applied_rules"] = [str(rule) for rule in applied_rules]

    catalog_id = getattr(raw, "catalog_id", None)
    if catalog_id is not None:
        section["catalog_id"] = catalog_id

    return section


def _character_section(scene: "SceneDescription") -> dict[str, Any]:
    payload = _as_mapping(scene.payload)
    character = payload.get("character")
    if isinstance(character, Mapping):
        return {str(k): v for k, v in character.items()}
    raw = getattr(scene.raw, "character", None)
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    return {}


def _scene_choices_from_model(raw_model: Any) -> list[dict[str, Any]]:
    slots = getattr(raw_model, "slots", None)
    if not isinstance(slots, Mapping):
        return []
    ordered = OrderedDict(slots)
    choices: list[dict[str, Any]] = []
    for slot, choice in ordered.items():
        option_id = getattr(choice, "option_id", None)
        label = getattr(choice, "label", None)
        probability = getattr(choice, "probability", None)
        entry = {
            "slot": slot,
            "id": option_id,
            "label": label,
        }
        if probability is not None:
            entry["probability"] = float(probability)
        metadata = getattr(choice, "metadata", None)
        if isinstance(metadata, Mapping) and metadata:
            entry["metadata"] = {str(k): v for k, v in metadata.items()}
        choices.append(entry)
    return choices


def _scene_section(scene: "SceneDescription") -> dict[str, Any]:
    section: dict[str, Any] = {
        "template_id": scene.template_id,
        "summary": scene.summary,
        "aspect_ratio": scene.aspect_ratio,
        "gene_ids": {str(k): v for k, v in scene.gene_ids.items()},
    }

    raw = scene.raw
    section["choices"] = _scene_choices_from_model(raw)
    conflicts = getattr(raw, "conflicts", None)
    if conflicts:
        section["conflicts"] = [
            _as_mapping(conflict) if isinstance(conflict, Mapping) else {"details": str(conflict)}
            for conflict in conflicts
        ]

    return section


def _collect_numeric_signals(source: Mapping[str, Any], layer: str) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for key, value in source.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        signals.append({"name": str(key), "value": numeric, "source": layer})
    return signals


def _top_signals(scene: "SceneDescription", *, limit: int = 5) -> list[dict[str, Any]]:
    raw = scene.raw
    macro = getattr(raw, "macro_snapshot", {}) if isinstance(getattr(raw, "macro_snapshot", {}), Mapping) else {}
    meso = getattr(raw, "meso_snapshot", {}) if isinstance(getattr(raw, "meso_snapshot", {}), Mapping) else {}

    signals = _collect_numeric_signals(_as_mapping(macro), "macro")
    signals.extend(_collect_numeric_signals(_as_mapping(meso), "meso"))

    if not signals:
        payload = _as_mapping(scene.payload)
        profile = _as_mapping(payload.get("style_profile"))
        focus = profile.get("component_focus")
        if isinstance(focus, Iterable):
            for entry in focus:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("component")
                try:
                    value = float(entry.get("weight"))
                except (TypeError, ValueError):
                    continue
                if not name:
                    continue
                signals.append({"name": str(name), "value": value, "source": "style"})

    signals.sort(key=lambda item: abs(item.get("value", 0.0)), reverse=True)
    return signals[:limit]


def build_caption_payload(scene: "SceneDescription") -> dict[str, Any]:
    """Compose the JSON payload expected by the Ollama caption contract."""

    return {
        "style": _style_section(scene),
        "character": _character_section(scene),
        "scene": _scene_section(scene),
        "top_signals": _top_signals(scene),
    }


__all__ = ["build_caption_payload"]
