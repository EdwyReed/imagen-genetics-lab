"""Structured representation of a sampled scene."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence


@dataclass(frozen=True)
class CatalogReference:
    """Reference to the source catalog section for a gene."""

    catalog_id: str
    section: str

    def to_dict(self) -> Dict[str, str]:
        return {"catalog_id": self.catalog_id, "section": self.section}


@dataclass(frozen=True)
class GeneOptionProbability:
    """Probability entry for a single gene option."""

    option_id: str
    probability: float
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.option_id,
            "probability": self.probability,
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SceneSlotChoice:
    """Concrete choice for a gene slot (pose, lighting, wardrobe, etc.)."""

    slot: str
    option_id: str
    label: str
    probability: float
    catalog_reference: CatalogReference
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "slot": self.slot,
            "id": self.option_id,
            "label": self.label,
            "probability": self.probability,
            "catalog_reference": self.catalog_reference.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SceneModel:
    """Serializable snapshot of a scene and the probabilities behind it."""

    catalog_id: str
    slots: Mapping[str, SceneSlotChoice]
    option_probabilities: Mapping[str, Sequence[GeneOptionProbability]]
    applied_rules: Sequence[str]
    conflicts: Sequence[Mapping[str, Any]]
    macro_snapshot: Mapping[str, Any]
    meso_snapshot: Mapping[str, Any]
    profile_id: str | None = None
    sfw_level: float | None = None
    temperature: float | None = None

    def _conflicts_payload(self) -> list[Dict[str, Any]]:
        payload: list[Dict[str, Any]] = []
        for conflict in self.conflicts:
            if isinstance(conflict, Mapping):
                payload.append(dict(conflict))
            else:
                payload.append({"details": str(conflict)})
        return payload

    def choices_payload(self) -> Dict[str, Any]:
        choices: Dict[str, Any] = {
            slot: choice.to_dict() for slot, choice in self.slots.items()
        }
        option_probs: Dict[str, Any] = {
            slot: [prob.to_dict() for prob in probs]
            for slot, probs in self.option_probabilities.items()
        }
        return {
            "catalog_id": self.catalog_id,
            "profile_id": self.profile_id,
            "sfw_level": self.sfw_level,
            "temperature": self.temperature,
            "macro_snapshot": dict(self.macro_snapshot),
            "meso_snapshot": dict(self.meso_snapshot),
            "choices": choices,
            "option_probabilities": option_probs,
            "applied_rules": [str(rule) for rule in self.applied_rules],
            "conflicts": self._conflicts_payload(),
        }

    def gene_choices_json(self) -> str:
        return json.dumps(self.choices_payload(), ensure_ascii=False, sort_keys=True)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "scene_model": self.choices_payload(),
            "gene_choices_json": self.gene_choices_json(),
        }

    def summary(self) -> str:
        ordered = [f"{slot}: {choice.label}" for slot, choice in self.slots.items()]
        return "; ".join(ordered)
