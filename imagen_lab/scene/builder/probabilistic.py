"""Probabilistic scene builder with bias-engine integration."""
from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Sequence

from imagen_lab.bias.engine.interfaces import BiasContext, BiasEngineProtocol
from imagen_lab.catalog import Catalog
from imagen_lab.scene.model import (
    CatalogReference,
    GeneOptionProbability,
    SceneModel,
    SceneSlotChoice,
)

from .interfaces import SceneBuilderProtocol, SceneDescription, SceneRequest


def _extract_float(option: Mapping[str, Any], key: str, default: float = 0.3) -> float:
    value = option.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _option_label(option: Mapping[str, Any]) -> str:
    for key in ("name", "label", "desc", "note"):
        value = option.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    option_id = option.get("id")
    return str(option_id) if option_id is not None else "unknown"


def _option_metadata(slot: str, option: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    desc = option.get("desc") or option.get("note")
    if isinstance(desc, str) and desc.strip():
        metadata["description"] = desc.strip()
    name = option.get("name")
    if isinstance(name, str) and name.strip():
        metadata["name"] = name.strip()
    nsfw_value = option.get("nsfw")
    if nsfw_value is not None:
        try:
            metadata["nsfw"] = float(nsfw_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    if slot == "wardrobe" and option.get("items"):
        metadata["items"] = list(option.get("items", []))
    if slot == "palette" and option.get("colors"):
        metadata["colors"] = list(option.get("colors", []))
    if slot == "mood" and option.get("words"):
        metadata["words"] = list(option.get("words", []))
    return metadata


@dataclass
class ProbabilisticSceneBuilder(SceneBuilderProtocol):
    """Builds a :class:`SceneModel` using catalog data and a bias engine."""

    catalog: Catalog
    bias_engine: BiasEngineProtocol
    catalog_id: str | None = None
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self._rng = self.rng if self.rng is not None else random
        self._catalog_id = (
            self.catalog_id
            if self.catalog_id
            else str(self.catalog.raw.get("version") or "catalog")
        )

    # ------------------------------------------------------------------
    # SceneBuilderProtocol implementation
    # ------------------------------------------------------------------
    def build_scene(self, request: SceneRequest) -> SceneDescription:
        context = BiasContext(
            profile_id=request.profile_id,
            macro_snapshot=request.macro_snapshot or {},
            meso_snapshot=request.meso_snapshot or {},
            sfw_level=request.sfw_level,
            temperature=request.temperature,
        )
        bias_data = self.bias_engine.compute_bias(context)
        slot_targets: Mapping[str, Mapping[str, Any]] = bias_data.get("slot_targets", {})
        applied_rules = list(bias_data.get("applied_rules", []))
        conflicts = list(bias_data.get("conflicts", []))
        effective_sfw = float(bias_data.get("sfw_level", request.sfw_level))

        slot_order = ["pose", "lighting", "palette", "wardrobe", "background", "mood"]
        slots: "OrderedDict[str, SceneSlotChoice]" = OrderedDict()
        option_probabilities: "OrderedDict[str, Sequence[GeneOptionProbability]]" = OrderedDict()
        gene_ids: Dict[str, str] = {}

        for slot in slot_order:
            options = self._slot_options(slot)
            if not options:
                raise ValueError(f"catalog does not define any options for slot '{slot}'")
            slot_bias = slot_targets.get(slot, {}) if slot_targets else {}
            probabilities, choice_idx, meta_entries = self._compute_slot_probabilities(
                slot,
                options,
                request,
                slot_bias,
                effective_sfw,
            )

            option_probabilities[slot] = tuple(probabilities)
            chosen_option = options[choice_idx]
            chosen_meta = meta_entries[choice_idx]
            option_id = str(chosen_option.get("id"))
            slots[slot] = SceneSlotChoice(
                slot=slot,
                option_id=option_id,
                label=_option_label(chosen_option),
                probability=probabilities[choice_idx].probability,
                catalog_reference=CatalogReference(self._catalog_id, slot),
                metadata=chosen_meta,
            )
            gene_ids[slot] = option_id

        scene_model = SceneModel(
            catalog_id=self._catalog_id,
            slots=slots,
            option_probabilities=option_probabilities,
            applied_rules=tuple(applied_rules),
            conflicts=tuple(conflicts),
            macro_snapshot=context.macro_snapshot,
            meso_snapshot=context.meso_snapshot,
            profile_id=request.profile_id,
            sfw_level=effective_sfw,
            temperature=request.temperature,
        )

        caption_bounds = self._caption_bounds()
        summary = scene_model.summary()
        return SceneDescription(
            template_id=request.template_id or "scene:probabilistic_v1",
            caption_bounds=caption_bounds,
            aspect_ratio=self._aspect_ratio(slots),
            gene_ids=gene_ids,
            payload=scene_model.to_payload(),
            summary=summary,
            raw=scene_model,
        )

    def rebuild_from_genes(
        self,
        genes: Mapping[str, str | None],
        request: SceneRequest,
    ) -> SceneDescription:
        # Rebuilding reuses the same probabilistic machinery but locks the chosen ids.
        description = self.build_scene(request)
        overridden_slots: Dict[str, SceneSlotChoice] = {}
        option_probabilities: Dict[str, Sequence[GeneOptionProbability]] = {}
        for slot, choice in description.raw.slots.items():
            override_id = genes.get(slot)
            if override_id:
                options = self._slot_options(slot)
                matches = [opt for opt in options if str(opt.get("id")) == override_id]
                if matches:
                    option = matches[0]
                    metadata = _option_metadata(slot, option)
                    probability_entry = GeneOptionProbability(
                        option_id=override_id,
                        probability=1.0,
                        label=_option_label(option),
                        metadata=metadata,
                    )
                    overridden_slots[slot] = SceneSlotChoice(
                        slot=slot,
                        option_id=override_id,
                        label=_option_label(option),
                        probability=1.0,
                        catalog_reference=CatalogReference(self._catalog_id, slot),
                        metadata=metadata,
                    )
                    option_probabilities[slot] = (probability_entry,)
                    continue
            overridden_slots[slot] = choice
            option_probabilities[slot] = description.raw.option_probabilities[slot]

        scene_model = SceneModel(
            catalog_id=description.raw.catalog_id,
            slots=OrderedDict(overridden_slots.items()),
            option_probabilities=OrderedDict(option_probabilities.items()),
            applied_rules=description.raw.applied_rules,
            conflicts=description.raw.conflicts,
            macro_snapshot=description.raw.macro_snapshot,
            meso_snapshot=description.raw.meso_snapshot,
            profile_id=description.raw.profile_id,
            sfw_level=description.raw.sfw_level,
            temperature=description.raw.temperature,
        )
        summary = scene_model.summary()
        return SceneDescription(
            template_id=description.template_id,
            caption_bounds=description.caption_bounds,
            aspect_ratio=description.aspect_ratio,
            gene_ids={slot: choice.option_id for slot, choice in scene_model.slots.items()},
            payload=scene_model.to_payload(),
            summary=summary,
            raw=scene_model,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _slot_options(self, slot: str) -> List[Mapping[str, Any]]:
        if slot == "pose":
            return self.catalog.section("poses")
        if slot == "lighting":
            return self.catalog.section("lighting_presets")
        if slot == "palette":
            return self.catalog.section("palettes")
        if slot == "wardrobe":
            return self.catalog.wardrobe_sets()
        if slot == "background":
            return self.catalog.section("backgrounds")
        if slot == "mood":
            return self.catalog.section("moods")
        raise ValueError(f"unsupported slot '{slot}'")

    def _compute_slot_probabilities(
        self,
        slot: str,
        options: Sequence[Mapping[str, Any]],
        request: SceneRequest,
        slot_bias: Mapping[str, Any],
        sfw_level: float,
    ) -> tuple[List[GeneOptionProbability], int, List[Mapping[str, Any]]]:
        weights: MutableSequence[float] = []
        metadata_entries: List[Mapping[str, Any]] = []
        labels: List[str] = []

        max_nsfw = slot_bias.get("max_nsfw") if slot_bias else None
        temp_boost = slot_bias.get("temperature_boost") if slot_bias else None

        for option in options:
            label = _option_label(option)
            nsfw_score = _extract_float(option, "nsfw")
            weight = max(1e-3, 1.0 - sfw_level * nsfw_score)
            if max_nsfw is not None and nsfw_score > float(max_nsfw):
                weight *= 0.05
            if slot == "palette" and temp_boost:
                try:
                    boost = float(temp_boost)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    boost = 1.0
                weight *= max(0.1, boost)
            weights.append(weight)
            metadata = _option_metadata(slot, option)
            metadata_entries.append(metadata)
            labels.append(label)

        temperature = max(0.05, request.temperature)
        adjusted = [w ** (1.0 / temperature) for w in weights]
        total = sum(adjusted)
        if total <= 0:
            probabilities = [1.0 / len(options)] * len(options)
        else:
            probabilities = [w / total for w in adjusted]

        probability_entries = [
            GeneOptionProbability(
                option_id=str(option.get("id")),
                probability=prob,
                label=label,
                metadata=metadata,
            )
            for option, prob, label, metadata in zip(options, probabilities, labels, metadata_entries)
        ]

        choice_idx = self._weighted_choice(range(len(options)), probabilities)
        return probability_entries, choice_idx, metadata_entries

    def _weighted_choice(self, population: Iterable[int], weights: Sequence[float]) -> int:
        population_list = list(population)
        if not population_list:
            raise ValueError("population for weighted choice is empty")
        return self._rng.choices(population_list, weights=weights, k=1)[0]

    def _caption_bounds(self) -> Mapping[str, int]:
        rules = self.catalog.rules()
        lengths = rules.get("caption_length") if isinstance(rules, Mapping) else None
        min_words = 18
        max_words = 80
        if isinstance(lengths, Mapping):
            try:
                min_words = int(lengths.get("min_words", min_words))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
            try:
                max_words = int(lengths.get("max_words", max_words))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        return {"min_words": min_words, "max_words": max_words}

    def _aspect_ratio(self, slots: Mapping[str, SceneSlotChoice]) -> str:
        # Until dedicated framing genes are migrated we default to a versatile ratio.
        wardrobe_choice = slots.get("wardrobe")
        if wardrobe_choice and "items" in wardrobe_choice.metadata:
            if any("poster" in str(item).lower() for item in wardrobe_choice.metadata.get("items", [])):
                return "9:16"
        return "3:4"
