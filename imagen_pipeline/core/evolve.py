"""Genetic algorithm utilities for the v0.9 genome."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from .assets import AssetLibrary
from .build import BuildResult
from .constraints import ScenarioContext
from .preferences import LockSet
from .selector import AssetSelector, EmptyPoolError

LOGGER = logging.getLogger("imagen.evolve")


GENE_SINGLE = {
    "model": "models",
    "wardrobe_set": "wardrobe_sets",
    "wardrobe_main": "wardrobe",
    "pose": "poses",
    "palette": "palettes",
    "lighting": "lighting",
    "camera": "camera",
}

GENE_LIST = {
    "characters": "characters",
    "style_tokens": "style",
    "rules": "rules",
}

SUPPORTED_GENES = set(GENE_SINGLE) | set(GENE_LIST)


class GenomeError(RuntimeError):
    """Raised when genome deserialisation fails."""


@dataclass
class Genome:
    """Minimal genome representation used by the GA."""

    model: str | None = None
    characters: List[str] = field(default_factory=list)
    wardrobe_set: str | None = None
    wardrobe_main: str | None = None
    pose: str | None = None
    palette: str | None = None
    lighting: str | None = None
    camera: str | None = None
    style_tokens: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)

    @classmethod
    def from_build(cls, build: BuildResult) -> "Genome":
        return cls(
            model=build.gene_ids.get("model"),
            characters=list(build.gene_ids.get("characters", [])),
            wardrobe_set=build.gene_ids.get("wardrobe_set"),
            wardrobe_main=build.gene_ids.get("wardrobe_main"),
            pose=build.gene_ids.get("pose"),
            palette=build.gene_ids.get("palette"),
            lighting=build.gene_ids.get("lighting"),
            camera=build.gene_ids.get("camera"),
            style_tokens=list(build.gene_ids.get("style_tokens", [])),
            rules=list(build.gene_ids.get("rules", [])),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Genome":
        unexpected = set(data.keys()) - SUPPORTED_GENES
        legacy_keys = {"template_id", "wardrobe_extras", "palette_id"}
        if unexpected or legacy_keys & set(data.keys()):
            raise GenomeError(
                "Unsupported genome format detected. Regenerate using the v0.9 pipeline."
            )
        return cls(
            model=data.get("model"),
            characters=list(data.get("characters", [])),
            wardrobe_set=data.get("wardrobe_set"),
            wardrobe_main=data.get("wardrobe_main"),
            pose=data.get("pose"),
            palette=data.get("palette"),
            lighting=data.get("lighting"),
            camera=data.get("camera"),
            style_tokens=list(data.get("style_tokens", [])),
            rules=list(data.get("rules", [])),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "characters": list(self.characters),
            "wardrobe_set": self.wardrobe_set,
            "wardrobe_main": self.wardrobe_main,
            "pose": self.pose,
            "palette": self.palette,
            "lighting": self.lighting,
            "camera": self.camera,
            "style_tokens": list(self.style_tokens),
            "rules": list(self.rules),
        }

    def as_selected(self, assets: AssetLibrary) -> Dict[str, Dict[str, Mapping[str, object]]]:
        selected: Dict[str, Dict[str, Mapping[str, object]]] = {}
        for gene, group in GENE_SINGLE.items():
            value = getattr(self, gene)
            if value:
                selected.setdefault(group, {})[value] = assets.get(group, value)
        for gene, group in GENE_LIST.items():
            values = getattr(self, gene)
            bucket = selected.setdefault(group, {})
            for value in values:
                bucket[value] = assets.get(group, value)
        return selected

    def copy(self) -> "Genome":
        return Genome.from_dict(self.to_dict())


def _lock_for(context: ScenarioContext, group: str) -> LockSet:
    return context.locks.get(group, LockSet()) if context.locks else LockSet()


def _predicate_for_lock(lock: LockSet, exclude: Sequence[str] | None = None):
    allowed = set(lock.allow)
    excluded = set(exclude or [])

    def predicate(asset: Mapping[str, object]) -> bool:
        asset_id = str(asset.get("id"))
        if allowed and asset_id not in allowed:
            return False
        if asset_id in excluded:
            return False
        return True

    return predicate


def mutate_genome(
    genome: Genome,
    *,
    selector: AssetSelector,
    assets: AssetLibrary,
    context: ScenarioContext,
    rng: random.Random | None = None,
    mutation_rate: float = 0.2,
) -> Genome:
    rng = rng or random.Random()
    mutated = genome.copy()
    selected = mutated.as_selected(assets)

    for gene, group in GENE_SINGLE.items():
        lock = _lock_for(context, group)
        if lock.is_pinned():
            continue
        if rng.random() > mutation_rate:
            continue
        current = getattr(mutated, gene)
        working_selected = {g: dict(items) for g, items in selected.items()}
        if current:
            if group in working_selected:
                working_selected[group].pop(current, None)
            if group in selected:
                selected[group].pop(current, None)
        predicate = _predicate_for_lock(lock, exclude=[current] if current else None)
        try:
            new_asset = selector.pick_one(group, working_selected, context, predicate=predicate)
        except EmptyPoolError:
            continue
        new_id = str(new_asset.get("id"))
        setattr(mutated, gene, new_id)
        selected.setdefault(group, {})[new_id] = new_asset

    for gene, group in GENE_LIST.items():
        lock = _lock_for(context, group)
        if lock.is_pinned():
            pinned_id = lock.allow[0]
            setattr(mutated, gene, [pinned_id])
            selected[group] = {pinned_id: assets.get(group, pinned_id)}
            continue
        if rng.random() > mutation_rate:
            continue
        current_values = list(getattr(mutated, gene))
        target_count = max(len(current_values), 1)
        working_selected = {g: dict(items) for g, items in selected.items()}
        working_selected.pop(group, None)
        selected.pop(group, None)
        predicate = _predicate_for_lock(lock, exclude=current_values)
        if group == "style":
            def style_predicate(asset: Mapping[str, object]) -> bool:
                return asset.get("kind") == "token" and predicate(asset)

            predicate_fn = style_predicate
        else:
            predicate_fn = predicate
        try:
            assets_list = selector.pick_many(group, target_count, working_selected, context, predicate=predicate_fn)
        except EmptyPoolError:
            continue
        new_ids = [str(asset.get("id")) for asset in assets_list]
        setattr(mutated, gene, new_ids)
        selected[group] = {asset_id: asset for asset_id, asset in zip(new_ids, assets_list)}

    return mutated


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    *,
    context: ScenarioContext,
    rng: random.Random | None = None,
) -> Genome:
    rng = rng or random.Random()
    child = Genome()
    for gene, group in GENE_SINGLE.items():
        lock = _lock_for(context, group)
        if lock.is_pinned():
            setattr(child, gene, lock.allow[0])
            continue
        choice = rng.choice([getattr(parent_a, gene), getattr(parent_b, gene)])
        if lock.allow and choice not in lock.allow:
            choice = lock.allow[0]
        setattr(child, gene, choice)
    for gene, group in GENE_LIST.items():
        lock = _lock_for(context, group)
        if lock.is_pinned():
            setattr(child, gene, [lock.allow[0]])
            continue
        source = parent_a if rng.random() < 0.5 else parent_b
        values = list(getattr(source, gene))
        if lock.allow:
            values = [value for value in values if value in lock.allow]
            if not values:
                values = list(lock.allow)
        if group == "style":
            values = [value for value in values]
        child_value = list(dict.fromkeys(values))
        setattr(child, gene, child_value)
    return child


def evolve_population(
    population: Sequence[Genome],
    *,
    selector: AssetSelector,
    assets: AssetLibrary,
    context: ScenarioContext,
    rng: random.Random | None = None,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.3,
) -> List[Genome]:
    """Produce a new population using crossover and mutation."""

    if not population:
        return []
    rng = rng or random.Random()
    next_generation: List[Genome] = []
    while len(next_generation) < len(population):
        if len(population) > 1 and rng.random() < crossover_rate:
            parents = rng.sample(population, 2)
            child = crossover(parents[0], parents[1], context=context, rng=rng)
        else:
            parent = rng.choice(population)
            child = mutate_genome(
                parent,
                selector=selector,
                assets=assets,
                context=context,
                rng=rng,
                mutation_rate=mutation_rate,
            )
        next_generation.append(child)
    return next_generation


__all__ = ["Genome", "GenomeError", "crossover", "evolve_population", "mutate_genome"]
