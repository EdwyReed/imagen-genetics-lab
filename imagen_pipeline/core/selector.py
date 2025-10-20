"""Asset selection utilities with weights and locks."""
from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .assets import AssetLibrary
from .constraints import ScenarioContext, is_compatible
from .preferences import BiasConfig, LockSet

SelectedState = MutableMapping[str, Dict[str, Mapping[str, object]]]


class EmptyPoolError(RuntimeError):
    """Raised when selector fails to build a candidate pool."""

    def __init__(
        self,
        group: str,
        *,
        allow: Iterable[str],
        deny: Iterable[str],
        selected: Mapping[str, Mapping[str, Mapping[str, object]]],
    ) -> None:
        message = (
            f"No candidates available for group '{group}'. "
            f"allow={list(allow) or '*'} deny={list(deny)} selected={format_selected(selected)}"
        )
        super().__init__(message)
        self.group = group
        self.allow = list(allow)
        self.deny = list(deny)
        self.selected = selected


def format_selected(selected: Mapping[str, Mapping[str, Mapping[str, object]]]) -> str:
    entries: List[str] = []
    for group, items in selected.items():
        for asset_id in items.keys():
            entries.append(f"{group}:{asset_id}")
    return ",".join(sorted(entries)) or "<none>"


def merge_locks(base: LockSet, override: LockSet | None) -> LockSet:
    allow = list(base.allow)
    deny = set(base.deny)
    if override:
        if override.allow:
            allow = list(override.allow)
        deny |= set(override.deny)
    merged = LockSet(allow=allow, deny=list(deny))
    return merged.normalized()


class AssetSelector:
    """Selector applying locks, weights and requires/excludes."""

    def __init__(
        self,
        assets: AssetLibrary,
        bias: BiasConfig,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.assets = assets
        self.bias = bias
        self.random = rng or random.Random()

    def _effective_lock(self, group: str, context: ScenarioContext | None) -> LockSet:
        base = self.bias.locks_for_group(group)
        override = None
        if context and context.locks:
            override = context.locks.get(group)
        return merge_locks(base, override)

    def _pool(
        self,
        group: str,
        selected: Mapping[str, Mapping[str, Mapping[str, object]]],
        context: ScenarioContext | None,
        *,
        exclude_ids: Iterable[str] | None = None,
        predicate: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> tuple[list[Mapping[str, object]], LockSet]:
        lock = self._effective_lock(group, context)
        allow_set = set(lock.allow)
        deny_set = set(lock.deny)
        exclude = set(exclude_ids or [])
        pool: list[Mapping[str, object]] = []
        for asset_id, asset in self.assets.catalog[group].items():
            if allow_set and asset_id not in allow_set:
                continue
            if asset_id in deny_set or asset_id in exclude:
                continue
            if predicate and not predicate(asset):
                continue
            if not is_compatible(asset, selected, context):
                continue
            pool.append(asset)
        if not pool:
            raise EmptyPoolError(group, allow=lock.allow, deny=lock.deny, selected=selected)
        return pool, lock

    def _weights(self, group: str, pool: Sequence[Mapping[str, object]]) -> list[float]:
        weights: list[float] = []
        for asset in pool:
            base = float(asset.get("weight", 1.0))
            asset_id = str(asset.get("id"))
            bias_weight = self.bias.weight_for(group, asset_id)
            weights.append(max(base * bias_weight, 0.0))
        if all(weight == 0 for weight in weights):
            weights = [1.0 for _ in pool]
        return weights

    def pick_one(
        self,
        group: str,
        selected: SelectedState,
        context: ScenarioContext | None = None,
        predicate: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> Mapping[str, object]:
        pool, _ = self._pool(group, selected, context, predicate=predicate)
        weights = self._weights(group, pool)
        choice = self.random.choices(pool, weights=weights, k=1)[0]
        return choice

    def pick_many(
        self,
        group: str,
        count: int,
        selected: SelectedState,
        context: ScenarioContext | None = None,
        predicate: Callable[[Mapping[str, object]], bool] | None = None,
    ) -> list[Mapping[str, object]]:
        results: list[Mapping[str, object]] = []
        working_selected: Dict[str, Dict[str, Mapping[str, object]]] = {
            g: dict(items) for g, items in selected.items()
        }
        for _ in range(count):
            exclude_ids = [str(asset.get("id")) for asset in results]
            pool, _ = self._pool(
                group,
                working_selected,
                context,
                exclude_ids=exclude_ids,
                predicate=predicate,
            )
            weights = self._weights(group, pool)
            choice = self.random.choices(pool, weights=weights, k=1)[0]
            asset_id = str(choice.get("id"))
            working_selected.setdefault(group, {})[asset_id] = choice
            results.append(choice)
        return results


__all__ = ["AssetSelector", "EmptyPoolError", "format_selected"]
