"""Scene builder and genome assembly."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping

from .assets import AssetLibrary
from .constraints import ScenarioContext
from .preferences import BiasConfig
from .scenarios import Stage
from .selector import AssetSelector, EmptyPoolError
from .system_prompt import SystemPromptBundle, system_prompt_for

LOGGER = logging.getLogger("imagen.build")


@dataclass
class BuildResult:
    """Result of build_struct."""

    scene: Dict[str, List[Mapping[str, object]]]
    selected: Dict[str, Dict[str, Mapping[str, object]]]
    system_prompt: SystemPromptBundle
    gene_ids: Dict[str, object]
    meta: Dict[str, object]


def _stage_locks_as_mapping(stage: Stage) -> Dict[str, Dict[str, List[str]]]:
    return {
        group: {"allow": list(lock.allow), "deny": list(lock.deny)}
        for group, lock in stage.locks.items()
    }


def _register(
    target: MutableMapping[str, Dict[str, Mapping[str, object]]],
    scene: MutableMapping[str, List[Mapping[str, object]]],
    group: str,
    assets: Iterable[Mapping[str, object]],
) -> None:
    entries = list(assets)
    if not entries:
        return
    bucket = target.setdefault(group, {})
    for asset in entries:
        asset_id = str(asset.get("id"))
        bucket[asset_id] = asset
    scene[group] = entries


def _pick_optional(
    selector: AssetSelector,
    group: str,
    selected: MutableMapping[str, Dict[str, Mapping[str, object]]],
    context: ScenarioContext,
) -> List[Mapping[str, object]]:
    if not selector.assets.catalog[group]:
        return []
    try:
        asset = selector.pick_one(group, selected, context)
        return [asset]
    except EmptyPoolError:
        LOGGER.debug("optional group %s skipped due to constraints", group)
        return []


def _style_profile(assets: AssetLibrary, profile_id: str) -> Mapping[str, object]:
    asset = assets.get("style", profile_id)
    if asset.get("kind") != "profile":
        raise ValueError(f"Style asset {profile_id} is not a profile")
    return asset


def _select_style_tokens(
    selector: AssetSelector,
    count: int,
    selected: MutableMapping[str, Dict[str, Mapping[str, object]]],
    context: ScenarioContext,
    profile_id: str,
) -> List[Mapping[str, object]]:
    if count <= 0:
        return []
    def predicate(asset: Mapping[str, object]) -> bool:
        return asset.get("kind") == "token" and asset.get("id") != profile_id

    try:
        tokens = selector.pick_many("style", count, selected, context, predicate=predicate)
    except EmptyPoolError:
        return []
    return list(tokens)


def _load_rules(assets: AssetLibrary, rule_ids: Iterable[str]) -> List[Mapping[str, object]]:
    rules: List[Mapping[str, object]] = []
    for rule_id in rule_ids:
        rules.append(assets.get("rules", rule_id))
    return rules


def build_struct(
    *,
    assets: AssetLibrary,
    selector: AssetSelector,
    bias: BiasConfig,
    stage: Stage,
    default_style_profile: str,
    style_token_limit: int = 0,
    extra_rule_ids: Iterable[str] | None = None,
) -> BuildResult:
    """Build a scene for the provided stage."""

    stage_override = _stage_locks_as_mapping(stage)
    effective_locks = bias.merge_locks(stage_override)
    context = ScenarioContext(
        locks=effective_locks,
        required_terms=stage.required_terms,
        inject_rules=stage.inject_rules,
    )
    selected: Dict[str, Dict[str, Mapping[str, object]]] = {}
    scene: Dict[str, List[Mapping[str, object]]] = {}

    wardrobe_set_assets = _pick_optional(selector, "wardrobe_sets", selected, context)
    _register(selected, scene, "wardrobe_sets", wardrobe_set_assets)

    for group in [
        "wardrobe",
        "characters",
        "models",
        "poses",
        "actions",
        "backgrounds",
        "lighting",
        "palettes",
        "moods",
        "camera",
        "props",
    ]:
        if not selector.assets.catalog[group]:
            continue
        asset = selector.pick_one(group, selected, context)
        _register(selected, scene, group, [asset])

    profile_id = stage.style_profile or default_style_profile
    profile = _style_profile(assets, profile_id)
    _register(selected, scene, "style", [profile])

    style_tokens = _select_style_tokens(selector, style_token_limit, selected, context, profile_id)
    _register(selected, scene, "style", style_tokens)

    extra_rules: List[Mapping[str, object]] = []
    extra_ids = [rule_id for rule_id in (extra_rule_ids or []) if rule_id not in stage.inject_rules]
    if extra_ids:
        extra_rules = _load_rules(assets, extra_ids)
    scenario_rules = _load_rules(assets, stage.inject_rules) + extra_rules
    _register(selected, scene, "rules", scenario_rules)

    system_prompt = system_prompt_for(
        profile,
        stage_required_terms=stage.required_terms,
        style_tokens=style_tokens,
        rules=scenario_rules,
        inject_rule_ids=stage.inject_rules,
    )

    gene_ids = {
        "model": next(iter(selected.get("models", {})), None),
        "characters": list(selected.get("characters", {}).keys()),
        "wardrobe_set": next(iter(selected.get("wardrobe_sets", {})), None),
        "wardrobe_main": next(iter(selected.get("wardrobe", {})), None),
        "pose": next(iter(selected.get("poses", {})), None),
        "palette": next(iter(selected.get("palettes", {})), None),
        "lighting": next(iter(selected.get("lighting", {})), None),
        "camera": next(iter(selected.get("camera", {})), None),
        "style_tokens": [str(asset.get("id")) for asset in style_tokens],
        "rules": [str(asset.get("id")) for asset in scenario_rules],
    }

    meta = {
        "stage_id": stage.stage_id,
        "style_profile": profile_id,
        "active_asset_packs": assets.active_roots,
        "run_cfg": {"temperature": stage.temperature, "cycles": stage.cycles},
        "inject_rules": stage.inject_rules,
        "applied_hard_rules": [
            str(rule.get("id"))
            for rule in scenario_rules
            if isinstance(rule.get("meta"), Mapping) and rule["meta"].get("hard")
        ],
    }

    return BuildResult(
        scene=scene,
        selected=selected,
        system_prompt=system_prompt,
        gene_ids=gene_ids,
        meta=meta,
    )


__all__ = ["BuildResult", "build_struct"]
