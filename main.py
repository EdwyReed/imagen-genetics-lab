"""Imagen pipeline v0.9 command-line interface."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

from imagen_pipeline.core.assets import AssetLibrary
from imagen_pipeline.core.build import BuildResult, build_struct
from imagen_pipeline.core.constraints import ScenarioContext
from imagen_pipeline.core.evolve import Genome, evolve_population
from imagen_pipeline.core.preferences import BiasConfig, LockSet
from imagen_pipeline.core.scenarios import ScenarioLoader, Stage
from imagen_pipeline.core.selector import AssetSelector

LOGGER = logging.getLogger("imagen.cli")
BASE_ASSET_ROOT = Path("data/catalog")
LEGACY_FLAGS = {"--allow", "--deny", "--template", "--evolve", "--ga"}
STYLE_TOKEN_LIMIT = 2

GENE_GROUPS: Dict[str, str] = {
    "model": "models",
    "wardrobe_set": "wardrobe_sets",
    "wardrobe_main": "wardrobe",
    "pose": "poses",
    "palette": "palettes",
    "lighting": "lighting",
    "camera": "camera",
    "characters": "characters",
    "style_tokens": "style",
    "rules": "rules",
}


def ensure_new_cli(argv: Sequence[str]) -> None:
    for token in argv:
        if token in LEGACY_FLAGS:
            raise RuntimeError(
                "Legacy CLI flag detected. v0.9 removed compatibility with older interfaces. "
                "Please review the README for the new commands."
            )


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def load_bias(path: Path) -> BiasConfig:
    LOGGER.info("loading bias from %s", path)
    return BiasConfig.from_file(path)


def load_assets(roots: Sequence[Path], *, fail_on_conflict: bool) -> AssetLibrary:
    ordered_roots = [BASE_ASSET_ROOT, *roots]
    LOGGER.info("asset roots: %s", [root.as_posix() for root in ordered_roots])
    return AssetLibrary(ordered_roots, fail_on_conflict=fail_on_conflict)


def load_scenario(path: Path) -> Sequence[Stage]:
    loader = ScenarioLoader()
    scenario = loader.load(path)
    return list(scenario.stages)


def stage_locks_dict(stage: Stage) -> Dict[str, Dict[str, List[str]]]:
    return {group: {"allow": list(lock.allow), "deny": list(lock.deny)} for group, lock in stage.locks.items()}


def stage_context(bias: BiasConfig, stage: Stage) -> ScenarioContext:
    merged = bias.merge_locks(stage_locks_dict(stage))
    LOGGER.info(
        "stage %s locks=%s", stage.stage_id, {group: {"allow": lock.allow, "deny": lock.deny} for group, lock in merged.items()}
    )
    LOGGER.info(
        "stage %s required_terms=%s inject_rules=%s", stage.stage_id, stage.required_terms, stage.inject_rules
    )
    return ScenarioContext(locks=merged, required_terms=stage.required_terms, inject_rules=stage.inject_rules)


def write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_payload(result: BuildResult, *, stage_id: str, cycle: int) -> dict:
    return {
        "stage_id": stage_id,
        "cycle": cycle,
        "gene_ids": result.gene_ids,
        "meta": result.meta,
        "system_prompt": {
            "text": result.system_prompt.text,
            "required_terms": result.system_prompt.required_terms,
            "style_tokens": result.system_prompt.style_tokens,
            "rules": result.system_prompt.rule_injections,
        },
        "scene": result.scene,
    }


def run_plain(args: argparse.Namespace) -> None:
    bias = load_bias(Path(args.bias))
    asset_packs = [Path(p) for p in args.asset_pack]
    assets = load_assets(asset_packs, fail_on_conflict=args.fail_on_conflict)
    selector = AssetSelector(assets, bias)
    stages = load_scenario(Path(args.scenario))
    if not stages:
        raise RuntimeError("Scenario must contain at least one stage")
    out_dir = Path(args.out)
    runs_log = out_dir / "runs.jsonl"
    run_id = uuid.uuid4().hex
    append_jsonl(runs_log, {"event": "run_start", "run_id": run_id, "mode": "plain"})
    for stage in stages:
        context = stage_context(bias, stage)
        for cycle in range(stage.cycles):
            result = build_struct(
                assets=assets,
                selector=selector,
                bias=bias,
                stage=stage,
                default_style_profile=args.style_profile,
                style_token_limit=STYLE_TOKEN_LIMIT,
            )
            record_id = f"{stage.stage_id}_cycle{cycle:03d}"
            payload = build_payload(result, stage_id=stage.stage_id, cycle=cycle)
            write_record(out_dir / f"{record_id}.json", payload)
            append_jsonl(
                runs_log,
                {
                    "event": "result",
                    "run_id": run_id,
                    "stage_id": stage.stage_id,
                    "cycle": cycle,
                    "gene_ids": result.gene_ids,
                    "meta": result.meta,
                },
            )
            LOGGER.info("%s gene_ids=%s", record_id, result.gene_ids)
    append_jsonl(runs_log, {"event": "run_end", "run_id": run_id})


def clone_stage(stage: Stage) -> Stage:
    locks = {group: LockSet(list(lock.allow), list(lock.deny)) for group, lock in stage.locks.items()}
    return Stage(
        stage_id=stage.stage_id,
        cycles=stage.cycles,
        temperature=stage.temperature,
        style_profile=stage.style_profile,
        locks=locks,
        required_terms=list(stage.required_terms),
        inject_rules=list(stage.inject_rules),
    )


def stage_with_genome(stage: Stage, genome: Genome) -> Stage:
    cloned = clone_stage(stage)
    for gene, group in GENE_GROUPS.items():
        values = []
        value = getattr(genome, gene)
        if isinstance(value, list):
            values = [item for item in value if item]
        elif value:
            values = [value]
        if not values:
            continue
        base_lock = cloned.locks.get(group, LockSet())
        cloned.locks[group] = LockSet(allow=list(dict.fromkeys(values)), deny=list(base_lock.deny))
    cloned.cycles = 1
    return cloned


def run_evolve(args: argparse.Namespace) -> None:
    bias = load_bias(Path(args.bias))
    asset_packs = [Path(p) for p in args.asset_pack]
    assets = load_assets(asset_packs, fail_on_conflict=args.fail_on_conflict)
    selector = AssetSelector(assets, bias)
    stages = load_scenario(Path(args.scenario))
    if not stages:
        raise RuntimeError("Scenario must contain at least one stage")
    if len(stages) > 1:
        LOGGER.warning("Scenario has multiple stages; evolve will use the first stage only")
    stage = stages[0]
    context = stage_context(bias, stage)
    out_dir = Path(args.out)
    runs_log = out_dir / "ga_runs.jsonl"
    run_id = uuid.uuid4().hex
    append_jsonl(runs_log, {"event": "run_start", "run_id": run_id, "mode": "evolve"})
    population: List[Genome] = []
    for idx in range(args.ga_pop):
        result = build_struct(
            assets=assets,
            selector=selector,
            bias=bias,
            stage=stage,
            default_style_profile=args.style_profile,
            style_token_limit=STYLE_TOKEN_LIMIT,
        )
        population.append(Genome.from_build(result))
        payload = build_payload(result, stage_id=stage.stage_id, cycle=idx)
        write_record(out_dir / f"gen00_ind{idx:03d}.json", payload)
        append_jsonl(
            runs_log,
            {
                "event": "population",
                "run_id": run_id,
                "generation": 0,
                "index": idx,
                "gene_ids": result.gene_ids,
                "meta": result.meta,
            },
        )
    rng = random.Random()
    for generation in range(1, args.ga_gen + 1):
        population = evolve_population(
            population,
            selector=selector,
            assets=assets,
            context=context,
            rng=rng,
        )
        for idx, genome in enumerate(population):
            stage_variant = stage_with_genome(stage, genome)
            result = build_struct(
                assets=assets,
                selector=selector,
                bias=bias,
                stage=stage_variant,
                default_style_profile=args.style_profile,
                style_token_limit=STYLE_TOKEN_LIMIT,
                extra_rule_ids=genome.rules,
            )
            payload = build_payload(result, stage_id=stage.stage_id, cycle=idx)
            write_record(out_dir / f"gen{generation:02d}_ind{idx:03d}.json", payload)
            append_jsonl(
                runs_log,
                {
                    "event": "population",
                    "run_id": run_id,
                    "generation": generation,
                    "index": idx,
                    "gene_ids": result.gene_ids,
                    "meta": result.meta,
                },
            )
            LOGGER.info("gen %s indiv %s gene_ids=%s", generation, idx, result.gene_ids)
    append_jsonl(runs_log, {"event": "run_end", "run_id": run_id})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Imagen pipeline v0.9")
    parser.add_argument("command", choices=["run"], help="Primary command")
    parser.add_argument("mode", choices=["plain", "evolve"], help="Run mode")
    parser.add_argument("--style-profile", required=True, dest="style_profile")
    parser.add_argument("--asset-pack", action="append", default=[], dest="asset_pack")
    parser.add_argument("--bias", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-on-conflict", action="store_true", dest="fail_on_conflict")
    parser.add_argument("--ga-pop", type=int, default=0, dest="ga_pop")
    parser.add_argument("--ga-gen", type=int, default=0, dest="ga_gen")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    ensure_new_cli(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if args.command != "run":  # pragma: no cover - argparse guarantees choices
        raise RuntimeError("Unsupported command")
    if args.mode == "plain":
        run_plain(args)
    elif args.mode == "evolve":
        if args.ga_pop <= 0 or args.ga_gen <= 0:
            raise RuntimeError("--ga-pop and --ga-gen must be positive for evolve mode")
        run_evolve(args)
    else:  # pragma: no cover - argparse enforces choices
        raise RuntimeError("Unsupported mode")


if __name__ == "__main__":  # pragma: no cover
    main()
