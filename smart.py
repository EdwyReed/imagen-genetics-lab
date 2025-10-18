#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entrypoint for Imagen + Ollama pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from imagen_lab.config import load_config
from imagen_lab.pipeline import run_evolve, run_plain
from imagen_lab.randomization import clamp


def _optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Imagen + Ollama + Dual scoring orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--outdir", default=None, help="Override output directory")
    parser.add_argument("--cycles", type=int, default=None, help="Number of cycles for plain mode")
    parser.add_argument("--per-cycle", type=int, default=None, help="Variants per cycle for plain mode")
    parser.add_argument("--sleep", type=float, default=None, help="Sleep seconds between cycles/individuals")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--sfw", type=float, default=None, help="Target SFW level [0..1]")
    parser.add_argument("--temperature", type=float, default=None, help="Caption temperature")
    parser.add_argument("--w-style", type=float, default=None, help="Fitness weight for style score")
    parser.add_argument("--w-nsfw", type=float, default=None, help="Fitness weight for NSFW score")
    parser.add_argument("--model", default=None, help="Override Imagen model name")
    parser.add_argument("--person-mode", default=None, help="Override person generation mode")
    parser.add_argument("--ollama-url", default=None, help="Override Ollama URL")
    parser.add_argument("--ollama-model", default=None, help="Override Ollama model")
    parser.add_argument("--ollama-top-p", type=float, default=None, help="Override Ollama top_p")

    parser.add_argument("--evolve", action="store_true", help="Run in genetic evolution mode")
    parser.add_argument("--pop", type=int, default=None, help="Population size for GA")
    parser.add_argument("--gens", type=int, default=None, help="Number of generations")
    parser.add_argument("--keep", type=float, default=None, help="Elite fraction to keep")
    parser.add_argument("--mut", type=float, default=None, help="Mutation probability per gene")
    parser.add_argument("--xover", type=float, default=None, help="Crossover probability")
    parser.add_argument("--resume-k", type=int, default=None, help="How many top gene sets to resume from DB")
    parser.add_argument("--resume-session", default=None, help="Restrict resume to a session id")
    parser.add_argument("--resume-mix", type=float, default=None, help="Mutation applied to resumed genes [0..1]")
    parser.set_defaults(resume_best=None)
    parser.add_argument("--resume-best", dest="resume_best", action="store_true", help="Seed GA population from DB")
    parser.add_argument("--no-resume-best", dest="resume_best", action="store_false", help="Disable DB seeding")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.model:
        config.imagen.model = args.model
    if args.person_mode:
        config.imagen.person_mode = args.person_mode
    if args.ollama_url:
        config.ollama.url = args.ollama_url
    if args.ollama_model:
        config.ollama.model = args.ollama_model
    if args.ollama_top_p is not None:
        config.ollama.top_p = args.ollama_top_p

    sfw_level = clamp(args.sfw, 0.0, 1.0) if args.sfw is not None else None
    outdir = _optional_path(args.outdir)

    if args.evolve:
        run_evolve(
            config,
            outdir=outdir,
            pop=args.pop,
            gens=args.gens,
            keep=args.keep,
            mut=args.mut,
            xover=args.xover,
            sleep_s=args.sleep,
            seed=args.seed,
            sfw_level=sfw_level,
            temperature=args.temperature,
            w_style=args.w_style,
            w_nsfw=args.w_nsfw,
            resume_best=args.resume_best,
            resume_k=args.resume_k,
            resume_session=args.resume_session,
            resume_mix=args.resume_mix,
        )
    else:
        run_plain(
            config,
            outdir=outdir,
            cycles=args.cycles,
            per_cycle=args.per_cycle,
            sfw_level=sfw_level,
            temperature=args.temperature,
            sleep_s=args.sleep,
            seed=args.seed,
            w_style=args.w_style,
            w_nsfw=args.w_nsfw,
        )


if __name__ == "__main__":
    main()
