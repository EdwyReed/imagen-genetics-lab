from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai

from scorer import DualScorer

from .catalog import Catalog
from .config import PipelineConfig
from .ga import GeneSet, crossover_genes, load_best_gene_sets, mutate_gene
from .prompting import (
    REQUIRED_STYLE_TERMS,
    enforce_bounds,
    enforce_once,
    imagen_call,
    ollama_generate,
    system_prompt_for,
    system_prompt_hash,
)
from .scene_builder import SceneBuilder
from .storage import ArtifactWriter, PromptLogger, save_and_score


@dataclass
class PipelineServices:
    scorer: DualScorer
    catalog: Catalog
    builder: SceneBuilder
    logger: PromptLogger
    writer: ArtifactWriter
    client: genai.Client


@dataclass
class GASettings:
    pop: int
    resume_k: int
    resume_mix: float


def _required_terms(config: PipelineConfig) -> List[str]:
    terms = [t for t in config.prompting.required_terms if t]
    return terms or list(REQUIRED_STYLE_TERMS)


def _prepare_services(config: PipelineConfig, output_dir: Optional[Path] = None) -> PipelineServices:
    scorer = DualScorer(
        device=config.scoring.device,
        batch=config.scoring.batch_size,
        db_path=config.paths.database,
        jsonl_path=config.paths.scores_jsonl,
        weights=config.scoring.weights,
        tau=config.scoring.tau,
        cal_style=config.scoring.cal_style,
        cal_illu=config.scoring.cal_illu,
    )
    catalog = Catalog.load(config.paths.catalog)
    builder = SceneBuilder(catalog, required_terms=_required_terms(config), template_ids=config.prompting.template_ids)
    logger = PromptLogger(config.paths.database)
    writer = ArtifactWriter(output_dir or config.paths.output_dir)
    client = genai.Client()
    return PipelineServices(scorer=scorer, catalog=catalog, builder=builder, logger=logger, writer=writer, client=client)


def run_plain(
    config: PipelineConfig,
    *,
    outdir: Optional[Path] = None,
    cycles: Optional[int] = None,
    per_cycle: Optional[int] = None,
    sfw_level: Optional[float] = None,
    temperature: Optional[float] = None,
    sleep_s: Optional[float] = None,
    seed: Optional[int] = None,
    w_style: Optional[float] = None,
    w_nsfw: Optional[float] = None,
) -> None:
    load_dotenv()
    cycles = cycles if cycles is not None else config.defaults.cycles
    per_cycle = per_cycle if per_cycle is not None else config.defaults.per_cycle
    sleep_s = sleep_s if sleep_s is not None else config.defaults.sleep_s
    temperature = temperature if temperature is not None else config.defaults.temperature
    sfw_level = sfw_level if sfw_level is not None else config.defaults.sfw_level
    seed = seed if seed is not None else config.defaults.seed
    w_style = w_style if w_style is not None else config.fitness.style
    w_nsfw = w_nsfw if w_nsfw is not None else config.fitness.nsfw

    if seed is not None:
        random.seed(seed)

    services = _prepare_services(config, output_dir=outdir)
    logger = services.logger
    writer = services.writer
    scorer = services.scorer
    builder = services.builder
    client = services.client

    session_id = f"plain-{int(time.time())}"
    logger.log_run(
        session_id,
        "plain",
        {
            "cycles": cycles,
            "per_cycle": per_cycle,
            "sfw": sfw_level,
            "temperature": temperature,
            "weights": {"style": w_style, "nsfw": w_nsfw},
        },
    )

    system_prompt = system_prompt_for(sfw_level)
    sys_hash = system_prompt_hash(system_prompt)

    for idx in range(1, cycles + 1):
        scene = builder.build_scene(sfw_level=sfw_level, temperature=temperature)
        try:
            caption = ollama_generate(
                config.ollama.url,
                config.ollama.model,
                system_prompt,
                scene.ollama_payload(),
                temperature=temperature,
                top_p=config.ollama.top_p,
                seed=seed,
            )
            caption = enforce_once(
                config.ollama.url,
                config.ollama.model,
                system_prompt,
                scene.ollama_payload(),
                caption,
                required_terms=_required_terms(config),
                temperature=max(0.45, temperature - 0.05),
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - network interaction
            print(f"[{idx:02d}/{cycles}] Ollama error: {exc}")
            time.sleep(sleep_s)
            continue

        bounds = scene.caption_bounds
        final_prompt = enforce_bounds(
            caption,
            int(bounds.get("min_words", 18)),
            int(bounds.get("max_words", 60)),
        )

        try:
            response = imagen_call(
                client,
                config.imagen.model,
                final_prompt,
                scene.aspect_ratio,
                per_cycle,
                config.imagen.person_mode,
                guidance_scale=config.imagen.guidance_scale,
            )
        except Exception as exc:  # pragma: no cover - API call
            print(f"[{idx:02d}/{cycles}] Imagen error: {exc}")
            time.sleep(sleep_s)
            continue

        cycle_id = f"rnd-{idx:03d}"
        meta_base = {
            "id": cycle_id,
            "model_imagen": config.imagen.model,
            "person_mode": config.imagen.person_mode,
            "variants": per_cycle,
            "seed": seed,
            "ollama": {
                "url": config.ollama.url,
                "model": config.ollama.model,
                "temperature": temperature,
                "top_p": config.ollama.top_p,
                "system_hash": sys_hash,
                "sfw_level": sfw_level,
                "style_mode": "inline+required_terms",
            },
        }

        triplets = save_and_score(
            response,
            writer,
            logger,
            scorer,
            meta_base,
            final_prompt,
            scene,
            session_id,
            gen=None,
            indiv=None,
            w_style=w_style,
            w_nsfw=w_nsfw,
        )

        if triplets:
            best_path, nsfw100, style100 = triplets[0]
            print(f"   [best] {best_path.name}  style={style100} nsfw={nsfw100}")
        time.sleep(sleep_s)


def _seed_population(
    builder: SceneBuilder,
    catalog: Catalog,
    settings: GASettings,
    sfw_level: float,
    temperature: float,
    db_path: Path,
    session_id: Optional[str],
    seed_resume: bool,
) -> List[GeneSet]:
    population: List[GeneSet] = []
    if seed_resume:
        seed_genes = load_best_gene_sets(db_path, settings.resume_k, session_id)
        if seed_genes:
            for genes in seed_genes:
                child = dict(genes)
                if settings.resume_mix > 0.0:
                    for key in list(child.keys()):
                        if random.random() < settings.resume_mix:
                            child[key] = mutate_gene(catalog, key, child.get(key), sfw_level, temperature)
                population.append(child)
            print(f"[resume] seeded from DB: {len(seed_genes)} genes (session={session_id or 'ANY'})")
    while len(population) < settings.pop:
        scene = builder.build_scene(sfw_level=sfw_level, temperature=temperature)
        population.append(dict(scene.gene_ids))
    return population


def run_evolve(
    config: PipelineConfig,
    *,
    outdir: Optional[Path] = None,
    pop: Optional[int] = None,
    gens: Optional[int] = None,
    keep: Optional[float] = None,
    mut: Optional[float] = None,
    xover: Optional[float] = None,
    sleep_s: Optional[float] = None,
    seed: Optional[int] = None,
    sfw_level: Optional[float] = None,
    temperature: Optional[float] = None,
    w_style: Optional[float] = None,
    w_nsfw: Optional[float] = None,
    resume_best: Optional[bool] = None,
    resume_k: Optional[int] = None,
    resume_session: Optional[str] = None,
    resume_mix: Optional[float] = None,
) -> None:
    load_dotenv()
    pop = pop if pop is not None else config.ga.pop
    gens = gens if gens is not None else config.ga.gens
    keep = keep if keep is not None else config.ga.keep
    mut = mut if mut is not None else config.ga.mut
    xover = xover if xover is not None else config.ga.xover
    sleep_s = sleep_s if sleep_s is not None else config.defaults.sleep_s
    temperature = temperature if temperature is not None else config.defaults.temperature
    sfw_level = sfw_level if sfw_level is not None else config.defaults.sfw_level
    seed = seed if seed is not None else config.defaults.seed
    w_style = w_style if w_style is not None else config.fitness.style
    w_nsfw = w_nsfw if w_nsfw is not None else config.fitness.nsfw
    resume_best = resume_best if resume_best is not None else config.ga.resume_best
    resume_k = resume_k if resume_k is not None else config.ga.resume_k
    resume_session = resume_session if resume_session is not None else config.ga.resume_session
    resume_mix = resume_mix if resume_mix is not None else config.ga.resume_mix

    if seed is not None:
        random.seed(seed)

    services = _prepare_services(config, output_dir=outdir)
    logger = services.logger
    writer = services.writer
    scorer = services.scorer
    builder = services.builder
    catalog = services.catalog
    client = services.client

    session_id = f"evolve-{int(time.time())}"
    logger.log_run(
        session_id,
        "evolve",
        {
            "pop": pop,
            "gens": gens,
            "keep": keep,
            "mut": mut,
            "xover": xover,
            "sfw": sfw_level,
            "temperature": temperature,
            "weights": {"style": w_style, "nsfw": w_nsfw},
        },
    )

    system_prompt = system_prompt_for(sfw_level)
    sys_hash = system_prompt_hash(system_prompt)

    settings = GASettings(
        pop=pop,
        resume_k=resume_k or min(pop, 12),
        resume_mix=resume_mix,
    )

    population = _seed_population(
        builder,
        catalog,
        settings,
        sfw_level,
        temperature,
        config.paths.database,
        resume_session,
        resume_best,
    )

    for gen_idx in range(1, gens + 1):
        print(f"\n===== Generation {gen_idx}/{gens} =====")
        scored: List[Tuple[float, GeneSet, Path, int, int]] = []

        for indiv_idx, genes in enumerate(population, start=1):
            scene = builder.rebuild_from_genes(genes, sfw_level=sfw_level, temperature=temperature)
            try:
                caption = ollama_generate(
                    config.ollama.url,
                    config.ollama.model,
                    system_prompt,
                    scene.ollama_payload(),
                    temperature=temperature,
                    top_p=config.ollama.top_p,
                    seed=seed,
                )
                caption = enforce_once(
                    config.ollama.url,
                    config.ollama.model,
                    system_prompt,
                    scene.ollama_payload(),
                    caption,
                    required_terms=_required_terms(config),
                    temperature=max(0.45, temperature - 0.05),
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - network interaction
                print(f"[G{gen_idx} I{indiv_idx}] Ollama error: {exc}")
                time.sleep(sleep_s)
                continue

            bounds = scene.caption_bounds
            final_prompt = enforce_bounds(
                caption,
                int(bounds.get("min_words", 18)),
                int(bounds.get("max_words", 60)),
            )

            try:
                response = imagen_call(
                    client,
                    config.imagen.model,
                    final_prompt,
                    scene.aspect_ratio,
                    variants=1,
                    person_mode=config.imagen.person_mode,
                    guidance_scale=config.imagen.guidance_scale,
                )
            except Exception as exc:  # pragma: no cover - API call
                print(f"[G{gen_idx} I{indiv_idx}] Imagen error: {exc}")
                time.sleep(sleep_s)
                continue

            if not getattr(response, "generated_images", None):
                print(f"[G{gen_idx} I{indiv_idx}] WARN: no image")
                time.sleep(sleep_s)
                continue

            indiv_id = f"G{gen_idx:02d}-I{indiv_idx:02d}"
            meta_base = {
                "id": indiv_id,
                "model_imagen": config.imagen.model,
                "person_mode": config.imagen.person_mode,
                "variants": 1,
                "seed": seed,
                "ollama": {
                    "url": config.ollama.url,
                    "model": config.ollama.model,
                    "temperature": temperature,
                    "top_p": config.ollama.top_p,
                    "system_hash": sys_hash,
                    "sfw_level": sfw_level,
                    "style_mode": "inline",
                },
            }

            triplets = save_and_score(
                response,
                writer,
                logger,
                scorer,
                meta_base,
                final_prompt,
                scene,
                session_id,
                gen=gen_idx,
                indiv=indiv_idx,
                w_style=w_style,
                w_nsfw=w_nsfw,
            )
            time.sleep(sleep_s)

            if triplets:
                best_path, nsfw100, style100 = triplets[0]
                fitness = w_style * style100 + w_nsfw * nsfw100
                scored.append((fitness, genes, best_path, style100, nsfw100))

        if not scored:
            print("[evolve] no scored individuals; stopping.")
            break

        scored.sort(key=lambda record: record[0], reverse=True)
        elite_n = max(1, int(round(keep * len(scored))))
        elites = scored[:elite_n]
        best = elites[0]
        print(
            f"[evolve] elite={elite_n}, best_fitness={best[0]:.2f}, style={best[3]}, nsfw={best[4]}"
        )

        new_population: List[GeneSet] = [dict(genes) for _, genes, _, _, _ in elites]
        while len(new_population) < pop:
            if random.random() < xover and len(elites) >= 2:
                parent_a = random.choice(elites)[1]
                parent_b = random.choice(elites)[1]
                child = crossover_genes(parent_a, parent_b)
            else:
                child = dict(random.choice(elites)[1])
            for key in list(child.keys()):
                if random.random() < mut:
                    child[key] = mutate_gene(catalog, key, child.get(key), sfw_level, temperature)
            new_population.append(child)
        population = new_population[:pop]

    print("\n[evolve] done.")
