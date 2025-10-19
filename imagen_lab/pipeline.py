from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai

from scorer import DualScorer

from .catalog import Catalog
from .config import PipelineConfig
from .scoring import DEFAULT_STYLE_WEIGHTS, WeightProfileTable
from .learning import StyleFeedback
from .ga import GeneSet, crossover_genes, load_best_gene_sets, mutate_gene
from .prompting import (
    DEFAULT_REQUIRED_TERMS,
    append_required_terms,
    enforce_bounds,
    enforce_once,
    imagen_call,
    ollama_generate,
    system_prompt_for,
    system_prompt_hash,
)
from .scene_builder import SceneBuilder, short_readable
from .embeddings import EmbeddingCache, EmbeddingHistoryConfig
from .storage import ArtifactWriter, PromptLogger, save_and_score
from .utils import OllamaServiceError, OllamaServiceManager
from .style_guide import StyleGuide


@dataclass
class PipelineServices:
    scorer: Optional[DualScorer]
    catalog: Catalog
    options_catalog: Catalog | None
    builder: SceneBuilder
    logger: PromptLogger
    writer: ArtifactWriter
    client: genai.Client
    history: EmbeddingCache
    feedback: Optional[StyleFeedback]
    style: StyleGuide
    required_terms: List[str]


@dataclass
class GASettings:
    pop: int
    resume_k: int
    resume_mix: float


def _prepare_services(
    config: PipelineConfig,
    output_dir: Optional[Path] = None,
    *,
    enable_scoring: bool = True,
) -> PipelineServices:
    print("[services] preparing pipeline services")
    history_cfg = EmbeddingHistoryConfig(
        enabled=config.history.enabled,
        max_embeddings=config.history.max_embeddings,
    )
    history_cache = EmbeddingCache(history_cfg)
    profiles_path = getattr(config.scoring, "weight_profiles_path", None)
    weight_table = None
    if profiles_path:
        print(f"[services] loading weight profiles from {profiles_path}")
        defaults = config.scoring.weights or DEFAULT_STYLE_WEIGHTS
        weight_table = WeightProfileTable.load(
            profiles_path,
            defaults=defaults,
            create=True,
        )

    scorer: Optional[DualScorer] = None
    if enable_scoring:
        print("[services] initializing scorer")
        scorer = DualScorer(
            device=config.scoring.device,
            batch=config.scoring.batch_size,
            db_path=config.paths.database,
            jsonl_path=config.paths.scores_jsonl,
            weights=config.scoring.weights,
            tau=config.scoring.tau,
            cal_style=config.scoring.cal_style,
            cal_illu=config.scoring.cal_illu,
            auto_weights=config.scoring.auto_weights.as_dict(),
            weight_table=weight_table,
            weight_profile=getattr(config.scoring, "weight_profile", "default"),
            persist_profile_updates=getattr(config.scoring, "persist_profile_updates", False),
            composition_enabled=getattr(config.scoring, "composition_metrics", True),
        )
        setattr(scorer, "embedding_cache", history_cache)
    print(f"[services] loading catalog from {config.paths.catalog}")
    catalog = Catalog.load(config.paths.catalog)
    options_catalog = None
    options_path = getattr(config.paths, "options_catalog", None)
    if options_path:
        try:
            print(f"[services] loading options catalog from {options_path}")
            options_catalog = Catalog.load(options_path)
        except FileNotFoundError:
            print(f"[services] options catalog not found at {options_path}")
            options_catalog = None
    configured_terms = [t for t in config.prompting.required_terms if t]
    fallback_terms = configured_terms or list(DEFAULT_REQUIRED_TERMS)
    style = StyleGuide.from_catalog(catalog, fallback_terms)
    required_terms = configured_terms or list(style.required_terms)

    builder = SceneBuilder(
        catalog,
        required_terms=required_terms,
        template_ids=config.prompting.template_ids,
    )
    logger = PromptLogger(config.paths.database)
    writer = ArtifactWriter(output_dir or config.paths.output_dir)
    print(f"[services] artifacts will be written to {writer.output_dir}")
    client = genai.Client()
    feedback = StyleFeedback(config.feedback) if config.feedback.enabled else None
    if feedback is not None:
        print("[services] feedback engine enabled")
    print("[services] pipeline services ready")
    return PipelineServices(
        scorer=scorer,
        catalog=catalog,
        options_catalog=options_catalog,
        builder=builder,
        logger=logger,
        writer=writer,
        client=client,
        history=history_cache,
        feedback=feedback,
        style=style,
        required_terms=required_terms,
    )


def _format_metrics(metrics: Dict[str, object]) -> Optional[str]:
    if not metrics:
        return None
    pieces: List[str] = []
    batch = metrics.get("batch", {}) if isinstance(metrics, dict) else {}
    history = metrics.get("history", {}) if isinstance(metrics, dict) else {}
    style = metrics.get("style", {}) if isinstance(metrics, dict) else {}
    composition = metrics.get("composition", {}) if isinstance(metrics, dict) else {}

    try:
        pairwise_mean = batch.get("pairwise_mean")  # type: ignore[assignment]
        if pairwise_mean is not None:
            pieces.append(f"batch_mean={float(pairwise_mean):.3f}")
    except Exception:
        pass
    try:
        pairwise_min = batch.get("pairwise_min")  # type: ignore[assignment]
        if pairwise_min is not None:
            pieces.append(f"batch_min={float(pairwise_min):.3f}")
    except Exception:
        pass
    try:
        history_mean = history.get("mean_distance")  # type: ignore[assignment]
        if history_mean is not None:
            pieces.append(f"history_mean={float(history_mean):.3f}")
    except Exception:
        pass
    try:
        history_min = history.get("min_distance")  # type: ignore[assignment]
        if history_min is not None:
            pieces.append(f"history_min={float(history_min):.3f}")
    except Exception:
        pass
    size = history.get("size") if isinstance(history, dict) else None
    if size:
        try:
            pieces.append(f"history_size={int(size)}")
        except Exception:
            pass
    try:
        style_mean = style.get("mean_total")  # type: ignore[assignment]
        if style_mean is not None:
            pieces.append(f"style_mean={float(style_mean):.3f}")
    except Exception:
        pass
    contributions = style.get("mean_contributions") if isinstance(style, dict) else None
    if isinstance(contributions, dict) and contributions:
        try:
            contrib_bits = [f"{key}:{float(val):.3f}" for key, val in sorted(contributions.items())]
            pieces.append(f"style_contribs={'/'.join(contrib_bits)}")
        except Exception:
            pass
    if isinstance(composition, dict) and composition:
        try:
            crop = composition.get("mean_cropping_tightness")
            thirds = composition.get("mean_thirds_alignment")
            neg = composition.get("mean_negative_space")
            comp_bits = []
            if crop is not None:
                comp_bits.append(f"crop={float(crop):.3f}")
            if thirds is not None:
                comp_bits.append(f"thirds={float(thirds):.3f}")
            if neg is not None:
                comp_bits.append(f"neg_space={float(neg):.3f}")
            if comp_bits:
                pieces.append(f"composition({' '.join(comp_bits)})")
        except Exception:
            pass
    if not pieces:
        return None
    return " ".join(pieces)


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
    enable_scoring: bool = True,
) -> None:
    print("[plain] starting plain pipeline run")
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

    services = _prepare_services(config, output_dir=outdir, enable_scoring=enable_scoring)
    logger = services.logger
    writer = services.writer
    scorer = services.scorer
    builder = services.builder
    client = services.client
    feedback = services.feedback
    style = services.style
    required_terms = services.required_terms

    session_id = f"plain-{int(time.time())}"
    if scorer is None:
        print("[plain] scoring disabled (--no-scoring)")
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

    system_prompt = system_prompt_for(style, sfw_level)
    sys_hash = system_prompt_hash(system_prompt)

    service_manager = OllamaServiceManager(manual_mode=config.ollama.manual_mode)

    try:
        with service_manager:
            for idx in range(1, cycles + 1):
                print(f"\n[plain] cycle {idx}/{cycles} started")
                if service_manager.enabled:
                    try:
                        service_manager.ensure_running()
                    except OllamaServiceError as exc:
                        print(f"[{idx:02d}/{cycles}] Ollama start error: {exc}")
                        time.sleep(sleep_s)
                        continue

                print(f"[{idx:02d}/{cycles}] building scene")
                scene = builder.build_scene(
                    sfw_level=sfw_level,
                    temperature=temperature,
                    feedback=feedback,
                )
                print(
                    f"[{idx:02d}/{cycles}] scene ready template={scene.template_id} summary={short_readable(scene)}"
                )
                try:
                    print(f"[{idx:02d}/{cycles}] requesting caption from Ollama")
                    caption = ollama_generate(
                        config.ollama.url,
                        config.ollama.model,
                        system_prompt,
                        scene.ollama_payload(),
                        temperature=temperature,
                        top_p=config.ollama.top_p,
                        seed=seed,
                    )
                    print(f"[{idx:02d}/{cycles}] enforcing required terms once")
                    caption = enforce_once(
                        config.ollama.url,
                        config.ollama.model,
                        system_prompt,
                        scene.ollama_payload(),
                        caption,
                        required_terms=required_terms,
                        temperature=max(0.45, temperature - 0.05),
                        seed=seed,
                    )
                except Exception as exc:  # pragma: no cover - network interaction
                    print(f"[{idx:02d}/{cycles}] Ollama error: {exc}")
                    print(f"[{idx:02d}/{cycles}] last caption payload: {scene.ollama_payload()}")
                    time.sleep(sleep_s)
                    continue

                bounds = scene.caption_bounds
                min_words = int(bounds.get("min_words", 18))
                max_words = int(bounds.get("max_words", 60))
                print(
                    f"[{idx:02d}/{cycles}] enforcing bounds min_words={min_words} max_words={max_words}"
                )
                final_prompt = enforce_bounds(
                    caption,
                    min_words,
                    max_words,
                )
                final_prompt = append_required_terms(
                    final_prompt,
                    required_terms,
                    max_words=max_words,
                )
                print(f"[{idx:02d}/{cycles}] final prompt prepared: {final_prompt}")

                try:
                    print(f"[{idx:02d}/{cycles}] requesting images from Imagen model")
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
                    if "400" in str(exc):
                        print(f"[{idx:02d}/{cycles}] final prompt that caused 400: {final_prompt}")
                    time.sleep(sleep_s)
                    continue

                if not getattr(response, "generated_images", None):
                    print(f"[{idx:02d}/{cycles}] no images returned; final prompt: {final_prompt}")
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

                print(f"[{idx:02d}/{cycles}] saving artifacts and scoring")
                batch = save_and_score(
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

                if not batch.is_empty():
                    best = batch.best(w_style, w_nsfw)
                    if best is not None:
                        print(f"   [best] {best.path.name}  style={best.style} nsfw={best.nsfw}")
                        if feedback is not None and scorer is not None:
                            style_metrics = batch.metrics.get("style", {}) if isinstance(batch.metrics, dict) else {}
                            style_weights = None
                            if isinstance(style_metrics, dict):
                                weights = style_metrics.get("weights")
                                if isinstance(weights, dict):
                                    style_weights = weights
                            feedback.update(
                                gene_ids=dict(scene.gene_ids),
                                template_id=scene.template_id,
                                summary=short_readable(scene),
                                batch_metrics=batch.metrics,
                                best_image=best,
                                weights=style_weights,
                            )
                    metrics_line = _format_metrics(batch.metrics)
                    if metrics_line:
                        print(f"   [metrics] {metrics_line}")
                time.sleep(sleep_s)
    except OllamaServiceError as exc:
        print(f"[plain] Ollama service error: {exc}")


def _seed_population(
    builder: SceneBuilder,
    catalog: Catalog,
    settings: GASettings,
    sfw_level: float,
    temperature: float,
    db_path: Path,
    session_id: Optional[str],
    seed_resume: bool,
    *,
    feedback: Optional[StyleFeedback] = None,
) -> List[GeneSet]:
    print("[evolve] seeding initial population")
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
        print(f"[evolve] generating seed individual {len(population)+1}/{settings.pop}")
        scene = builder.build_scene(
            sfw_level=sfw_level,
            temperature=temperature,
            feedback=feedback,
        )
        population.append(dict(scene.gene_ids))
    print(f"[evolve] initial population size: {len(population)}")
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
    enable_scoring: bool = True,
) -> None:
    print("[evolve] starting evolutionary pipeline run")
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

    services = _prepare_services(config, output_dir=outdir, enable_scoring=enable_scoring)
    logger = services.logger
    writer = services.writer
    scorer = services.scorer
    builder = services.builder
    catalog = services.catalog
    client = services.client
    feedback = services.feedback
    style = services.style
    required_terms = services.required_terms

    session_id = f"evolve-{int(time.time())}"
    if scorer is None:
        print("[evolve] scoring disabled (--no-scoring)")
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

    system_prompt = system_prompt_for(style, sfw_level)
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
        feedback=feedback,
    )

    service_manager = OllamaServiceManager(manual_mode=config.ollama.manual_mode)

    try:
        with service_manager:
            for gen_idx in range(1, gens + 1):
                print(f"\n===== Generation {gen_idx}/{gens} =====")
                scored: List[Tuple[float, GeneSet, Path, int, int]] = []

                for indiv_idx, genes in enumerate(population, start=1):
                    print(f"[G{gen_idx} I{indiv_idx}] evaluating individual")
                    if service_manager.enabled:
                        try:
                            service_manager.ensure_running()
                        except OllamaServiceError as exc:
                            print(f"[G{gen_idx} I{indiv_idx}] Ollama start error: {exc}")
                            time.sleep(sleep_s)
                            continue

                    print(f"[G{gen_idx} I{indiv_idx}] rebuilding scene from genes")
                    scene = builder.rebuild_from_genes(
                        genes,
                        sfw_level=sfw_level,
                        temperature=temperature,
                        feedback=feedback,
                    )
                    print(
                        f"[G{gen_idx} I{indiv_idx}] scene ready template={scene.template_id} summary={short_readable(scene)}"
                    )
                    try:
                        print(f"[G{gen_idx} I{indiv_idx}] requesting caption from Ollama")
                        caption = ollama_generate(
                            config.ollama.url,
                            config.ollama.model,
                            system_prompt,
                            scene.ollama_payload(),
                            temperature=temperature,
                            top_p=config.ollama.top_p,
                            seed=seed,
                        )
                        print(f"[G{gen_idx} I{indiv_idx}] enforcing required terms once")
                        caption = enforce_once(
                            config.ollama.url,
                            config.ollama.model,
                            system_prompt,
                            scene.ollama_payload(),
                            caption,
                            required_terms=required_terms,
                            temperature=max(0.45, temperature - 0.05),
                            seed=seed,
                        )
                    except Exception as exc:  # pragma: no cover - network interaction
                        print(f"[G{gen_idx} I{indiv_idx}] Ollama error: {exc}")
                        print(f"[G{gen_idx} I{indiv_idx}] last caption payload: {scene.ollama_payload()}")
                        time.sleep(sleep_s)
                        continue

                    bounds = scene.caption_bounds
                    min_words = int(bounds.get("min_words", 18))
                    max_words = int(bounds.get("max_words", 60))
                    print(
                        f"[G{gen_idx} I{indiv_idx}] enforcing bounds min_words={min_words} max_words={max_words}"
                    )
                    final_prompt = enforce_bounds(
                        caption,
                        min_words,
                        max_words,
                    )
                    final_prompt = append_required_terms(
                        final_prompt,
                        required_terms,
                        max_words=max_words,
                    )
                    print(f"[G{gen_idx} I{indiv_idx}] final prompt prepared: {final_prompt}")

                    try:
                        print(f"[G{gen_idx} I{indiv_idx}] requesting images from Imagen model")
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
                        if "400" in str(exc):
                            print(
                                f"[G{gen_idx} I{indiv_idx}] final prompt that caused 400: {final_prompt}"
                            )
                        time.sleep(sleep_s)
                        continue

                    if not getattr(response, "generated_images", None):
                        print(
                            f"[G{gen_idx} I{indiv_idx}] WARN: no image returned; final prompt: {final_prompt}"
                        )
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

                    print(f"[G{gen_idx} I{indiv_idx}] saving artifacts and scoring")
                    batch = save_and_score(
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

                    if not batch.is_empty():
                        best = batch.best(w_style, w_nsfw)
                        if best is not None:
                            fitness = w_style * best.style + w_nsfw * best.nsfw
                            scored.append((fitness, genes, best.path, best.style, best.nsfw))
                            if feedback is not None and scorer is not None:
                                style_metrics = batch.metrics.get("style", {}) if isinstance(batch.metrics, dict) else {}
                                style_weights = None
                                if isinstance(style_metrics, dict):
                                    weights = style_metrics.get("weights")
                                    if isinstance(weights, dict):
                                        style_weights = weights
                                feedback.update(
                                    gene_ids=dict(scene.gene_ids),
                                    template_id=scene.template_id,
                                    summary=short_readable(scene),
                                    batch_metrics=batch.metrics,
                                    best_image=best,
                                    weights=style_weights,
                                )
                        metrics_line = _format_metrics(batch.metrics)
                        if metrics_line:
                            print(f"   [metrics] {metrics_line}")

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
    except OllamaServiceError as exc:
        print(f"[evolve] Ollama service error: {exc}")
        return

    print("\n[evolve] done.")
