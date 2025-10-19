from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from imagen_lab.caption.ollama.interfaces import CaptionRequest
from imagen_lab.config import PipelineConfig
from imagen_lab.db.repo.interfaces import RunRecord
from imagen_lab.ga.engine.adapter import GARunParameters
from imagen_lab.ga.session_state import SessionGeneStats
from imagen_lab.pipeline_factory import PipelineContainer, create_pipeline_container
from imagen_lab.scene.builder.interfaces import SceneRequest
from imagen_lab.scoring.core.interfaces import ScoringRequest
from imagen_lab.scoring.core.utils import format_metrics
from imagen_lab.image.imagen.interfaces import ImagenRequest
from imagen_lab.utils import OllamaServiceError, OllamaServiceManager
from imagen_lab.validation import ValidationContext, validate_run_parameters


def _build_container(
    config: PipelineConfig,
    *,
    outdir: Optional[Path] = None,
    enable_scoring: bool = True,
) -> PipelineContainer:
    return create_pipeline_container(config, output_dir=outdir, enable_scoring=enable_scoring)


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

    macro_snapshot = dict(config.bias.macro_weights)
    meso_snapshot = dict(config.bias.meso_aggregates)
    validation = validate_run_parameters(
        ValidationContext(
            sfw_level=sfw_level,
            macro_snapshot=macro_snapshot,
            meso_snapshot=meso_snapshot,
            weights={"style": w_style, "nsfw": w_nsfw},
        )
    )
    sfw_level = validation.sfw_level
    macro_snapshot = validation.macro_snapshot
    meso_snapshot = validation.meso_snapshot
    w_style = validation.weights.get("style", w_style)
    w_nsfw = validation.weights.get("nsfw", w_nsfw)

    for note in validation.notifications:
        print(f"[validation] {note}")

    if seed is not None:
        random.seed(seed)

    container = _build_container(config, outdir=outdir, enable_scoring=enable_scoring)
    session_id = f"plain-{int(time.time())}"

    if container.scorer is None:
        print("[plain] scoring disabled (--no-scoring)")

    gene_tracker = SessionGeneStats(container.repository.iter_gene_stats())

    container.repository.log_run(
        RunRecord(
            session_id=session_id,
            mode="plain",
            payload={
                "cycles": cycles,
                "per_cycle": per_cycle,
                "sfw": sfw_level,
                "temperature": temperature,
                "weights": {"style": w_style, "nsfw": w_nsfw},
            },
            macro_snapshot=macro_snapshot,
            meso_snapshot=meso_snapshot,
            seed=seed,
            conflicts=validation.conflicts_payload(),
        )
    )

    manager = OllamaServiceManager(manual_mode=config.ollama.manual_mode)

    try:
        with manager:
            for idx in range(1, cycles + 1):
                print(f"\n[plain] cycle {idx}/{cycles} started")
                if manager.enabled:
                    try:
                        manager.ensure_running()
                    except OllamaServiceError as exc:
                        print(f"[{idx:02d}/{cycles}] Ollama start error: {exc}")
                        time.sleep(sleep_s)
                        continue

                gene_tracker.decay_penalties()
                scene_request = SceneRequest(
                    sfw_level=sfw_level,
                    temperature=temperature,
                    feedback=container.feedback,
                    macro_snapshot=macro_snapshot,
                    meso_snapshot=meso_snapshot,
                    gene_fitness=gene_tracker.ema_snapshot(),
                    penalties=gene_tracker.penalty_snapshot(),
                )
                scene = container.scene_builder.build_scene(scene_request)
                attempts = 0
                while gene_tracker.should_penalize(scene.gene_ids) and attempts < 3:
                    gene_tracker.decay_penalties()
                    scene = container.scene_builder.build_scene(scene_request)
                    attempts += 1
                print(
                    f"[{idx:02d}/{cycles}] scene ready template={scene.template_id} summary={scene.summary}"
                )

                try:
                    caption = container.caption_engine.generate(
                        CaptionRequest(
                            scene=scene,
                            sfw_level=sfw_level,
                            temperature=temperature,
                            top_p=config.ollama.top_p,
                            seed=seed,
                        )
                    )
                    if caption.enforced:
                        print(f"[{idx:02d}/{cycles}] enforced required terms once")
                except Exception as exc:  # pragma: no cover - network interaction
                    print(f"[{idx:02d}/{cycles}] Ollama error: {exc}")
                    gene_tracker.record_failure(scene.gene_ids)
                    time.sleep(sleep_s)
                    continue

                bounds = caption.bounds
                min_words = int(bounds.get("min_words", 18))
                max_words = int(bounds.get("max_words", 60))
                print(
                    f"[{idx:02d}/{cycles}] enforcing bounds min_words={min_words} max_words={max_words}"
                )
                print(f"[{idx:02d}/{cycles}] final prompt prepared: {caption.final_prompt}")

                try:
                    imagen_result = container.imagen_engine.generate(
                        ImagenRequest(
                            prompt=caption.final_prompt,
                            aspect_ratio=scene.aspect_ratio,
                            variants=per_cycle,
                            person_mode=config.imagen.person_mode,
                            guidance_scale=config.imagen.guidance_scale,
                            seed=seed,
                        )
                    )
                except Exception as exc:  # pragma: no cover - API call
                    print(f"[{idx:02d}/{cycles}] Imagen error: {exc}")
                    if "400" in str(exc):
                        print(f"[{idx:02d}/{cycles}] final prompt that caused 400: {caption.final_prompt}")
                    gene_tracker.record_failure(scene.gene_ids)
                    time.sleep(sleep_s)
                    continue

                response = imagen_result.response
                if not getattr(response, "generated_images", None):
                    print(f"[{idx:02d}/{cycles}] no images returned; final prompt: {caption.final_prompt}")
                    gene_tracker.record_failure(scene.gene_ids)
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
                        "system_hash": caption.system_hash,
                        "sfw_level": sfw_level,
                        "style_mode": "inline+required_terms",
                    },
                }

                scoring_result = container.scoring_engine.score(
                    ScoringRequest(
                        response=response,
                        prompt=caption.final_prompt,
                        scene=scene.raw,
                        session_id=session_id,
                        meta=meta_base,
                        weights={"style": w_style, "nsfw": w_nsfw},
                        ga_context={"gen": None, "indiv": None},
                    )
                )

                batch = scoring_result.batch
                if not batch.is_empty():
                    best = batch.best(w_style, w_nsfw)
                    if best is not None:
                        fitness_value = best.fitness(w_style, w_nsfw)
                        gene_tracker.record_success(scene.gene_ids, fitness_value)
                        print(f"   [best] {best.path.name}  style={best.style} nsfw={best.nsfw}")
                        if container.feedback is not None and container.scorer is not None:
                            style_metrics = batch.metrics.get("style", {}) if isinstance(batch.metrics, dict) else {}
                            style_weights = None
                            if isinstance(style_metrics, dict):
                                weights = style_metrics.get("weights")
                                if isinstance(weights, dict):
                                    style_weights = weights
                            container.feedback.update(
                                gene_ids=dict(scene.gene_ids),
                                template_id=scene.template_id,
                                summary=scene.summary,
                                batch_metrics=batch.metrics,
                                best_image=best,
                                weights=style_weights,
                            )
                    metrics_line = format_metrics(batch.metrics)
                    if metrics_line:
                        print(f"   [metrics] {metrics_line}")
                else:
                    gene_tracker.record_failure(scene.gene_ids)
                time.sleep(sleep_s)
    except OllamaServiceError as exc:
        print(f"[plain] Ollama service error: {exc}")
    finally:
        try:
            gene_tracker.flush(container.repository)
        except Exception as exc:
            print(f"[plain] failed to flush gene stats: {exc}")


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

    macro_snapshot = dict(config.bias.macro_weights)
    meso_snapshot = dict(config.bias.meso_aggregates)
    validation = validate_run_parameters(
        ValidationContext(
            sfw_level=sfw_level,
            macro_snapshot=macro_snapshot,
            meso_snapshot=meso_snapshot,
            weights={"style": w_style, "nsfw": w_nsfw},
        )
    )
    sfw_level = validation.sfw_level
    macro_snapshot = validation.macro_snapshot
    meso_snapshot = validation.meso_snapshot
    w_style = validation.weights.get("style", w_style)
    w_nsfw = validation.weights.get("nsfw", w_nsfw)

    for note in validation.notifications:
        print(f"[validation] {note}")

    container = _build_container(config, outdir=outdir, enable_scoring=enable_scoring)
    session_id = f"evolve-{int(time.time())}"

    if container.scorer is None:
        print("[evolve] scoring disabled (--no-scoring)")

    gene_tracker = SessionGeneStats(container.repository.iter_gene_stats())

    container.repository.log_run(
        RunRecord(
            session_id=session_id,
            mode="evolve",
            payload={
                "pop": pop,
                "gens": gens,
                "keep": keep,
                "mut": mut,
                "xover": xover,
                "sfw": sfw_level,
                "temperature": temperature,
                "weights": {"style": w_style, "nsfw": w_nsfw},
            },
            macro_snapshot=macro_snapshot,
            meso_snapshot=meso_snapshot,
            seed=seed,
            conflicts=validation.conflicts_payload(),
        )
    )

    manager = OllamaServiceManager(manual_mode=config.ollama.manual_mode)

    params = GARunParameters(
        session_id=session_id,
        generations=gens,
        pop_size=pop,
        keep=keep,
        mutation=mut,
        crossover=xover,
        sleep_s=sleep_s,
        sfw_level=sfw_level,
        temperature=temperature,
        seed=seed,
        w_style=w_style,
        w_nsfw=w_nsfw,
        top_p=config.ollama.top_p,
        imagen_variants=1,
        imagen_person_mode=config.imagen.person_mode,
        imagen_guidance=config.imagen.guidance_scale,
        imagen_model=config.imagen.model,
        ollama_url=config.ollama.url,
        ollama_model=config.ollama.model,
        resume_best=resume_best,
        resume_k=resume_k or min(pop, 12),
        resume_session=resume_session,
        resume_mix=resume_mix,
        db_path=config.paths.database,
        service_manager=manager,
        gene_tracker=gene_tracker,
        profile_id=None,
        macro_snapshot=macro_snapshot,
        meso_snapshot=meso_snapshot,
    )

    try:
        container.ga_engine.run(params)
    finally:
        try:
            gene_tracker.flush(container.repository)
        except Exception as exc:
            print(f"[evolve] failed to flush gene stats: {exc}")
