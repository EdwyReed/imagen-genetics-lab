from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from dotenv import load_dotenv

from imagen_lab.catalog import Catalog
from imagen_lab.ga.genes import (
    GeneSet,
    crossover_genes,
    load_best_gene_sets,
    mutate_gene,
)
from imagen_lab.scene.builder.interfaces import SceneBuilderProtocol, SceneRequest
from imagen_lab.caption.ollama.interfaces import CaptionEngineProtocol, CaptionRequest
from imagen_lab.image.imagen.interfaces import ImagenEngineProtocol, ImagenRequest
from imagen_lab.scoring.core.interfaces import ScoringEngineProtocol, ScoringRequest
from imagen_lab.scoring.core.utils import format_metrics
from imagen_lab.utils import OllamaServiceError, OllamaServiceManager

from .interfaces import GAEngineProtocol
from ..session_state import SessionGeneStats


@dataclass(frozen=True)
class GARunParameters:
    session_id: str
    generations: int
    pop_size: int
    keep: float
    mutation: float
    crossover: float
    sleep_s: float
    sfw_level: float
    temperature: float
    seed: Optional[int]
    w_style: float
    w_nsfw: float
    top_p: float
    imagen_variants: int
    imagen_person_mode: Optional[str]
    imagen_guidance: Optional[float]
    imagen_model: str
    ollama_url: str
    ollama_model: str
    resume_best: bool
    resume_k: int
    resume_session: Optional[str]
    resume_mix: float
    db_path: Path
    service_manager: OllamaServiceManager
    gene_tracker: SessionGeneStats
    profile_id: Optional[str] = None
    macro_snapshot: Mapping[str, float] | None = None
    meso_snapshot: Mapping[str, float] | None = None


class DefaultGAEngine(GAEngineProtocol):
    def __init__(
        self,
        builder: SceneBuilderProtocol,
        catalog: Catalog,
        caption_engine: CaptionEngineProtocol,
        imagen_engine: ImagenEngineProtocol,
        scoring_engine: ScoringEngineProtocol,
    ) -> None:
        self._builder = builder
        self._catalog = catalog
        self._caption_engine = caption_engine
        self._imagen_engine = imagen_engine
        self._scoring_engine = scoring_engine

    def _seed_population(
        self,
        params: GARunParameters,
    ) -> List[GeneSet]:
        load_dotenv()
        population: List[GeneSet] = []
        tracker = params.gene_tracker
        if params.resume_best:
            seed_genes = load_best_gene_sets(params.db_path, params.resume_k, params.resume_session)
            if seed_genes:
                for genes in seed_genes:
                    if tracker.should_penalize(genes):
                        continue
                    child = dict(genes)
                    if params.resume_mix > 0.0:
                        for key in list(child.keys()):
                            if random.random() < params.resume_mix:
                                child[key] = mutate_gene(
                                    self._catalog,
                                    key,
                                    child.get(key),
                                    params.sfw_level,
                                    params.temperature,
                                )
                    population.append(child)
        while len(population) < params.pop_size:
            tracker.decay_penalties()
            scene = self._builder.build_scene(
                SceneRequest(
                    sfw_level=params.sfw_level,
                    temperature=params.temperature,
                    profile_id=params.profile_id,
                    macro_snapshot=params.macro_snapshot,
                    meso_snapshot=params.meso_snapshot,
                    gene_fitness=tracker.ema_snapshot(),
                    penalties=tracker.penalty_snapshot(),
                )
            )
            if tracker.should_penalize(scene.gene_ids):
                continue
            population.append(dict(scene.gene_ids))
        return population

    def run(self, params: GARunParameters) -> None:
        if params.seed is not None:
            random.seed(params.seed)

        population = self._seed_population(params)
        manager = params.service_manager
        tracker = params.gene_tracker

        try:
            with manager:
                for gen_idx in range(1, params.generations + 1):
                    print(f"\n===== Generation {gen_idx}/{params.generations} =====")
                    scored: List[tuple[float, GeneSet, Path, float, float]] = []
                    tracker.decay_penalties()

                    for indiv_idx, genes in enumerate(population, start=1):
                        print(f"[G{gen_idx} I{indiv_idx}] evaluating individual")
                        if manager.enabled:
                            try:
                                manager.ensure_running()
                            except OllamaServiceError as exc:
                                print(f"[G{gen_idx} I{indiv_idx}] Ollama start error: {exc}")
                                time.sleep(params.sleep_s)
                                continue

                        scene = self._builder.rebuild_from_genes(
                            genes,
                            SceneRequest(
                                sfw_level=params.sfw_level,
                                temperature=params.temperature,
                                profile_id=params.profile_id,
                                macro_snapshot=params.macro_snapshot,
                                meso_snapshot=params.meso_snapshot,
                                gene_fitness=tracker.ema_snapshot(),
                                penalties=tracker.penalty_snapshot(),
                            ),
                        )
                        if tracker.should_penalize(scene.gene_ids):
                            tracker.record_failure(scene.gene_ids)
                            continue
                        print(
                            f"[G{gen_idx} I{indiv_idx}] scene ready template={scene.template_id} summary={scene.summary}"
                        )
                        try:
                            caption = self._caption_engine.generate(
                                CaptionRequest(
                                    scene=scene,
                                    sfw_level=params.sfw_level,
                                    temperature=params.temperature,
                                    top_p=params.top_p,
                                    seed=params.seed,
                                )
                            )
                            if caption.enforced:
                                print(f"[G{gen_idx} I{indiv_idx}] enforced required terms once")
                        except Exception as exc:  # pragma: no cover - network interaction
                            print(f"[G{gen_idx} I{indiv_idx}] Ollama error: {exc}")
                            tracker.record_failure(scene.gene_ids)
                            time.sleep(params.sleep_s)
                            continue

                        try:
                            imagen_result = self._imagen_engine.generate(
                                ImagenRequest(
                                    prompt=caption.final_prompt,
                                    aspect_ratio=scene.aspect_ratio,
                                    variants=params.imagen_variants,
                                    person_mode=params.imagen_person_mode,
                                    guidance_scale=params.imagen_guidance,
                                    seed=params.seed,
                                )
                            )
                        except Exception as exc:  # pragma: no cover - API call
                            print(f"[G{gen_idx} I{indiv_idx}] Imagen error: {exc}")
                            tracker.record_failure(scene.gene_ids)
                            time.sleep(params.sleep_s)
                            continue

                        response = imagen_result.response
                        if not getattr(response, "generated_images", None):
                            print(
                                f"[G{gen_idx} I{indiv_idx}] WARN: no image returned; final prompt: {caption.final_prompt}"
                            )
                            tracker.record_failure(scene.gene_ids)
                            time.sleep(params.sleep_s)
                            continue

                        indiv_id = f"G{gen_idx:02d}-I{indiv_idx:02d}"
                        meta_base = {
                            "id": indiv_id,
                            "model_imagen": params.imagen_model,
                            "person_mode": params.imagen_person_mode,
                            "variants": params.imagen_variants,
                            "seed": params.seed,
                            "ollama": {
                                "url": params.ollama_url,
                                "model": params.ollama_model,
                                "temperature": params.temperature,
                                "top_p": params.top_p,
                                "system_hash": caption.system_hash,
                                "sfw_level": params.sfw_level,
                                "style_mode": "inline",
                            },
                        }

                        scoring_result = self._scoring_engine.score(
                            ScoringRequest(
                                imagen=imagen_result,
                                prompt=caption.final_prompt,
                                caption=caption.caption,
                                scene=scene.raw,
                                session_id=params.session_id,
                                meta=meta_base,
                                weights={"style": params.w_style, "nsfw": params.w_nsfw},
                                ga_context={"gen": gen_idx, "indiv": indiv_idx},
                            )
                        )
                        time.sleep(params.sleep_s)

                        if scoring_result.is_empty():
                            tracker.record_failure(scene.gene_ids)
                            continue

                        best_variant = scoring_result.best(params.w_style, params.w_nsfw)
                        if best_variant is None:
                            tracker.record_failure(scene.gene_ids)
                            continue

                        fitness = best_variant.weighted_fitness(params.w_style, params.w_nsfw)
                        tracker.record_success(scene.gene_ids, fitness)
                        style_component = best_variant.fitness_style
                        nsfw_component = 1.0 - best_variant.fitness_coverage
                        scored.append((fitness, genes, Path(best_variant.prompt_path), style_component, nsfw_component))
                        metrics_line = format_metrics(best_variant.report)
                        if metrics_line:
                            print(f"   [metrics] {metrics_line}")

                    if not scored:
                        print("[evolve] no scored individuals; stopping.")
                        break

                    scored.sort(key=lambda record: record[0], reverse=True)
                    elite_n = max(1, int(round(params.keep * len(scored))))
                    elites = scored[:elite_n]
                    best = elites[0]
                    print(
                        f"[evolve] elite={elite_n}, best_fitness={best[0]:.2f}, style={best[3]}, nsfw={best[4]}"
                    )

                    new_population: List[GeneSet] = [dict(genes) for _, genes, _, _, _ in elites]
                    while len(new_population) < params.pop_size:
                        if random.random() < params.crossover and len(elites) >= 2:
                            parent_a = random.choice(elites)[1]
                            parent_b = random.choice(elites)[1]
                            child = crossover_genes(parent_a, parent_b)
                        else:
                            child = dict(random.choice(elites)[1])
                        for key in list(child.keys()):
                            if random.random() < params.mutation:
                                child[key] = mutate_gene(
                                    self._catalog,
                                    key,
                                    child.get(key),
                                    params.sfw_level,
                                    params.temperature,
                                )
                        new_population.append(child)
                    population = new_population[: params.pop_size]
        except OllamaServiceError as exc:
            print(f"[evolve] Ollama service error: {exc}")

        print("\n[evolve] done.")
