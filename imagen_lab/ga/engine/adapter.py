from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from imagen_lab.catalog import Catalog
from imagen_lab.ga import GeneSet, crossover_genes, load_best_gene_sets, mutate_gene
from imagen_lab.learning import StyleFeedback
from imagen_lab.scene.builder.interfaces import SceneBuilderProtocol, SceneRequest
from imagen_lab.caption.ollama.interfaces import CaptionEngineProtocol, CaptionRequest
from imagen_lab.image.imagen.interfaces import ImagenEngineProtocol, ImagenRequest
from imagen_lab.scoring.core.interfaces import ScoringEngineProtocol, ScoringRequest
from imagen_lab.scoring.core.utils import format_metrics
from imagen_lab.utils import OllamaServiceError, OllamaServiceManager

from .interfaces import GAEngineProtocol


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


class DefaultGAEngine(GAEngineProtocol):
    def __init__(
        self,
        builder: SceneBuilderProtocol,
        catalog: Catalog,
        caption_engine: CaptionEngineProtocol,
        imagen_engine: ImagenEngineProtocol,
        scoring_engine: ScoringEngineProtocol,
        feedback: StyleFeedback | None,
    ) -> None:
        self._builder = builder
        self._catalog = catalog
        self._caption_engine = caption_engine
        self._imagen_engine = imagen_engine
        self._scoring_engine = scoring_engine
        self._feedback = feedback

    def _seed_population(
        self,
        params: GARunParameters,
    ) -> List[GeneSet]:
        load_dotenv()
        population: List[GeneSet] = []
        if params.resume_best:
            seed_genes = load_best_gene_sets(params.db_path, params.resume_k, params.resume_session)
            if seed_genes:
                for genes in seed_genes:
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
            scene = self._builder.build_scene(
                SceneRequest(
                    sfw_level=params.sfw_level,
                    temperature=params.temperature,
                    feedback=self._feedback,
                )
            )
            population.append(dict(scene.gene_ids))
        return population

    def run(self, params: GARunParameters) -> None:
        if params.seed is not None:
            random.seed(params.seed)

        population = self._seed_population(params)
        manager = params.service_manager

        try:
            with manager:
                for gen_idx in range(1, params.generations + 1):
                    print(f"\n===== Generation {gen_idx}/{params.generations} =====")
                    scored: List[tuple[float, GeneSet, Path, float, float]] = []

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
                                feedback=self._feedback,
                            ),
                        )
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
                            time.sleep(params.sleep_s)
                            continue

                        response = imagen_result.response
                        if not getattr(response, "generated_images", None):
                            print(
                                f"[G{gen_idx} I{indiv_idx}] WARN: no image returned; final prompt: {caption.final_prompt}"
                            )
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
                                response=response,
                                prompt=caption.final_prompt,
                                scene=scene.raw,
                                session_id=params.session_id,
                                meta=meta_base,
                                weights={"style": params.w_style, "nsfw": params.w_nsfw},
                                ga_context={"gen": gen_idx, "indiv": indiv_idx},
                            )
                        )
                        time.sleep(params.sleep_s)

                        batch = scoring_result.batch
                        if not batch.is_empty():
                            best = batch.best(params.w_style, params.w_nsfw)
                            if best is not None:
                                fitness = params.w_style * best.style + params.w_nsfw * best.nsfw
                                scored.append((fitness, genes, best.path, best.style, best.nsfw))
                                if self._feedback is not None:
                                    style_metrics = batch.metrics.get("style", {}) if isinstance(batch.metrics, dict) else {}
                                    style_weights = None
                                    if isinstance(style_metrics, dict):
                                        weights = style_metrics.get("weights")
                                        if isinstance(weights, dict):
                                            style_weights = weights
                                    self._feedback.update(
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
