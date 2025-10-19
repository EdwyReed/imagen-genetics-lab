from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from imagen_lab.db.repo.interfaces import PromptRecord, RepositoryProtocol, ScoreRecord
from imagen_lab.image.store import ImageArtifactStore, SavedVariant

from .interfaces import ScoredVariant, ScoringEngineProtocol, ScoringRequest, ScoringResult
from .metrics import ScoreInputs, ScoreReport, compute_score_report


def _scene_payload(scene: Any) -> Mapping[str, Any]:
    if scene is None:
        return {}
    if isinstance(scene, Mapping):
        return dict(scene)
    choices_payload = getattr(scene, "choices_payload", None)
    if callable(choices_payload):
        try:
            payload = choices_payload()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:  # pragma: no cover - defensive
            pass
    to_payload = getattr(scene, "to_payload", None)
    if callable(to_payload):
        try:
            payload = to_payload()
            if isinstance(payload, Mapping):
                model = payload.get("scene_model")
                if isinstance(model, Mapping):
                    return dict(model)
                return dict(payload)
        except Exception:  # pragma: no cover - defensive
            pass
    return {}


def _score_inputs(saved: SavedVariant, caption: str) -> ScoreInputs:
    return ScoreInputs(image=saved.image, caption=caption, scene=saved.scene_payload)


@dataclass
class CoreScoringEngine(ScoringEngineProtocol):
    repository: RepositoryProtocol
    artifacts: ImageArtifactStore
    temperatures: Mapping[str, Any] | None = None
    compute_metrics: bool = True

    def score(self, request: ScoringRequest) -> ScoringResult:
        scene_payload = _scene_payload(request.scene)
        timestamp = int(request.timestamp or time.time())
        saved_variants = self.artifacts.persist_variants(
            session_id=request.session_id,
            imagen=request.imagen,
            prompt=request.prompt,
            caption=request.caption,
            scene_payload=scene_payload,
            meta_base=request.meta,
        )
        if not saved_variants:
            return ScoringResult(variants=())

        scored = [
            self._score_variant(saved, request, timestamp)
            for saved in saved_variants
        ]
        return ScoringResult(tuple(scored))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _score_variant(
        self,
        saved: SavedVariant,
        request: ScoringRequest,
        timestamp: int,
    ) -> ScoredVariant:
        report = self._compute_report(saved, request)
        fitness_style = float(report.meso.get("fitness_style", 0.0))
        fitness_coverage = float(report.meso.get("fitness_coverage", 0.0))
        fitness_visual = float(report.meso.get("fitness_visual", 0.0))
        w_style = float(request.weights.get("style", 0.0))
        w_nsfw = float(request.weights.get("nsfw", 0.0))
        composite = w_style * fitness_style + w_nsfw * (1.0 - fitness_coverage)

        prompt_record = PromptRecord(
            path=saved.prompt_path,
            session_id=request.session_id,
            prompt=request.prompt,
            params={
                "scene": saved.scene_payload,
                "imagen": dict(request.imagen.metadata),
                "variant_metadata": dict(saved.metadata.get("variant_metadata", {})),
                "meta": dict(request.meta),
            },
            gene_choices=saved.gene_choices,
            option_probabilities=saved.option_probabilities,
            caption=request.caption,
            imagen_version=str(request.imagen.metadata.get("imagen_version", "unknown")),
            fitness=composite,
            parents=request.ga_context.get("parents"),
            op=request.ga_context.get("op"),
            gen=request.ga_context.get("gen"),
            indiv=request.ga_context.get("indiv"),
            created_at=timestamp,
            status="ok",
        )

        score_record: ScoreRecord | None = None
        if self.compute_metrics:
            score_record = ScoreRecord(
                prompt_path=saved.prompt_path,
                micro_metrics=report.micro,
                meso_metrics=report.meso,
                fitness_visual=fitness_visual,
                fitness_body_focus=report.meso.get("fitness_body_focus"),
                fitness_alignment=report.meso.get("fitness_alignment"),
                fitness_cleanliness=report.meso.get("fitness_cleanliness"),
                fitness_era_match=report.meso.get("fitness_era_match"),
                fitness_novelty=report.meso.get("fitness_novelty"),
                clip_alignment=report.micro.get("clip_prompt_alignment"),
                ai_artifacts=report.micro.get("ai_artifacts"),
                created_at=timestamp,
            )
        self.repository.record_cycle(prompt=prompt_record, score=score_record)

        return ScoredVariant(
            prompt_path=saved.prompt_path,
            report=report,
            metadata=saved.metadata,
            fitness_style=fitness_style,
            fitness_coverage=fitness_coverage,
            fitness_visual=fitness_visual,
            composite_fitness=composite,
        )

    def _compute_report(self, saved: SavedVariant, request: ScoringRequest) -> ScoreReport:
        if not self.compute_metrics:
            return ScoreReport(micro={}, meso={}, temperatures={})
        inputs = _score_inputs(saved, request.caption)
        if self.temperatures:
            return compute_score_report(inputs, self.temperatures)
        return compute_score_report(inputs)


__all__ = ["CoreScoringEngine"]
