"""Test-friendly orchestrators for the Normal/Evolve/Dry-run modes."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Sequence

from imagen_lab.db.repo.interfaces import PromptRecord, RepositoryProtocol, RunRecord, ScoreRecord


@dataclass(frozen=True)
class ModeContext:
    session_id: str
    sfw_level: float
    temperature: float
    weights: Mapping[str, float]
    macro_snapshot: Mapping[str, float] | None = None
    meso_snapshot: Mapping[str, float] | None = None
    profile_id: str | None = None
    seed: int | None = None
    top_p: float = 0.9
    conflicts: Sequence[Mapping[str, object]] | None = None


@dataclass(frozen=True)
class SceneRequest:
    sfw_level: float
    temperature: float
    profile_id: str | None = None
    macro_snapshot: Mapping[str, Any] | None = None
    meso_snapshot: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SceneDescription:
    template_id: str
    caption_bounds: Mapping[str, Any]
    aspect_ratio: str
    gene_ids: Mapping[str, Optional[str]]
    payload: Mapping[str, Any]
    summary: str


class SceneBuilderProtocol(Protocol):
    def build_scene(self, request: SceneRequest) -> SceneDescription: ...

    def rebuild_from_genes(
        self, genes: Mapping[str, Optional[str]], request: SceneRequest
    ) -> SceneDescription: ...


@dataclass(frozen=True)
class CaptionRequest:
    scene: SceneDescription
    sfw_level: float
    temperature: float
    top_p: float
    seed: int | None


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    final_prompt: str
    enforced: bool
    bounds: Mapping[str, Any]
    system_hash: str


class CaptionEngineProtocol(Protocol):
    def generate(self, request: CaptionRequest) -> CaptionResult: ...


@dataclass(frozen=True)
class ImagenRequest:
    prompt: str
    aspect_ratio: str
    variants: int
    person_mode: str | None
    guidance_scale: float | None
    seed: int | None


@dataclass(frozen=True)
class ImagenVariant:
    index: int
    image_bytes: bytes
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ImagenResult:
    response: Any
    variants: Sequence[ImagenVariant]
    metadata: Mapping[str, Any]


class ImagenEngineProtocol(Protocol):
    def generate(self, request: ImagenRequest) -> ImagenResult: ...


@dataclass(frozen=True)
class PipelineDependencies:
    scene_builder: SceneBuilderProtocol
    caption_engine: CaptionEngineProtocol
    imagen_engine: ImagenEngineProtocol
    repository: RepositoryProtocol


@dataclass(frozen=True)
class ModeResult:
    scene: SceneDescription
    caption: CaptionResult
    prompt_path: str
    imagen_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _MetricsBundle:
    micro: Mapping[str, float]
    meso: Mapping[str, float]


def _default_metrics() -> _MetricsBundle:
    micro = {
        "clip_prompt_alignment": 0.82,
        "ai_artifacts": 0.08,
        "style_core": 0.9,
        "gloss_intensity": 0.72,
        "softness_blur": 0.68,
        "coverage_ratio": 0.58,
        "skin_exposure": 0.35,
        "coverage_target_alignment": 0.61,
        "identity_consistency": 0.77,
        "pose_coherence": 0.73,
    }
    meso = {
        "fitness_style": 0.78,
        "fitness_body_focus": 0.62,
        "fitness_coverage": 0.69,
        "fitness_alignment": 0.76,
        "fitness_cleanliness": 0.91,
        "fitness_era_match": 0.72,
        "fitness_novelty": 0.57,
        "fitness_visual": 0.79,
    }
    return _MetricsBundle(micro=micro, meso=meso)


def _sentence_count(text: str) -> int:
    return len([segment for segment in re.split(r"[.!?]+", text) if segment.strip()])


def _scene_payload(scene: SceneDescription) -> Mapping[str, Any]:
    payload = scene.payload.get("scene_model") if isinstance(scene.payload, Mapping) else None
    if isinstance(payload, Mapping):
        return payload
    return {}


def _run_cycle(
    mode: str,
    deps: PipelineDependencies,
    context: ModeContext,
    *,
    ga_context: Mapping[str, Optional[int | str]] | None = None,
    skip_scoring: bool = False,
) -> ModeResult:
    timestamp = int(time.time())
    ga_context = ga_context or {}

    deps.repository.log_run(
        RunRecord(
            session_id=context.session_id,
            mode=mode,
            payload={
                "sfw_level": context.sfw_level,
                "temperature": context.temperature,
                "weights": dict(context.weights),
            },
            macro_snapshot=context.macro_snapshot,
            meso_snapshot=context.meso_snapshot,
            profile_id=context.profile_id,
            seed=context.seed,
            conflicts=list(context.conflicts or []),
            created_at=timestamp,
        )
    )

    scene = deps.scene_builder.build_scene(
        SceneRequest(
            sfw_level=context.sfw_level,
            temperature=context.temperature,
            profile_id=context.profile_id,
            macro_snapshot=context.macro_snapshot,
            meso_snapshot=context.meso_snapshot,
        )
    )

    caption = deps.caption_engine.generate(
        CaptionRequest(
            scene=scene,
            sfw_level=context.sfw_level,
            temperature=context.temperature,
            top_p=context.top_p,
            seed=context.seed,
        )
    )
    if _sentence_count(caption.caption) > 2:
        raise ValueError("caption exceeds two sentences")

    imagen_result = deps.imagen_engine.generate(
        ImagenRequest(
            prompt=caption.final_prompt,
            aspect_ratio=scene.aspect_ratio,
            variants=1,
            person_mode=None,
            guidance_scale=None,
            seed=context.seed,
        )
    )
    if not imagen_result.variants:
        raise ValueError("imagen engine returned no variants")
    variant = imagen_result.variants[0]

    scene_payload = _scene_payload(scene)
    choices = scene_payload.get("choices") if isinstance(scene_payload, Mapping) else None
    option_probs = scene_payload.get("option_probabilities") if isinstance(scene_payload, Mapping) else None

    prompt_path = f"{context.session_id}-{variant.index:02d}.jpg"

    prompt_record = PromptRecord(
        path=prompt_path,
        session_id=context.session_id,
        prompt=caption.final_prompt,
        params={
            "scene": scene_payload,
            "imagen": dict(imagen_result.metadata),
            "variant_metadata": dict(variant.metadata),
        },
        gene_choices=choices if isinstance(choices, Mapping) else dict(scene.gene_ids),
        option_probabilities=option_probs if isinstance(option_probs, Mapping) else None,
        caption=caption.caption,
        imagen_version=str(imagen_result.metadata.get("imagen_version", "unknown")),
        fitness=0.0,
        parents=ga_context.get("parents"),
        op=ga_context.get("op"),
        gen=ga_context.get("gen"),
        indiv=ga_context.get("indiv"),
        created_at=timestamp,
        status="ok",
    )

    if skip_scoring:
        deps.repository.record_cycle(prompt=prompt_record, score=None)
        return ModeResult(scene=scene, caption=caption, prompt_path=prompt_path, imagen_metadata=imagen_result.metadata)

    metrics = _default_metrics()
    prompt_record = PromptRecord(
        **{
            **prompt_record.__dict__,
            "fitness": metrics.meso["fitness_visual"],
        }
    )

    score_record = ScoreRecord(
        prompt_path=prompt_path,
        micro_metrics=metrics.micro,
        meso_metrics=metrics.meso,
        fitness_visual=metrics.meso["fitness_visual"],
        fitness_body_focus=metrics.meso["fitness_body_focus"],
        fitness_alignment=metrics.meso["fitness_alignment"],
        fitness_cleanliness=metrics.meso["fitness_cleanliness"],
        fitness_era_match=metrics.meso["fitness_era_match"],
        fitness_novelty=metrics.meso["fitness_novelty"],
        clip_alignment=metrics.micro["clip_prompt_alignment"],
        ai_artifacts=metrics.micro["ai_artifacts"],
        created_at=timestamp,
    )

    deps.repository.record_cycle(prompt=prompt_record, score=score_record)
    return ModeResult(scene=scene, caption=caption, prompt_path=prompt_path, imagen_metadata=imagen_result.metadata)


def run_normal_mode(deps: PipelineDependencies, context: ModeContext) -> ModeResult:
    """Execute a single normal-mode cycle using deterministic stubs."""

    return _run_cycle("normal", deps, context)


def run_evolve_mode(
    deps: PipelineDependencies,
    context: ModeContext,
    *,
    gen: int,
    indiv: int,
    op: str | None = None,
    parents: Sequence[str] | None = None,
) -> ModeResult:
    """Execute a single GA evaluation cycle with metadata recorded."""

    ga_context: MutableMapping[str, Optional[int | str]] = {
        "gen": gen,
        "indiv": indiv,
        "op": op,
        "parents": list(parents) if parents is not None else None,
    }
    return _run_cycle("evolve", deps, context, ga_context=ga_context)


def run_dry_run_mode(deps: PipelineDependencies, context: ModeContext) -> ModeResult:
    """Execute a dry-run cycle without writing score rows."""

    return _run_cycle("dry-run", deps, context, skip_scoring=True)


__all__ = [
    "ModeContext",
    "SceneRequest",
    "SceneDescription",
    "SceneBuilderProtocol",
    "CaptionRequest",
    "CaptionResult",
    "CaptionEngineProtocol",
    "ImagenRequest",
    "ImagenVariant",
    "ImagenResult",
    "ImagenEngineProtocol",
    "PipelineDependencies",
    "ModeResult",
    "run_normal_mode",
    "run_evolve_mode",
    "run_dry_run_mode",
]

