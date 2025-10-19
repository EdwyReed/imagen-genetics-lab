import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.db.repo.interfaces import PromptRecord, ScoreRecord
from imagen_lab.db.repo.sqlite import SQLiteRepository
from imagen_lab.pipeline_modes import (
    CaptionEngineProtocol,
    CaptionRequest,
    CaptionResult,
    ImagenEngineProtocol,
    ImagenRequest,
    ImagenResult,
    ImagenVariant,
    ModeContext,
    PipelineDependencies,
    SceneBuilderProtocol,
    SceneDescription,
    SceneRequest,
    run_dry_run_mode,
    run_evolve_mode,
    run_normal_mode,
)
from imagen_lab.scoring.core.interfaces import ScoredVariant, ScoringEngineProtocol, ScoringRequest, ScoringResult
from imagen_lab.scoring.core.metrics import ScoreReport
from imagen_lab.pipeline_modes import (
    CaptionEngineProtocol,
    CaptionRequest,
    CaptionResult,
    ImagenEngineProtocol,
    ImagenRequest,
    ImagenResult,
    ImagenVariant,
    ModeContext,
    PipelineDependencies,
    SceneBuilderProtocol,
    SceneDescription,
    SceneRequest,
    run_dry_run_mode,
    run_evolve_mode,
    run_normal_mode,
)


class StubCaptionEngine(CaptionEngineProtocol):
    def generate(self, request: CaptionRequest) -> CaptionResult:
        summary = request.scene.summary
        caption = f"{summary}. Captured via stub."
        return CaptionResult(
            caption=caption,
            final_prompt=f"Prompt: {summary}",
            enforced=False,
            bounds=request.scene.caption_bounds,
            system_hash="stub",
        )


class StubImagenEngine(ImagenEngineProtocol):
    def generate(self, request: ImagenRequest) -> ImagenResult:
        response = SimpleNamespace(
            generated_images=[SimpleNamespace(image=b"binary")],
            model_version="imagen-test@v1",
        )
        variant = ImagenVariant(index=1, image_bytes=b"binary", metadata={"seed": request.seed or 0})
        metadata = {"imagen_version": "imagen-test@v1"}
        return ImagenResult(response=response, variants=(variant,), metadata=metadata)


class StubScoringEngine(ScoringEngineProtocol):
    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository

    def score(self, request: ScoringRequest) -> ScoringResult:
        prompt_path = f"{request.session_id}-01.jpg"
        scene = request.scene if isinstance(request.scene, Mapping) else {}
        gene_choices = scene.get("choices") if isinstance(scene, Mapping) else None
        option_probs = scene.get("option_probabilities") if isinstance(scene, Mapping) else None

        prompt_record = PromptRecord(
            path=prompt_path,
            session_id=request.session_id,
            prompt=request.prompt,
            params={
                "scene": scene,
                "imagen": dict(request.meta),
                "variant_metadata": {},
            },
            gene_choices=gene_choices,
            option_probabilities=option_probs,
            caption=request.caption,
            imagen_version=str(request.meta.get("mode", "stub")),
            fitness=0.79,
            parents=request.ga_context.get("parents"),
            op=request.ga_context.get("op"),
            gen=request.ga_context.get("gen"),
            indiv=request.ga_context.get("indiv"),
            created_at=int(time.time()),
            status="ok",
        )

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

        score_record = ScoreRecord(
            prompt_path=prompt_path,
            micro_metrics=micro,
            meso_metrics=meso,
            fitness_visual=meso["fitness_visual"],
            fitness_body_focus=meso["fitness_body_focus"],
            fitness_alignment=meso["fitness_alignment"],
            fitness_cleanliness=meso["fitness_cleanliness"],
            fitness_era_match=meso["fitness_era_match"],
            fitness_novelty=meso["fitness_novelty"],
            clip_alignment=micro["clip_prompt_alignment"],
            ai_artifacts=micro["ai_artifacts"],
            created_at=int(time.time()),
        )

        self._repository.record_cycle(prompt=prompt_record, score=score_record)

        report = ScoreReport(micro=micro, meso=meso, temperatures={})
        variant = ScoredVariant(
            prompt_path=prompt_path,
            report=report,
            metadata={},
            fitness_style=meso["fitness_style"],
            fitness_coverage=meso["fitness_coverage"],
            fitness_visual=meso["fitness_visual"],
            composite_fitness=0.79,
        )
        return ScoringResult((variant,))


class StubSceneBuilder(SceneBuilderProtocol):
    def __init__(self, payload: Mapping[str, object]):
        self._payload = payload

    def build_scene(self, request: SceneRequest) -> SceneDescription:
        model = json.loads(json.dumps(self._payload))
        return SceneDescription(
            template_id="stub:scene",
            caption_bounds={"min_words": 5, "max_words": 24},
            aspect_ratio="16:9",
            gene_ids={slot: data["id"] for slot, data in model["choices"].items()},
            payload={"scene_model": model},
            summary="; ".join(f"{slot} {data['label']}" for slot, data in model["choices"].items()),
        )

    def rebuild_from_genes(
        self, genes: Mapping[str, Optional[str]], request: SceneRequest
    ) -> SceneDescription:
        return self.build_scene(request)


def _make_dependencies(tmp_path: Path) -> tuple[PipelineDependencies, ModeContext]:
    scene_json = {
        "choices": {
            "pose": {"id": "pose:hero", "label": "Hero Pose"},
            "palette": {"id": "palette:retro", "label": "Retro"},
            "lighting": {"id": "light:soft", "label": "Soft"},
        },
        "option_probabilities": {
            "pose": [{"id": "pose:hero", "probability": 1.0}],
            "palette": [{"id": "palette:retro", "probability": 1.0}],
            "lighting": [{"id": "light:soft", "probability": 1.0}],
        },
        "macro_snapshot": {"coverage_target": 0.45},
        "meso_snapshot": {"fitness_body_focus": 0.4},
    }
    builder = StubSceneBuilder(scene_json)
    repo_path = tmp_path / "repo.sqlite"
    repository = SQLiteRepository(repo_path)
    deps = PipelineDependencies(
        scene_builder=builder,
        caption_engine=StubCaptionEngine(),
        imagen_engine=StubImagenEngine(),
        scoring_engine=StubScoringEngine(repository),
        repository=repository,
    )
    context = ModeContext(
        session_id="session-normal",
        sfw_level=0.65,
        temperature=0.75,
        weights={"style": 0.7, "nsfw": 0.3},
        macro_snapshot={"coverage_target": 0.45},
        meso_snapshot={"fitness_body_focus": 0.4},
        seed=42,
    )
    return deps, context


def _sentence_count(text: str) -> int:
    return len([segment for segment in text.split(".") if segment.strip()])


def test_normal_mode_records_prompt_and_score(tmp_path: Path) -> None:
    deps, context = _make_dependencies(tmp_path)

    result = run_normal_mode(deps, context)

    repo = deps.repository
    prompt = repo.get_prompt(result.prompt_path)
    score = repo.get_score(result.prompt_path)

    assert prompt is not None and score is not None
    assert prompt["caption"].endswith("Captured via stub.")
    assert score["fitness_visual"] == pytest.approx(0.79)
    assert _sentence_count(result.caption.caption) <= 2
    choices = result.scene.payload["scene_model"]["choices"]
    assert choices["pose"]["label"] == "Hero Pose"


def test_evolve_mode_records_ga_metadata(tmp_path: Path) -> None:
    deps, context = _make_dependencies(tmp_path)

    context = replace(context, session_id="session-evolve")
    result = run_evolve_mode(deps, context, gen=1, indiv=2, op="mutation", parents=["seed"])

    prompt = deps.repository.get_prompt(result.prompt_path)
    assert prompt is not None
    assert prompt["gen"] == 1
    assert prompt["indiv"] == 2
    assert prompt["op"] == "mutation"
    assert deps.repository.get_score(result.prompt_path) is not None


def test_dry_run_mode_skips_score_entries(tmp_path: Path) -> None:
    deps, context = _make_dependencies(tmp_path)

    context = replace(context, session_id="session-dry")
    result = run_dry_run_mode(deps, context)

    prompt = deps.repository.get_prompt(result.prompt_path)
    assert prompt is not None
    assert deps.repository.get_score(result.prompt_path) is None
