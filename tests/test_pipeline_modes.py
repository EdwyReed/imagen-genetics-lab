import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    deps = PipelineDependencies(
        scene_builder=builder,
        caption_engine=StubCaptionEngine(),
        imagen_engine=StubImagenEngine(),
        repository=SQLiteRepository(repo_path),
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
