import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

from imagen_lab.caption.ollama.contract import build_caption_payload


@dataclass
class _Choice:
    option_id: str
    label: str
    probability: float
    metadata: dict[str, object]


@dataclass
class _RawScene:
    catalog_id: str
    slots: dict[str, _Choice]
    option_probabilities: dict[str, tuple[dict[str, object], ...]]
    applied_rules: tuple[str, ...]
    conflicts: tuple[dict[str, object], ...]
    macro_snapshot: dict[str, object]
    meso_snapshot: dict[str, object]
    profile_id: str
    sfw_level: float
    temperature: float


@dataclass
class _SceneDescription:
    template_id: str
    caption_bounds: dict[str, object]
    aspect_ratio: str
    gene_ids: dict[str, str]
    payload: dict[str, object]
    summary: str
    raw: _RawScene


def make_scene_description() -> _SceneDescription:
    slots = {
        "pose": _Choice(
            option_id="pose_a",
            label="dynamic pose",
            probability=0.7,
            metadata={"description": "dramatic"},
        ),
        "lighting": _Choice(
            option_id="light_soft",
            label="soft lighting",
            probability=0.6,
            metadata={"nsfw": 0.2},
        ),
    }
    option_probabilities = {
        "pose": (
            {"id": "pose_a", "probability": 0.7, "label": "dynamic"},
            {"id": "pose_b", "probability": 0.3, "label": "static"},
        ),
        "lighting": (
            {"id": "light_soft", "probability": 0.6, "label": "soft"},
            {"id": "light_drama", "probability": 0.4, "label": "dramatic"},
        ),
    }
    raw_scene = _RawScene(
        catalog_id="catalog:test@v1",
        slots=slots,
        option_probabilities=option_probabilities,
        applied_rules=("novelty_palette_boost=1.20",),
        conflicts=(),
        macro_snapshot={
            "sfw_level": 0.6,
            "novelty_preference": 0.85,
            "lighting_softness": 0.9,
            "coverage_target": 0.5,
        },
        meso_snapshot={
            "fitness_style": 0.72,
            "fitness_body_focus": 0.31,
            "fitness_alignment": 0.52,
        },
        profile_id="profile:demo",
        sfw_level=0.6,
        temperature=0.75,
    )

    payload = {
        "style_profile": {
            "component_focus": [
                {"component": "gloss", "weight": 0.8},
                {"component": "retro", "weight": 0.6},
            ],
            "notes": ["boost gloss"],
        },
        "feedback_notes": ["Prefer soft glow"],
        "character": {
            "id": "char:avery",
            "name": "Avery",
            "summary": "Confident performer",
        },
    }

    return _SceneDescription(
        template_id="scene:probabilistic_v1",
        caption_bounds={"min_words": 18, "max_words": 40},
        aspect_ratio="3:4",
        gene_ids={"pose": "pose_a", "lighting": "light_soft"},
        payload=payload,
        summary="pose: dynamic pose; lighting: soft lighting",
        raw=raw_scene,
    )


def test_caption_payload_structure_contains_required_sections():
    scene = make_scene_description()

    payload = build_caption_payload(scene)

    assert set(payload) == {"style", "character", "scene", "top_signals"}

    style_section = payload["style"]
    assert style_section["profile_id"] == "profile:demo"
    assert "applied_rules" in style_section and "novelty_palette_boost=1.20" in style_section["applied_rules"]

    character = payload["character"]
    assert character["name"] == "Avery"

    scene_section = payload["scene"]
    assert scene_section["template_id"] == "scene:probabilistic_v1"
    assert any(choice["slot"] == "pose" for choice in scene_section["choices"])

    top_signals = payload["top_signals"]
    assert len(top_signals) <= 5
    signal_sources = {entry["source"] for entry in top_signals}
    assert {"macro", "meso"}.issubset(signal_sources)
    names = {entry["name"] for entry in top_signals}
    assert "sfw_level" in names
