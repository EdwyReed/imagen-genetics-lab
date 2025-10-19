import pytest

np = pytest.importorskip("numpy")

from imagen_lab.config import TemperatureConfig
from imagen_lab.scoring.core.metrics import ScoreInputs, compute_score_report


def _make_scene() -> dict:
    return {
        "summary": "dynamic pose with soft lighting and retro palette",
        "macro_snapshot": {"coverage_target": 0.6, "era_target": 0.75},
        "pose": {"label": "dynamic pose", "pose_confidence": 0.8},
        "lighting": {"label": "soft"},
        "palette": {"label": "retro sunset", "palette_era_score": 0.7},
        "background": {"label": "retro club"},
        "character": {"name": "Avery", "aliases": ["Av"]},
        "signals": {"chest_focus": 0.65, "thigh_focus": 0.5},
        "pose_metrics": {"coverage": 0.55, "skin_ratio": 0.32},
        "gene_ids": {"pose": "pose_a", "palette": "palette_vivid", "props": "prop_hat"},
        "option_probabilities": {
            "pose": (
                {"id": "pose_a", "probability": 0.3},
                {"id": "pose_b", "probability": 0.7},
            ),
            "palette": (
                {"id": "palette_vivid", "probability": 0.2},
                {"id": "palette_safe", "probability": 0.8},
            ),
            "props": (
                {"id": "prop_hat", "probability": 0.45},
                {"id": "prop_none", "probability": 0.55},
            ),
        },
    }


def _make_image() -> np.ndarray:
    base = np.array(
        [
            [[210, 120, 140], [230, 140, 150], [190, 110, 130], [180, 100, 120]],
            [[220, 150, 160], [240, 160, 170], [205, 135, 150], [195, 125, 140]],
            [[160, 90, 110], [170, 100, 120], [150, 80, 100], [140, 70, 90]],
            [[130, 60, 80], [140, 70, 90], [135, 65, 85], [145, 75, 95]],
        ],
        dtype=np.float32,
    )
    return base / 255.0


def test_compute_score_report_produces_expected_keys_and_ranges():
    caption = "Avery strikes a dynamic pose under soft retro lighting showing confident legs"
    scene = _make_scene()
    inputs = ScoreInputs(image=_make_image(), caption=caption, scene=scene)

    report = compute_score_report(inputs)

    assert set(report.micro) >= {
        "style_core",
        "gloss_intensity",
        "softness_blur",
        "clip_prompt_alignment",
        "ai_artifacts",
        "novelty_palette",
    }
    assert set(report.meso) == {
        "fitness_style",
        "fitness_body_focus",
        "fitness_coverage",
        "fitness_alignment",
        "fitness_cleanliness",
        "fitness_era_match",
        "fitness_novelty",
        "fitness_visual",
    }
    for value in report.micro.values():
        assert 0.0 <= value <= 1.0
    for value in report.meso.values():
        assert 0.0 <= value <= 1.0


def test_meso_aggregates_follow_documented_formulas():
    caption = "Avery poses with alluring flair under soft retro lighting"
    inputs = ScoreInputs(image=_make_image(), caption=caption, scene=_make_scene())
    report = compute_score_report(inputs)

    micro = report.micro
    meso = report.meso

    assert meso["fitness_style"] == pytest.approx(
        0.5 * micro["style_core"]
        + 0.3 * micro["gloss_intensity"]
        + 0.2 * micro["softness_blur"]
    )
    assert meso["fitness_body_focus"] == pytest.approx(
        0.4 * micro["chest_focus"]
        + 0.4 * micro["thigh_focus"]
        + 0.2 * micro["pose_suggestiveness"]
    )
    assert meso["fitness_coverage"] == pytest.approx(
        0.5 * micro["coverage_ratio"]
        + 0.3 * (1.0 - micro["skin_exposure"])
        + 0.2 * micro["coverage_target_alignment"]
    )
    assert meso["fitness_alignment"] == pytest.approx(
        0.45 * micro["clip_prompt_alignment"]
        + 0.35 * micro["identity_consistency"]
        + 0.20 * micro["pose_coherence"]
    )
    assert meso["fitness_cleanliness"] == pytest.approx(
        0.6 * (1.0 - micro["ai_artifacts"]) + 0.4 * (1.0 - micro["visual_noise_level"])
    )
    assert meso["fitness_era_match"] == pytest.approx(
        0.4 * micro["palette_era_match"]
        + 0.35 * micro["wardrobe_era_match"]
        + 0.25 * micro["environment_era_match"]
    )
    assert meso["fitness_novelty"] == pytest.approx(
        0.34 * micro["novelty_palette"]
        + 0.33 * micro["novelty_pose"]
        + 0.33 * micro["novelty_props"]
    )


def test_temperatures_change_alignment_behavior_without_exposing_micro():
    caption = "Avery dynamic pose retro lighting"
    scene = _make_scene()
    image = _make_image()
    inputs = ScoreInputs(image=image, caption=caption, scene=scene)

    baseline = compute_score_report(inputs, TemperatureConfig(default=0.0))
    sharpened = compute_score_report(inputs, TemperatureConfig(default=0.01))

    assert sharpened.micro["clip_prompt_alignment"] >= baseline.micro["clip_prompt_alignment"]
    assert sharpened.micro["ai_artifacts"] != baseline.micro["ai_artifacts"]

    public_payload = sharpened.public_payload()
    assert "micro" not in public_payload
    assert set(public_payload["aggregates"]) == set(sharpened.meso)


def test_score_report_clamps_out_of_range_micro_values():
    image = np.ones((2, 2, 3), dtype=np.float32)
    scene = _make_scene()
    caption = "Two sentence caption. Second sentence."
    inputs = ScoreInputs(image=image * 10.0, caption=caption, scene=scene)

    report = compute_score_report(inputs)

    assert all(0.0 <= value <= 1.0 for value in report.micro.values())
    assert all(0.0 <= value <= 1.0 for value in report.meso.values())

