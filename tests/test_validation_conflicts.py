import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.validation import ValidationContext, validate_run_parameters


def test_high_sfw_raises_coverage_and_body_focus():
    context = ValidationContext(
        sfw_level=0.82,
        macro_snapshot={"coverage_target": 0.2},
        meso_snapshot={"fitness_body_focus": 0.25},
        weights={"style": 0.8, "nsfw": 0.4},
    )

    result = validate_run_parameters(context)

    assert result.macro_snapshot["coverage_target"] == pytest.approx(0.45)
    assert result.meso_snapshot["fitness_body_focus"] == pytest.approx(0.35)
    assert any(conflict.rule.startswith("macro.coverage_vs_sfw") for conflict in result.conflicts)
    assert any("coverage_target" in str(c.corrections) for c in result.conflicts)
    assert result.notifications, "expected user notifications for conflict"


def test_weight_clamping_and_normalisation():
    context = ValidationContext(
        sfw_level=0.5,
        macro_snapshot={},
        meso_snapshot={},
        weights={"style": 1.2, "nsfw": 0.6},
    )

    result = validate_run_parameters(context)

    assert result.weights["style"] == pytest.approx(0.625)
    assert result.weights["nsfw"] == pytest.approx(0.375)
    assert any(conflict.rule.startswith("weights") for conflict in result.conflicts)
    assert any("normalise" in (conflict.message or "").lower() for conflict in result.conflicts)
    assert result.notifications, "expected at least one notification"


def test_validation_restores_defaults_when_weights_zero():
    context = ValidationContext(
        sfw_level=0.4,
        macro_snapshot={"coverage_target": 0.5},
        meso_snapshot={},
        weights={"style": 0.0, "nsfw": 0.0},
    )

    result = validate_run_parameters(context)

    assert result.weights["style"] == pytest.approx(0.7)
    assert result.weights["nsfw"] == pytest.approx(0.3)
    assert not result.conflicts or any(conflict.rule.startswith("weights") for conflict in result.conflicts)
