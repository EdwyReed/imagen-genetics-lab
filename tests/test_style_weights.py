from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
pytest.importorskip("yaml")

from imagen_lab.config import DEFAULT_WEIGHT_PROFILE_PATH, PipelineConfig
from imagen_lab.scoring.weights_table import (
    DEFAULT_STYLE_WEIGHTS,
    StyleMixer,
    WeightProfileTable,
    normalize_weights,
)


def test_style_mixer_composition_matches_manual_sum() -> None:
    mixer = StyleMixer(weights=DEFAULT_STYLE_WEIGHTS)
    composition = mixer.compose(0.8, 0.6, 0.4)
    expected = sum(
        DEFAULT_STYLE_WEIGHTS[key] * value
        for key, value in {"clip": 0.8, "spec": 0.6, "illu": 0.4}.items()
    )
    assert composition.total == pytest.approx(expected)
    assert composition.contributions["clip"] == pytest.approx(DEFAULT_STYLE_WEIGHTS["clip"] * 0.8)
    assert composition.contributions["spec"] == pytest.approx(DEFAULT_STYLE_WEIGHTS["spec"] * 0.6)
    assert composition.contributions["illu"] == pytest.approx(DEFAULT_STYLE_WEIGHTS["illu"] * 0.4)


def test_weight_profile_table_load_and_persist(tmp_path: Path) -> None:
    profile_text = """
default: default
profiles:
  default:
    clip: 0.5
    spec: 0.3
    illu: 0.2
  alt:
    clip: 0.6
    spec: 0.2
    illu: 0.2
"""
    path = tmp_path / "weights.yaml"
    path.write_text(profile_text.strip() + "\n", encoding="utf-8")

    table = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    assert set(table.profile_names()) == {"alt", "default"}

    updated = {"clip": 0.7, "spec": 0.2, "illu": 0.1}
    table.update_profile("alt", updated, persist=True)

    reloaded = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    expected = normalize_weights(updated, defaults=DEFAULT_STYLE_WEIGHTS)
    assert reloaded.get_profile("alt") == pytest.approx(expected)  # type: ignore[arg-type]


def test_style_mixer_persists_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    table = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS, create=True)
    mixer = StyleMixer(weight_table=table, profile="experiment", persist_updates=True)
    mixer.set_weights({"clip": 0.6, "spec": 0.25, "illu": 0.15})

    reloaded = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    profile = reloaded.get_profile("experiment")
    assert pytest.approx(profile["clip"], rel=1e-6) == 0.6
    assert pytest.approx(profile["spec"], rel=1e-6) == 0.25
    assert pytest.approx(profile["illu"], rel=1e-6) == 0.15


def test_pipeline_config_defaults_include_profile_path() -> None:
    config = PipelineConfig.from_dict({})
    assert config.scoring.weight_profiles_path == DEFAULT_WEIGHT_PROFILE_PATH


def test_pipeline_config_allows_disabling_profiles() -> None:
    config = PipelineConfig.from_dict({"scoring": {"weight_profiles_path": None}})
    assert config.scoring.weight_profiles_path is None
