from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("yaml")
import yaml

from imagen_lab.config import DEFAULT_WEIGHT_PROFILE_PATH, PipelineConfig
from imagen_lab.scoring.weights_table import (
    DEFAULT_STYLE_WEIGHTS,
    StyleMixer,
    WeightProfileTable,
    normalize_weights,
)


def test_style_mixer_composition_matches_manual_sum() -> None:
    mixer = StyleMixer(weights=DEFAULT_STYLE_WEIGHTS)
    components = {
        key: (idx + 1) / (len(DEFAULT_STYLE_WEIGHTS) + 1)
        for idx, key in enumerate(DEFAULT_STYLE_WEIGHTS.keys())
    }
    composition = mixer.compose(components)
    expected = sum(
        DEFAULT_STYLE_WEIGHTS[key] * components.get(key, 0.0)
        for key in DEFAULT_STYLE_WEIGHTS.keys()
    )
    assert composition.total == pytest.approx(expected)
    for key, weight in DEFAULT_STYLE_WEIGHTS.items():
        assert composition.contributions[key] == pytest.approx(weight * components.get(key, 0.0))


def test_weight_profile_table_load_and_persist(tmp_path: Path) -> None:
    default_profile = {key: float(val) for key, val in DEFAULT_STYLE_WEIGHTS.items()}
    alt_profile = default_profile.copy()
    alt_profile.update({"clip": 0.25, "spec": 0.2, "retro": 0.12})
    payload = {
        "default": "default",
        "profiles": {
            "default": default_profile,
            "alt": alt_profile,
        },
    }
    path = tmp_path / "weights.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    table = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    assert set(table.profile_names()) == {"alt", "default"}

    updated = {
        "clip": 0.28,
        "spec": 0.22,
        "illu": 0.1,
        "retro": 0.12,
        "medium": 0.08,
        "sensual": 0.08,
        "pose": 0.04,
        "camera": 0.03,
        "color": 0.03,
        "accessories": 0.01,
        "composition": 0.005,
        "skin_glow": 0.005,
    }
    table.update_profile("alt", updated, persist=True)

    reloaded = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    expected = normalize_weights(updated, defaults=DEFAULT_STYLE_WEIGHTS)
    assert reloaded.get_profile("alt") == pytest.approx(expected)  # type: ignore[arg-type]


def test_style_mixer_persists_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    table = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS, create=True)
    mixer = StyleMixer(weight_table=table, profile="experiment", persist_updates=True)
    new_weights = {
        "clip": 0.26,
        "spec": 0.21,
        "illu": 0.09,
        "retro": 0.1,
        "medium": 0.1,
        "sensual": 0.08,
        "pose": 0.05,
        "camera": 0.03,
        "color": 0.03,
        "accessories": 0.02,
        "composition": 0.02,
        "skin_glow": 0.01,
    }
    mixer.set_weights(new_weights)

    reloaded = WeightProfileTable.load(path, defaults=DEFAULT_STYLE_WEIGHTS)
    profile = reloaded.get_profile("experiment")
    expected = normalize_weights(new_weights, defaults=DEFAULT_STYLE_WEIGHTS)
    for key, value in expected.items():
        assert pytest.approx(profile[key], rel=1e-6) == value


def test_pipeline_config_defaults_include_profile_path() -> None:
    config = PipelineConfig.from_dict({})
    assert config.scoring.weight_profiles_path == DEFAULT_WEIGHT_PROFILE_PATH


def test_pipeline_config_allows_disabling_profiles() -> None:
    config = PipelineConfig.from_dict({"scoring": {"weight_profiles_path": None}})
    assert config.scoring.weight_profiles_path is None
