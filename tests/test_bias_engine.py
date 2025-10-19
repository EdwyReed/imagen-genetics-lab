import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.bias.engine.simple import SimpleBiasEngine
from imagen_lab.bias.engine.interfaces import BiasContext


def test_bias_engine_clamps_sfw_and_records_conflicts() -> None:
    engine = SimpleBiasEngine(coverage_floor=0.5)
    context = BiasContext(
        profile_id="profile-a",
        macro_snapshot={"sfw_level": 1.2, "coverage_target": 0.2},
        meso_snapshot={},
        sfw_level=0.95,
        temperature=0.8,
        gene_fitness={},
        penalties={},
    )

    result = engine.compute_bias(context)

    assert result["sfw_level"] == pytest.approx(1.0)
    assert any(conflict["type"] == "coverage_vs_sfw" for conflict in result["conflicts"])
    assert any(rule.startswith("coverage_floor_enforced") for rule in result["applied_rules"])


def test_bias_engine_applies_gene_penalties() -> None:
    engine = SimpleBiasEngine()
    context = BiasContext(
        profile_id="profile-b",
        macro_snapshot={"novelty_preference": 0.85},
        meso_snapshot={},
        sfw_level=0.3,
        temperature=0.6,
        gene_fitness={"pose:hero": 0.8, "palette:retro": 0.2},
        penalties={"pose:hero": 0.5},
    )

    result = engine.compute_bias(context)

    gene_bias = result["gene_bias"]
    assert gene_bias["pose:hero"] < gene_bias["palette:retro"]
    assert any(rule.startswith("novelty_palette_boost") for rule in result["applied_rules"])
