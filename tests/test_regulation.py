import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.bias.regulation import (
    BiasMixer,
    BiasSource,
    DEFAULT_REGULATORS,
    RegulationProjector,
    RegulationState,
    RegulatorLevel,
    RegulatorProfile,
    parse_rule,
    parse_rules,
)


def test_catalog_structure():
    specs = DEFAULT_REGULATORS
    assert len(specs) == 12
    macro = [spec for spec in specs.values() if spec.level is RegulatorLevel.MACRO]
    meso = [spec for spec in specs.values() if spec.level is RegulatorLevel.MESO]
    assert len(macro) == 6
    assert len(meso) == 6
    # every regulator links to at least one gene
    assert all(spec.gene_links for spec in specs.values())


def test_profile_resolve_and_rule_parsing():
    profile = RegulatorProfile(
        name="heroic",
        regulators={"subject_focus": 1.5, "nsfw_pressure": 0.1},
    )
    resolved = profile.resolve(DEFAULT_REGULATORS)
    assert resolved["subject_focus"] == pytest.approx(1.5)
    # nsfw_pressure is clamped by its spec min/max
    assert resolved["nsfw_pressure"] == pytest.approx(0.1)
    # unresolved regulators fallback to baseline
    assert resolved["prop_density"] == pytest.approx(1.0)

    rule_mul = parse_rule("subject_focus *= 0.7")
    rule_clamp = parse_rule("prop_density clamp 0.6..1.2")
    rule_forbid = parse_rule("!nsfw_pressure")
    assert rule_mul.regulator_id == "subject_focus"
    assert rule_clamp.regulator_id == "prop_density"
    assert rule_forbid.regulator_id == "nsfw_pressure"

    rules = parse_rules(["subject_focus *= 0.7", "!nsfw_pressure", "prop_density clamp 0.6..1.2"])
    assert len(rules) == 3


@pytest.mark.parametrize("line", ["", "unknown syntax"])
def test_bad_rules(line):
    with pytest.raises(Exception):
        parse_rule(line)


def test_projection_pipeline():
    projector = RegulationProjector(DEFAULT_REGULATORS)
    observed_genes = {
        "hero_focus": 0.1,
        "environment_story": -0.2,
        "prop_frequency": 0.3,
        "clean_background": -0.1,
        "nsfw_gate": 0.4,
        "safe_pose": -0.2,
    }
    observed_state = RegulationState.from_genes(observed_genes, DEFAULT_REGULATORS)
    assert observed_state.values["subject_focus"] == pytest.approx(1.14, rel=1e-3)
    assert observed_state.values["prop_density"] == pytest.approx(1.3, rel=1e-3)
    assert observed_state.values["nsfw_pressure"] == pytest.approx(0.38, rel=1e-3)

    rules = parse_rules(["subject_focus *= 0.7", "!nsfw_pressure", "prop_density clamp 0.6..1.2"])
    desired_adjustments = projector.project(observed_genes, rules)

    assert desired_adjustments["hero_focus"] == pytest.approx(-0.1212, abs=1e-4)
    assert desired_adjustments["environment_story"] == pytest.approx(0.0808, abs=1e-4)
    assert desired_adjustments["prop_frequency"] == pytest.approx(0.114285, abs=1e-5)
    assert desired_adjustments["clean_background"] == pytest.approx(-0.085714, abs=1e-5)
    assert desired_adjustments["nsfw_gate"] == pytest.approx(0.492308, abs=1e-6)
    assert desired_adjustments["safe_pose"] == pytest.approx(-0.307692, abs=1e-6)


def test_bias_mixer_with_temperature_and_ema():
    mixer = BiasMixer(
        DEFAULT_REGULATORS,
        floor=0.5,
        ceiling=1.5,
        temperature=2.0,
        ema_decay=0.25,
    )
    sources = [
        BiasSource("profile", {"subject_focus": 1.3, "nsfw_pressure": 0.6}),
        BiasSource("style", {"subject_focus": 0.9, "prop_density": 1.4}, weight=2.0),
        BiasSource("character", {"subject_focus": 1.1, "prop_density": 0.7}, weight=0.5),
    ]
    ema_state = {"subject_focus": 1.4, "nsfw_pressure": 0.3, "prop_density": 1.0}

    mixed = mixer.mix(sources, ema_state=ema_state)

    assert mixed["subject_focus"] == pytest.approx(1.113, abs=1e-3)
    assert mixed["nsfw_pressure"] == pytest.approx(0.72, abs=1e-2)
    assert mixed["prop_density"] == pytest.approx(1.06, abs=1e-2)
    # Regulators without explicit bias stay near baseline and within clamps
    assert mixed["texture_detail"] == pytest.approx(1.0, abs=1e-6)

