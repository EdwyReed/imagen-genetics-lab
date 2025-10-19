import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.bias.engine.simple import SimpleBiasEngine
from imagen_lab.catalog import Catalog
from imagen_lab.scene.builder.interfaces import SceneRequest
from imagen_lab.scene.builder.probabilistic import ProbabilisticSceneBuilder


def make_catalog() -> Catalog:
    raw_catalog = {
        "version": "catalog:test@v1",
        "poses": [
            {"id": "pose_a", "desc": "pose A", "nsfw": 0.2},
            {"id": "pose_b", "desc": "pose B", "nsfw": 0.4},
        ],
        "lighting_presets": [
            {"id": "light_soft", "desc": "soft light", "nsfw": 0.1},
            {"id": "light_drama", "desc": "dramatic", "nsfw": 0.3},
        ],
        "palettes": [
            {
                "id": "palette_cool",
                "name": "Cool",
                "colors": [{"name": "Blue", "hex": "#00F"}],
                "nsfw": 0.1,
            },
            {
                "id": "palette_warm",
                "name": "Warm",
                "colors": [{"name": "Red", "hex": "#F00"}],
                "nsfw": 0.2,
            },
        ],
        "wardrobe_sets": [
            {
                "id": "wardrobe_layered",
                "items": ["jacket", "skirt"],
                "note": "layered outfit",
                "nsfw": 0.2,
            },
            {
                "id": "wardrobe_bold",
                "items": ["dress", "heels"],
                "note": "bold outfit",
                "nsfw": 0.45,
            },
        ],
        "backgrounds": [
            {"id": "bg_clean", "desc": "clean background", "nsfw": 0.1},
            {"id": "bg_city", "desc": "city lights", "nsfw": 0.35},
        ],
        "moods": [
            {"id": "mood_bright", "words": ["bright", "fresh"], "nsfw": 0.2},
            {"id": "mood_noir", "words": ["noir", "moody"], "nsfw": 0.4},
        ],
        "rules": {"caption_length": {"min_words": 12, "max_words": 30}},
    }
    return Catalog(raw=raw_catalog)


def test_scene_builder_returns_scene_model_with_probabilities():
    catalog = make_catalog()
    engine = SimpleBiasEngine()
    builder = ProbabilisticSceneBuilder(catalog=catalog, bias_engine=engine, rng=random.Random(42))

    request = SceneRequest(
        sfw_level=0.6,
        temperature=0.8,
        profile_id="profile:demo",
        macro_snapshot={"sfw_level": 0.6, "novelty_preference": 0.75},
        meso_snapshot={"fitness_novelty": 0.8},
        gene_fitness={"pose_a": 80.0, "pose_b": 20.0},
    )

    scene = builder.build_scene(request)

    assert scene.template_id == "scene:probabilistic_v1"
    assert scene.aspect_ratio in {"3:4", "9:16"}
    assert set(scene.gene_ids) == {"pose", "lighting", "palette", "wardrobe", "background", "mood"}
    payload = scene.payload["scene_model"]
    assert payload["catalog_id"] == "catalog:test@v1"
    assert payload["choices"]["pose"]["id"] in {"pose_a", "pose_b"}
    pose_probs = payload["option_probabilities"]["pose"]
    assert abs(sum(entry["probability"] for entry in pose_probs) - 1.0) < 1e-6
    assert any("novelty_palette_boost" in rule for rule in payload["applied_rules"])

    gene_json = json.loads(scene.payload["gene_choices_json"])
    assert gene_json["choices"]["palette"]["catalog_reference"]["section"] == "palette"


def test_rebuild_from_genes_preserves_override():
    catalog = make_catalog()
    engine = SimpleBiasEngine()
    builder = ProbabilisticSceneBuilder(catalog=catalog, bias_engine=engine, rng=random.Random(7))

    request = SceneRequest(sfw_level=0.5, temperature=0.7)
    scene = builder.build_scene(request)
    pose_choice = scene.gene_ids["pose"]

    rebuilt = builder.rebuild_from_genes({"pose": pose_choice}, request)
    assert rebuilt.gene_ids["pose"] == pose_choice
    payload = json.loads(rebuilt.payload["gene_choices_json"])
    assert payload["choices"]["pose"]["id"] == pose_choice


def test_gene_penalties_reduce_option_weight():
    catalog = make_catalog()
    engine = SimpleBiasEngine()
    builder = ProbabilisticSceneBuilder(catalog=catalog, bias_engine=engine, rng=random.Random(3))

    base_request = SceneRequest(
        sfw_level=0.4,
        temperature=0.7,
        gene_fitness={"wardrobe_layered": 70.0, "wardrobe_bold": 30.0},
    )
    scene = builder.build_scene(base_request)
    wardrobe_choice = scene.payload["scene_model"]["choices"]["wardrobe"]
    assert wardrobe_choice["id"] in {"wardrobe_layered", "wardrobe_bold"}

    penalized_request = SceneRequest(
        sfw_level=0.4,
        temperature=0.7,
        gene_fitness={"wardrobe_layered": 70.0, "wardrobe_bold": 30.0},
        penalties={"wardrobe_layered": 0.9},
    )
    penalized_scene = builder.build_scene(penalized_request)
    penalized_choice = penalized_scene.payload["scene_model"]["choices"]["wardrobe"]
    assert penalized_choice["id"] == "wardrobe_bold"
