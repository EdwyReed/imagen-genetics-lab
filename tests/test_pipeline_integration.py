from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3

from imagen_lab.pipeline.configuration import PipelineConfig
from imagen_lab.pipeline.orchestrator import Pipeline


def build_config(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    styles = assets / "styles"
    characters = assets / "characters"
    styles.mkdir(parents=True)
    characters.mkdir()
    (styles / "style.json").write_text(
        """
        {
          "id": "style",
          "name": "Style",
          "description": "Test style",
          "macro_defaults": {"sfw_level": 0.6, "coverage_target": 0.5, "gloss_priority": 0.5, "illustration_strength": 0.8, "novelty_preference": 0.5, "lighting_softness": 0.6, "retro_authenticity": 0.6, "focus_mode": "style_first"},
          "meso_defaults": {"fitness_style": 60},
          "genes": {
            "pose": [{"id": "pose/base", "weight": 1.0}],
            "lighting": [{"id": "lighting/base", "weight": 1.0}],
            "palette": [{"id": "palette/base", "weight": 1.0}],
            "wardrobe": [{"id": "wardrobe/base", "weight": 1.0}]
          }
        }
        """,
        encoding="utf-8",
    )
    (characters / "char.json").write_text(
        """
        {
          "id": "char",
          "name": "Character",
          "summary": "Hero",
          "macro_overrides": {"coverage_target": 0.7},
          "traits": {}
        }
        """,
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
styles_path: {styles}
characters_path: {characters}
style_preset: style
character_preset: char
macro_weights:
  sfw_level: 0.9
  gloss_priority: 0.4
  coverage_target: 0.2
meso_overrides:
  fitness_style: 70
meso_aggregates:
  fitness_style:
    components:
      style_core: 0.5
      gloss_intensity: 0.5
storage:
  database: {tmp_path / 'runs.sqlite'}
  artifacts: {tmp_path / 'artifacts'}
        """,
        encoding="utf-8",
    )
    return config


def test_pipeline_records_prompt(tmp_path: Path) -> None:
    config_path = build_config(tmp_path)
    config = PipelineConfig.load(config_path)
    pipeline = Pipeline(config)
    result = pipeline.run(session_id="session-test")
    pipeline.close()

    assert Path(result.image_path).exists()
    assert result.pre_prompt.selected_genes["wardrobe"]
    assert result.conflicts  # sfw 0.9 with coverage from macro 0.3 triggers clamp

    conn = sqlite3.connect(tmp_path / "runs.sqlite")
    cur = conn.execute("SELECT metric, value FROM scores WHERE prompt_id = ?", (result.prompt_id,))
    scores = {metric: value for metric, value in cur.fetchall()}
    conn.close()

    assert "style_core" in scores
    assert "fitness_visual" in scores
    assert scores["clip_prompt_alignment"] >= 70.0

    conn = sqlite3.connect(tmp_path / "runs.sqlite")
    cur = conn.execute("SELECT average_fitness FROM gene_stats WHERE category = ?", ("wardrobe",))
    gene_row = cur.fetchone()
    conn.close()
    assert gene_row is not None and gene_row[0] > 0.0
