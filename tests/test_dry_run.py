from pathlib import Path
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.pipeline.configuration import PipelineConfig
from imagen_lab.pipeline.orchestrator import Pipeline


def make_config(tmp_path: Path) -> Path:
    styles = tmp_path / "styles"
    styles.mkdir()
    (styles / "style.json").write_text(
        """
        {
          "id": "style",
          "name": "Style",
          "description": "",
          "macro_defaults": {"sfw_level": 0.5, "coverage_target": 0.5, "focus_mode": "style_first"},
          "meso_defaults": {},
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
    config = tmp_path / "config.yaml"
    template = """
styles_path: {styles}
characters_path:
style_preset: style
character_preset:
macro_weights:
  sfw_level: 0.5
meso_overrides: {{}}
meso_aggregates: {{}}
runtime:
  dry_run_no_scoring: true
storage:
  database: {database}
  artifacts: {artifacts}
"""
    config.write_text(
        template.format(
            styles=styles,
            database=tmp_path / "runs.sqlite",
            artifacts=tmp_path / "artifacts",
        ),
        encoding="utf-8",
    )
    return config


def test_dry_run_skips_scoring(tmp_path: Path) -> None:
    config = PipelineConfig.load(make_config(tmp_path))
    pipeline = Pipeline(config)
    result = pipeline.run()
    pipeline.close()

    assert result.conflicts == ()
    assert Path(result.image_path).exists()
