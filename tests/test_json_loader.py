from pathlib import Path
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.io.json_loader import JsonLoader


def test_json_loader_parses_bias_rules(tmp_path: Path) -> None:
    style_path = tmp_path / "styles"
    style_path.mkdir()
    payload = {
        "id": "sample",
        "name": "Sample Style",
        "description": "",
        "macro_defaults": {"sfw_level": 0.5},
        "genes": {
            "pose": [
                {"id": "pose/a", "weight": 0.5},
                {"id": "pose/b", "weight": 0.5}
            ]
        },
        "bias_rules": [
            {"when": {"sfw_level": "> 0.4"}, "adjust": {"pose": 0.3}}
        ]
    }
    (style_path / "sample.json").write_text(__import__("json").dumps(payload), encoding="utf-8")

    loader = JsonLoader(style_path)
    styles = loader.load_styles()

    assert styles["sample"].bias_rules[0].adjust["pose"] == 0.3
