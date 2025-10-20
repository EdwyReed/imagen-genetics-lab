import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

TEMPLATE_ROOT = Path(__file__).parent / "templates"
CATEGORIES = {
    "pose": "poses",
    "item": "items",
    "character": "characters",
    "environment": "environment",
    "style": "styles",
}


@dataclass
class Template:
    category: str
    name: str
    prompt: str
    nsfw_weight: float
    source: Path

    @classmethod
    def from_file(cls, category: str, path: Path) -> "Template":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            category=category,
            name=data["name"],
            prompt=data["prompt"],
            nsfw_weight=float(data.get("nsfw_weight", 0.0)),
            source=path.relative_to(TEMPLATE_ROOT),
        )


def load_templates() -> Dict[str, List[Template]]:
    templates: Dict[str, List[Template]] = {}
    for category, folder_name in CATEGORIES.items():
        folder = TEMPLATE_ROOT / folder_name
        entries = [Template.from_file(category, path) for path in sorted(folder.glob("*.json"))]
        if not entries:
            raise RuntimeError(f"No templates found for category '{category}' in {folder}")
        templates[category] = entries
    return templates


def score_weight(nsfw_level: float, template: Template) -> float:
    distance = abs(nsfw_level - template.nsfw_weight)
    return max(1.0 - distance, 0.05)


def choose_template(options: List[Template], nsfw_level: float) -> Template:
    weights = [score_weight(nsfw_level, option) for option in options]
    return random.choices(options, weights=weights, k=1)[0]


def build_preprompt(nsfw_level: float, seed: int | None = None) -> Dict[str, object]:
    if seed is not None:
        random.seed(seed)
    templates = load_templates()
    selection: Dict[str, Template] = {}
    for category, options in templates.items():
        selection[category] = choose_template(options, nsfw_level)

    prompt_parts = [template.prompt for template in selection.values()]
    full_prompt = ", ".join(prompt_parts)

    current_time = datetime.now(timezone.utc)
    return {
        "generated_at": current_time.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "nsfw_level": nsfw_level,
        "seed": seed,
        "prompt": full_prompt,
        "components": {
            category: {
                "name": template.name,
                "prompt": template.prompt,
                "nsfw_weight": template.nsfw_weight,
                "source": str(template.source),
            }
            for category, template in selection.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a randomized preprompt object.")
    parser.add_argument(
        "--nsfw",
        type=float,
        default=0.0,
        help="Desired NSFW level between 0 and 1 (default: 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated"),
        help="Directory where the preprompt metadata will be written (default: ./generated).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.nsfw <= 1.0:
        raise SystemExit("The NSFW level must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprompt = build_preprompt(args.nsfw, seed=args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = args.output_dir / f"preprompt-{timestamp}.json"
    with filename.open("w", encoding="utf-8") as handle:
        json.dump(preprompt, handle, ensure_ascii=False, indent=2)

    print(json.dumps(preprompt, ensure_ascii=False, indent=2))
    print(f"\nSaved metadata to {filename}")


if __name__ == "__main__":
    main()
