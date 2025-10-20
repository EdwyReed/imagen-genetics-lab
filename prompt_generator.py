import argparse
import base64
import json
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

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


def build_caption_payload(preprompt: Mapping[str, object]) -> Dict[str, object]:
    components = preprompt.get("components", {})
    return {
        "prompt": preprompt.get("prompt", ""),
        "components": components,
        "nsfw_level": preprompt.get("nsfw_level"),
    }


def _post_json(url: str, body: Mapping[str, object], timeout: float) -> Mapping[str, object]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:  # pragma: no cover - network error handling
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTP {exc.code} error from {url}: {detail.strip()}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network error handling
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    try:
        return json.loads(payload.decode(encoding))
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


DEFAULT_CAPTION_SYSTEM_PROMPT = """
You are a creative captioning assistant. Based on the provided structured
preprompt object, write a concise and vivid text-to-image caption that can be fed
to an Imagen-like text-to-image model. Keep it under 120 words, weave together
the pose, character, environment, and style cues, and do not add any extra
explanations.
""".strip()


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout: float = 30.0
    system_prompt: str = DEFAULT_CAPTION_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def generate_caption(
        self,
        payload: Mapping[str, object],
        *,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> str:
        prompt = f"{self.system_prompt}\n\n" + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body: Dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": 1.05,
            },
        }
        if seed is not None:
            body["options"]["seed"] = int(seed)

        url = f"{self.base_url}/api/generate"
        data = _post_json(url, body, timeout=self.timeout)
        text = data.get("response", "") if isinstance(data, Mapping) else ""
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama response did not include a caption")
        return " ".join(text.split())


@dataclass
class ImagenClient:
    base_url: str
    model: str
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def generate_images(
        self,
        caption: str,
        *,
        variants: int,
        aspect_ratio: str,
        guidance_scale: float,
        seed: int | None,
    ) -> Mapping[str, object]:
        body: Dict[str, object] = {
            "model": self.model,
            "prompt": caption,
            "variants": int(max(1, variants)),
            "aspect_ratio": aspect_ratio,
            "guidance_scale": float(guidance_scale),
        }
        if seed is not None:
            body["seed"] = int(seed)

        url = f"{self.base_url}/generate"
        response = _post_json(url, body, timeout=self.timeout)
        if not isinstance(response, Mapping):
            raise RuntimeError("Unexpected response type from Imagen endpoint")
        return response


def _extract_images(entries: Iterable[object]) -> List[Mapping[str, object]]:
    images: List[Mapping[str, object]] = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            images.append({"index": idx, "image": entry, "metadata": {}})
            continue
        if isinstance(entry, Mapping):
            payload: Dict[str, object] = {
                "index": idx,
                "metadata": dict(entry.get("metadata", {})) if isinstance(entry.get("metadata", {}), Mapping) else {},
            }
            image_data = None
            for key in ("image", "image_base64", "imageBytes", "image_bytes"):
                value = entry.get(key)
                if isinstance(value, str):
                    image_data = value
                    break
            if image_data is None and isinstance(entry.get("image"), Mapping):
                nested = entry["image"]
                for key in ("imageBytes", "image_bytes", "base64"):
                    value = nested.get(key) if isinstance(nested, Mapping) else None
                    if isinstance(value, str):
                        image_data = value
                        break
            if image_data:
                payload["image"] = image_data
                images.append(payload)
    return images


def save_imagen_outputs(
    response: Mapping[str, object],
    *,
    output_dir: Path,
    stem: str,
) -> Dict[str, object]:
    raw_images: Iterable[object] = []
    images_value = response.get("images")
    if isinstance(images_value, Iterable) and not isinstance(images_value, (str, bytes, bytearray)):
        raw_images = images_value  # type: ignore[assignment]
    else:
        generated_value = response.get("generated_images")
        if isinstance(generated_value, Iterable) and not isinstance(
            generated_value, (str, bytes, bytearray)
        ):
            raw_images = generated_value  # type: ignore[assignment]

    extracted = _extract_images(raw_images)
    saved_variants: List[Dict[str, object]] = []
    for variant in extracted:
        image_b64 = variant.get("image")
        if not isinstance(image_b64, str):
            continue
        try:
            blob = base64.b64decode(image_b64)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            continue

        index = variant.get("index", len(saved_variants) + 1)
        filename = output_dir / f"{stem}-v{int(index):02d}.png"
        with filename.open("wb") as handle:
            handle.write(blob)

        try:
            relative_path = filename.relative_to(output_dir)
        except ValueError:  # pragma: no cover - defensive
            relative_path = filename.name

        saved_variants.append(
            {
                "index": int(index),
                "image_path": str(relative_path),
                "metadata": variant.get("metadata", {}),
            }
        )

    metadata = dict(response.get("metadata", {})) if isinstance(response.get("metadata", {}), Mapping) else {}
    if "model" in response:
        metadata.setdefault("model", response["model"])
    if "model_version" in response:
        metadata.setdefault("model_version", response["model_version"])

    return {"metadata": metadata, "variants": saved_variants}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompts, captions, and images via the Ollama + Imagen pipeline."
    )
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
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama server (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3",
        help="Model name registered with Ollama (default: llama3).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for Ollama requests (default: 30).",
    )
    parser.add_argument(
        "--caption-temperature",
        type=float,
        default=0.55,
        help="Sampling temperature for the captioning model (default: 0.55).",
    )
    parser.add_argument(
        "--caption-top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value for the captioning model (default: 0.9).",
    )
    parser.add_argument(
        "--imagen-url",
        type=str,
        default="http://localhost:9090",
        help="Base URL for the Imagen-compatible service (default: http://localhost:9090).",
    )
    parser.add_argument(
        "--imagen-model",
        type=str,
        default="imagen",
        help="Target Imagen model identifier (default: imagen).",
    )
    parser.add_argument(
        "--imagen-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for Imagen requests (default: 60).",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of image variants to request from Imagen (default: 1).",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        help="Aspect ratio to request from Imagen (default: 1:1).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale parameter for Imagen (default: 7.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.nsfw <= 1.0:
        raise SystemExit("The NSFW level must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"run-{run_timestamp}"

    preprompt = build_preprompt(args.nsfw, seed=args.seed)
    caption_payload = build_caption_payload(preprompt)

    ollama_client = OllamaClient(
        base_url=args.ollama_url,
        model=args.ollama_model,
        timeout=args.ollama_timeout,
    )
    caption_prompt = ollama_client.generate_caption(
        caption_payload,
        temperature=args.caption_temperature,
        top_p=args.caption_top_p,
        seed=args.seed,
    )

    imagen_client = ImagenClient(
        base_url=args.imagen_url,
        model=args.imagen_model,
        timeout=args.imagen_timeout,
    )
    imagen_response = imagen_client.generate_images(
        caption_prompt,
        variants=args.variants,
        aspect_ratio=args.aspect_ratio,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    imagen_result = save_imagen_outputs(
        imagen_response,
        output_dir=args.output_dir,
        stem=stem,
    )

    artifact = dict(preprompt)
    artifact["caption"] = {
        "payload": caption_payload,
        "prompt": caption_prompt,
        "temperature": args.caption_temperature,
        "top_p": args.caption_top_p,
    }
    artifact["ollama"] = {
        "base_url": args.ollama_url,
        "model": args.ollama_model,
        "timeout": args.ollama_timeout,
    }
    artifact["imagen"] = {
        "base_url": args.imagen_url,
        "model": args.imagen_model,
        "timeout": args.imagen_timeout,
        "variants_requested": int(args.variants),
        "aspect_ratio": args.aspect_ratio,
        "guidance_scale": args.guidance_scale,
        "result": imagen_result,
    }

    metadata_path = args.output_dir / f"{stem}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)

    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"\nSaved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
