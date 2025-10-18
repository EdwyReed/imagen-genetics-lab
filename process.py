"""Resize images and create style-free captions or tags for LoRA datasets."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from PIL import Image, ImageOps

from imagen_lab.utils import OllamaServiceError, OllamaServiceManager


STYLE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwatercolou?r\b",
        r"\bglossy\b",
        r"\billustration\b",
        r"\bpastel\b",
        r"\bdigital\s+(?:art|painting)\b",
        r"\boil\s+painting\b",
        r"\banime\b",
        r"\bcartoon\b",
        r"\bstyl(?:e|ized)\b",
        r"\b3d\b",
        r"\b2d\b",
        r"\brender(?:ed|ing)?\b",
    ]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare images and sanitized text labels")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with original images and prompts")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for resized images")
    parser.add_argument(
        "--max-size", type=int, default=512, help="Maximum size (pixels) for the longest image side"
    )
    parser.add_argument("--tagging", choices=["tags", "caption"], required=True, help="Output text format")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama endpoint base URL")
    parser.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--ollama-temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--ollama-top-p", type=float, default=0.8, help="top_p sampling value")
    parser.add_argument("--ollama-timeout", type=float, default=45.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--manual-ollama",
        action="store_true",
        help="Skip automatic Ollama lifecycle control if the service is already running",
    )
    return parser.parse_args()


def strip_style_terms(text: str) -> str:
    cleaned = text
    for pattern in STYLE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    return cleaned.strip()


def deduplicate(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        lowered = item.lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        result.append(item)
    return result


def format_tags(text: str) -> str:
    tokens = []
    for chunk in re.split(r"[,\n]", text):
        chunk = chunk.strip().lower()
        chunk = re.sub(r"[^a-z0-9\s\-]", "", chunk)
        chunk = chunk.replace("  ", " ")
        if chunk:
            tokens.append(chunk)
    tokens = deduplicate(tokens)
    return ", ".join(tokens)


def build_system_prompt(mode: str) -> str:
    if mode == "tags":
        return (
            "You refine prompts into factual content tags."
            " Extract the depicted characters, actions, wardrobe, props, setting, camera, and lighting."
            " Remove every reference to art mediums, rendering technology, or visual style."
            " Respond with comma-separated lowercase tags only."
        )
    return (
        "You rewrite prompts into a single descriptive caption about the scene."
        " Focus strictly on subjects, actions, clothing, background, and lighting."
        " Remove any words that describe drawing or rendering style, materials, or artistic techniques."
        " Reply with one or two sentences." 
    )


def call_ollama(
    *,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": system_prompt.strip() + "\n\nOriginal prompt:\n" + user_prompt.strip(),
        "stream": False,
        "raw": False,
        "options": {"temperature": float(temperature), "top_p": float(top_p), "repeat_penalty": 1.05},
    }
    response = requests.post(f"{url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data.get("response", "")
    return text.strip()


def sanitize_prompt(
    original: str,
    *,
    mode: str,
    url: str,
    model: str,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    system_prompt = build_system_prompt(mode)
    try:
        raw = call_ollama(
            url=url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=original,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    except (requests.RequestException, ValueError):
        raw = original
    cleaned = strip_style_terms(raw)
    if mode == "tags":
        formatted = format_tags(cleaned)
        if formatted:
            return formatted
        return format_tags(strip_style_terms(original))
    cleaned = " ".join(cleaned.split())
    if cleaned:
        return cleaned
    fallback = strip_style_terms(original)
    return " ".join(fallback.split())


def load_prompt_for(image_path: Path) -> Optional[str]:
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            final_prompt = data.get("final_prompt") or data.get("prompt")
            if isinstance(final_prompt, str) and final_prompt.strip():
                return final_prompt.strip()
    return None


def process_image(
    src_path: Path,
    dst_path: Path,
    *,
    max_size: int,
) -> None:
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if w == 0 or h == 0:
            raise ValueError("Image has zero dimension")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        scale = min(1.0, max_size / max(w, h))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        if new_size != im.size:
            im = im.resize(new_size, Image.LANCZOS)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, format="JPEG", quality=92, optimize=True)


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        yield path


def main() -> None:
    args = parse_args()
    src_dir: Path = args.input_dir
    dst_dir: Path = args.output_dir

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Input directory {src_dir} does not exist or is not a directory")
    dst_dir.mkdir(parents=True, exist_ok=True)

    service_manager = OllamaServiceManager(manual_mode=args.manual_ollama)

    try:
        with service_manager:
            index = 1
            for src_path in iter_image_files(src_dir):
                prompt = load_prompt_for(src_path)
                if not prompt:
                    print(f"[skip] No prompt found for {src_path.name}")
                    continue
                try:
                    sanitized = sanitize_prompt(
                        prompt,
                        mode=args.tagging,
                        url=args.ollama_url,
                        model=args.ollama_model,
                        temperature=args.ollama_temperature,
                        top_p=args.ollama_top_p,
                        timeout=args.ollama_timeout,
                    )
                except Exception as exc:
                    print(f"[skip] Failed to sanitize prompt for {src_path.name}: {exc}")
                    continue

                out_path = dst_dir / f"{index}.jpg"
                try:
                    process_image(src_path, out_path, max_size=args.max_size)
                except Exception as exc:
                    print(f"[skip] Failed to process {src_path.name}: {exc}")
                    continue

                text_path = out_path.with_suffix(".txt")
                text_path.write_text(sanitized + "\n", encoding="utf-8")
                print(f"[ok] {src_path.name} -> {out_path.name}")
                index += 1
    except OllamaServiceError as exc:
        raise SystemExit(f"Ollama service error: {exc}")


if __name__ == "__main__":
    main()
