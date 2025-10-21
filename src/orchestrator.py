from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .captioner_ollama import Captioner, CaptionResult, build_payload
from .collector import DICT_FILES, OptionCollector, load_dictionaries
from .generator_imagen import ImagenGenerator, ImagenMeta
from .io_utils import create_session, hash_dictionary_files, hash_text, save_variant
from .logging_utils import create_logger
from .rng import DeterministicRNG
from .schema import Config, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset orchestrator for Imagen Genetics Lab")
    parser.add_argument("--config", type=Path, default=Path("conf/config.toml"), help="Path to config TOML")
    parser.add_argument("--count", type=int, help="Override run.count")
    parser.add_argument("--seed", type=int, help="Override run.seed")
    parser.add_argument("--out", type=Path, help="Override output directory")
    parser.add_argument("--dict-dir", type=Path, default=Path("data/dictionaries"), help="Dictionary directory")
    parser.add_argument("--session-id", type=str, help="Explicit session id")
    parser.add_argument("--dry-run", action="store_true", help="Skip Imagen calls and emit placeholders")
    return parser.parse_args()


def _resolve_seed(config: Config) -> int:
    if config.run.seed is not None:
        return int(config.run.seed)
    seed = int(time.time() * 1000) & 0xFFFFFFFF
    config.run.seed = seed
    return seed


def _common_meta(
    config: Config,
    dict_hashes: Dict[str, str],
    system_hash: str,
    imagen_meta: ImagenMeta,
    caption_result: CaptionResult,
) -> Dict[str, object]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "caption_model": config.ollama.model,
        "system_hash": system_hash,
        "imagen_model": imagen_meta.model,
        "aspect_ratio": imagen_meta.aspect_ratio,
        "dicts_sha256": dict_hashes,
        "person_mode": imagen_meta.person_mode,
        "resolution": imagen_meta.resolution,
        "safety_filter": imagen_meta.safety_filter,
        "created_at": now,
        "required_terms": config.enforce.required_terms,
        "rewrites": caption_result.rewrites,
        "missing_terms": caption_result.missing_terms or None,
        "errors": None,
    }


def _fallback_caption(payload: Dict[str, object]) -> str:
    character = payload.get("character", "a character")
    action = payload.get("action", "posing")
    pose = payload.get("pose", "standing")
    clothes = payload.get("clothes", "styled wardrobe")
    style = payload.get("style", "soft lighting")
    camera = payload.get("camera", "studio framing")
    mood = payload.get("mood")
    parts = [
        f"{character} {action}",
        f"in {pose}",
        f"wearing {clothes}",
        f"captured with {camera}",
        f"set in {style}",
    ]
    if mood:
        parts.append(f"evoking {mood}")
    caption = ", ".join(parts)
    return caption


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    config.apply_overrides(count=args.count, seed=args.seed, out_dir=args.out)

    dict_dir = args.dict_dir.resolve()
    dictionaries = load_dictionaries(dict_dir)
    dictionary_paths = [dict_dir / filename for _, filename in DICT_FILES]

    system_prompt = config.ollama.system_prompt_path.read_text(encoding="utf-8")
    system_hash = hash_text(system_prompt)
    dict_hashes = hash_dictionary_files(dictionary_paths)

    seed = _resolve_seed(config)
    rng = DeterministicRNG(seed)

    session = create_session(config.run.out_dir, args.session_id, prefix=config.cli.default_session_prefix)
    logger = create_logger(config.logging.level, session.log_file if config.logging.to_file else None)
    try:
        logger.log("BOOT", f"config={config.path} dictionaries={dict_dir}")
        logger.log("SESSION", f"start id={session.session_id} seed={seed} count={config.run.count}")

        collector = OptionCollector(dictionaries, config.select, rng)
        captioner = None if args.dry_run else Captioner(config.ollama, config.enforce, system_prompt, seed)
        imagen = ImagenGenerator(config.imagen, dry_run=args.dry_run)

        for index in range(1, config.run.count + 1):
            try:
                options = logger.timed(
                    "COLLECT",
                    lambda opt: f"i={index}/{config.run.count} {OptionCollector.summarize(opt)}",
                    collector.collect,
                )
            except Exception as exc:
                logger.log("COLLECT", f"i={index} failed: {exc}", level="ERROR")
                continue

            payload = build_payload(options, config.enforce.required_terms)

            if captioner is None:
                caption_result = CaptionResult(
                    caption=_fallback_caption(payload),
                    rewrites=0,
                    missing_terms=[],
                )
                logger.log(
                    "OLLAMA",
                    f"i={index}/{config.run.count} dry-run placeholder words={len(caption_result.caption.split())}",
                )
            else:
                try:
                    caption_result = logger.timed(
                        "OLLAMA",
                        lambda res: f"i={index}/{config.run.count} words={len(res.caption.split())} rewrites={res.rewrites}",
                        captioner.generate,
                        payload,
                    )
                except Exception as exc:
                    logger.log("OLLAMA", f"i={index} failed: {exc}", level="ERROR")
                    continue

                if caption_result.rewrites or caption_result.missing_terms:
                    level = "WARN" if caption_result.missing_terms else "INFO"
                    missing_str = ",".join(caption_result.missing_terms)
                    logger.log(
                        "ENFORCE",
                        f"i={index}/{config.run.count} rewrites={caption_result.rewrites} missing=[{missing_str}]",
                        level=level,
                    )

            try:
                variants, imagen_meta = logger.timed(
                    "IMAGEN",
                    lambda res: f"i={index}/{config.run.count} variants={len(res[0])}",
                    imagen.generate,
                    caption_result.caption,
                )
            except Exception as exc:
                logger.log("IMAGEN", f"i={index} failed: {exc}", level="ERROR")
                continue

            if not variants:
                logger.log("IMAGEN", f"i={index} produced no variants", level="WARN")
                continue

            meta_template = _common_meta(config, dict_hashes, system_hash, imagen_meta, caption_result)

            def _save() -> List[str]:
                saved = []
                for variant in variants:
                    artifact = save_variant(
                        base_id=index,
                        session=session,
                        index=index,
                        variant_number=variant.index,
                        caption=caption_result.caption,
                        options=options,
                        image_bytes=variant.image_bytes,
                        meta_payload=dict(meta_template),
                        zero_pad=config.naming.zero_pad,
                        seed=seed,
                    )
                    saved.append(artifact.file_id)
                return saved

            try:
                logger.timed(
                    "SAVE",
                    lambda saved: f"i={index}/{config.run.count} -> {', '.join(saved)}",
                    _save,
                )
            except Exception as exc:
                logger.log("SAVE", f"i={index} failed: {exc}", level="ERROR")
                continue

            if config.scoring.enabled:
                logger.log("SCORE", "scoring pipeline not implemented", level="WARN")

        logger.log("DONE", f"session={session.session_id} completed")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
