#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imagen generator + Ollama captionifier + Dual scoring + Genetic evolution.

Modes:
- default batch: --cycles N --per-cycle M
- genetic evolve: --evolve --pop P --gens G [--keep 0.25 --mut 0.15 --xover 0.30]

DB:
- uses the same SQLite (scores.sqlite) as scorer.py and adds:
  runs(session_id TEXT PRIMARY KEY, started_ts INT, mode TEXT, cfg_json TEXT)
  prompts(path TEXT PRIMARY KEY, ts INT, gen INT, indiv INT, prompt TEXT,
          params TEXT, sfw REAL, temperature REAL,
          weights_style REAL, weights_nsfw REAL, fitness REAL,
          parents TEXT, op TEXT)

Notes:
- GA fitness = w_style*style_score + w_nsfw*nsfw_score  (0..100 scale)
- On GTX 1060 6GB keep per-cycle small (1) in GA to save VRAM time for scoring.

Dependencies: google-genai, requests, piexif (optional), scorer.py (DualScorer)
"""

import argparse
import hashlib
import logging
import json
import random
import requests
import sqlite3
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from scorer import DualScorer  # твой двухканальный скорер
from imagen_pipeline import BiasConfig, PromptAssets, WeightedSelector

try:
  import piexif
except ImportError:
  piexif = None

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "data" / "catalog"
DB_PATH = HERE / "scores.sqlite"


# ---------------- EXIF ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("imagen.pipeline")
def _xp_utf16le(t: str) -> bytes: return t.encode("utf-16le")


def write_exif_text(jpg: Path, text: str):
  if piexif is None: return
  exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
  try:
    exif = piexif.load(str(jpg))
  except Exception:
    pass
  exif.setdefault("0th", {})[piexif.ImageIFD.ImageDescription] = text.encode("utf-8", "ignore")
  exif["0th"][piexif.ImageIFD.XPComment] = _xp_utf16le(text)
  piexif.insert(piexif.dump(exif), str(jpg))


# ---------------- DB helpers ----------------
def db_init():
  conn = sqlite3.connect(DB_PATH)
  c = conn.cursor()
  c.execute("""CREATE TABLE IF NOT EXISTS runs
               (
                   session_id TEXT PRIMARY KEY,
                   started_ts INTEGER,
                   mode       TEXT,
                   cfg_json   TEXT
               )""")
  c.execute("""CREATE TABLE IF NOT EXISTS prompts
               (
                   path          TEXT PRIMARY KEY,
                   ts            INTEGER,
                   gen           INTEGER,
                   indiv         INTEGER,
                   prompt        TEXT,
                   params        TEXT,
                   sfw           REAL,
                   temperature   REAL,
                   weights_style REAL,
                   weights_nsfw  REAL,
                   fitness       REAL,
                   parents       TEXT,
                   op            TEXT
               )""")
  conn.commit();
  conn.close()


def db_log_run(session_id: str, mode: str, cfg: dict):
  conn = sqlite3.connect(DB_PATH);
  c = conn.cursor()
  c.execute("INSERT OR REPLACE INTO runs(session_id,started_ts,mode,cfg_json) VALUES(?,?,?,?)",
            (session_id, int(time.time()), mode, json.dumps(cfg, ensure_ascii=False)))
  conn.commit();
  conn.close()


def db_log_prompt(row: dict):
  conn = sqlite3.connect(DB_PATH);
  c = conn.cursor()
  c.execute("""INSERT OR REPLACE INTO prompts
        (path,ts,gen,indiv,prompt,params,sfw,temperature,weights_style,weights_nsfw,fitness,parents,op)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row["path"], row["ts"], row["gen"], row["indiv"], row["prompt"],
             json.dumps(row["params"], ensure_ascii=False),
             row["sfw"], row["temperature"], row["w_style"], row["w_nsfw"],
             row["fitness"], json.dumps(row.get("parents")), row.get("op", "")))
  conn.commit();
  conn.close()


# ---------------- utils ----------------
def clamp(v, lo, hi): return max(lo, min(hi, v))


def clamp01(x: float) -> float: return clamp(float(x), 0.0, 1.0)


def pick(arr, k=1): return random.sample(arr, k) if k > 1 else random.choice(arr)


def maybe(p=0.5): return random.random() < p


def enforce_bounds(text: str, mn: int, mx: int) -> str:
  ws = text.split()
  if len(ws) > mx: ws = ws[:mx]
  return " ".join(ws)


def ratio_from_camera(fr: dict) -> str: return fr.get("ratio", "3:4")


# ---------- SFW-biased randomization ----------
def short_readable(meta: dict) -> str:
  parts = [
    f"tpl={meta.get('template_id')}",
    f"palette={meta.get('palette', {}).get('id')}",
    f"light={meta.get('lighting', {}).get('id')}",
    f"bg={meta.get('background', {}).get('id')}",
    f"angle={meta.get('camera_angle', {}).get('id')}",
    f"frame={meta.get('camera_framing', {}).get('id')}",
    f"pose={meta.get('pose', {}).get('id')}",
    f"action={meta.get('action', {}).get('id')}",
    f"main={meta.get('wardrobe_main')}",
    f"extras={','.join(meta.get('wardrobe_extras', []))}",
  ]
  return " | ".join(filter(None, parts))


# ---------------- scene assembly ----------------
def build_struct(data: dict, selector: WeightedSelector, required_terms: list[str],
                 template_id: str | None = None):
  style = data.get("style_controller", {})
  rules = data.get("rules", {})
  bounds = rules.get("caption_length", {"min_words": 18, "max_words": 48})

  selection_trace: dict[str, Any] = {}

  def record(label: str, choice: dict[str, Any]) -> dict[str, Any]:
    meta = dict(choice.get("selection_meta", {}))
    meta.update({"id": choice.get("id"), "desc": choice.get("desc")})
    selection_trace[label] = meta
    return choice

  palette = record("palette", selector.pick_one("palettes", data["palettes"]))
  lighting = record("lighting", selector.pick_one("lighting_presets", data["lighting_presets"]))
  background = record("background", selector.pick_one("backgrounds", data["backgrounds"]))

  cam = data["camera"]
  cam_angle = record("camera_angle", selector.pick_one("camera:angles", cam["angles"]))
  cam_frame = record("camera_framing", selector.pick_one("camera:framing", cam["framing"]))
  cam_lens = record("camera_lens", selector.pick_one("camera:lenses", cam.get("lenses", [])))
  cam_depth = record("camera_depth", selector.pick_one("camera:depth", cam.get("depth", [])))

  mood = record("mood", selector.pick_one("moods", data["moods"]))
  pose = record("pose", selector.pick_one("poses", data["poses"]))
  action = record("action", selector.pick_one("actions", data["actions"]))
  model_choice = record("model", selector.pick_one("models", data["model_descriptions"]))

  wardrobe = data["wardrobe"]
  sets = data.get("wardrobe_sets", [])
  use_set = maybe(0.55) and bool(sets)
  chosen_set = None
  main_piece = None
  extras: list[str] = []

  def find_desc(iid: str) -> str:
    for g, items in wardrobe.items():
      for it in items:
        if it["id"] == iid:
          return it["desc"]
    for p in data.get("props", []):
      if p["id"] == iid:
        return p["desc"]
    return iid

  if use_set:
    chosen_set = record("wardrobe_set", selector.pick_one("wardrobe_sets", sets))
    ids = chosen_set.get("items", [])
    readable = [(iid, find_desc(iid)) for iid in ids]
    priority = [("dresses",), ("one_piece",), ("tops", "bottoms")]
    main = None
    for bucket in priority:
      for iid, desc in readable:
        for grp in bucket:
          if any(it["id"] == iid for it in wardrobe.get(grp, [])):
            main = (iid, desc)
            break
        if main:
          break
      if main:
        break
    if not main:
      main = readable[0] if readable else ("custom", "styled set")
    main_piece = main[1]
    rest = [d for iid, d in readable if d != main_piece]
    random.shuffle(rest)
    extras = rest[:random.choice([1, 2, 3])] if rest else []
    if "swimsuit" in main_piece.lower():
      extras = [e for e in extras if not any(x in e.lower() for x in ["stocking", "tights", "thigh-high"])]
  else:
    choose_dress = maybe(0.35) and wardrobe.get("dresses")
    choose_one = maybe(0.35) and wardrobe.get("one_piece") and not choose_dress
    if choose_dress:
      main_piece = record("wardrobe_main", selector.pick_one("wardrobe:dresses", wardrobe["dresses"]))["desc"]
    elif choose_one:
      main_piece = record("wardrobe_main", selector.pick_one("wardrobe:one_piece", wardrobe["one_piece"]))["desc"]
    else:
      top = record("wardrobe_top", selector.pick_one("wardrobe:tops", wardrobe["tops"]))["desc"]
      bot = record("wardrobe_bottom", selector.pick_one("wardrobe:bottoms", wardrobe["bottoms"]))["desc"]
      main_piece = f"{top} with {bot}"
    pools = []
    for key in ["footwear", "hosiery", "socks", "headwear", "jewelry", "belts", "scarves", "gloves", "outerwear"]:
      pools.extend(wardrobe.get(key, []))

    def allowed_extra(d: str) -> bool:
      if "swimsuit" in (main_piece or "").lower() and any(x in d.lower() for x in ["stocking", "tights", "thigh-high"]):
        return False
      if "stockings" in d.lower() and all(x not in (main_piece or "").lower() for x in ["skirt", "dress"]):
        return False
      return True

    pools = [it for it in pools if allowed_extra(it["desc"])]
    if pools:
      take = random.choice([1, 2, 3])
      chosen = selector.pick_many("wardrobe:extras", pools, take)
      extras = [it["desc"] for it in chosen]
      if chosen:
        selection_trace["wardrobe_extras"] = [dict(it.get("selection_meta", {}), id=it.get("id"), desc=it.get("desc")) for it in chosen]

  props = data.get("props", [])
  chosen_props: list[str] = []
  prop_meta: list[dict[str, Any]] = []
  if props and maybe(0.45):
    first = selector.pick_one("props", props)
    chosen_props = [first["desc"]]
    prop_meta.append(dict(first.get("selection_meta", {}), id=first.get("id"), desc=first.get("desc")))
    if props and maybe(0.2):
      p2_candidates = [p for p in props if p["id"] != first["id"]]
      if p2_candidates:
        second = selector.pick_one("props", p2_candidates)
        chosen_props.append(second["desc"])
        prop_meta.append(dict(second.get("selection_meta", {}), id=second.get("id"), desc=second.get("desc")))
  if prop_meta:
    selection_trace["props"] = prop_meta

  struct = {
    "template_id": template_id or random.choice(["caption_v1", "caption_v2", "caption_v3", "caption_v4"]),
    "palette": palette,
    "lighting": lighting,
    "background": background,
    "camera_angle": cam_angle,
    "camera_framing": cam_frame,
    "camera_lens": cam_lens,
    "camera_depth": cam_depth,
    "aspect_ratio": ratio_from_camera(cam_frame),
    "mood_words": mood.get("words", []),
    "pose": pose,
    "action": action,
    "model": model_choice["desc"],
    "wardrobe_main": main_piece,
    "wardrobe_extras": extras,
    "props": chosen_props,
    "caption_bounds": bounds,
    "selection_trace": selection_trace,
    "gene_ids": {
      "palette": palette.get("id"),
      "lighting": lighting.get("id"),
      "background": background.get("id"),
      "camera_angle": cam_angle.get("id"),
      "camera_framing": cam_frame.get("id"),
      "lens": cam_lens.get("id"),
      "depth": cam_depth.get("id"),
      "mood": mood.get("id", ""),
      "pose": pose.get("id", ""),
      "action": action.get("id", "")
    }
  }

  style = data.get("style_controller", {})
  struct["ollama_payload"] = {
    "mood": ", ".join(random.sample(mood.get("words", []), k=min(len(mood.get("words", [])), random.choice([1, 2])))) if mood.get("words") else "",
    "model": model_choice["desc"],
    "pose": pose["desc"],
    "action": action["desc"],
    "wardrobe": main_piece + (("; extras: " + ", ".join(extras)) if extras else "") + (
      ("; props: " + ", ".join(chosen_props)) if chosen_props else ""),
    "camera": f"{cam_angle['desc']}; {struct['aspect_ratio']}; lens={cam_lens['desc']}; depth={cam_depth['desc']}",
    "lighting": lighting["desc"],
    "background": background["desc"],
    "style": {
      "aesthetic": style.get("aesthetic"),
      "paper_texture": bool(style.get("paper_texture")),
      "watercolor": style.get("watercolor"),
      "highlights": style.get("highlights"),
      "subsurface_glow": style.get("subsurface_glow"),
      "background_norm": style.get("background_norm"),
      "palette_preference": style.get("palette_preference"),
    },
    "required_terms": required_terms,
  }
  return struct


# ---------------- Ollama ----------------
def system_prompt_for(sfw_level: float, required_terms: list[str]) -> str:
  sfw_level = clamp(sfw_level, 0.0, 1.0)
  tone_desc = (
    "wholesome and innocent" if sfw_level < 0.25 else
    "flirty but tasteful" if sfw_level < 0.7 else
    "bold adult tone"
  )
  required = ", ".join(required_terms) if required_terms else "illustration, watercolor, glossy, paper, pastel"
  return f"""You are an imaginative art director following the Generative Art Prompt Bible. Craft a single cinematic caption (22–58 words) for a retro pin-up watercolor illustration with an ultra-glossy jelly aesthetic.

Obey the Bible's structure: lead with the protagonist, clarify pose and action, set the environment, then layer materials, lighting, camera notes, and mood. Blend sensory adjectives with storytelling verbs. Keep the tone calibrated to SFW level {sfw_level:.2f} ({tone_desc})—suggestive but never explicit.

Weave these mandatory terms naturally: {required}. Maintain fluent sentences (no bullet lists), 1–2 sentences maximum. Ensure cohesion, vivid imagery, and professional polish."""


def ollama_generate(url: str, model: str, system_prompt: str, payload: dict,
                    temperature: float = 0.55, top_p: float = 0.9, seed: int | None = None, timeout: int = 30) -> str:
  prompt = system_prompt.strip() + "\n\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  opts = {"temperature": float(temperature), "top_p": float(top_p), "repeat_penalty": 1.05}
  if seed is not None: opts["seed"] = int(seed)
  r = requests.post(f"{url.rstrip('/')}/api/generate", json={
    "model": model, "prompt": prompt, "options": opts, "stream": False, "raw": False
  }, timeout=timeout)
  r.raise_for_status()
  text = r.json().get("response", "").strip()
  text = " ".join(text.split()).replace("..", ".")
  return text


def needs_enforcement(txt: str, required_terms: list[str]) -> list[str]:
  return [w for w in required_terms if w.lower() not in txt.lower()]


def enforce_once(url, model, system_prompt, payload, base_caption, required_terms,
                 temperature=0.5, seed=None):
  missing = needs_enforcement(base_caption, required_terms)
  if not missing: return base_caption
  enforce_sys = system_prompt + (
      "\n\nRewrite the caption naturally (18–60 words) and include the missing words: "
      + ", ".join(missing) + ". Keep it one or two sentences."
  )
  return ollama_generate(url, model, enforce_sys, payload, temperature=temperature, seed=seed)


# ---------------- Imagen ----------------
def imagen_call(client, model_name: str, prompt: str, aspect_ratio: str, variants: int, person_mode: str):
  cfg = types.GenerateImagesConfig(
    number_of_images=variants,
    aspect_ratio=aspect_ratio,
    person_generation=person_mode,
    safety_filter_level="block_low_and_above",
    output_mime_type='image/jpeg',
    guidance_scale=0.5,
  )
  return client.models.generate_images(model=model_name, prompt=prompt, config=cfg)


def load_best_gene_sets(k: int, session_id: str | None = None) -> list[dict]:
  """
  Берёт top-K по fitness из таблицы prompts, парсит params.struct.gene_ids.
  Возвращает список словарей генов (уникальных).
  """
  conn = sqlite3.connect(DB_PATH)
  c = conn.cursor()
  if session_id:
    c.execute(
      "SELECT path, params, fitness FROM prompts WHERE fitness IS NOT NULL AND params LIKE '%\"struct\"%' AND params LIKE ? ORDER BY fitness DESC LIMIT ?",
      (f'%{session_id}%', k * 3))
  else:
    c.execute(
      "SELECT path, params, fitness FROM prompts WHERE fitness IS NOT NULL AND params LIKE '%\"struct\"%' ORDER BY fitness DESC LIMIT ?",
      (k * 3,))
  rows = c.fetchall()
  conn.close()

  best = []
  seen_keys = set()
  for path, params_json, fit in rows:
    try:
      params = json.loads(params_json)
      genes = params.get("struct", {}).get("gene_ids")
      if not isinstance(genes, dict):
        continue
      # сделаем маленький ключ для дедупликации
      key = json.dumps(genes, sort_keys=True)
      if key in seen_keys:
        continue
      seen_keys.add(key)
      best.append(genes)
      if len(best) >= k:
        break
    except Exception:
      continue
  return best


# ---------------- core save + score ----------------
def save_and_score(resp, out: Path, meta_base: dict, final_prompt: str, s_struct: dict,
                   dual: DualScorer, per_cycle: int, session_id: str,
                   gen: int | None, indiv: int | None, w_style: float, w_nsfw: float):
  saved_paths = []
  for k, gi in enumerate(resp.generated_images, 1):
    img = gi.image
    img_bytes = getattr(img, "image_bytes", None) or getattr(img, "imageBytes", None)
    if not img_bytes:
      logger.warning("[%s] empty bytes for variant %s", meta_base['id'], k)
      continue

    img_path = out / f"{meta_base['id']}_{k}.jpg"
    json_sidecar = img_path.with_suffix(".json")
    txt_sidecar = img_path.with_suffix(".txt")

    try:
      with open(img_path, "wb") as f:
        f.write(img_bytes)
      exif_line = f"{short_readable({'template_id': s_struct['template_id'], 'palette': s_struct['palette'], 'lighting': s_struct['lighting'], 'background': s_struct['background'], 'camera_angle': s_struct['camera_angle'], 'camera_framing': s_struct['camera_framing'], 'pose': s_struct['pose'], 'action': s_struct['action'], 'wardrobe_main': s_struct['wardrobe_main'], 'wardrobe_extras': s_struct['wardrobe_extras']})} | prompt={final_prompt}"
      write_exif_text(img_path, exif_line)

      meta = dict(meta_base);
      meta["id"] = f"{meta_base['id']}_{k}"
      meta["final_prompt"] = final_prompt
      meta["parameters"] = s_struct

      with open(json_sidecar, "w", encoding="utf-8") as sf:
        json.dump(meta, sf, ensure_ascii=False, indent=2)
      with open(txt_sidecar, "w", encoding="utf-8") as tf:
        tf.write(final_prompt + "\n")

      saved_paths.append(img_path)
    except Exception as e:
      logger.error("[%s] save error: %s", meta_base['id'], e)

  # scoring
  triplets = dual.score_and_save(saved_paths, notes="imagen jelly pin-up")
  # write scores & fitness; also log prompts to DB
  for p, nsfw100, style100 in triplets:
    sid = p.with_suffix(".json")
    try:
      side = json.loads(sid.read_text(encoding="utf-8"))
      side["nsfw_score"] = nsfw100
      side["style_score"] = style100
      fitness = round(w_style * style100 + w_nsfw * nsfw100, 2)
      side["fitness"] = fitness
      sid.write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")

      db_log_prompt({
        "path": str(p),
        "ts": int(time.time()),
        "gen": int(gen) if gen is not None else -1,
        "indiv": int(indiv) if indiv is not None else -1,
        "prompt": final_prompt,
        "params": {
          "session": session_id,
          "meta": meta_base,
          "struct": s_struct
        },
        "sfw": meta_base["ollama"]["sfw_level"],
        "temperature": meta_base["ollama"]["temperature"],
        "w_style": w_style,
        "w_nsfw": w_nsfw,
        "fitness": fitness,
        "parents": meta_base.get("ga_parents"),
        "op": meta_base.get("ga_op", "plain")
      })
    except Exception as e:
      logger.error("[score] update/log failed for %s: %s", sid.name, e)

  triplets.sort(key=lambda x: w_style * x[2] + w_nsfw * x[1], reverse=True)
  if triplets:
    logger.info("[score] best=%s style=%.2f nsfw=%.2f",
                triplets[0][0].name, triplets[0][2], triplets[0][1])
  return triplets


# ---------------- batch mode ----------------
def run_plain(outdir="output", model_imagen="imagen-3.0-generate-002", person_mode="allow_adult",
              per_cycle=2, cycles=10, sleep_s=1.0, seed=None,
              ollama_url="http://localhost:11434", ollama_model="qwen2.5:3b",
              sfw_level=0.6, temperature=0.55, w_style=0.7, w_nsfw=0.3,
              bias_config: BiasConfig | None = None, assets: PromptAssets | None = None):
  load_dotenv()
  db_init()
  client = genai.Client()

  bias = bias_config or BiasConfig()
  assets = assets or PromptAssets(ASSETS_DIR)
  data = assets.as_dict()
  required_terms = assets.required_terms()
  selector = WeightedSelector(bias, sfw_level, temperature)

  session_id = f"plain-{int(time.time())}"
  run_cfg = {
    "cycles": cycles,
    "per_cycle": per_cycle,
    "sfw": sfw_level,
    "temperature": temperature,
    "weights": {"style": w_style, "nsfw": w_nsfw},
    "bias": bias.to_dict(),
  }
  db_log_run(session_id, "plain", run_cfg)
  logger.info("[session %s] starting plain run with cfg=%s", session_id, run_cfg)

  dual = DualScorer(device="auto", batch=4,
                    db_path=DB_PATH,
                    jsonl_path=HERE / "scores.jsonl")

  if seed is not None:
    random.seed(seed)
    logger.info("Seeded python RNG with %s", seed)

  out = Path(outdir)
  out.mkdir(parents=True, exist_ok=True)
  logger.info("Output directory: %s", out.resolve())

  sys_prompt = system_prompt_for(sfw_level, required_terms)
  sys_hash = hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()[:12]
  logger.debug("System prompt hash: %s", sys_hash)

  for i in range(1, cycles + 1):
    s = build_struct(data, selector, required_terms)
    aspect = s["aspect_ratio"]
    cycle_id = f"rnd-{i:03d}"
    logger.info("[%s] selection: %s", cycle_id, short_readable(s))
    logger.debug("[%s] selection trace: %s", cycle_id, s.get("selection_trace", {}))

    try:
      cap = ollama_generate(ollama_url, ollama_model, sys_prompt, s["ollama_payload"],
                            temperature=temperature, seed=seed)
      cap = enforce_once(ollama_url, ollama_model, sys_prompt, s["ollama_payload"], cap,
                         required_terms, temperature=max(0.45, temperature - 0.05), seed=seed)
    except Exception as e:
      logger.error("[%s] Ollama error: %s", cycle_id, e)
      time.sleep(sleep_s)
      continue

    bounds = s["caption_bounds"]
    final_prompt = enforce_bounds(cap, bounds.get("min_words", 18), bounds.get("max_words", 60))
    logger.info("[%s] caption: %s", cycle_id, final_prompt)

    try:
      resp = imagen_call(client, model_imagen, final_prompt, aspect, per_cycle, person_mode)
    except Exception as e:
      logger.error("[%s] Imagen error: %s", cycle_id, e)
      time.sleep(sleep_s)
      continue

    meta_base = {
      "id": cycle_id,
      "model_imagen": model_imagen,
      "person_mode": person_mode,
      "variants": per_cycle,
      "seed": seed,
      "ollama": {
        "url": ollama_url,
        "model": ollama_model,
        "temperature": temperature,
        "system_hash": sys_hash,
        "sfw_level": sfw_level,
        "style_mode": "inline+required_terms"
      },
      "selector": {
        "temperature": temperature,
        "sfw_level": sfw_level,
        "bias": bias.to_dict(),
      }
    }

    trips = save_and_score(resp, out, meta_base, final_prompt, s, dual, per_cycle, session_id,
                           gen=None, indiv=None, w_style=w_style, w_nsfw=w_nsfw)
    logger.info("[%s] saved %d variants", cycle_id, len(trips))
    time.sleep(sleep_s)


# ---------------- genetic helpers ----------------
def mutate_gene(data, selector: WeightedSelector, key: str, current_id: str | None):
  if key == "palette":
    seq = data["palettes"]
    group = "palettes"
  elif key == "lighting":
    seq = data["lighting_presets"]
    group = "lighting_presets"
  elif key == "background":
    seq = data["backgrounds"]
    group = "backgrounds"
  elif key == "camera_angle":
    seq = data["camera"]["angles"]
    group = "camera:angles"
  elif key == "camera_framing":
    seq = data["camera"]["framing"]
    group = "camera:framing"
  elif key == "lens":
    seq = data["camera"]["lenses"]
    group = "camera:lenses"
  elif key == "depth":
    seq = data["camera"]["depth"]
    group = "camera:depth"
  elif key == "mood":
    seq = data["moods"]
    group = "moods"
  elif key == "pose":
    seq = data["poses"]
    group = "poses"
  elif key == "action":
    seq = data["actions"]
    group = "actions"
  else:
    return current_id
  candidates = [it for it in seq if it.get("id") != current_id] or list(seq)
  choice = selector.pick_one(group, candidates)
  return choice.get("id")


def crossover_genes(a: dict, b: dict):
  child = {}
  for k in a.keys():
    child[k] = a[k] if maybe(0.5) else b.get(k, a[k])
  return child


def genes_to_struct(data, selector: WeightedSelector, genes: dict, required_terms: list[str]):
  """Ресинтез сцены из набора id-генов (ближайшие описания подтягиваем из словарей)."""
  # простой путь: сэмплим заново через build_struct с приоритетом выбранных id
  s = build_struct(data, selector, required_terms)

  # заменить id там, где можем
  def pick_by_id(seq, iid):
    for it in seq:
      if it.get("id") == iid: return it
    return None

  if genes.get("palette"):    s["palette"] = pick_by_id(data["palettes"], genes["palette"]) or s["palette"]
  if genes.get("lighting"):   s["lighting"] = pick_by_id(data["lighting_presets"], genes["lighting"]) or s["lighting"]
  if genes.get("background"): s["background"] = pick_by_id(data["backgrounds"], genes["background"]) or s["background"]
  if genes.get("camera_angle"):   s["camera_angle"] = pick_by_id(data["camera"]["angles"], genes["camera_angle"]) or s[
    "camera_angle"]
  if genes.get("camera_framing"): s["camera_framing"] = pick_by_id(data["camera"]["framing"],
                                                                   genes["camera_framing"]) or s["camera_framing"]
  if genes.get("lens"):       s["camera_lens"] = pick_by_id(data["camera"]["lenses"], genes["lens"]) or s["camera_lens"]
  if genes.get("depth"):      s["camera_depth"] = pick_by_id(data["camera"]["depth"], genes["depth"]) or s[
    "camera_depth"]
  if genes.get("mood"):
    # заменить mood_words
    m = pick_by_id(data["moods"], genes["mood"])
    if m: s["mood_words"] = m["words"]
  if genes.get("pose"):   s["pose"] = pick_by_id(data["poses"], genes["pose"]) or s["pose"]
  if genes.get("action"): s["action"] = pick_by_id(data["actions"], genes["action"]) or s["action"]
  s["gene_ids"] = genes
  return s


# ---------------- evolve mode ----------------
def run_evolve(outdir="output", model_imagen="imagen-3.0-generate-002", person_mode="allow_adult",
               pop=16, gens=4, keep=0.25, mut=0.15, xover=0.30,
               sleep_s=1.0, seed=None,
               ollama_url="http://localhost:11434", ollama_model="qwen2.5:3b",
               sfw_level=0.6, temperature=0.55, w_style=0.7, w_nsfw=0.3,
               resume_best: bool = False, resume_k: int = 0, resume_session: str | None = None,
               resume_mix: float = 0.10,
               bias_config: BiasConfig | None = None, assets: PromptAssets | None = None):
  load_dotenv()
  db_init()
  client = genai.Client()

  bias = bias_config or BiasConfig()
  assets = assets or PromptAssets(ASSETS_DIR)
  data = assets.as_dict()
  required_terms = assets.required_terms()
  selector = WeightedSelector(bias, sfw_level, temperature)

  session_id = f"evolve-{int(time.time())}"
  run_cfg = {
    "pop": pop,
    "gens": gens,
    "keep": keep,
    "mut": mut,
    "xover": xover,
    "sfw": sfw_level,
    "temperature": temperature,
    "weights": {"style": w_style, "nsfw": w_nsfw},
    "bias": bias.to_dict(),
  }
  db_log_run(session_id, "evolve", run_cfg)
  logger.info("[session %s] starting evolve run with cfg=%s", session_id, run_cfg)

  dual = DualScorer(device="auto", batch=4,
                    db_path=DB_PATH,
                    jsonl_path=HERE / "scores.jsonl")

  if seed is not None:
    random.seed(seed)
    logger.info("Seeded python RNG with %s", seed)

  out = Path(outdir)
  out.mkdir(parents=True, exist_ok=True)
  logger.info("Output directory: %s", out.resolve())

  sys_prompt = system_prompt_for(sfw_level, required_terms)
  sys_hash = hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()[:12]

  population: list[dict[str, str]] = []
  if resume_best:
    seed_genes = load_best_gene_sets(resume_k or pop, resume_session)
    if seed_genes:
      logger.info("[resume] seeded %s gene sets from DB (session=%s)", len(seed_genes), resume_session or "ANY")
      for gset in seed_genes:
        child = dict(gset)
        if resume_mix > 0:
          for kname in list(child.keys()):
            if random.random() < resume_mix:
              child[kname] = mutate_gene(data, selector, kname, child.get(kname))
        population.append(child)
    while len(population) < pop:
      s = build_struct(data, selector, required_terms)
      population.append(s["gene_ids"])
  else:
    for _ in range(pop):
      s = build_struct(data, selector, required_terms)
      population.append(s["gene_ids"])

  indiv_counter = 0
  for g in range(1, gens + 1):
    logger.info("===== Generation %s/%s =====", g, gens)
    scored: list[tuple[float, dict, Path, float, float]] = []

    for idx, genes in enumerate(population, start=1):
      indiv_counter += 1
      s = genes_to_struct(data, selector, genes, required_terms)
      aspect = s["aspect_ratio"]
      indiv_id = f"G{g:02d}-I{idx:02d}"
      logger.info("[%s] genes=%s", indiv_id, genes)

      try:
        cap = ollama_generate(ollama_url, ollama_model, sys_prompt, s["ollama_payload"],
                              temperature=temperature, seed=seed)
        cap = enforce_once(ollama_url, ollama_model, sys_prompt, s["ollama_payload"], cap,
                           required_terms, temperature=max(0.45, temperature - 0.05), seed=seed)
      except Exception as e:
        logger.error("[%s] Ollama error: %s", indiv_id, e)
        time.sleep(sleep_s)
        continue

      bounds = s["caption_bounds"]
      final_prompt = enforce_bounds(cap, bounds.get("min_words", 18), bounds.get("max_words", 60))

      try:
        resp = imagen_call(client, model_imagen, final_prompt, aspect, 1, person_mode)
      except Exception as e:
        logger.error("[%s] Imagen error: %s", indiv_id, e)
        time.sleep(sleep_s)
        continue

      if not getattr(resp, "generated_images", None):
        logger.warning("[%s] no image returned", indiv_id)
        time.sleep(sleep_s)
        continue

      meta_base = {
        "id": indiv_id,
        "model_imagen": model_imagen,
        "person_mode": person_mode,
        "variants": 1,
        "seed": seed,
        "ollama": {
          "url": ollama_url,
          "model": ollama_model,
          "temperature": temperature,
          "system_hash": sys_hash,
          "sfw_level": sfw_level,
          "style_mode": "inline"
        },
        "selector": {
          "temperature": temperature,
          "sfw_level": sfw_level,
          "bias": bias.to_dict(),
        }
      }
      trips = save_and_score(resp, out, meta_base, final_prompt, s, dual, 1, session_id,
                             gen=g, indiv=idx, w_style=w_style, w_nsfw=w_nsfw)
      time.sleep(sleep_s)

      if trips:
        p, nsfw100, style100 = trips[0]
        fitness = w_style * style100 + w_nsfw * nsfw100
        scored.append((fitness, genes, p, style100, nsfw100))
        logger.info("[%s] fitness=%.2f style=%.2f nsfw=%.2f", indiv_id, fitness, style100, nsfw100)

    if not scored:
      logger.warning("[evolve] no scored individuals; stopping.")
      break

    scored.sort(key=lambda t: t[0], reverse=True)
    elite_n = max(1, int(round(keep * len(scored))))
    elites = scored[:elite_n]
    logger.info("[evolve] elite=%s best_fitness=%.2f style=%.2f nsfw=%.2f",
                elite_n, elites[0][0], elites[0][3], elites[0][4])

    new_pop = [genes for _, genes, _, _, _ in elites]
    while len(new_pop) < pop:
      op = "clone"
      if random.random() < xover and len(elites) >= 2:
        pa = random.choice(elites)[1]
        pb = random.choice(elites)[1]
        child = crossover_genes(pa, pb)
        op = "xover"
      else:
        child = dict(random.choice(elites)[1])

      for k in list(child.keys()):
        if random.random() < mut:
          child[k] = mutate_gene(data, selector, k, child.get(k))
          op = (op + "+mut") if op else "mut"

      new_pop.append(child)
      logger.debug("[evolve] offspring op=%s genes=%s", op, child)
    population = new_pop[:pop]

  logger.info("[evolve] done.")


# ---------------- CLI ----------------
if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Imagen + Ollama + Dual scoring + Genetic evolution.")
  ap.add_argument("--outdir", default="output")
  ap.add_argument("--model", default="imagen-3.0-generate-002")
  ap.add_argument("--person-mode", default="allow_adult", choices=["dont_allow", "allow_adult", "allow_all"])
  ap.add_argument("--per-cycle", type=int, default=2)
  ap.add_argument("--cycles", type=int, default=10)
  ap.add_argument("--sleep", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--ollama-url", default="http://localhost:11434")
  ap.add_argument("--ollama-model", default="qwen2.5:3b")
  ap.add_argument("--sfw", type=float, default=0.6)
  ap.add_argument("--temperature", type=float, default=0.55)
  ap.add_argument("--w-style", type=float, default=0.7)
  ap.add_argument("--w-nsfw", type=float, default=0.3)
  ap.add_argument("--config", default=None, help="Path to bias configuration JSON")
  ap.add_argument("--weight", action="append", default=[],
                  help="Override preference e.g. pose.sexuality=0.8 or wardrobe.era.1980s=1.0")

  # evolve flags
  ap.add_argument("--evolve", action="store_true")
  ap.add_argument("--pop", type=int, default=16)
  ap.add_argument("--gens", type=int, default=4)
  ap.add_argument("--keep", type=float, default=0.25)
  ap.add_argument("--mut", type=float, default=0.15)
  ap.add_argument("--xover", type=float, default=0.30)
  ap.add_argument("--resume-best", action="store_true", help="Seed initial population from top prompts in SQLite")
  ap.add_argument("--resume-k", type=int, default=0, help="How many top gene sets to pull from DB (default min(pop,12))")
  ap.add_argument("--resume-session", default=None, help="Restrict seeding to a specific past session_id")
  ap.add_argument("--resume-mix", type=float, default=0.10, help="Light mutation rate applied to resumed genes [0..1]")

  args = ap.parse_args()

  default_config = HERE / "config" / "default_bias.json"
  config_path = Path(args.config) if args.config else (default_config if default_config.exists() else None)
  try:
    bias_config = BiasConfig.from_file(config_path)
  except FileNotFoundError:
    logger.warning("Bias configuration not found at %s, using defaults", config_path)
    bias_config = BiasConfig()
  bias_config.apply_overrides(args.weight)

  try:
    assets = PromptAssets(ASSETS_DIR)
  except FileNotFoundError as exc:
    raise SystemExit(f"Failed to load prompt assets: {exc}")

  if args.evolve:
    run_evolve(outdir=args.outdir, model_imagen=args.model, person_mode=args.person_mode,
               pop=args.pop, gens=args.gens, keep=args.keep, mut=args.mut, xover=args.xover,
               sleep_s=args.sleep, seed=args.seed,
               ollama_url=args.ollama_url, ollama_model=args.ollama_model,
               sfw_level=clamp(args.sfw, 0.0, 1.0), temperature=args.temperature,
               w_style=args.w_style, w_nsfw=args.w_nsfw,
               resume_best=args.resume_best, resume_k=args.resume_k,
               resume_session=args.resume_session, resume_mix=args.resume_mix,
               bias_config=bias_config, assets=assets)

  else:
    run_plain(outdir=args.outdir, model_imagen=args.model, person_mode=args.person_mode,
              per_cycle=args.per_cycle, cycles=args.cycles, sleep_s=args.sleep, seed=args.seed,
              ollama_url=args.ollama_url, ollama_model=args.ollama_model,
              sfw_level=clamp(args.sfw, 0.0, 1.0), temperature=args.temperature,
              w_style=args.w_style, w_nsfw=args.w_nsfw,
              bias_config=bias_config, assets=assets)
