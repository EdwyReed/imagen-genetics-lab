from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .catalog import Catalog
from .characters import CharacterLibrary, CharacterProfile
from .randomization import WeightedSelector, maybe, pick_from_ids, pick_one
from .learning import StyleFeedback


@dataclass
class PromptPayload:
    mood: str
    model: str
    pose: str
    action: str
    wardrobe: str
    camera: str
    lighting: str
    background: str
    style: Dict[str, Any]
    required_terms: List[str]
    style_profile: Dict[str, Any]
    scene_summary: str = ""
    feedback_notes: List[str] = field(default_factory=list)
    character: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "mood": self.mood,
            "model": self.model,
            "pose": self.pose,
            "action": self.action,
            "wardrobe": self.wardrobe,
            "camera": self.camera,
            "lighting": self.lighting,
            "background": self.background,
            "style": self.style,
            "required_terms": list(self.required_terms),
            "style_profile": dict(self.style_profile),
            "scene_summary": self.scene_summary,
            "feedback_notes": list(self.feedback_notes),
        }
        if self.character:
            data["character"] = dict(self.character)
        return data


@dataclass
class SceneStruct:
    template_id: str
    palette: Dict[str, Any]
    lighting: Dict[str, Any]
    background: Dict[str, Any]
    camera_angle: Dict[str, Any]
    camera_framing: Dict[str, Any]
    camera_lens: Dict[str, Any]
    camera_depth: Dict[str, Any]
    aspect_ratio: str
    mood_words: List[str]
    pose: Dict[str, Any]
    action: Dict[str, Any]
    model: str
    wardrobe_main: str
    wardrobe_extras: List[str]
    props: List[str]
    caption_bounds: Dict[str, Any]
    gene_ids: Dict[str, Optional[str]]
    payload: PromptPayload
    character: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "template_id": self.template_id,
            "palette": self.palette,
            "lighting": self.lighting,
            "background": self.background,
            "camera_angle": self.camera_angle,
            "camera_framing": self.camera_framing,
            "camera_lens": self.camera_lens,
            "camera_depth": self.camera_depth,
            "aspect_ratio": self.aspect_ratio,
            "mood_words": self.mood_words,
            "pose": self.pose,
            "action": self.action,
            "model": self.model,
            "wardrobe_main": self.wardrobe_main,
            "wardrobe_extras": self.wardrobe_extras,
            "props": self.props,
            "caption_bounds": self.caption_bounds,
            "gene_ids": self.gene_ids,
            "ollama_payload": self.payload.to_dict(),
        }
        if self.character:
            data["character"] = dict(self.character)
        return data

    def ollama_payload(self) -> Dict[str, Any]:
        return self.payload.to_dict()


class SceneBuilder:
    def __init__(
        self,
        catalog: Catalog,
        required_terms: List[str],
        template_ids: List[str],
        *,
        character_library: CharacterLibrary | None = None,
        default_character: str | None = None,
        variant_character_defaults: Dict[str, str] | None = None,
    ):
        self.catalog = catalog
        self.required_terms = required_terms
        self.template_ids = template_ids
        self.character_library = character_library
        self.default_character = default_character
        self.variant_character_defaults = variant_character_defaults or {}

    def _pick_template(self, template_id: Optional[str], feedback: Optional[StyleFeedback]) -> str:
        if template_id:
            return template_id
        if not self.template_ids:
            raise ValueError("No caption templates configured")
        if feedback is None:
            return random.choice(self.template_ids)
        base_weights = [1.0] * len(self.template_ids)
        biased = feedback.apply_bias("template", self.template_ids, base_weights)
        total = sum(biased)
        if total <= 0:
            biased = base_weights
        return random.choices(self.template_ids, weights=biased, k=1)[0]

    def _pick_with_feedback(
        self,
        selector: WeightedSelector,
        seq: List[dict],
        slot: str,
        feedback: Optional[StyleFeedback],
    ):
        if not seq:
            raise ValueError(f"Sequence for slot '{slot}' is empty")
        weights = selector.weights(seq)
        if feedback is not None:
            weights = feedback.apply_bias(slot, seq, weights)
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(seq)
        return random.choices(seq, weights=weights, k=1)[0]

    def _sample_with_feedback(
        self,
        selector: WeightedSelector,
        seq: List[dict],
        slot: str,
        k: int,
        feedback: Optional[StyleFeedback],
    ) -> List[dict]:
        k = min(int(k), len(seq))
        if k <= 0:
            return []
        pool = list(seq)
        out: List[dict] = []
        for _ in range(k):
            if not pool:
                break
            weights = selector.weights(pool)
            if feedback is not None:
                weights = feedback.apply_bias(slot, pool, weights)
            total = sum(weights)
            if total <= 0:
                weights = [1.0] * len(pool)
            choice = random.choices(pool, weights=weights, k=1)[0]
            out.append(choice)
            pool.remove(choice)
        return out

    # ------------------------------------------------------------------
    # character helpers
    # ------------------------------------------------------------------
    def _resolve_variant(self, *items: dict) -> Optional[str]:
        for item in items:
            if isinstance(item, dict):
                variant = item.get("variant")
                if isinstance(variant, str) and variant.strip():
                    return variant.strip()
        brand = self.catalog.raw.get("brand") if isinstance(self.catalog.raw, Mapping) else None
        if isinstance(brand, str) and brand.strip():
            return brand.strip()
        return None

    def _variant_default_character(self, variant: Optional[str]) -> Optional[str]:
        if variant and variant in self.variant_character_defaults:
            return self.variant_character_defaults[variant]
        return self.default_character

    def _select_character(
        self,
        variant: Optional[str],
    ) -> Optional[CharacterProfile]:
        if self.character_library is None:
            return None
        default_id = self._variant_default_character(variant)
        try:
            return self.character_library.choose(variant, default_id=default_id)
        except ValueError:
            return self.character_library.find(default_id) or None

    def _fallback_model_description(self) -> str:
        entries = self.catalog.section("model_descriptions")
        if not entries:
            return "featured muse"
        choice = pick_one(entries)
        if isinstance(choice, dict):
            for key in ("description", "desc", "name"):
                value = choice.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return str(choice)
        return str(choice)

    def build_scene(
        self,
        sfw_level: float,
        temperature: float,
        template_id: Optional[str] = None,
        *,
        feedback: Optional[StyleFeedback] = None,
    ) -> SceneStruct:
        selector = WeightedSelector(sfw_level=sfw_level, temperature=temperature)
        palettes = self.catalog.section("palettes")
        lighting_presets = self.catalog.section("lighting_presets")
        backgrounds = self.catalog.section("backgrounds")

        palette = self._pick_with_feedback(selector, palettes, "palette", feedback)
        lighting = self._pick_with_feedback(selector, lighting_presets, "lighting", feedback)
        background = self._pick_with_feedback(selector, backgrounds, "background", feedback)

        camera = self.catalog.raw.get("camera", {})
        camera_angles = camera.get("angles", [])
        camera_framing = camera.get("framing", [])
        camera_lenses = camera.get("lenses", []) or [
            {
                "id": "standard_50",
                "desc": "50mm equivalent; natural proportion",
                "nsfw": 0.35,
            }
        ]
        camera_depths = camera.get("depth", []) or [
            {
                "id": "soft_f2_8",
                "desc": "soft background; sparkling speculars",
                "nsfw": 0.5,
            }
        ]

        cam_angle = self._pick_with_feedback(selector, camera_angles, "camera_angle", feedback)
        cam_frame = self._pick_with_feedback(selector, camera_framing, "camera_framing", feedback)
        cam_lens = self._pick_with_feedback(selector, camera_lenses, "lens", feedback)
        cam_depth = self._pick_with_feedback(selector, camera_depths, "depth", feedback)

        moods = self.catalog.section("moods")
        poses = self.catalog.section("poses")
        actions = self.catalog.section("actions")

        mood = self._pick_with_feedback(selector, moods, "mood", feedback)
        pose = self._pick_with_feedback(selector, poses, "pose", feedback)
        action = self._pick_with_feedback(selector, actions, "action", feedback)

        variant_id = self._resolve_variant(palette, lighting, background, mood, pose, action)
        character_profile = self._select_character(variant_id)
        character_info: Optional[Dict[str, Any]] = None
        if character_profile is not None:
            character_info = character_profile.to_dict()
            if variant_id and "variant" not in character_info:
                character_info["variant"] = variant_id
        model_desc = (
            character_profile.summary
            if character_profile is not None and character_profile.summary
            else self._fallback_model_description()
        )
        model_desc = str(model_desc).strip()

        wardrobe_groups = self.catalog.wardrobe_groups()
        wardrobe_sets = self.catalog.wardrobe_sets()

        main_piece: Optional[str] = None
        extras: List[str] = []
        chosen_set = None

        def readable_desc(item_id: str) -> str:
            desc = self.catalog.find_description(item_id)
            return desc or item_id

        if wardrobe_sets and maybe(0.55):
            chosen_set = self._pick_with_feedback(selector, wardrobe_sets, "wardrobe_set", feedback)
            ids = chosen_set.get("items", [])
            readable = [(iid, readable_desc(iid)) for iid in ids]
            priority = [("dresses",), ("one_piece",), ("tops", "bottoms")]
            for bucket in priority:
                main = None
                for iid, desc in readable:
                    for grp in bucket:
                        items = wardrobe_groups.get(grp, [])
                        if any(it.get("id") == iid for it in items):
                            main = desc
                            break
                    if main:
                        break
                if main:
                    main_piece = main
                    break
            if not main_piece and readable:
                main_piece = readable[0][1]
            remaining = [desc for _, desc in readable if desc != main_piece]
            random.shuffle(remaining)
            extras = remaining[: random.choice([1, 2, 3])] if remaining else []
            if main_piece and "swimsuit" in main_piece.lower():
                extras = [
                    e
                    for e in extras
                    if not any(token in e.lower() for token in ["stocking", "tights", "thigh-high"])
                ]
        else:
            choose_dress = maybe(0.35) and wardrobe_groups.get("dresses")
            choose_one_piece = maybe(0.35) and wardrobe_groups.get("one_piece") and not choose_dress
            selector_groups = selector
            if choose_dress:
                main_piece = selector_groups.pick(wardrobe_groups.get("dresses", []))["desc"]
            elif choose_one_piece:
                main_piece = selector_groups.pick(wardrobe_groups.get("one_piece", []))["desc"]
            else:
                top = selector_groups.pick(wardrobe_groups.get("tops", []))["desc"]
                bottom = selector_groups.pick(wardrobe_groups.get("bottoms", []))["desc"]
                main_piece = f"{top} with {bottom}"

            pools: List[dict] = []
            for key in [
                "footwear",
                "hosiery",
                "socks",
                "headwear",
                "jewelry",
                "belts",
                "scarves",
                "gloves",
                "outerwear",
            ]:
                pools.extend(wardrobe_groups.get(key, []))

            def allowed_extra(description: str) -> bool:
                if main_piece and "swimsuit" in main_piece.lower() and any(
                    token in description.lower() for token in ["stocking", "tights", "thigh-high"]
                ):
                    return False
                if "stockings" in description.lower() and main_piece and all(
                    token not in main_piece.lower() for token in ["skirt", "dress"]
                ):
                    return False
                return True

            pools = [item for item in pools if allowed_extra(item.get("desc", ""))]
            if pools:
                take = random.choice([1, 2, 3])
                sampled = self._sample_with_feedback(selector, pools, "wardrobe_extra", take, feedback)
                extras = [item.get("desc", "") for item in sampled]

        props = self.catalog.section("props")
        chosen_props: List[str] = []
        if props and maybe(0.45):
            first_prop = self._pick_with_feedback(selector, props, "prop", feedback)
            chosen_props.append(first_prop.get("desc", ""))
            if props and maybe(0.2):
                remaining = [p for p in props if p is not first_prop]
                if remaining:
                    second = self._pick_with_feedback(selector, remaining, "prop", feedback)
                    chosen_props.append(second.get("desc", ""))

        rules = self.catalog.rules()
        bounds = rules.get("caption_length", {"min_words": 18, "max_words": 48})

        template = self._pick_template(template_id, feedback)
        aspect_ratio = cam_frame.get("ratio", "3:4")

        style = self.catalog.style_controller()
        mood_words = mood.get("words", [])
        mood_selection = ", ".join(
            random.sample(mood_words, k=min(len(mood_words), random.choice([1, 2])))
        )

        main_text = main_piece or "styled ensemble"
        wardrobe_text = main_text
        if extras:
            wardrobe_text += "; extras: " + ", ".join(extras)
        if chosen_props:
            wardrobe_text += "; props: " + ", ".join(chosen_props)

        style_snapshot = feedback.snapshot() if feedback is not None else StyleFeedback.baseline_snapshot()
        payload = PromptPayload(
            mood=mood_selection,
            model=model_desc,
            pose=pose.get("desc", ""),
            action=action.get("desc", ""),
            wardrobe=wardrobe_text,
            camera=f"{cam_angle['desc']}; {aspect_ratio}; lens={cam_lens['desc']}; depth={cam_depth['desc']}",
            lighting=lighting.get("desc", ""),
            background=background.get("desc", ""),
            style={
                "aesthetic": style.get("aesthetic"),
                "paper_texture": bool(style.get("paper_texture")),
                "watercolor": style.get("watercolor"),
                "highlights": style.get("highlights"),
                "subsurface_glow": style.get("subsurface_glow"),
                "background_norm": style.get("background_norm"),
                "palette_preference": style.get("palette_preference"),
            },
            required_terms=self.required_terms,
            style_profile=style_snapshot,
            feedback_notes=list(style_snapshot.get("notes", [])),
            character=character_info,
        )

        gene_ids = {
            "palette": palette.get("id"),
            "lighting": lighting.get("id"),
            "background": background.get("id"),
            "camera_angle": cam_angle.get("id"),
            "camera_framing": cam_frame.get("id"),
            "lens": cam_lens.get("id"),
            "depth": cam_depth.get("id"),
            "mood": mood.get("id"),
            "pose": pose.get("id"),
            "action": action.get("id"),
            "template": template,
        }

        scene = SceneStruct(
            template_id=template,
            palette=palette,
            lighting=lighting,
            background=background,
            camera_angle=cam_angle,
            camera_framing=cam_frame,
            camera_lens=cam_lens,
            camera_depth=cam_depth,
            aspect_ratio=aspect_ratio,
            mood_words=mood_words,
            pose=pose,
            action=action,
            model=model_desc,
            wardrobe_main=main_text,
            wardrobe_extras=extras,
            props=chosen_props,
            caption_bounds=bounds,
            gene_ids=gene_ids,
            payload=payload,
            character=character_info,
        )
        scene.payload.scene_summary = short_readable(scene)
        return scene

    def rebuild_from_genes(
        self,
        genes: Dict[str, Optional[str]],
        sfw_level: float,
        temperature: float,
        *,
        feedback: Optional[StyleFeedback] = None,
    ) -> SceneStruct:
        base = self.build_scene(sfw_level=sfw_level, temperature=temperature, feedback=feedback)
        palette = pick_from_ids(self.catalog.section("palettes"), genes.get("palette")) or base.palette
        lighting = pick_from_ids(self.catalog.section("lighting_presets"), genes.get("lighting")) or base.lighting
        background = pick_from_ids(self.catalog.section("backgrounds"), genes.get("background")) or base.background
        camera = self.catalog.raw.get("camera", {})
        base.camera_angle = pick_from_ids(camera.get("angles", []), genes.get("camera_angle")) or base.camera_angle
        base.camera_framing = pick_from_ids(camera.get("framing", []), genes.get("camera_framing")) or base.camera_framing
        base.camera_lens = pick_from_ids(camera.get("lenses", []), genes.get("lens")) or base.camera_lens
        base.camera_depth = pick_from_ids(camera.get("depth", []), genes.get("depth")) or base.camera_depth

        moods = self.catalog.section("moods")
        poses = self.catalog.section("poses")
        actions = self.catalog.section("actions")
        mood = pick_from_ids(moods, genes.get("mood"))
        pose = pick_from_ids(poses, genes.get("pose"))
        action = pick_from_ids(actions, genes.get("action"))

        if mood:
            base.mood_words = mood.get("words", [])
        if pose:
            base.pose = pose
        if action:
            base.action = action

        base.template_id = genes.get("template", base.template_id) or base.template_id
        base.palette = palette
        base.lighting = lighting
        base.background = background
        updated_genes = dict(base.gene_ids)
        updated_genes.update({k: v for k, v in genes.items() if v is not None})
        base.gene_ids = updated_genes
        base.aspect_ratio = base.camera_framing.get("ratio", base.aspect_ratio)

        # refresh payload with new descriptive text
        mood_words = base.mood_words
        mood_selection = ", ".join(
            random.sample(mood_words, k=min(len(mood_words), random.choice([1, 2])))
        ) if mood_words else ""
        wardrobe_text = base.wardrobe_main
        if base.wardrobe_extras:
            wardrobe_text += "; extras: " + ", ".join(base.wardrobe_extras)
        if base.props:
            wardrobe_text += "; props: " + ", ".join(base.props)

        style_snapshot = feedback.snapshot() if feedback is not None else StyleFeedback.baseline_snapshot()
        base.payload = PromptPayload(
            mood=mood_selection,
            model=base.model,
            pose=base.pose.get("desc", ""),
            action=base.action.get("desc", ""),
            wardrobe=wardrobe_text,
            camera=f"{base.camera_angle['desc']}; {base.aspect_ratio}; lens={base.camera_lens['desc']}; depth={base.camera_depth['desc']}",
            lighting=base.lighting.get("desc", ""),
            background=base.background.get("desc", ""),
            style=base.payload.style,
            required_terms=self.required_terms,
            style_profile=style_snapshot,
            feedback_notes=list(style_snapshot.get("notes", [])),
            character=base.character,
        )
        base.payload.scene_summary = short_readable(base)
        return base

def short_readable(scene: SceneStruct) -> str:
    parts = [
        f"tpl={scene.template_id}",
        f"palette={scene.palette.get('id')}",
        f"light={scene.lighting.get('id')}",
        f"bg={scene.background.get('id')}",
        f"angle={scene.camera_angle.get('id')}",
        f"frame={scene.camera_framing.get('id')}",
        f"pose={scene.pose.get('id') if isinstance(scene.pose, dict) else ''}",
        f"action={scene.action.get('id') if isinstance(scene.action, dict) else ''}",
        f"char={scene.character.get('id')}" if scene.character else "",
        f"main={scene.wardrobe_main}",
        f"extras={','.join(scene.wardrobe_extras)}" if scene.wardrobe_extras else "",
    ]
    return " | ".join(filter(None, parts))
