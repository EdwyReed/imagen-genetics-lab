from __future__ import annotations

import json
from itertools import cycle, islice
from pathlib import Path


BASE_DIR = Path("data/catalog")


def write_assets(folder: str, items: list[dict[str, object]]) -> None:
    folder_path = BASE_DIR / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    for item in items:
        payload = json.dumps(item, ensure_ascii=False, indent=2)
        (folder_path / f"{item['id']}.json").write_text(payload + "\n", encoding="utf-8")


def build_actions() -> list[dict[str, object]]:
    movements = [
        ("hip_sway", "Hip sway", "Rolls hips slowly side to side, drawing every highlight across the curves."),
        ("hip_pop", "Hip pop", "Pops one hip with languid control, framing a plush silhouette."),
        ("hip_circle", "Hip circle", "Traces a slow glossy circle with her hips, catching the studio gleam."),
        ("cross_step", "Cross-step", "Slides one leg across the other so the hips glide forward."),
        ("back_arch", "Back arch", "Arches the lower back to push hips forward with satin confidence."),
        ("lean_back", "Soft lean-back", "Leans back just enough to spotlight the hips."),
        ("hip_twist", "Hip twist", "Twists at the waist to show the hips from a teasing angle."),
        ("dip_sway", "Dip sway", "Dips the knees while swaying hips in a playful rhythm."),
    ]
    expressions = [
        ("sugared_smile", "with sugared smile", "flashes a sugared smile that feels innocently indulgent."),
        ("playful_wink", "with playful wink", "sends a playful wink that says she is in on the fun."),
        ("shy_grin", "with shy grin", "lets a shy, syrupy grin bloom across her lips."),
        ("bold_pout", "with bold pout", "flaunts a glossy pout that gleams like candy glaze."),
        ("lip_bite", "with lip bite", "teases a gentle bite of her lip, equal parts bashful and daring."),
        ("soft_laugh", "with soft laugh", "releases a soft laugh that keeps the mood light and sweet."),
    ]
    hands = [
        ("hand_on_hip", "hand on hip", "one hand resting just above the hip bone."),
        ("fingers_in_hair", "fingers in hair", "fingers combing slowly through soft curls."),
        ("lollipop_spin", "twirling lollipop", "twirls a candy-bright lollipop near her smile."),
        ("hat_brim", "tipping brim", "tips a satin sunhat brim toward the camera."),
    ]
    expression_cycle = cycle(expressions)
    hand_cycle = cycle(hands)
    tag_cycle = cycle([["flirty"], ["sweet"], ["confident"], ["tease"], ["playful"], ["sultry"]])
    items: list[dict[str, object]] = []
    for movement in movements:
        for _ in range(3):
            expression = next(expression_cycle)
            hand = next(hand_cycle)
            movement_id, movement_label, movement_desc = movement
            expression_id, expression_label, expression_desc = expression
            hand_id, hand_label, hand_desc = hand
            asset_id = f"{movement_id}_{expression_id}_{hand_id}"
            label = f"{movement_label} {expression_label} ({hand_label})"
            description = (
                f"{movement_desc} She {expression_desc} Her free hand keeps {hand_desc} "
                "All focus lands on the hipline."
            )
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "movement": movement_label,
                        "expression": expression_label.strip(),
                        "gesture": hand_label,
                        "description": description,
                    },
                }
            )
    return items[:24]


def build_backgrounds() -> list[dict[str, object]]:
    colors = [
        ("peach_milkshake", "Peach milkshake fade", ["peach", "melon"], "Peach into pale melon"),
        ("strawberry_cream", "Strawberry cream wash", ["rose", "cream"], "Faded strawberry cream"),
        ("citrus_sorbet", "Citrus sorbet haze", ["citrus", "vanilla"], "Citrus sorbet"),
        ("lavender_fog", "Lavender fog glow", ["lavender", "cloud"], "Lavender fog"),
        ("honey_butter", "Honey butter blush", ["honey", "champagne"], "Honey butter"),
        ("mint_mousse", "Mint mousse drift", ["mint", "foam"], "Mint mousse"),
        ("rose_quartz", "Rose quartz mist", ["rose", "quartz"], "Rose quartz"),
        ("apricot_cloud", "Apricot cloud", ["apricot", "pearl"], "Apricot cloud"),
        ("blueberry_custard", "Blueberry custard", ["blueberry", "cream"], "Blueberry custard"),
        ("cocoa_vanilla", "Cocoa vanilla", ["cocoa", "vanilla"], "Cocoa vanilla"),
        ("sunset_sherbet", "Sunset sherbet", ["sunset", "sherbet"], "Sunset sherbet"),
        ("bubblegum_sky", "Bubblegum sky", ["bubblegum", "sky"], "Bubblegum sky"),
    ]
    finishes = [
        ("jelly_gloss", "jelly gloss sheen", "Wet-look gleam hugging the subject outline."),
        ("powder_soft", "powder-soft diffusion", "Powder-soft diffusion that keeps edges simple."),
        ("halo_glow", "halo glow", "A thin halo glow right behind the hip line."),
        ("floor_reflect", "floor reflection", "A satin floor reflection fading near the knees."),
        ("misty_spark", "misty spark", "Misty sparkle hugging the silhouette."),
        ("clean_gradient", "clean gradient", "Nothing but a clean gradient and a whisper of shine."),
    ]
    tag_cycle = cycle(
        [["studio", "simple"], ["pastel", "simple"], ["gloss", "studio"], ["minimal"], ["soft", "studio"], ["retro", "simple"]]
    )
    items: list[dict[str, object]] = []
    for index, color in enumerate(colors):
        color_id, color_label, color_palette, color_note = color
        for offset in range(2):
            finish = finishes[(index + offset) % len(finishes)]
            finish_id, finish_label, finish_desc = finish
            asset_id = f"{color_id}_{finish_id}"
            label = f"{color_label} with {finish_label}"
            meta_desc = f"Simple {color_note.lower()} gradient with {finish_desc}".strip()
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "colors": color_palette,
                        "finish": finish_label,
                        "description": meta_desc,
                    },
                }
            )
            if len(items) >= 24:
                return items
    return items


def build_camera() -> list[dict[str, object]]:
    framings = [
        ("three_quarter", "Three-quarter", "Frames from head to mid-thigh for an easy hip emphasis."),
        ("waist_up", "Waist-up", "Keeps the hips centered at the lower third."),
        ("full_body", "Full body", "Shows the whole silhouette without clutter."),
        ("hip_close", "Hip-close crop", "Zooms near the hips and lower torso."),
    ]
    angles = [
        ("straight_on", "straight-on", "Straight-on view that lets the hips lead the shot."),
        ("low_angle", "low angle", "Slight low angle that lengthens the legs and hips."),
        ("high_perch", "high perch", "Gentle overhead tilt that still keeps hips pronounced."),
        ("side_slant", "side slant", "Angled from the side to highlight curvature."),
        ("over_shoulder", "over-shoulder", "Over-the-shoulder glance back at the camera."),
        ("fortyfive_spin", "45° spin", "Forty-five degree twist to trace the hip line."),
    ]
    tag_cycle = cycle([["portrait", "hip_focus"], ["fashion", "hip_focus"], ["pinup"], ["studio"], ["playful"], ["confident"]])
    items: list[dict[str, object]] = []
    for frame_index, framing in enumerate(framings):
        framing_id, framing_label, framing_desc = framing
        for offset in range(len(angles)):
            angle = angles[(frame_index + offset) % len(angles)]
            angle_id, angle_label, angle_desc = angle
            asset_id = f"{framing_id}_{angle_id}"
            label = f"{framing_label} {angle_label}"
            description = f"{framing_desc} {angle_desc}".strip()
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "framing": framing_label,
                        "angle": angle_label,
                        "description": description,
                    },
                }
            )
    return items[:24]


def build_lighting() -> list[dict[str, object]]:
    keys = [
        ("sugar_soft", "Sugar-soft key", "Large jelly-soft key hugging the front."),
        ("butter_glow", "Butter glow key", "Butter-yellow glow that kisses the hips."),
        ("pearl_wrap", "Pearl wrap key", "Pearl-toned wrap that adds wet shine."),
        ("candy_beam", "Candy beam key", "Directional candy-color beam to trace curves."),
    ]
    fills = [
        ("bounce_creme", "Cream bounce fill", "Cream bounce to keep shadows gentle."),
        ("frosted_fill", "Frosted fill", "Frosted fill light that polishes highlights."),
        ("lavender_air", "Lavender air fill", "Lavender-tinted fill smoothing skin."),
        ("satin_reflect", "Satin reflector", "Satin reflector placed near the hip line."),
        ("peach_catch", "Peach catch fill", "Peach catch light lifting the cheeks."),
        ("mint_soft", "Mint soft fill", "Minty fill balancing the jelly sheen."),
    ]
    rims = [
        ("halo_rim", "halo rim", "Slim halo rim hugging the silhouette."),
        ("spark_rim", "spark rim", "Sparkling rim hugging the outer hip."),
        ("sunset_rim", "sunset rim", "Warm sunset rim behind the subject."),
        ("frost_rim", "frost rim", "Cool frost rim for contrast."),
        ("rose_rim", "rose rim", "Rose rim light tracing the back hip."),
        ("butterfly_rim", "butterfly rim", "Soft butterfly rim meeting above the shoulders."),
    ]
    tag_cycle = cycle([["studio", "gloss"], ["pinup"], ["soft"], ["vintage"], ["anime_glow"], ["semi_realism"]])
    items: list[dict[str, object]] = []
    for key_index, key in enumerate(keys):
        key_id, key_label, key_desc = key
        for offset in range(len(rims)):
            fill = fills[(key_index + offset) % len(fills)]
            rim_id, rim_label, rim_desc = rims[offset]
            fill_id, fill_label, fill_desc = fill
            asset_id = f"{key_id}_{fill_id}_{rim_id}"
            label = f"{key_label} with {fill_label} and {rim_label}"
            description = f"{key_desc} {fill_desc} {rim_desc}".strip()
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "key": key_label,
                        "fill": fill_label,
                        "rim": rim_label,
                        "description": description,
                    },
                }
            )
    return items[:24]


def build_moods() -> list[dict[str, object]]:
    moods = [
        ("cotton_coy", "Cotton-candy coy", "A coy sweetness like spun sugar flirting with the camera."),
        ("sunny_sass", "Sunny sass", "Warm sunshine attitude with a wink."),
        ("buttercup_bashful", "Buttercup bashful", "Bashful charm that still shows off the hips."),
        ("honeyed_tease", "Honeyed tease", "Honey-sweet teasing energy."),
        ("lollipop_glow", "Lollipop glow", "Glow of someone twirling a lollipop mid-flirt."),
        ("sparkle_serene", "Sparkle serene", "Serenity with sparkling eyes."),
        ("blooming_confidence", "Blooming confidence", "Confidence blooming like a retro pin-up poster."),
        ("daydream_daring", "Daydream daring", "Dreamy but daring presence."),
        ("neighborly_flirt", "Neighborly flirt", "Girl-next-door warmth with a flirtatious twist."),
        ("peachy_poise", "Peachy poise", "Polished poise with peachy sweetness."),
        ("twinkle_tease", "Twinkle tease", "Twinkling eyes that tease while staying innocent."),
        ("cherry_charm", "Cherry charm", "Cherry-bright charm with gentle provocation."),
        ("marshmallow_mischief", "Marshmallow mischief", "Soft mischief cushioned by sweetness."),
        ("satin_selfassured", "Satin self-assured", "Self-assured but silky demeanor."),
        ("retro_romp", "Retro romp", "Playful retro romp energy."),
        ("velvet_vulnerability", "Velvet vulnerability", "Velvety vulnerability that invites closeness."),
        ("dollop_of_daring", "Dollop of daring", "Just a dollop of daring atop the sweetness."),
        ("gossamer_grin", "Gossamer grin", "Featherlight grin hinting at more."),
        ("sparkler_soft", "Sparkler soft", "Soft yet sparkling enthusiasm."),
        ("sundae_confessional", "Sundae confessional", "Like sharing secrets over sundaes."),
        ("vintage_velour", "Vintage velour", "Velour-smooth vintage flirtation."),
        ("starlet_sigh", "Starlet sigh", "A starlet's sigh that draws attention to her curves."),
        ("sugarcoat_strength", "Sugarcoat strength", "Strength hidden under sugared charm."),
        ("cupid_caper", "Cupid caper", "Playful mischief as if Cupid staged the scene."),
    ]
    items: list[dict[str, object]] = []
    for mood_id, label, description in moods:
        items.append(
            {
                "id": mood_id,
                "label": label,
                "weight": 1.0,
                "tags": ["pinup"],
                "requires": [],
                "excludes": [],
                "meta": {"description": description},
            }
        )
    return items


def build_palettes() -> list[dict[str, object]]:
    bases = [
        ("peach_rose_cream", "Peach rose cream", ["peach", "rose", "cream"], "Soft dessert trio."),
        ("melon_mist_gold", "Melon mist gold", ["melon", "mint", "pale gold"], "Cooling melon with gold glints."),
        ("berry_milk", "Berry milk", ["raspberry", "milk", "powder"], "Milkshake berry mix."),
        ("sunset_puff", "Sunset puff", ["apricot", "petal", "twilight"], "Sunset with puffed pastels."),
        ("mint_cotton", "Mint cotton", ["mint", "cotton candy", "pearl"], "Mint cotton candy dream."),
        ("honey_peony", "Honey peony", ["honey", "peony", "champagne"], "Peony kissed honey."),
        ("soda_fizz", "Soda fizz", ["cherry soda", "vanilla", "sky"], "Cherry vanilla soda fizz."),
        ("sherbet_swirl", "Sherbet swirl", ["sherbet", "cream", "cantaloupe"], "Swirling sherbet hues."),
        ("vintage_coral", "Vintage coral", ["coral", "sepia", "powder blue"], "Vintage coral poster vibes."),
        ("butter_mauve", "Butter mauve", ["butter", "mauve", "ivory"], "Buttercream mauve glow."),
        ("blush_mint", "Blush mint", ["blush", "mint", "opal"], "Blush and mint harmony."),
        ("amber_plum", "Amber plum", ["amber", "plum", "cream"], "Amber plum sundae."),
    ]
    accents = [
        ("glaze_highlight", "glaze highlight", "Glossy glaze accent"),
        ("satin_shadow", "satin shadow", "Satin-soft low contrast shadow"),
        ("sugar_spark", "sugar spark", "Micro sugar sparkles"),
        ("poster_grain", "poster grain", "Light vintage poster grain"),
    ]
    tag_cycle = cycle([["warm"], ["cool"], ["retro"], ["anime"], ["semi_realism"], ["pinup"]])
    items: list[dict[str, object]] = []
    for index, base in enumerate(bases):
        base_id, base_label, colors, base_desc = base
        for offset in range(2):
            accent = accents[(index + offset) % len(accents)]
            accent_id, accent_label, accent_desc = accent
            asset_id = f"{base_id}_{accent_id}"
            label = f"{base_label} with {accent_label}"
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "colors": colors,
                        "accent": accent_label,
                        "description": f"{base_desc} finished with {accent_desc}.",
                    },
                }
            )
            if len(items) >= 24:
                return items
    return items


def build_props() -> list[dict[str, object]]:
    props = [
        ("candy_heart_lollipop", "Candy heart lollipop", "Heart-shaped lollipop catching glossy highlights."),
        ("sparkle_soda_bottle", "Sparkle soda bottle", "Vintage soda bottle with pastel fizz."),
        ("silk_ribbon_bundle", "Silk ribbon bundle", "Bundle of silk ribbons tied near the hip."),
        ("pearl_phone", "Pearl rotary phone", "Pearlescent rotary phone for playful calls."),
        ("bouquet_babybreath", "Baby's breath bouquet", "Tiny bouquet resting near the waist."),
        ("vinyl_record", "Mint vinyl record", "Mint-colored vinyl for vintage touch."),
        ("compact_mirror", "Compact mirror", "Glossy compact reflecting hip curve."),
        ("sundae_dish", "Sundae dish", "Pastel sundae dish with melting gloss."),
        ("sparkler_star", "Sparkler star", "Small sparkler shaped like a star."),
        ("heart_balloon", "Helium heart balloon", "Floating heart balloon brushing hips."),
        ("silk_scarf", "Silk polka-dot scarf", "Polka-dot scarf draped loosely."),
        ("marshmallow_pillow", "Marshmallow pillow", "Fluffy pillow to lean against."),
        ("retro_camera", "Retro snapshot camera", "Vintage camera held at the hip."),
        ("perfume_atomizer", "Perfume atomizer", "Glass atomizer misting shimmer."),
        ("candy_box", "Candy sampler box", "Open candy box with bright centers."),
        ("ribbon_curtain", "Ribbon curtain", "Simple ribbon strands framing hips."),
        ("sunhat", "Wide-brim sunhat", "Soft sunhat with satin ribbon."),
        ("cherry_earrings", "Cherry earrings", "Cherry drops that sway with movement."),
        ("stacked_bangles", "Stacked bangles", "Glossy bangles sliding near the wrist."),
        ("feather_duster", "Feather duster", "Playful feather duster matching palette."),
        ("macaron_plate", "Macaron plate", "Plate of macarons resting at the side."),
        ("boudoir_stool", "Boudoir stool", "Low stool that keeps focus on hips."),
        ("glitter_clutch", "Glitter clutch", "Mini clutch sparkling at hip level."),
        ("bubble_blower", "Bubble blower", "Bubble wand for iridescent orbs."),
    ]
    items: list[dict[str, object]] = []
    for prop_id, label, description in props:
        items.append(
            {
                "id": prop_id,
                "label": label,
                "weight": 1.0,
                "tags": ["pinup"],
                "requires": [],
                "excludes": [],
                "meta": {"description": description},
            }
        )
    return items


def build_poses() -> list[dict[str, object]]:
    bases = [
        ("swaying_stand", "Swaying stand", "Weight shifted onto one leg with hips leading."),
        ("cross_leg_pop", "Cross-leg pop", "Legs crossed with hip popped outward."),
        ("leaning_counter", "Leaning counter", "Lean onto a counter while hips rest back."),
        ("perch_pose", "Perched seat", "Perch on a stool with hips angled forward."),
        ("kneel_arch", "Kneeling arch", "Kneel with lower back arched and hips pushed back."),
        ("floor_siren", "Floor siren", "Lay on side propped on elbow with hips stacked."),
        ("step_forward", "Step forward", "One step forward while hips follow the stride."),
        ("retro_salute", "Retro salute", "Salute with hips cocked to the side."),
    ]
    legs = [
        ("toe_point", "pointed toes", "Toes pointed to elongate legs."),
        ("knee_pop", "bent knee", "One knee bent to drive the hip curve."),
        ("heel_lift", "lifted heel", "Heel lifted off the ground for extra hip tilt."),
    ]
    arms = [
        ("hand_cascade", "cascading hand", "Hand cascading along the hip."),
        ("arms_overhead", "arms overhead", "Arms overhead stretching the torso."),
        ("double_frame", "double frame", "Both hands framing the waist."),
    ]
    leg_cycle = cycle(legs)
    arm_cycle = cycle(arms)
    tag_cycle = cycle([["standing"], ["floor"], ["stool"], ["pinup"], ["playful"], ["confident"]])
    items: list[dict[str, object]] = []
    for base in bases:
        base_id, base_label, base_desc = base
        for _ in range(3):
            leg_id, leg_label, leg_desc = next(leg_cycle)
            arm_id, arm_label, arm_desc = next(arm_cycle)
            asset_id = f"{base_id}_{leg_id}_{arm_id}"
            label = f"{base_label} with {leg_label} and {arm_label}"
            description = f"{base_desc} {leg_desc} {arm_desc}".strip()
            items.append(
                {
                    "id": asset_id,
                    "label": label,
                    "weight": 1.0,
                    "tags": next(tag_cycle),
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "base": base_label,
                        "legs": leg_label,
                        "arms": arm_label,
                        "description": description,
                    },
                }
            )
    return items[:24]


def build_characters() -> list[dict[str, object]]:
    characters = [
        ("sundae_starlet", "Sundae starlet", "Classic pin-up darling with dripping gloss."),
        ("neighborly_muse", "Neighborly muse", "Girl-next-door spark with coy polish."),
        ("velvet_vixen", "Velvet vixen", "Velvet-soft glamour that still feels warm."),
        ("cotton_candy_dreamer", "Cotton candy dreamer", "Dreamy sweetness that invites a smile."),
        ("jellybean_jester", "Jellybean jester", "Playful jokester with pastel punch."),
        ("apricot_angel", "Apricot angel", "Sun-kissed apricot charm and playful innocence."),
        ("retro_rocket", "Retro rocket", "Vintage poster heroine ready for fun."),
        ("starlit_neighbor", "Starlit neighbor", "Friendly neighbor with starlet energy."),
        ("bubblegum_bard", "Bubblegum bard", "Storytelling sweetheart with candy flair."),
        ("pillowtalk_poet", "Pillowtalk poet", "Soft-spoken muse with flirtatious prose."),
        ("diner_daydream", "Diner daydream", "Charming soda-shop muse."),
        ("sunny_showgirl", "Sunny showgirl", "Showgirl sparkle toned down by warmth."),
        ("peachy_pinup", "Peachy pin-up", "Peach-forward glamour with a wink."),
        ("glossy_girl_next", "Glossy girl-next-door", "Neighborly softness and high-gloss finish."),
        ("satin_storyteller", "Satin storyteller", "Narrative-driven flirt with satin notes."),
        ("butterscotch_belle", "Butterscotch belle", "Caramel-sweet confidence."),
        ("cherub_chanteuse", "Cherub chanteuse", "Singer-inspired sweetness and coquette flair."),
        ("macaron_maven", "Macaron maven", "Elegant dessert expert with hip-swaying grace."),
        ("velour_valentine", "Velour valentine", "Velour softness matched with daring pose."),
        ("sugarcoated_stylist", "Sugarcoated stylist", "Stylish creative who loves accentuating hips."),
        ("strawberry_spark", "Strawberry spark", "Sparkling energy with strawberry gloss."),
        ("minty_muse", "Minty muse", "Cool-toned muse balancing innocence and allure."),
        ("poster_princess", "Poster princess", "Poster-inspired heroine with bold hips."),
        ("anime_darling", "Anime darling", "Anime-flavored sweetheart with soft curves."),
    ]
    items: list[dict[str, object]] = []
    for char_id, label, description in characters:
        items.append(
            {
                "id": char_id,
                "label": label,
                "weight": 1.0,
                "tags": ["pinup"],
                "requires": [],
                "excludes": [],
                "meta": {"description": description},
            }
        )
    return items


def build_models() -> list[dict[str, object]]:
    families = [
        ("gelatin_glamour", "Gelatin glamour", "semi-realism", "Semi-realistic jelly sheen pin-up render"),
        ("jelly_luxe", "Jelly luxe", "semi-realism", "High-gloss semi-real render with wet textures"),
        ("nectar_glow", "Nectar glow", "semi-realism", "Honeyed semi-real look with luminous hips"),
        ("holo_satin", "Holo satin", "semi-realism", "Holographic satin finish, slick and juicy"),
        ("anime_spark", "Anime spark", "anime", "Shōjo-inspired sparkle with soft shading"),
        ("pastel_idol", "Pastel idol", "anime", "Idol-style anime render with candy palettes"),
        ("manga_muse", "Manga muse", "anime", "Manga lines softened with pastel washes"),
        ("chibi_coquette", "Chibi coquette", "anime", "Stylized chibi proportions with hip emphasis"),
        ("poster_velvet", "Poster velvet", "vintage", "Vintage poster grain with velvety tones"),
        ("sepia_glam", "Sepia glam", "vintage", "Sepia-infused photo look with pin-up lighting"),
        ("retro_litho", "Retro lithograph", "vintage", "Lithograph texture minus typography"),
        ("film_blush", "Film blush", "vintage", "Film grain blush with satin highlights"),
    ]
    variants = [("v1", "Version 1"), ("v2", "Version 2")]
    items: list[dict[str, object]] = []
    for family_id, family_label, style_family, family_desc in families:
        for variant_id, variant_label in variants:
            asset_id = f"{family_id}_{variant_id}"
            items.append(
                {
                    "id": asset_id,
                    "label": f"{family_label} {variant_label}",
                    "weight": 1.0,
                    "tags": [style_family],
                    "requires": [],
                    "excludes": [],
                    "meta": {
                        "style_family": style_family,
                        "variant": variant_label,
                        "description": family_desc,
                    },
                }
            )
            if len(items) >= 24:
                return items
    return items


def build_rules() -> list[dict[str, object]]:
    rules = [
        ("hips_center_stage", "Keep hips center stage", "Ensure prompts keep hips as the visual anchor."),
        ("gloss_highlight", "Emphasize glossy highlights", "Call for wet-look highlights along curves."),
        ("sugar_soft_shadows", "Sugar-soft shadows", "Keep shadows tender and low contrast."),
        ("simple_backdrops", "Simple backdrops", "Restrict backgrounds to gradients or minimal props."),
        ("no_text", "No poster text", "Avoid lettering even in vintage frames."),
        ("hipline_guides", "Hipline guiding pose", "Ask for poses that guide the eye along the hipline."),
        ("sweet_expression", "Sweet expression first", "Expressions should feel sweet before daring."),
        ("clean_palette", "Clean dessert palette", "Limit palette to dessert-inspired hues."),
        ("anime_sparkle_eyes", "Anime sparkle eyes", "Add eye sparkles for anime moods."),
        ("semi_real_skin", "Semi-real skin sheen", "Describe skin as semi-real with jelly gleam."),
        ("vintage_grain", "Vintage grain touch", "Apply subtle film grain without text."),
        ("hips_forward", "Ask for hips forward", "Remind the model to angle hips toward camera."),
        ("simple_props", "Simple props", "Use at most one playful prop."),
        ("warm_light", "Warm light bias", "Favor warm butter-toned light."),
        ("cool_balance", "Cool balance", "Balance warmth with a minty fill."),
        ("poster_crop", "Poster crop", "Frame like a poster with breathing room."),
        ("anime_pose_flow", "Anime pose flow", "Encourage flowing limbs like anime key art."),
        ("pinup_posture", "Pin-up posture", "Call for straight spine and dramatic hips."),
        ("neighborly_vibes", "Neighborly vibes", "Mention approachable, friendly energy."),
        ("prop_at_hip", "Prop near hip", "If props appear, place them by the hips."),
        ("gloss_on_fabric", "Gloss on fabric", "Describe fabrics as glossy or satin."),
        ("pastel_back", "Pastel background", "Keep backgrounds pastel and uncluttered."),
        ("eyes_camera", "Eyes to camera", "Maintain eye contact for intimacy."),
        ("soft_smile", "Soft smile cue", "Default to soft smiles unless told otherwise."),
    ]
    items: list[dict[str, object]] = []
    for rule_id, label, description in rules:
        items.append(
            {
                "id": rule_id,
                "label": label,
                "weight": 1.0,
                "tags": ["pinup"],
                "requires": [],
                "excludes": [],
                "meta": {"prompt": description},
            }
        )
    return items


def build_wardrobe() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    tops = [
        ("satin_wrap_blouse", "Satin wrap blouse", "top", ["top", "wrap"], "Blousy wrap top tied above the hips."),
        ("sugar_bustier", "Sugar-glaze bustier", "top", ["top", "structured"], "Structured bustier with jelly gloss."),
        ("bow_peplum", "Bow peplum blouse", "top", ["top", "peplum"], "Peplum hem that flares over hips."),
        ("lace_camisole", "Lace camisole", "top", ["top", "lace"], "Delicate lace cami tucked in."),
        ("offshoulder_ripple", "Off-shoulder ripple top", "top", ["top", "offshoulder"], "Rippled off-shoulder neckline."),
        ("cropped_cardigan", "Cropped cardigan", "top", ["top", "cardigan"], "Snug cardigan fastened at bust."),
        ("halter_glow", "Halter glow top", "top", ["top", "halter"], "Halter top with liquid sheen."),
        ("tie_front_blouse", "Tie-front blouse", "top", ["top", "knot"], "Tie-front blouse knotted above waist."),
        ("bubble_sleeve_bodysuit", "Bubble sleeve bodysuit", "top", ["top", "bodysuit"], "Bodysuit with bubble sleeves emphasizing hips."),
        ("pleated_peterpan", "Pleated Peter Pan blouse", "top", ["top", "collar"], "Peter Pan collar with gentle pleats."),
    ]
    for item_id, label, slot, tags, description in tops:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    bottoms = [
        ("satin_pencil_skirt", "Satin pencil skirt", "bottom", ["bottom", "skirt"], "Glossy pencil skirt hugging hips."),
        ("swing_circle_skirt", "Swing circle skirt", "bottom", ["bottom", "skirt"], "High-waist circle skirt twirling."),
        ("highrise_shorts", "High-rise cuffed shorts", "bottom", ["bottom", "shorts"], "Cuffed shorts accentuating legs."),
        ("ruffle_mini", "Ruffle mini skirt", "bottom", ["bottom", "skirt"], "Ruffled hem flirting with thighs."),
        ("belted_capris", "Belted capris", "bottom", ["bottom", "pants"], "Capris cinched with a bow belt."),
        ("pleated_skort", "Pleated skort", "bottom", ["bottom", "skort"], "Pleated skort with playful swing."),
        ("satin_hotpants", "Satin hot pants", "bottom", ["bottom", "shorts"], "Ultra-gloss hot pants hugging hips."),
        ("sugarflare_trousers", "Sugar-flare trousers", "bottom", ["bottom", "pants"], "Flared trousers with shimmer."),
        ("button_front_skirt", "Button-front skirt", "bottom", ["bottom", "skirt"], "Button row guiding the eye."),
        ("wrap_sarong", "Wrap sarong", "bottom", ["bottom", "wrap"], "Wrap sarong knotted at hip."),
    ]
    for item_id, label, slot, tags, description in bottoms:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    dresses = [
        ("wiggle_dress", "Wiggle dress", "dress", ["dress"], "Curve-hugging wiggle dress with sweetheart neckline."),
        ("swing_dress", "Swing dress", "dress", ["dress"], "Swing dress with cinched waist and flaring hips."),
        ("satin_slip_dress", "Satin slip dress", "dress", ["dress", "slip"], "Satin slip with jelly sheen."),
        ("tea_length_dress", "Tea-length twirl dress", "dress", ["dress"], "Tea-length twirl with sweetheart vibe."),
        ("bow_wrap_dress", "Bow wrap dress", "dress", ["dress", "wrap"], "Wrap dress with oversized hip bow."),
        ("retro_polka_dress", "Retro polka dress", "dress", ["dress", "polka"], "Polka dot dress hugging the waist."),
    ]
    for item_id, label, slot, tags, description in dresses:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    accessories = [
        ("pearl_belt", "Pearl hip belt", "accessory", ["accessory", "belt"], "Pearl belt resting across hips."),
        ("satin_gloves", "Satin opera gloves", "accessory", ["accessory", "gloves"], "Elbow gloves with glossy finish."),
        ("heart_sunglasses", "Heart sunglasses", "accessory", ["accessory", "eyewear"], "Heart-shaped frames."),
        ("ribbon_choker", "Ribbon choker", "accessory", ["accessory", "neck"], "Satin ribbon choker with bow."),
        ("ankle_scarf", "Ankle scarf", "accessory", ["accessory", "ankle"], "Small scarf tied at ankle."),
        ("waist_bow", "Waist bow", "accessory", ["accessory", "belt"], "Oversized bow perched on hip."),
    ]
    for item_id, label, slot, tags, description in accessories:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    hosiery = [
        ("backseam_stockings", "Back-seam stockings", "hosiery", ["hosiery"], "Sheer stockings with center seam."),
        ("lace_thighhighs", "Lace thigh-highs", "hosiery", ["hosiery"], "Lace tops hugging thighs."),
        ("pastel_tights", "Pastel sheen tights", "hosiery", ["hosiery"], "Glossy pastel tights."),
        ("fishnet_sparkle", "Sparkle fishnets", "hosiery", ["hosiery"], "Fishnets with glitter strands."),
    ]
    for item_id, label, slot, tags, description in hosiery:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    footwear = [
        ("peep_toe_pumps", "Peep-toe pumps", "footwear", ["footwear", "heels"], "Classic pumps with gloss."),
        ("ankle_strap_heels", "Ankle-strap heels", "footwear", ["footwear", "heels"], "Strappy heels showing off ankles."),
        ("kitten_heels", "Kitten heels", "footwear", ["footwear", "heels"], "Comfort kitten heels for neighborly vibe."),
        ("ribbon_wedges", "Ribbon wedges", "footwear", ["footwear", "wedges"], "Wedges tied with ribbons."),
        ("platform_maryjanes", "Platform Mary Janes", "footwear", ["footwear", "heels"], "Mary Janes with playful platform."),
        ("satin_mules", "Satin mules", "footwear", ["footwear", "mules"], "Slip-on mules with satin sheen."),
    ]
    for item_id, label, slot, tags, description in footwear:
        items.append(
            {
                "id": item_id,
                "label": label,
                "weight": 1.0,
                "tags": tags,
                "requires": [],
                "excludes": [],
                "meta": {"slot": slot, "description": description},
            }
        )

    return items


def build_wardrobe_sets() -> list[dict[str, object]]:
    combos = [
        ("wrap_blouse_pencil", "Wrap blouse and pencil skirt", ["satin_wrap_blouse", "satin_pencil_skirt", "pearl_belt", "peep_toe_pumps"], "Glossy wrap blouse tucked into pencil skirt with pearl belt."),
        ("bow_peplum_circle", "Bow peplum with circle skirt", ["bow_peplum", "swing_circle_skirt", "waist_bow", "ankle_strap_heels"], "Peplum flounce over swirling skirt."),
        ("halter_hotpants", "Halter and hot pants", ["halter_glow", "satin_hotpants", "ankle_scarf", "platform_maryjanes"], "Halter top with ultra-gloss shorts."),
        ("cardigan_skort", "Cardigan and pleated skort", ["cropped_cardigan", "pleated_skort", "ribbon_choker", "kitten_heels"], "Girl-next-door cardigan with playful skort."),
        ("laced_neighbor", "Lace cami and button skirt", ["lace_camisole", "button_front_skirt", "pearl_belt", "kitten_heels"], "Lace cami tucked into button skirt."),
        ("sunny_shortset", "Tie blouse and cuffed shorts", ["tie_front_blouse", "highrise_shorts", "heart_sunglasses", "peep_toe_pumps"], "Sunny short set with heart shades."),
        ("bow_wrap_dress_set", "Bow wrap dress look", ["bow_wrap_dress", "satin_gloves", "peep_toe_pumps"], "Wrap dress with opera gloves."),
        ("wiggle_dress_gloss", "Wiggle dress glam", ["wiggle_dress", "pearl_belt", "ankle_strap_heels"], "Classic wiggle dress with pearl belt."),
        ("swing_dress_vintage", "Swing dress vintage", ["swing_dress", "satin_gloves", "platform_maryjanes"], "Vintage swing with gloves."),
        ("slip_dress_glow", "Slip dress glow", ["satin_slip_dress", "ribbon_choker", "satin_mules"], "Satin slip with ribbon choker."),
        ("tea_length_neighbor", "Tea-length neighbor", ["tea_length_dress", "heart_sunglasses", "kitten_heels"], "Tea-length twirl with heart glasses."),
        ("polka_dress_pinup", "Polka pin-up", ["retro_polka_dress", "satin_gloves", "ankle_strap_heels"], "Polka dots with gloves and straps."),
        ("peplum_capri", "Peplum and capris", ["bow_peplum", "belted_capris", "ribbon_choker", "ribbon_wedges"], "Peplum top with cinched capris."),
        ("wrap_sarong_set", "Sarong sweetness", ["satin_wrap_blouse", "wrap_sarong", "waist_bow", "satin_mules"], "Wrap blouse tied with sarong."),
        ("bodysuit_flare", "Bodysuit with flare trousers", ["bubble_sleeve_bodysuit", "sugarflare_trousers", "pearl_belt", "ankle_strap_heels"], "Bodysuit tucked into flare trousers."),
        ("halter_skirt_combo", "Halter and ruffle mini", ["halter_glow", "ruffle_mini", "ankle_scarf", "peep_toe_pumps"], "Halter matched with ruffle mini."),
        ("neighborly_shortset", "Neighborly short set", ["cropped_cardigan", "satin_hotpants", "heart_sunglasses", "kitten_heels"], "Glossy shorts with cardigan."),
        ("bodysuit_shortset", "Bodysuit and cuffed shorts", ["bubble_sleeve_bodysuit", "highrise_shorts", "waist_bow", "ankle_strap_heels"], "Bodysuit tucked into cuffed shorts."),
        ("wrap_dress_stockings", "Wrap dress with stockings", ["bow_wrap_dress", "backseam_stockings", "peep_toe_pumps"], "Wrap dress with classic hosiery."),
        ("swing_stockings", "Swing dress with stockings", ["swing_dress", "lace_thighhighs", "platform_maryjanes"], "Swing dress and lace thigh-highs."),
        ("sarong_swim", "Sarong swim tease", ["satin_wrap_blouse", "satin_hotpants", "ankle_scarf", "satin_mules"], "Sarong-inspired wrap over shorts."),
        ("poster_neighbor", "Poster neighbor", ["pleated_peterpan", "swing_circle_skirt", "ribbon_choker", "kitten_heels"], "Poster-perfect neighborly ensemble."),
        ("pearl_trouser_set", "Pearl trouser set", ["satin_wrap_blouse", "sugarflare_trousers", "pearl_belt", "satin_mules"], "Pearl belt tying blouse and trousers."),
        ("wiggle_stockings", "Wiggle dress stockings", ["wiggle_dress", "fishnet_sparkle", "ankle_strap_heels"], "Wiggle dress with sparkle fishnets."),
    ]
    items: list[dict[str, object]] = []
    for set_id, label, requires, description in combos:
        items.append(
            {
                "id": set_id,
                "label": label,
                "weight": 1.0,
                "tags": ["pinup"],
                "requires": [f"wardrobe:{req}" for req in requires],
                "excludes": [],
                "meta": {"description": description},
            }
        )
    return items


def main() -> None:
    write_assets("actions", build_actions())
    write_assets("backgrounds", build_backgrounds())
    write_assets("camera", build_camera())
    write_assets("lighting", build_lighting())
    write_assets("moods", build_moods())
    write_assets("palettes", build_palettes())
    write_assets("props", build_props())
    write_assets("poses", build_poses())
    write_assets("characters", build_characters())
    write_assets("models", build_models())
    write_assets("rules", build_rules())
    write_assets("wardrobe", build_wardrobe())
    write_assets("wardrobe_sets", build_wardrobe_sets())


if __name__ == "__main__":
    main()
