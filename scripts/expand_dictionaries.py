"""Utility to expand action, clothing, pose, and style dictionaries.

The generator creates a wide range of new options spanning safe, playful,
and provocative vibes so downstream sampling has richer diversity.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dictionaries"


def slugify(text: str, *, limit: int = 6) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(tokens[:limit]) or "item"


def short_phrase(text: str, *, max_words: int = 9) -> str:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9']+", text) if t]
    if len(tokens) <= max_words:
        return " ".join(tokens)
    return " ".join(tokens[: max_words - 1] + [tokens[-1]])


def cycle_from(seq: Sequence[Sequence[str]]) -> Iterable[Sequence[str]]:
    while True:
        for item in seq:
            yield item


def generate_actions() -> List[dict]:
    categories = [
        {
            "prefix": "fresh",
            "verb_phrases": [
                "twirling a silk parasol by the ribbon",
                "balancing a stack of art books on one hip",
                "sipping sparkling yuzu water through a glass straw",
                "adjusting oversized sunglasses with a bright smile",
                "pinning a polaroid to a string of fairy lights",
                "holding a bouquet of farmer's market ranunculus",
                "rolling a straw hat brim between both palms",
                "zipping a windbreaker halfway while laughing",
                "stretching arms overhead with a sunrise yawn",
                "tying a satin scarf around a messy ponytail",
                "buttoning a chambray shirt over a ribbed tank",
                "twisting the cap on a mason jar lemonade",
                "tracing raindrops on a café window",
                "peeling an orange with meticulous focus",
                "checking a vintage film camera light meter",
                "pouring oat milk into a foamy latte heart",
                "kicking up one heel while texting",
                "planting seedlings in a balcony planter",
                "brushing watercolor across an easel canvas",
                "snapping a fresh daisy between fingertips",
            ],
            "posture_phrases": [
                "hips angled toward the camera and shoulders relaxed",
                "standing tall with shoulders rolled back",
                "leaning into a counter with elbows tucked in",
                "perched on a stool with knees together",
                "weight shifting from one leg to the other",
                "seated sideways with ankles neatly crossed",
                "arched slightly with collarbone catching the light",
                "head tilted toward the sun and eyes closed",
                "gaze over shoulder with loose braid falling forward",
                "spine elongated and core engaged",
            ],
            "setting_phrases": [
                "on a sun-warmed rooftop garden",
                "beside a breezy seaside railing",
                "inside a plant-filled greenhouse",
                "along a cobblestone market alley",
                "under string lights at golden hour",
                "near a mural-splashed warehouse wall",
                "within a minimalist Scandinavian loft",
                "in front of a citrus orchard backdrop",
                "beside a train window streaked with rain",
                "beneath a lace parasol in a rose garden",
            ],
            "moods": [
                ["fresh", "breezy"],
                ["calm", "content"],
                ["playful", "soft"],
                ["curious", "bright"],
            ],
            "target": 90,
        },
        {
            "prefix": "flirt",
            "verb_phrases": [
                "licking powdered sugar from a fingertip",
                "tracing a heart on steamed glass",
                "biting the tip of a fountain pen cap",
                "looping pearl strands around one wrist",
                "spritzing perfume along a swan neck",
                "dipping a toe into a bubble bath overflow",
                "dangling a lollipop teasingly between lips",
                "snapping a suspenders strap with a grin",
                "tugging a cardigan off one shoulder",
                "drawing lipstick with a slow deliberate curve",
                "cinching a corset ribbon tight",
                "rolling stockings up each thigh",
                "dusting glitter highlighter across collarbones",
                "grazing fingertips along garter clips",
                "pressing palms flat on a mirrored vanity",
                "balancing on tiptoes to reach a high shelf",
                "lifting hair to cool the back of her neck",
                "sliding a silk robe sleeve past the elbow",
                "leaning into a cherry-red convertible door",
                "holding a cherry stem knotted on her tongue",
            ],
            "posture_phrases": [
                "hips popped and ribs creating a gentle S-curve",
                "spine arched with chin tilted toward camera",
                "knees grazing together while toes point inward",
                "standing on one leg with the other bent forward",
                "elbows propped behind for a graceful lean",
                "leaning forward with shoulders slightly hunched",
                "back to the camera with head turned over shoulder",
                "perched on countertop with calves dangling",
                "one knee planted while the other extends behind",
                "lying sideways with torso twisted up toward lens",
            ],
            "setting_phrases": [
                "inside a glossy art deco dressing room",
                "against a rain-fogged penthouse window",
                "across a velvet chaise in low evening light",
                "atop a mirrored dance floor with spotlight glare",
                "in a chrome-trimmed laundromat after hours",
                "beside a jukebox glowing neon pink",
                "on a retro motel balcony with palm shadows",
                "inside a candy-striped photo booth",
                "under a streetlamp during midnight drizzle",
                "framed by a beaded doorway curtain",
            ],
            "moods": [
                ["flirty", "bold"],
                ["teasing", "electric"],
                ["magnetic", "sultry"],
                ["glam", "charged"],
            ],
            "target": 110,
        },
        {
            "prefix": "daring",
            "verb_phrases": [
                "arching backward over a chrome workout bench",
                "crawling forward with a feline prowl",
                "sprawling half inside a commercial dryer drum",
                "straddling a motorcycle seat with engine idling",
                "bracing both palms on a shower wall streaked with steam",
                "hooking heels over the top rung of a ladder",
                "suspending from silk aerial ribbons in a split",
                "pressing thighs together while sliding down a pole",
                "resting ankles on a partner chair back while laughing",
                "pressing knees apart while tugging a harness strap",
                "perching on a countertop with toes barely touching",
                "lying upside-down off a bed with hair cascading",
                "climbing onto a washing machine mid-cycle",
                "stretching across a piano lid with fingers trailing keys",
                "balancing on roller skates while gripping a doorframe",
                "arching into a backbend bridge with wrists crossed",
                "writhing under a spotlight with wet-look skin",
                "leaning into a window frame with lower back exposed",
                "hooking a finger into fishnet waistband elastic",
                "pressing a forearm against a mirrored wall",
            ],
            "posture_phrases": [
                "hips swiveling with a dramatic scoop",
                "spine bending in a hyperextended arc",
                "knees parted with toes pointed",
                "legs tangled with one knee dropped low",
                "shoulders pressing into the surface for counterweight",
                "torso twisted toward the lens with breathless energy",
                "back arched so ribs jut into the light",
                "glutes lifted toward the camera in a power stance",
                "neck elongated with jawline flexed",
                "core taut as arms extend overhead",
            ],
            "setting_phrases": [
                "beneath industrial warehouse spotlights",
                "inside a crimson-lit locker room",
                "against a rain-slicked city alley",
                "on a revolving stage with strobes pulsing",
                "within a chrome-walled laundromat bay",
                "inside a photography cyclorama drenched in gel light",
                "along a mirrored fitness studio wall",
                "atop a kitchen island cluttered with baking props",
                "under a vintage hairdryer hood in a salon",
                "inside a fisheye hallway with checkerboard tiles",
            ],
            "moods": [
                ["provocative", "charged"],
                ["feral", "intense"],
                ["seductive", "dramatic"],
                ["bold", "siren"],
            ],
            "target": 120,
        },
    ]

    generated: List[dict] = []

    for cat in categories:
        mood_cycle = cycle_from(cat["moods"])
        combos = itertools.product(
            cat["verb_phrases"], cat["posture_phrases"], cat["setting_phrases"]
        )
        for combo_index, (verb, posture, setting) in enumerate(combos, start=1):
            if combo_index > cat["target"]:
                break
            desc = f"{verb} with {posture} {setting}"
            summary = short_phrase(f"{verb} {setting}")
            generated.append(
                {
                    "id": f"action_{cat['prefix']}_{combo_index:03d}",
                    "description": desc,
                    "mood": next(mood_cycle),
                    "summary": summary,
                }
            )

    return generated


def generate_clothes() -> List[dict]:
    categories = [
        {
            "prefix": "streetlux",
            "primary": [
                "sunset satin bomber with ruched sleeves",
                "powder-blue cropped moto jacket with perforated vents",
                "charcoal pinstripe blazer cinched with a patent belt",
                "ivory waffle-knit wrap sweater with deep cuffs",
                "glossy vinyl trench with clear lapels",
                "buttercup silk camp shirt tied above the waist",
                "mint mesh hoodie layered over a bralette",
                "denim corset top with brass grommets",
                "terracotta suede chore coat with exaggerated collar",
                "holographic windbreaker with translucent panels",
                "cropped shearling biker jacket with satin lining",
                "emerald utility vest with gold zipper tape",
                "graphite ribbed bodysuit under a cutaway blazer",
                "checkerboard knit polo with zip neckline",
                "longline varsity cardigan edged in lurex",
                "two-tone trench cape with detachable hood",
                "frosted lilac shacket with pearl snaps",
                "striped rugby crop top with velvet appliqué",
                "sandstone safari blouse with rolled sleeves",
                "papaya silk bomber with embroidered koi",
            ],
            "bottoms": [
                "high-rise cigarette pants with ankle slits",
                "paperbag waist shorts in caramel leather",
                "pleated tennis skirt with satin stripes",
                "distressed straight-leg denim with patchwork",
                "faux leather mini skirt with cargo pockets",
                "ultra-wide satin trousers pooling at the heel",
                "bias-cut midi skirt with sheen",
                "holographic biker shorts",
                "micro-pleated skort with scalloped hem",
                "tailored culottes with contrast piping",
                "ombre joggers with zipped cuffs",
                "mesh-paneled leggings",
                "cargo parachute pants with cinched hems",
                "wrap sarong skirt in watercolor chiffon",
                "striped palazzo pants",
            ],
            "footwear": [
                "white platform sneakers",
                "lacquered ankle-strap heels",
                "chunky lug-soled loafers",
                "translucent wedge sandals",
                "chrome cap-toe booties",
                "ombre roller skates",
                "patent combat boots",
                "mesh-paneled trainers",
                "pointed slingback flats",
                "denim-wrapped stilettos",
            ],
            "accessories": [
                "mirrored visor shades",
                "chain-link belt with crystal charms",
                "layered enamel bangles",
                "micro saddle bag",
                "wide suede obi belt",
                "pearlescent bucket hat",
                "fingerless leather gloves",
                "oversized hoop earrings",
                "silk bandana tied at the throat",
                "iridescent fanny pack",
                "mesh tote with vinyl trim",
                "layered padlock necklaces",
                "crystal harness suspenders",
                "metallic arm cuff",
                "colorblock knee socks",
            ],
            "target": 120,
        },
        {
            "prefix": "evening",
            "primary": [
                "liquid satin slip dress with cowl back",
                "draped velvet column gown with thigh reveal",
                "sequined blazer dress with sharp shoulders",
                "off-shoulder corset gown with boned bodice",
                "asymmetric chiffon gown with feather trim",
                "sheer-paneled catsuit with crystal lattice",
                "bias-cut metallic gown with puddle hem",
                "laser-cut leather midi dress with nude lining",
                "charcoal tuxedo dress with satin lapels",
                "mirror-tile mini dress with plunge neckline",
                "latex hourglass dress with molded cups",
                "tulle tea-length dress with corseted waist",
                "liquid lamé wrap gown",
                "mesh illusion gown with star appliqués",
                "open-back jersey gown with cowl hood",
                "floor-length kimono dress with obi sash",
                "beaded flapper sheath with fringe",
                "fishtail sequin gown with horsehair hem",
                "strapless organza bubble dress",
                "ombre feathered cocktail dress",
            ],
            "bottoms": [
                "crystal-slit stocking leggings",
                "silk cigarette trousers",
                "corseted waist cincher",
                "mesh mermaid underskirt",
                "lace-paneled tights",
                "sheer train overlay",
                "satin opera stockings",
                "fishnet sparkle tights",
                "latex thigh-high leggings",
                "satin tap shorts",
            ],
            "footwear": [
                "stiletto sandals with ankle crystals",
                "pearl-encrusted pumps",
                "metallic cage heels",
                "vinyl thigh boots",
                "satin platform mules",
                "strappy gladiator heels",
                "clear lucite stilettos",
                "beaded kitten heels",
                "lace-up corset boots",
                "feathered slingbacks",
            ],
            "accessories": [
                "opera-length gloves",
                "rhinestone body chain",
                "swarovski choker",
                "mirror clutch",
                "feathered capelet",
                "gloss latex harness",
                "crystal ear cuff",
                "velvet ribbon anklet",
                "crushed velvet stole",
                "sheer elbow gloves",
                "diamond drop earrings",
                "chainmail bolero",
            ],
            "target": 110,
        },
        {
            "prefix": "lounge",
            "primary": [
                "cashmere lounge set with cropped hoodie",
                "sheer chiffon babydoll with satin cups",
                "ribbed knit bodysuit with high-cut hips",
                "silk pajama shirt half-buttoned",
                "plush velour robe with quilted collar",
                "mesh-paneled bralette with satin bows",
                "gauzy kimono with watercolor print",
                "lace chemise with scalloped hem",
                "modal tank dress with side ruching",
                "seamless bamboo camisole with lace inset",
                "cloud-soft fleece cardigan",
                "matte latex teddy with zipper front",
                "mesh longline bra with boning",
                "corseted sleep romper with silk ties",
                "stretch velvet playsuit with plunging back",
                "sheer tulle dressing gown with ostrich trim",
                "satin-trimmed bandeau lounge set",
                "jersey one-shoulder sleep dress",
                "marabou-trim baby tee",
                "wet-look wrap bodysuit",
            ],
            "bottoms": [
                "matching drawstring joggers",
                "high-waist tap shorts",
                "lace boyshorts",
                "silk palazzo lounge pants",
                "ribbed biker shorts",
                "mesh-paneled leggings",
                "feather-trim lounge pants",
                "satin bloomers",
                "sheer maxi skirt overlay",
                "backless briefs",
            ],
            "footwear": [
                "faux fur slide slippers",
                "satin ballet flats",
                "barefoot with ankle jewels",
                "cashmere socks",
                "feather pom slippers",
                "lace-up satin slippers",
                "sheer anklet socks",
                "velvet mary janes",
                "transparent mule heels",
                "pearl anklet sandals",
            ],
            "accessories": [
                "silk sleep mask",
                "satin scrunchie stack",
                "delicate waist chain",
                "puffball earmuffs",
                "body shimmer oil",
                "feather pen prop",
                "plush heart pillow",
                "long knit leg warmers",
                "lace garter belt",
                "oversized cardigan drape",
                "sheer opera gloves",
                "rhinestone anklet set",
            ],
            "target": 120,
        },
    ]

    generated: List[dict] = []

    for cat in categories:
        combos = itertools.product(
            cat["primary"], cat["bottoms"], cat["footwear"], cat["accessories"]
        )
        for combo_index, (primary, bottom, footwear, accessory) in enumerate(
            combos, start=1
        ):
            if combo_index > cat["target"]:
                break
            extras = [bottom, footwear, accessory]
            summary = (
                f"{primary}; paired with {bottom}, {footwear}, {accessory}"
            )
            generated.append(
                {
                    "id": f"clothes_{cat['prefix']}_{combo_index:03d}",
                    "primary": primary,
                    "extras": extras,
                    "summary": summary,
                }
            )

    return generated


def generate_poses() -> List[dict]:
    camera_options = [
        ("eye-level", "shot at eye level for direct connection"),
        ("slightly above", "captured from a gentle overhead tilt"),
        ("slightly below", "captured from just below the waistline"),
        ("low angle", "framed from a dramatic low angle"),
        ("top-down", "seen from directly overhead"),
        ("from behind", "viewed from behind with shoulders turning"),
        ("worm's-eye", "captured from the floor with limbs stretching upward"),
        ("bird's-eye", "filmed from a sweeping aerial perspective"),
        ("dutch tilt", "shot with a stylized dutch tilt"),
        ("fisheye", "captured with a fisheye exaggerating proportions"),
    ]

    framing_options = [
        "full",
        "three-quarter",
        "mid",
        "knee-up",
        "waist-up",
        "close",
        "detail",
        "panoramic",
    ]

    base_postures = [
        "standing contrapposto with one hip hiked and toe skimming the floor",
        "kneeling on both shins with torso twisted toward camera",
        "perched on a barstool with legs crossed and back arched",
        "lying supine with knees lifted and ankles crossed",
        "leaning against a wall with shoulder blades pressed flat",
        "crouching low with one palm on the ground for balance",
        "sitting on heels with back to the camera and head turned over shoulder",
        "arching over a yoga ball with fingertips grazing the floor",
        "straddling a chair backward with arms draped across the top",
        "lounging sideways on a chaise with one leg extended toward lens",
        "balancing on tiptoe with arms raised into a V",
        "walking mid-stride with coat billowing behind",
        "pressing chest against a countertop with spine curved",
        "bending at the waist with palms flat on thighs",
        "planking on forearms with gaze toward camera",
        "suspended in aerial silks with split legs",
        "crawling forward with back arched like a cat",
        "sitting cross-legged with elbows resting on knees",
        "lying prone with heels kicked up and chin in palms",
        "backbend bridge with hair cascading to the floor",
        "kneeling on a sofa back, leaning over the cushions",
        "standing with back to viewer, glancing over shoulder",
        "kneeling on a bed with spine arched and hands in hair",
        "leaning out of a car window with torso twisted",
        "sprawling across stairs with one leg on a higher step",
    ]

    modifiers = [
        "hands tracing along outer thighs",
        "fingers hooked into a waistband",
        "head tilted with eyes closed",
        "mouth parted mid-laugh",
        "hair gathered in one fist",
        "one hand pressing against the lens",
        "shoulders draped in a loose jacket",
        "ankles bound together with ribbon",
        "toes pointed dramatically",
        "back illuminated by rim light",
        "knees pressed together demurely",
        "legs spread wide with confidence",
        "spine lengthened into a graceful arc",
        "hips swung away from the camera",
        "torso twisting to reveal oblique lines",
        "chin tucked with gaze up through lashes",
        "arms folding across the chest",
        "wrists crossed behind the back",
        "hands planted wide for support",
        "palms framing the face",
    ]

    settings = [
        "on a mirrored studio floor",
        "across a velvet chaise",
        "inside a laundromat aisle",
        "beside a pool edge",
        "under a rainfall shower",
        "on a spiral staircase",
        "against a floor-to-ceiling window",
        "on a plush rug surrounded by candles",
        "within a graffiti alley",
        "over a kitchen island",
        "beneath hanging jungle plants",
        "within a boxing ring",
        "on a rooftop helipad",
        "inside a vintage car",
        "atop a grand piano",
        "inside a giant birdcage prop",
        "in a photo booth",
        "on a sandy shoreline",
        "at the edge of a diving board",
        "beneath a skylight",
        "in front of a neon wall",
        "inside a washer door frame",
        "halfway down a ladder",
        "on an ottoman piled with cushions",
        "against a ballet barre",
    ]

    generated: List[dict] = []

    combo_counter = 0
    for posture, modifier, setting in itertools.product(
        base_postures, modifiers, settings
    ):
        camera_angle, camera_detail = camera_options[combo_counter % len(camera_options)]
        framing = framing_options[combo_counter % len(framing_options)]
        combo_counter += 1
        if combo_counter > 230:
            break
        description = (
            f"{posture} with {modifier} {camera_detail}, set {setting}."
        )
        summary = short_phrase(
            f"{posture} {modifier} {setting}", max_words=16
        )
        generated.append(
            {
                "id": f"pose_dynamic_{combo_counter:03d}",
                "description": description,
                "camera_angle": camera_angle,
                "framing": framing,
                "summary": summary,
            }
        )

    return generated


def generate_styles() -> List[dict]:
    lighting_options = [
        "diffused skylight filtered through gauze panels",
        "sharp strobe with contrasting color gels",
        "candlelit ambience layered with string lights",
        "cinematic rim light with smoke haze",
        "noon sun bounced through mirrored panels",
        "bioluminescent glow simulated with LED strips",
        "retro tungsten practicals casting amber halos",
        "neon noir lighting with magenta key and cyan fill",
        "silvered moonlight with fog machine scatter",
        "studio flash with specular beauty dish",
        "vintage projector beams with dust motes",
        "cool aquarium caustics rippling across skin",
        "volumetric sunrise shafts through Venetian blinds",
        "polarized fashion lighting with hard shadow edges",
        "360-degree light ring for chrome reflections",
        "sodium vapor streetlights paired with blue bounce",
        "torchlight flicker with warm gradients",
        "ultraviolet blacklight with fluorescent accents",
        "glitter cannon sparkles catching spotlight beams",
        "backlit translucent scrim with silhouette highlight",
        "moving LED panels animating color waves",
        "laser fan lighting cutting through haze",
        "softboxes arranged in infinity cove",
        "ice blue soft light with warm kicker",
        "hard noon sunlight with polarizer",
    ]

    palette_options = [
        "sun-bleached corals and seafoam",
        "chrome silver, jet black, and opal",
        "molten gold with scarlet accents",
        "sherbet orange, blush pink, mint",
        "emerald, chartreuse, deep indigo",
        "desert neutrals with burnished copper",
        "high-contrast monochrome",
        "ultra-violet, magenta, cobalt",
        "rose quartz with smoky mauve",
        "citrus brights with white lacquer",
        "gunmetal and acid lime",
        "sepia film fade with teal highlights",
        "rainbow iridescence",
        "powder pastels with mirror silver",
        "ink black with holographic prisms",
        "aquamarine with coral punch",
        "amber whiskey tones",
        "lavender haze and midnight blue",
        "buttercream, latte, vanilla",
        "sapphire, garnet, pearl",
        "desaturated cyberpunk neons",
        "forest moss and bronze",
        "icy lilac with ultraviolet sparkle",
        "charcoal with copper patina",
        "opaline neutrals",
    ]

    background_options = [
        "infinite cyclorama with chrome cubes",
        "rooftop cityscape viewed through fisheye glass",
        "floating staircase inside a greenhouse",
        "desert salt flat stretching to horizon",
        "candy-striped diner booth",
        "submerged aquarium tunnel",
        "art deco hotel lobby",
        "warehouse runway with fog curtain",
        "mirror maze with repeating reflections",
        "tropical boardwalk at sunset",
        "retro laundromat aisle",
        "zero-gravity capsule interior",
        "opera stage with velvet drapes",
        "circular infinity pool",
        "marble bathhouse with steam",
        "Tokyo side street drenched in rain",
        "mountaintop helipad",
        "ornate library mezzanine",
        "futuristic subway platform",
        "liminal parking garage",
        "fisheye lens view of locker room",
        "giant candy factory conveyor",
        "glittering ice cave",
        "suspended glass skywalk",
        "retro arcade tunnel",
    ]

    perspective_notes = [
        "wide-lens exaggeration elongates limbs",
        "compressed telephoto framing sculpts curves",
        "tilted horizon adds kinetic motion",
        "diorama scale shift miniaturizes environment",
        "fisheye bend wraps scene into a sphere",
        "macro crop focuses on glossed textures",
        "architectural vanishing lines converge behind subject",
        "split-tone overlay emulates retro print",
        "anamorphic flare streaks along the frame",
        "360 panorama collage stitches perspectives",
        "Dutch tilt amplifies tension",
        "top-down orthographic flattening",
        "mirrored kaleidoscope reflections",
        "under-cranked motion blur smears highlights",
        "lensbaby selective focus blurs edges",
        "vogue collage layering cuts silhouettes",
        "double exposure silhouette reveals pattern",
        "split-diopter keeps foreground and background sharp",
        "circular vignette centers the pose",
        "overcranked shutter freezes cascading fabric",
    ]

    generated: List[dict] = []

    for index, (lighting, palette, background, perspective) in enumerate(
        itertools.product(lighting_options, palette_options, background_options, perspective_notes),
        start=1,
    ):
        if index > 220:
            break
        summary = short_phrase(
            f"{lighting} with {perspective} against {background}", max_words=22
        )
        generated.append(
            {
                "id": f"style_atlas_{index:03d}",
                "lighting": f"{lighting}; {perspective}",
                "palette": palette,
                "background": background,
                "summary": summary,
            }
        )

    return generated


def merge_json(path: Path, new_entries: Iterable[dict], key: str = "id") -> None:
    existing = json.loads(path.read_text())
    index = {item[key]: item for item in existing}
    for item in new_entries:
        index[item[key]] = item
    merged = sorted(index.values(), key=lambda item: item[key])
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    merge_json(DATA_DIR / "actions.json", generate_actions())
    merge_json(DATA_DIR / "clothes.json", generate_clothes())
    merge_json(DATA_DIR / "poses.json", generate_poses())
    merge_json(DATA_DIR / "styles.json", generate_styles())


if __name__ == "__main__":
    main()
