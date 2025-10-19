from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .catalog import Catalog


@dataclass
class StyleGuide:
    """Lightweight description of the active style catalog.

    The guide distils the free-form JSON descriptors that ship with each
    catalog into structured hints for the prompting layer.  This keeps the
    system prompt focused on the currently selected catalog instead of the
    historical jelly pin-up defaults.
    """

    brand: str
    purpose: str
    aesthetic: str
    mediums: List[str]
    palette_preference: str
    background: str
    glam_hint: str
    highlight_notes: List[str]
    gloss_notes: List[str]
    required_terms: List[str]

    @classmethod
    def from_catalog(
        cls, catalog: Catalog, fallback_terms: Sequence[str] = ()
    ) -> "StyleGuide":
        raw = catalog.to_dict()
        brand = str(raw.get("brand") or "Custom Style")
        purpose = str(raw.get("purpose") or "").strip()

        style_data = catalog.style_controller()
        if not isinstance(style_data, dict):
            style_data = {}

        aesthetic = str(style_data.get("aesthetic") or brand).strip()
        palette_preference = str(style_data.get("palette_preference") or "").strip()
        background = str(style_data.get("background_norm") or "").strip()
        glam_hint = str(style_data.get("glam_level_hint") or "").strip()

        mediums: List[str] = []
        if style_data.get("watercolor"):
            mediums.append("watercolor washes")
        if style_data.get("paper_texture"):
            mediums.append("paper texture")

        highlight_notes: List[str] = []
        highlights = style_data.get("highlights")
        if isinstance(highlights, dict):
            for key, value in highlights.items():
                if isinstance(value, str) and value:
                    highlight_notes.append(value)
                elif value:
                    highlight_notes.append(key.replace("_", " "))
        subsurface = style_data.get("subsurface_glow")
        if isinstance(subsurface, list):
            highlight_notes.extend(
                str(item) for item in subsurface if isinstance(item, str)
            )

        gloss_notes: List[str] = []
        gloss = style_data.get("gloss_pipeline")
        if isinstance(gloss, dict):
            gloss_notes.extend(
                str(value) for value in gloss.values() if isinstance(value, str)
            )

        required_terms = _derive_required_terms(
            fallback_terms,
            style_data,
            aesthetic,
            purpose,
            palette_preference,
            glam_hint,
        )

        return cls(
            brand=brand,
            purpose=purpose,
            aesthetic=aesthetic,
            mediums=mediums,
            palette_preference=palette_preference,
            background=background,
            glam_hint=glam_hint,
            highlight_notes=_unique_words(highlight_notes),
            gloss_notes=_unique_words(gloss_notes),
            required_terms=required_terms,
        )

    def context_lines(self) -> List[str]:
        """Return descriptive sentences for inclusion in the system prompt."""

        lines: List[str] = []
        if self.purpose:
            lines.append(self.purpose.rstrip(".") + ".")
        if self.mediums:
            lines.append("Medium cues: " + ", ".join(self.mediums) + ".")
        if self.gloss_notes:
            lines.append("Finish cues: " + ", ".join(self.gloss_notes) + ".")
        if self.highlight_notes:
            lines.append(
                "Highlight focus: " + ", ".join(self.highlight_notes) + "."
            )
        if self.background:
            lines.append("Typical backgrounds: " + self.background.rstrip(".") + ".")
        if self.glam_hint:
            lines.append("Overall vibe: " + self.glam_hint.rstrip(".") + ".")
        if self.palette_preference:
            lines.append(
                "Palette preference: " + self.palette_preference.rstrip(".") + "."
            )
        return lines


def _derive_required_terms(
    fallback_terms: Sequence[str],
    style_data: dict,
    aesthetic: str,
    purpose: str,
    palette_preference: str,
    glam_hint: str,
) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        clean = term.strip()
        if not clean:
            return
        lowered = clean.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        ordered.append(clean)

    for term in fallback_terms:
        add(term)

    joined = " ".join(
        [aesthetic, purpose, palette_preference, glam_hint]
    ).lower()

    add("illustration")
    if style_data.get("watercolor") or "watercolor" in joined:
        add("watercolor")
    if style_data.get("paper_texture") or "paper" in joined:
        add("paper")
    if style_data.get("gloss_pipeline") or "gloss" in joined:
        add("glossy")
    if "pastel" in joined:
        add("pastel")

    return ordered


def _unique_words(values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        lowered = clean.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(clean)
    return ordered
