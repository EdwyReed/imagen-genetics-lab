from pathlib import Path
import sys

import pytest

pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.prompting import append_required_terms


def test_append_required_terms_adds_fallback_sentence() -> None:
    text = "Retro girl tying her shoes."
    required = ["illustration", "watercolor", "glossy", "paper", "pastel"]

    result = append_required_terms(text, required)

    for term in required:
        assert term in result.lower()
    assert text.split()[0].lower() in result.lower()


def test_append_required_terms_trims_base_caption_for_max_words() -> None:
    text = " ".join([f"word{i}" for i in range(15)])
    required = ["illustration", "watercolor", "glossy", "paper", "pastel"]

    result = append_required_terms(text, required, max_words=20)

    assert len(result.split()) <= 20
    for term in required:
        assert term in result.lower()
