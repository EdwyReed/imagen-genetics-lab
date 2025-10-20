from __future__ import annotations

from imagen_pipeline.core.system_prompt import system_prompt_for


def test_system_prompt_merges_terms_and_rules():
    profile = {
        "id": "retro_glossy",
        "system_prompt": "Base profile",
        "required_terms": ["retro", "glossy finish"],
    }
    stage_terms = ["holographic", "retro"]
    tokens = [{"id": "glossy_micro", "label": "Glossy microtexture"}]
    rules = [
        {"id": "base_guideline", "label": "Base rule", "meta": {"prompt": "Base balance."}},
        {"id": "no_vignette", "label": "No vignette", "meta": {"hard": True}},
    ]
    bundle = system_prompt_for(
        profile,
        stage_required_terms=stage_terms,
        style_tokens=tokens,
        rules=rules,
        inject_rule_ids=["base_guideline"],
    )
    assert bundle.required_terms == ["retro", "glossy finish", "holographic"]
    assert "Glossy microtexture" in bundle.style_tokens
    assert "Base balance." in bundle.rule_injections
    assert any("No vignette" in rule for rule in bundle.rule_injections)
    assert "Base profile" in bundle.text
