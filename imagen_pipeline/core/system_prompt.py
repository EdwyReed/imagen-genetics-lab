"""System prompt assembly utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence


@dataclass
class SystemPromptBundle:
    """Final system prompt with metadata."""

    base_prompt: str
    required_terms: List[str] = field(default_factory=list)
    rule_injections: List[str] = field(default_factory=list)
    style_tokens: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines: List[str] = [self.base_prompt.strip()]
        if self.required_terms:
            lines.append("Required terms: " + ", ".join(self.required_terms))
        if self.style_tokens:
            lines.append("Style tokens: " + ", ".join(self.style_tokens))
        if self.rule_injections:
            lines.append("Rules: " + "; ".join(self.rule_injections))
        return "\n".join(line for line in lines if line)


def _unique(sequence: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in sequence:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _rule_text(rule: Mapping[str, object]) -> str:
    meta = rule.get("meta", {}) if isinstance(rule, Mapping) else {}
    if isinstance(meta, Mapping):
        prompt = meta.get("prompt") or meta.get("text")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        if meta.get("hard") and isinstance(rule.get("label"), str):
            return rule["label"].strip()
    label = rule.get("label") if isinstance(rule, Mapping) else None
    return str(label) if label else ""


def system_prompt_for(
    profile: Mapping[str, object],
    *,
    stage_required_terms: Iterable[str] = (),
    style_tokens: Iterable[Mapping[str, object]] | None = None,
    rules: Iterable[Mapping[str, object]] = (),
    inject_rule_ids: Sequence[str] | None = None,
) -> SystemPromptBundle:
    """Compose the final system prompt."""

    base_prompt = str(profile.get("system_prompt", "")).strip()
    profile_terms = profile.get("required_terms", []) or []
    combined_terms = _unique(list(profile_terms) + list(stage_required_terms))
    style_token_labels = _unique(
        str(token.get("label"))
        for token in (style_tokens or [])
        if isinstance(token, Mapping) and isinstance(token.get("label"), str)
    )
    inject_set = set(inject_rule_ids or [])
    rule_texts: List[str] = []
    for rule in rules:
        rule_id = str(rule.get("id", ""))
        meta = rule.get("meta", {}) if isinstance(rule, Mapping) else {}
        hard = bool(meta.get("hard")) if isinstance(meta, Mapping) else False
        if rule_id in inject_set or hard:
            text = _rule_text(rule)
            if text:
                rule_texts.append(text)
    rule_texts = _unique(rule_texts)
    return SystemPromptBundle(
        base_prompt=base_prompt,
        required_terms=combined_terms,
        rule_injections=rule_texts,
        style_tokens=style_token_labels,
    )


__all__ = ["SystemPromptBundle", "system_prompt_for"]
