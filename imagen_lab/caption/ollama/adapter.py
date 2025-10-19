from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, MutableMapping

import requests


class OllamaResponseError(RuntimeError):
    """Raised when the Ollama API returns an unexpected response."""


class OllamaTimeoutError(OllamaResponseError):
    """Raised when the Ollama API request times out."""


def _normalise_text(text: str) -> str:
    """Collapse whitespace and fix duplicated periods."""

    collapsed = " ".join(text.split())
    return collapsed.replace("..", ".")


@dataclass
class OllamaClient:
    """Thin wrapper around the Ollama `/api/generate` endpoint."""

    base_url: str
    model: str
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/") or self.base_url

    def _request(
        self,
        payload: Mapping[str, object] | MutableMapping[str, object],
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> requests.Response:
        prompt = system_prompt.strip() + "\n\n" + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": 1.05,
            },
        }
        if seed is not None:
            body["options"]["seed"] = int(seed)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=body,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise OllamaTimeoutError("Timed out waiting for Ollama response") from exc
        except requests.RequestException as exc:
            raise OllamaResponseError("Failed to reach Ollama endpoint") from exc

        return response

    def generate(
        self,
        system_prompt: str,
        payload: Mapping[str, object] | MutableMapping[str, object],
        *,
        temperature: float = 0.55,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> str:
        response = self._request(
            payload,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.text.strip() or f"HTTP {response.status_code}"
            raise OllamaResponseError(f"Ollama error: {message}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Invalid JSON payload from Ollama") from exc

        text = data.get("response", "") if isinstance(data, dict) else ""
        if not isinstance(text, str):
            raise OllamaResponseError("Unexpected response format from Ollama")

        return _normalise_text(text)
