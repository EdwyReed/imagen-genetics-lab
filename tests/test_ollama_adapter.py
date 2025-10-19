from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from imagen_lab.caption.ollama.adapter import (
    OllamaClient,
    OllamaResponseError,
    OllamaTimeoutError,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, body: dict[str, object] | None = None):
        self.status_code = status_code
        self._body = body or {"response": "  Hello   world..  "}
        self.text = json.dumps(self._body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict[str, object]:
        return self._body


def test_generate_returns_normalised_text() -> None:
    client = OllamaClient(base_url="http://localhost:11434/", model="caption")

    with patch("requests.post", return_value=_FakeResponse()) as mock_post:
        result = client.generate("Write", {"scene": "demo"})

    assert result == "Hello world."
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "caption"
    assert "\n\n" in payload["prompt"]
    assert payload["options"]["temperature"] == pytest.approx(0.55)


def test_generate_raises_on_timeout() -> None:
    client = OllamaClient(base_url="http://localhost", model="caption")

    with patch("requests.post", side_effect=requests.Timeout):
        with pytest.raises(OllamaTimeoutError):
            client.generate("Write", {})


def test_generate_raises_on_http_error() -> None:
    client = OllamaClient(base_url="http://localhost", model="caption")

    with patch("requests.post", return_value=_FakeResponse(status_code=500)):
        with pytest.raises(OllamaResponseError):
            client.generate("Write", {})


def test_generate_validates_json_response() -> None:
    client = OllamaClient(base_url="http://localhost", model="caption")

    bad_body = {"response": 123}
    with patch("requests.post", return_value=_FakeResponse(body=bad_body)):
        with pytest.raises(OllamaResponseError):
            client.generate("Write", {})
