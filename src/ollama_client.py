from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import requests


def _normalize_ollama_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        if scheme == "https":
            port = 443
        elif host in {"localhost", "127.0.0.1"}:
            port = 11434
        else:
            port = 80
    return f"{scheme}://{host}:{port}"


@contextmanager
def _ollama_session(url: str, timeout: int) -> Iterator[str]:
    base_url = _normalize_ollama_url(url)
    proc: Optional[subprocess.Popen] = None
    started = False
    try:
        requests.get(f"{base_url}/api/tags", timeout=3)
    except requests.RequestException:
        binary = shutil.which("ollama")
        if not binary:
            raise RuntimeError(
                "Ollama CLI not found. Install Ollama or configure an accessible Ollama host."
            )
        env = os.environ.copy()
        proc = subprocess.Popen(  # type: ignore[arg-type]
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        started = True
        deadline = time.time() + max(timeout, 10)
        while time.time() < deadline:
            try:
                requests.get(f"{base_url}/api/tags", timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.25)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError(f"Timed out waiting for Ollama serve to start at {base_url}")
    try:
        yield base_url
    finally:
        if started and proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _ensure_model_available(base_url: str, model: str, timeout: int) -> None:
    need_pull = False
    try:
        resp = requests.post(f"{base_url}/api/show", json={"name": model}, timeout=timeout)
        if resp.status_code == 404:
            need_pull = True
        else:
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                need_pull = True
    except (requests.RequestException, ValueError):
        need_pull = True

    if not need_pull:
        return

    try:
        resp = requests.post(
            f"{base_url}/api/pull",
            json={"name": model, "stream": False},
            timeout=max(timeout, 60),
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(
                f"Failed to pull Ollama model '{model}': {data.get('error') or data}"
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to pull Ollama model '{model}' from {base_url}") from exc


def _serialize_payload(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def ollama_generate(
    url: str,
    model: str,
    system_prompt: str,
    payload: Dict[str, object],
    temperature: float = 0.55,
    top_p: float = 0.9,
    timeout: int = 30,
    seed: Optional[int] = None,
) -> str:
    prompt = system_prompt.strip() + "\n\n" + _serialize_payload(payload)
    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": 1.05,
    }
    if seed is not None:
        options["seed"] = int(seed)

    with _ollama_session(url, timeout) as base_url:
        _ensure_model_available(base_url, model, timeout)
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "options": options,
                    "stream": False,
                    "raw": False,
                    "keep_alive": 0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama generation failed for model '{model}'") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned a non-JSON response") from exc

    text = data.get("response", "").strip()
    text = " ".join(text.split()).replace("..", ".")
    return text


def _extract_required_terms(payload: Dict[str, object]) -> List[str]:
    terms = payload.get("required_terms")
    if isinstance(terms, list):
        normalized: List[str] = []
        for term in terms:
            if isinstance(term, str):
                normalized.append(term)
        return normalized
    return []


def enforce_once(
    url: str,
    model: str,
    system_prompt: str,
    payload: Dict[str, object],
    base_caption: str,
    temperature: float = 0.5,
    seed: Optional[int] = None,
    timeout: Optional[int] = None,
) -> str:
    required_terms = _extract_required_terms(payload)
    missing = [term for term in required_terms if term.lower() not in base_caption.lower()]
    if not missing:
        return base_caption
    enforce_system = (
        system_prompt
        + "\n\nRewrite the caption naturally (18–60 words) and include the missing words: "
        + ", ".join(missing)
        + ". Keep it one or two sentences."
    )
    return ollama_generate(
        url=url,
        model=model,
        system_prompt=enforce_system,
        payload=payload,
        temperature=temperature,
        timeout=timeout if timeout is not None else 30,
        seed=seed,
    )


__all__ = ["ollama_generate", "enforce_once"]
