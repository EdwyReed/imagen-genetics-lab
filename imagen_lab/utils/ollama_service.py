"""Helpers for managing the Ollama service lifecycle."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Optional


class OllamaServiceError(RuntimeError):
    """Raised when the Ollama service cannot be controlled automatically."""


@dataclass
class OllamaServiceManager:
    """Utility for starting and stopping the Ollama service on demand.

    The manager relies on ``ollama list`` to detect an active service and
    ``ollama serve`` to launch one when needed.  It keeps track of whether the
    current process owns the service in order to avoid terminating an instance
    started manually by the user.
    """

    warmup_seconds: float = 15.0
    check_interval: float = 1.0
    max_retries: int = 3
    manual_mode: bool = False

    def __post_init__(self) -> None:
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._started_by_manager = False

    @property
    def enabled(self) -> bool:
        """Whether automatic control is enabled."""

        return not self.manual_mode

    def __enter__(self) -> "OllamaServiceManager":
        if self.enabled:
            self.ensure_running()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.shutdown()

    def ensure_running(self) -> bool:
        """Ensure that the Ollama service is up.

        Returns ``True`` if the manager started a new service instance.  When in
        manual mode, the call is a no-op.
        """

        if not self.enabled:
            return False

        self._cleanup_finished_process()
        if self._service_is_available():
            self._started_by_manager = False
            return False

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._start_process()
                self._wait_until_ready()
                self._started_by_manager = True
                return True
            except Exception as exc:  # pragma: no cover - subprocess interaction
                last_error = exc
                self._terminate_process()
                if attempt < self.max_retries:
                    time.sleep(self.check_interval)

        raise OllamaServiceError("Failed to start Ollama service") from last_error

    def shutdown(self) -> None:
        """Terminate the service if it was started by the manager."""

        if not self.enabled:
            return

        self._cleanup_finished_process()
        if not self._started_by_manager:
            return

        self._terminate_process()
        self._started_by_manager = False

    # Internal helpers -------------------------------------------------

    def _cleanup_finished_process(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._started_by_manager = False

    def _service_is_available(self) -> bool:
        try:
            subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

    def _start_process(self) -> None:
        try:
            self._process = subprocess.Popen(  # pragma: no cover - subprocess interaction
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:  # pragma: no cover - subprocess interaction
            raise OllamaServiceError("'ollama' executable not found") from exc

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.warmup_seconds
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise OllamaServiceError(
                    f"ollama serve exited prematurely with code {self._process.returncode}"
                )
            if self._service_is_available():
                return
            time.sleep(self.check_interval)

        if self._service_is_available():
            return

        raise OllamaServiceError("Timed out waiting for Ollama service to become ready")

    def _terminate_process(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - subprocess interaction
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None
