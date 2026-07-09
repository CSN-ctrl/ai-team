"""
OpenHands Bridge Module

Connects the OpenClaw orchestrator to OpenHands (headless REST API)
for automated code task execution.

OpenHands is an open-source AI software engineering agent (formerly OpenDevin)
that exposes a headless REST API for programmatic task submission.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class OpenHandsError(Exception):
    """Base exception for all OpenHands bridge errors."""


class OpenHandsConnectionError(OpenHandsError):
    """Raised when the OpenHands server cannot be reached."""


class OpenHandsSessionError(OpenHandsError):
    """Raised when a session-level failure occurs."""


class OpenHandsTimeoutError(OpenHandsError):
    """Raised when a task exceeds its configured timeout."""


class OpenHandsTaskError(OpenHandsError):
    """Raised when task execution itself fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OpenHandsConfig:
    """Configuration for the OpenHands bridge."""

    base_url: str = "http://localhost:8000"
    api_key: str = ""
    sandbox_dir: str = "~/.openclaw/sandboxes/"
    repo_path: str = ""
    default_timeout: int = 300
    max_retries: int = 3

    @property
    def resolved_sandbox_dir(self) -> str:
        """Return sandbox_dir with ~ expanded to the user's home directory."""
        return os.path.expanduser(self.sandbox_dir)


# ---------------------------------------------------------------------------
# Model-config presets
# ---------------------------------------------------------------------------

ENGINEER_PRESETS: dict[str, dict[str, str]] = {
    "coding": {
        "model": "qwen3",
        "agent": "CodeActAgent",
        "description": "General-purpose coding agent using Qwen3",
    },
    "refactoring": {
        "model": "deepseek",
        "agent": "CodeActAgent",
        "description": "Refactoring-specialised agent using DeepSeek",
    },
}

# ---------------------------------------------------------------------------
# Internal HTTP client
# ---------------------------------------------------------------------------


class OpenHandsClient:
    """Low-level HTTP client for the OpenHands headless REST API."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # -- lazy lifecycle ---------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # -- core request helper ---------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the OpenHands API.

        Raises:
            OpenHandsConnectionError: If the server is unreachable.
            OpenHandsSessionError: If the server returns a non-2xx status.
        """
        client = await self._ensure_client()
        url = f"{self._base_url}{path}"

        try:
            response = await client.request(method, url, json=json)
        except httpx.ConnectError as exc:
            raise OpenHandsConnectionError(
                f"Unable to connect to OpenHands at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OpenHandsConnectionError(
                f"Request to OpenHands timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenHandsConnectionError(
                f"HTTP error communicating with OpenHands: {exc}"
            ) from exc

        if not response.is_success:
            body = ""
            try:
                body = response.text
            except Exception:
                pass
            raise OpenHandsSessionError(
                f"OpenHands API returned {response.status_code} for {method} {path}: {body}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise OpenHandsSessionError(
                f"Failed to parse OpenHands response as JSON: {exc}"
            ) from exc

    # -- API methods -----------------------------------------------------

    async def create_session(self, config: dict[str, Any]) -> dict[str, Any]:
        """POST /api/sessions — create a new agent session."""
        return await self._request("POST", "/api/sessions", json=config)

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """GET /api/sessions/{session_id} — get session status."""
        return await self._request("GET", f"/api/sessions/{session_id}")

    async def send_message(
        self, session_id: str, message: str
    ) -> dict[str, Any]:
        """POST /api/sessions/{session_id}/messages — send a task message."""
        return await self._request(
            "POST",
            f"/api/sessions/{session_id}/messages",
            json={"message": message},
        )

    async def get_session_diff(self, session_id: str) -> dict[str, Any]:
        """GET /api/sessions/{session_id}/diff — retrieve the git diff."""
        return await self._request("GET", f"/api/sessions/{session_id}/diff")

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """DELETE /api/sessions/{session_id} — clean up a session."""
        return await self._request(
            "DELETE", f"/api/sessions/{session_id}"
        )

    async def health(self) -> bool:
        """GET /api/health — lightweight connectivity check.

        Returns True when the server responds with HTTP 200.
        All exceptions (connection, auth, etc.) return False.
        """
        try:
            await self._request("GET", "/api/health")
            return True
        except OpenHandsError:
            return False


# ---------------------------------------------------------------------------
# Session handle
# ---------------------------------------------------------------------------


class OpenHandsSession:
    """Represents a single OpenHands agent session."""

    def __init__(
        self,
        session_id: str,
        config: OpenHandsConfig,
        client: OpenHandsClient,
    ) -> None:
        self._session_id = session_id
        self._config = config
        self._client = client
        self._created_at = datetime.now(timezone.utc)
        self._status: str = "created"
        self._closed = False

    # -- properties ------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def status(self) -> str:
        return self._status

    @property
    def created_at(self) -> datetime:
        return self._created_at

    # -- task lifecycle --------------------------------------------------

    async def submit_task(self, task_prompt: str) -> dict[str, Any]:
        """Send a coding task to the OpenHands session.

        Returns the session status response from the API.
        """
        if self._closed:
            raise OpenHandsSessionError(
                f"Session {self._session_id} is closed."
            )
        result = await self._client.send_message(self._session_id, task_prompt)
        self._status = "running"
        return result

    async def get_status(self) -> dict[str, Any]:
        """Poll the current session status.

        Returns a dict with keys:
            status, steps_completed, current_step, output
        """
        try:
            data = await self._client.get_session_status(self._session_id)
        except OpenHandsConnectionError:
            return {
                "status": "unknown",
                "steps_completed": 0,
                "current_step": "",
                "output": "",
            }

        raw_status = data.get("status", "unknown")
        self._status = raw_status

        return {
            "status": raw_status,
            "steps_completed": data.get("steps_completed", 0),
            "current_step": data.get("current_step", ""),
            "output": data.get("output", ""),
        }

    async def get_diff(self) -> str | None:
        """Retrieve the git diff / patch produced by the session.

        Returns a unified-diff string, or ``None`` if no changes exist
        or the endpoint is not yet available.
        """
        try:
            data = await self._client.get_session_diff(self._session_id)
        except OpenHandsSessionError:
            return None

        diff = data.get("diff") or data.get("patch")
        if diff is None:
            return None
        return str(diff)

    async def close(self) -> None:
        """Tear down the session on the server and mark it closed."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.delete_session(self._session_id)
        except OpenHandsError as exc:
            logger.warning(
                "Error closing session %s: %s", self._session_id, exc
            )
        self._status = "closed"

    async def wait_for_completion(
        self,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Block (async) until the session reaches a terminal status.

        Terminal statuses are ``completed``, ``failed``, ``stopped``,
        and ``error``.

        Args:
            poll_interval: Seconds between status polls (default 5.0).
            timeout: Maximum seconds to wait. Falls back to the config
                default when ``None``.

        Returns:
            The final status dict from ``get_status()``.

        Raises:
            OpenHandsTimeoutError: If the session does not finish
                within *timeout* seconds.
            OpenHandsTaskError: If the session enters a ``failed``
                or ``error`` terminal status.
        """
        deadline = time = timeout if timeout is not None else float(
            self._config.default_timeout
        )
        deadline = time
        started = datetime.now(timezone.utc)

        terminal_states = {"completed", "failed", "stopped", "error"}

        while True:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            if elapsed >= deadline:
                raise OpenHandsTimeoutError(
                    f"Session {self._session_id} did not complete within "
                    f"{deadline}s (elapsed={elapsed:.1f}s)."
                )

            status_data = await self.get_status()
            cur = status_data.get("status", "unknown")

            if cur in terminal_states:
                if cur in ("failed", "error"):
                    raise OpenHandsTaskError(
                        f"Session {self._session_id} terminated with "
                        f"status={cur!r}: {status_data.get('output', '')}"
                    )
                # completed / stopped
                return status_data

            await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Main bridge
# ---------------------------------------------------------------------------


class OpenHandsBridge:
    """High-level bridge that manages OpenHands agent sessions.

    Usage::

        config = OpenHandsConfig(base_url="http://localhost:8000")
        bridge = OpenHandsBridge(config)
        session = await bridge.start_session(
            session_id="task-42",
            task_prompt="Refactor the auth module to use async/await.",
        )
        final = await session.wait_for_completion()
        diff = await session.get_diff()
        await bridge.close_session(session.session_id)
    """

    def __init__(self, config: OpenHandsConfig) -> None:
        self._config = config
        self._client = OpenHandsClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.default_timeout,
        )
        self._sessions: dict[str, OpenHandsSession] = {}

    # -- session management ----------------------------------------------

    async def start_session(
        self,
        session_id: str,
        task_prompt: str,
        engineer_type: str = "coding",
    ) -> OpenHandsSession:
        """Create a new OpenHands agent session and submit the task.

        Args:
            session_id: Unique identifier for this session.
            task_prompt: The coding task description to execute.
            engineer_type: ``"coding"`` (Qwen3) or ``"refactoring"``
                (DeepSeek).

        Returns:
            An ``OpenHandsSession`` handle.

        Raises:
            OpenHandsConnectionError: If the server cannot be reached.
            ValueError: If *engineer_type* is unknown.
        """
        if engineer_type not in ENGINEER_PRESETS:
            valid = list(ENGINEER_PRESETS)
            raise ValueError(
                f"Unknown engineer_type={engineer_type!r}. "
                f"Valid options: {valid}"
            )

        preset = ENGINEER_PRESETS[engineer_type]

        # Build a configuration dict the OpenHands API understands.
        session_config: dict[str, Any] = {
            "session_id": session_id,
            **preset,
        }
        if self._config.repo_path:
            session_config["repo_path"] = self._config.repo_path

        await self._client.create_session(session_config)

        session = OpenHandsSession(
            session_id=session_id,
            config=self._config,
            client=self._client,
        )
        self._sessions[session_id] = session

        # Submit the initial task prompt.
        await session.submit_task(task_prompt)
        return session

    async def get_session(
        self, session_id: str
    ) -> OpenHandsSession | None:
        """Retrieve a previously started session by ID."""
        return self._sessions.get(session_id)

    async def list_sessions(self) -> list[str]:
        """Return the IDs of all tracked sessions."""
        return list(self._sessions)

    async def close_session(self, session_id: str) -> None:
        """Close and remove a single session."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        """Close and remove every tracked session."""
        for session_id in list(self._sessions):
            await self.close_session(session_id)

    # -- health ----------------------------------------------------------

    async def health_check(self) -> bool:
        """Check whether the OpenHands server is reachable.

        Returns ``True`` if ``GET /api/health`` responds with 2xx.
        """
        return await self._client.health()
