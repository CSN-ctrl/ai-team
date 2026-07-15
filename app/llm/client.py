"""
OpenAI-compatible HTTP client for the NVIDIA NIM API.

Wraps ``httpx.AsyncClient`` with model-specific error types and
streaming support.  All network calls are async.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Raised on HTTP 429 — rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class ModelTimeoutError(Exception):
    """Raised when the upstream request times out."""

    def __init__(self, message: str = "Model request timed out") -> None:
        super().__init__(message)


class ModelError(Exception):
    """Raised on non-retryable API errors (HTTP 4xx, 5xx)."""

    def __init__(self, message: str, status_code: int, body: Optional[dict] = None) -> None:
        self.status_code = status_code
        self.body = body or {}
        super().__init__(message)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_TIMEOUT_S: float = 25.0
_DEFAULT_MAX_TOKENS: int = 4096
_DEFAULT_TEMPERATURE: float = 0.7


class NIMClient:
    """Async HTTP client for the NVIDIA NIM OpenAI-compatible API.

    Parameters
    ----------
    api_key:
        NVIDIA API key (Bearer token).
    base_url:
        Base URL for the NIM API.  Defaults to the NVIDIA-hosted endpoint.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    # ------------------------------------------------------------------
    # Chat completion (non-streaming)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
    ) -> dict:
        """Send a chat completion request.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"qwen3-coder"``).
        messages:
            List of message dicts (``{"role": ..., "content": ...}``).
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens in the response.
        stream:
            If ``True``, the method still returns a dict but internally
            consumes the full SSE stream.  Prefer
            ``chat_completion_stream`` for true streaming.

        Returns
        -------
        Parsed JSON response dict (see OpenAI chat completion schema).

        Raises
        ------
        RateLimitError
            On HTTP 429.
        ModelTimeoutError
            On timeout.
        ModelError
            On other HTTP errors.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"Request to model {model!r} timed out"
            ) from exc
        except httpx.ConnectError as exc:
            raise ModelError(
                message=f"Connection error for model {model!r}: {exc}",
                status_code=0,
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(
                message=f"HTTP error for model {model!r}: {exc}",
                status_code=getattr(exc, "status_code", 0) or 0,
            ) from exc

        if response.status_code == 429:
            retry_after = _parse_retry_after(response)
            raise RateLimitError(
                message=f"Rate limit exceeded for model {model!r}",
                retry_after=retry_after,
            )

        if response.is_error:
            body = _safe_json(response)
            raise ModelError(
                message=(
                    f"Model {model!r} returned HTTP {response.status_code}: "
                    f"{body.get('error', {}).get('message', response.text)}"
                ),
                status_code=response.status_code,
                body=body,
            )

        return response.json()

    # ------------------------------------------------------------------
    # Chat completion (streaming)
    # ------------------------------------------------------------------

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion deltas via SSE.

        Yields content delta strings as they arrive from the API.

        Raises the same exceptions as ``chat_completion``.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response)
                    raise RateLimitError(
                        message=f"Rate limit exceeded for model {model!r}",
                        retry_after=retry_after,
                    )

                if response.is_error:
                    body_text = await response.aread()
                    body = _safe_json_from_bytes(body_text)
                    raise ModelError(
                        message=(
                            f"Model {model!r} returned HTTP {response.status_code}: "
                            f"{body.get('error', {}).get('message', body_text.decode(errors='replace'))}"
                        ),
                        status_code=response.status_code,
                        body=body,
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed SSE chunk: %s", data_str)
                        continue

                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta

        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"Streaming request to model {model!r} timed out"
            ) from exc
        except httpx.ConnectError as exc:
            raise ModelError(
                message=f"Connection error for model {model!r}: {exc}",
                status_code=0,
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(
                message=f"HTTP error for model {model!r}: {exc}",
                status_code=getattr(exc, "status_code", 0) or 0,
            ) from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self, model: Optional[str] = None) -> bool:
        """Verify that the NIM endpoint is reachable.

        If *model* is provided, the check filters the model list for
        that specific model.  Otherwise it ensures the endpoint responds
        at all.

        Returns ``True`` if reachable, ``False`` otherwise.
        """
        try:
            response = await self._client.get("/models")
            if response.is_error:
                logger.warning("Health check failed: HTTP %d", response.status_code)
                return False

            if model is not None:
                data = response.json()
                models = data.get("data", [])
                return any(m.get("id") == model for m in models)

            return True
        except httpx.HTTPError as exc:
            logger.warning("Health check connection error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    """Extract ``Retry-After`` header as a float, if present."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _safe_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return {}


def _safe_json_from_bytes(data: bytes) -> dict:
    try:
        return json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return {}
