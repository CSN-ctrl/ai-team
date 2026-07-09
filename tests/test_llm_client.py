"""Tests for the NVIDIA NIM LLM client.

Uses monkeypatching to avoid real HTTP calls.  Tests cover initialization,
exception classes, health check, and error paths.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm.client import (
    ModelError,
    ModelTimeoutError,
    NIMClient,
    RateLimitError,
)


class TestExceptions:
    """Exception classes carry the right metadata."""

    def test_rate_limit_error_defaults(self) -> None:
        exc = RateLimitError()
        assert "Rate limit exceeded" in str(exc)
        assert exc.retry_after is None

    def test_rate_limit_error_with_retry_after(self) -> None:
        exc = RateLimitError(retry_after=30.0)
        assert exc.retry_after == 30.0

    def test_model_timeout_error(self) -> None:
        exc = ModelTimeoutError()
        assert "timed out" in str(exc)

    def test_model_error(self) -> None:
        exc = ModelError("Bad request", status_code=400, body={"error": "bad"})
        assert exc.status_code == 400
        assert exc.body == {"error": "bad"}

    def test_model_error_default_body(self) -> None:
        exc = ModelError("Server error", status_code=500)
        assert exc.body == {}


class TestNIMClientInit:
    def test_init_sets_base_url(self) -> None:
        client = NIMClient(api_key="test-key", base_url="https://example.com/v1")
        assert client._base_url == "https://example.com/v1"

    def test_init_strips_trailing_slash(self) -> None:
        client = NIMClient(api_key="test-key", base_url="https://example.com/v1/")
        assert client._base_url == "https://example.com/v1"

    def test_init_sets_auth_header(self) -> None:
        client = NIMClient(api_key="my-secret-key")
        assert client._client.headers["Authorization"] == "Bearer my-secret-key"

    def test_init_default_base_url(self) -> None:
        client = NIMClient(api_key="x")
        assert "nvidia.com" in client._base_url


class TestHealthCheck:
    async def test_health_check_returns_false_on_connection_error(
        self, nim_client: NIMClient
    ) -> None:
        """When the endpoint is unreachable, health_check should return False."""
        result = await nim_client.health_check()
        assert result is False

    async def test_health_check_returns_false_on_http_error(
        self, nim_client: NIMClient
    ) -> None:
        """Simulate an HTTP error response."""
        # Monkey-patch the client's transport to return an error
        async def mock_get(*args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(500, request=httpx.Request("GET", "http://localhost:0/models"))

        nim_client._client.get = mock_get  # type: ignore[method-assign]
        result = await nim_client.health_check()
        assert result is False


class TestChatCompletion:
    async def test_chat_completion_raises_model_error_on_connection_error(
        self, nim_client: NIMClient
    ) -> None:
        """When the endpoint is unreachable, chat_completion should raise ModelError."""
        with pytest.raises(ModelError) as exc_info:
            await nim_client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert "Connection error" in str(exc_info.value)
        assert exc_info.value.status_code == 0

    async def test_chat_completion_raises_model_error_on_http_500(
        self, nim_client: NIMClient
    ) -> None:
        """Simulate a 500 response."""

        async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                500,
                json={"error": {"message": "Internal error"}},
                request=httpx.Request("POST", "http://localhost:0/chat/completions"),
            )

        nim_client._client.post = mock_post  # noqa
        with pytest.raises(ModelError) as exc_info:
            await nim_client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.status_code == 500

    async def test_chat_completion_raises_rate_limit_error(
        self, nim_client: NIMClient
    ) -> None:
        """Simulate a 429 response."""

        async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "30"},
                request=httpx.Request("POST", "http://localhost:0/chat/completions"),
            )

        nim_client._client.post = mock_post  # noqa
        with pytest.raises(RateLimitError) as exc_info:
            await nim_client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.retry_after == 30.0


class TestClose:
    async def test_close_does_not_raise(self, nim_client: NIMClient) -> None:
        await nim_client.close()
        # Calling close again should be safe
        await nim_client.close()