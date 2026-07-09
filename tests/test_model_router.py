"""Tests for the Model Router — registry, selector, queue, fallback, and facade.

All tests are self-contained; no real HTTP calls are made.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from app.router import ModelRouter, ModelRouterError
from app.router.fallback import FallbackHandler
from app.router.queue import RateLimitQueue
from app.router.registry import (
    ROUTING_TABLE,
    get_fallback,
    get_model_for_capability,
    list_available_models,
)
from app.router.selector import ModelSelector


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_get_model_for_capability_returns_primary_and_fallback(self) -> None:
        primary, fallback = get_model_for_capability("coding")
        assert primary is not None
        # coding has a fallback
        assert fallback is not None

    def test_get_model_for_capability_vision_no_fallback(self) -> None:
        primary, fallback = get_model_for_capability("vision")
        assert primary is not None
        assert fallback is None

    def test_get_model_for_capability_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model_for_capability("nonexistent")

    def test_get_fallback_known_model(self) -> None:
        fallback = get_fallback("qwen/qwen3-next-80b-a3b-instruct")
        assert fallback is not None

    def test_get_fallback_unknown_model(self) -> None:
        fallback = get_fallback("unknown/model")
        assert fallback is None

    def test_all_model_ids_have_provider_prefix(self) -> None:
        """Every model ID in the routing table must contain '/' (provider prefix)."""
        for capability, entry in ROUTING_TABLE.items():
            for key in ("primary", "fallback"):
                model_id = entry.get(key)
                if model_id is not None:
                    assert "/" in model_id, (
                        f"Model {model_id!r} for capability {capability!r} "
                        f"({key}) lacks provider prefix '/'"
                    )

    def test_list_available_models_returns_sorted(self) -> None:
        models = list_available_models()
        assert len(models) > 0
        assert models == sorted(models)

    def test_all_capabilities_have_primary(self) -> None:
        for capability, entry in ROUTING_TABLE.items():
            assert entry["primary"] is not None, (
                f"Capability {capability!r} has no primary model"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Selector
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelector:
    def test_select_returns_model_for_valid_capability(
        self, model_selector: ModelSelector
    ) -> None:
        model = model_selector.select("coding")
        assert isinstance(model, str)
        assert "/" in model

    def test_select_unknown_capability_raises_key_error(
        self, model_selector: ModelSelector
    ) -> None:
        with pytest.raises(KeyError, match="Unknown capability"):
            model_selector.select("nonexistent")

    def test_select_uses_fallback_when_primary_degraded(
        self, model_selector: ModelSelector
    ) -> None:
        primary, fallback = get_model_for_capability("coding")
        assert fallback is not None
        model_selector.mark_degraded(primary)
        selected = model_selector.select("coding")
        assert selected == fallback

    def test_select_raises_runtime_error_when_all_degraded(
        self, model_selector: ModelSelector
    ) -> None:
        primary, fallback = get_model_for_capability("coding")
        model_selector.mark_degraded(primary)
        if fallback:
            model_selector.mark_degraded(fallback)
        with pytest.raises(RuntimeError, match="All models degraded"):
            model_selector.select("coding")

    def test_mark_healthy_restores_model(
        self, model_selector: ModelSelector
    ) -> None:
        primary, _ = get_model_for_capability("coding")
        model_selector.mark_degraded(primary)
        assert model_selector._is_degraded(primary)
        model_selector.mark_healthy(primary)
        assert not model_selector._is_degraded(primary)

    def test_get_health_status(self, model_selector: ModelSelector) -> None:
        status = model_selector.get_health_status()
        assert isinstance(status, dict)
        assert len(status) > 0
        for name, healthy in status.items():
            assert isinstance(name, str)
            assert isinstance(healthy, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Queue
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueue:
    async def test_acquire_returns_true(self, rate_limit_queue: RateLimitQueue) -> None:
        result = await rate_limit_queue.acquire("test-model")
        assert result is True

    async def test_acquire_release_cycle(
        self, rate_limit_queue: RateLimitQueue
    ) -> None:
        assert await rate_limit_queue.acquire("test-model") is True
        rate_limit_queue.release("test-model")
        # Give the async release task time to run
        await asyncio.sleep(0.05)
        assert await rate_limit_queue.acquire("test-model") is True

    async def test_global_rate_limit(self, rate_limit_queue: RateLimitQueue) -> None:
        """Fill the global window to capacity, then verify rejection."""
        # Acquire up to the limit across different models
        for i in range(40):
            assert await rate_limit_queue.acquire(f"model-{i}") is True
        # Next acquire should fail (global limit hit)
        assert await rate_limit_queue.acquire("another-model") is False

    async def test_per_model_rate_limit(self, rate_limit_queue: RateLimitQueue) -> None:
        """Fill a single model's window to capacity."""
        for _ in range(40):
            assert await rate_limit_queue.acquire("busy-model") is True
        assert await rate_limit_queue.acquire("busy-model") is False

    async def test_release_frees_slot(self, rate_limit_queue: RateLimitQueue) -> None:
        for _ in range(40):
            assert await rate_limit_queue.acquire("model-a") is True
        assert await rate_limit_queue.acquire("model-a") is False
        rate_limit_queue.release("model-a")
        await asyncio.sleep(0.05)
        assert await rate_limit_queue.acquire("model-a") is True

    async def test_get_usage_stats_structure(
        self, rate_limit_queue: RateLimitQueue
    ) -> None:
        await rate_limit_queue.acquire("stats-model")
        stats = await rate_limit_queue.get_usage_stats()
        assert "global" in stats
        assert "models" in stats
        assert "current_rpm" in stats["global"]
        assert "limit" in stats["global"]
        assert "available" in stats["global"]
        assert stats["global"]["current_rpm"] >= 1
        assert "stats-model" in stats["models"]

    async def test_release_after_failed_acquire(
        self, rate_limit_queue: RateLimitQueue
    ) -> None:
        """Release should not error when called without a prior acquire."""
        rate_limit_queue.release("never-acquired")
        await asyncio.sleep(0.05)
        # Should still be able to acquire
        assert await rate_limit_queue.acquire("never-acquired") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback
# ═══════════════════════════════════════════════════════════════════════════════


class FailingLLMClient:
    """LLM client that always raises an exception."""

    def __init__(self, exc_class: type[Exception] = Exception) -> None:
        self._exc_class = exc_class

    async def chat_completion(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        raise self._exc_class(f"Simulated failure for {model}")


class TestFallback:
    async def test_execute_with_fallback_success(
        self,
        fallback_handler: FallbackHandler,
        dummy_llm_client: object,
    ) -> None:
        result = await fallback_handler.execute_with_fallback(
            capability="coding",
            messages=[{"role": "user", "content": "hi"}],
            llm_client=dummy_llm_client,
        )
        assert isinstance(result, dict)
        assert result["model"] is not None

    async def test_execute_with_fallback_all_fail_raises(
        self, fallback_handler: FallbackHandler
    ) -> None:
        client = FailingLLMClient()
        with pytest.raises(ModelRouterError) as exc_info:
            await fallback_handler.execute_with_fallback(
                capability="coding",
                messages=[{"role": "user", "content": "hi"}],
                llm_client=client,
            )
        assert exc_info.value.capability == "coding"
        assert len(exc_info.value.models_tried) > 0

    async def test_fallback_retryable_error_retries(
        self, fallback_handler: FallbackHandler
    ) -> None:
        """RateLimitError is retryable; the handler should attempt retries."""
        from app.llm.client import RateLimitError

        client = FailingLLMClient(exc_class=RateLimitError)
        with pytest.raises(ModelRouterError):
            await fallback_handler.execute_with_fallback(
                capability="coding",
                messages=[{"role": "user", "content": "hi"}],
                llm_client=client,
            )

    async def test_fallback_unknown_capability(
        self, fallback_handler: FallbackHandler, dummy_llm_client: object
    ) -> None:
        with pytest.raises(KeyError):
            await fallback_handler.execute_with_fallback(
                capability="nonexistent",
                messages=[{"role": "user", "content": "hi"}],
                llm_client=dummy_llm_client,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Facade (ModelRouter)
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelRouterFacade:
    async def test_route_success(self, model_router: ModelRouter) -> None:
        result = await model_router.route(
            "coding",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, dict)
        assert "choices" in result

    async def test_route_unknown_capability(self, model_router: ModelRouter) -> None:
        with pytest.raises(ModelRouterError):
            await model_router.route(
                "nonexistent",
                messages=[{"role": "user", "content": "Hello"}],
            )

    async def test_check_health(self, model_router: ModelRouter) -> None:
        health = await model_router.check_health()
        assert isinstance(health, dict)

    async def test_selector_property(self, model_router: ModelRouter) -> None:
        assert isinstance(model_router.selector, ModelSelector)

    async def test_queue_property(self, model_router: ModelRouter) -> None:
        assert isinstance(model_router.queue, RateLimitQueue)

    async def test_fallback_property(self, model_router: ModelRouter) -> None:
        assert isinstance(model_router.fallback, FallbackHandler)

    async def test_route_with_rate_limit_hit_triggers_fallback(
        self, model_router: ModelRouter
    ) -> None:
        """When rate limit is hit, the router should fall through to fallback."""
        # Fill the queue for a specific model
        for _ in range(40):
            await model_router.queue.acquire("qwen/qwen3-next-80b-a3b-instruct")
        # The next route for 'coding' should hit rate limit and use fallback
        result = await model_router.route(
            "coding",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, dict)