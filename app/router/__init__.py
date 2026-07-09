"""
Model Router — the main facade for dynamic model selection, rate-limit
management, and fallback orchestration for the OpenClaw CEO server.

Usage::

    from app.router import ModelRouter
    from app.llm.client import NIMClient

    client = NIMClient(api_key="...")
    router = ModelRouter(llm_client=client, rpm_limit=40)

    response = await router.route("coding", messages=[...])
"""

import logging
from typing import Optional

from app.router.selector import ModelSelector
from app.router.queue import RateLimitQueue, DEFAULT_RPM_LIMIT
from app.router.fallback import FallbackHandler, ModelRouterError

logger = logging.getLogger(__name__)


class ModelRouter:
    """Facade combining model selection, rate-limit tracking, and fallback.

    Parameters
    ----------
    llm_client:
        An object with an ``async chat_completion(model, messages)``
        method (e.g. ``NIMClient``).
    rpm_limit:
        Maximum requests per minute allowed per model.
        Defaults to ``DEFAULT_RPM_LIMIT`` (40).
    fallback_max_retries:
        Maximum retries per model in the fallback chain.
    """

    def __init__(
        self,
        llm_client,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        fallback_max_retries: int = 3,
    ) -> None:
        self._llm_client = llm_client
        self._selector = ModelSelector()
        self._queue = RateLimitQueue(rpm_limit=rpm_limit)
        self._fallback = FallbackHandler(max_retries=fallback_max_retries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(self, capability: str, messages: list[dict]) -> dict:
        """Route *messages* through the best available model for
        *capability*.

        Workflow:

        1. Select the best (non-degraded) model via ``ModelSelector``.
        2. Acquire a rate-limit slot via ``RateLimitQueue``.
        3. Call the LLM client.
        4. On failure, release the slot and trigger the fallback handler.
        5. On success, return the LLM response dict.

        Raises
        ------
        ModelRouterError
            When every model in the fallback chain is exhausted.
        """
        # Step 1: select model
        try:
            model = self._selector.select(capability)
        except (KeyError, RuntimeError) as exc:
            raise ModelRouterError(
                message=str(exc),
                capability=capability,
                models_tried=[],
                original_exception=exc,
            ) from exc

        logger.info("Routing capability=%s -> model=%s", capability, model)

        # Step 2: acquire rate-limit slot
        allowed = await self._queue.acquire(model)
        if not allowed:
            logger.warning("Rate limit hit for %s; releasing slot", model)
            self._queue.release(model)
            # Fall through to fallback handler rather than silently
            # failing — the fallback may try a different model.
            return await self._fallback.execute_with_fallback(
                capability=capability,
                messages=messages,
                llm_client=self._llm_client,
            )

        # Step 3: attempt the primary call
        try:
            response = await self._llm_client.chat_completion(
                model=model,
                messages=messages,
            )
            return response
        except Exception as exc:
            logger.warning(
                "Primary model %s failed for %s: %s",
                model,
                capability,
                exc,
            )
            # Release the rate-limit slot so fallback attempts (and
            # other requests) can use it.
            self._queue.release(model)
            self._selector.mark_degraded(model)

            # Step 4: fallback
            return await self._fallback.execute_with_fallback(
                capability=capability,
                messages=messages,
                llm_client=self._llm_client,
            )

    async def check_health(self) -> dict:
        """Return health status for all known models.

        Output::

            {
                "<model_name>": <bool>,
                ...
            }
        """
        return self._selector.get_health_status()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def selector(self) -> ModelSelector:
        """The underlying ``ModelSelector`` instance."""
        return self._selector

    @property
    def queue(self) -> RateLimitQueue:
        """The underlying ``RateLimitQueue`` instance."""
        return self._queue

    @property
    def fallback(self) -> FallbackHandler:
        """The underlying ``FallbackHandler`` instance."""
        return self._fallback


__all__ = [
    "ModelRouter",
    "ModelRouterError",
]
