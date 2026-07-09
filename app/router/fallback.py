"""
Fallback handler — tries the primary model, cascades to fallback on
failure, and raises ``ModelRouterError`` when all attempts are exhausted.
"""

import asyncio
import logging
from typing import Optional

from app.router.registry import get_model_for_capability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelRouterError(Exception):
    """Raised when the router cannot fulfil a request after all fallbacks."""

    def __init__(
        self,
        message: str,
        capability: str,
        models_tried: list[str],
        original_exception: Optional[Exception] = None,
    ) -> None:
        self.capability = capability
        self.models_tried = models_tried
        self.original_exception = original_exception
        super().__init__(message)


# ---------------------------------------------------------------------------
# Fallback handler
# ---------------------------------------------------------------------------

_DEFAULT_BASE_DELAY_S: float = 1.0
_DEFAULT_MAX_RETRIES: int = 3


class FallbackHandler:
    """Executes an LLM request with automatic fallback across models.

    The handler will attempt the primary model for the given capability.
    If the call fails with a retryable error (HTTP 429, timeout, network
    error) it waits with exponential backoff and then tries the fallback
    model.  Up to *max_retries* attempts are made across all models in
    the fallback chain.
    """

    def __init__(
        self,
        base_delay: float = _DEFAULT_BASE_DELAY_S,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._base_delay = base_delay
        self._max_retries = max_retries

    async def execute_with_fallback(
        self,
        capability: str,
        messages: list[dict],
        llm_client,  # duck-typed: needs chat_completion(model, messages, ...)
        max_retries: Optional[int] = None,
    ) -> dict:
        """Attempt the LLM call, falling back through the model chain.

        Parameters
        ----------
        capability:
            The task capability used to look up the routing table.
        messages:
            The conversation / prompt messages.
        llm_client:
            An object with an ``async chat_completion(model, messages,
            ...)`` method that returns a dict (parsed JSON response).
            It may raise ``RateLimitError``, ``ModelTimeoutError``,
            or ``ModelError``.
        max_retries:
            Override the instance-level default.

        Returns
        -------
        The raw LLM response dict.

        Raises
        ------
        ModelRouterError
            When every model in the chain has been exhausted.
        """
        retries = self._max_retries if max_retries is None else max_retries
        primary, fallback = get_model_for_capability(capability)
        models_to_try: list[str | None] = [primary, fallback]

        last_exception: Optional[Exception] = None
        models_tried: list[str] = []
        attempt = 0

        for model in models_to_try:
            if model is None:
                continue

            for attempt_in_this_model in range(retries):
                attempt += 1
                try:
                    response = await llm_client.chat_completion(
                        model=model,
                        messages=messages,
                    )
                    return response
                except Exception as exc:
                    last_exception = exc
                    models_tried.append(model)
                    logger.warning(
                        "Fallback attempt %d/%d with model %s failed: %s",
                        attempt,
                        retries * len([m for m in models_to_try if m]),
                        model,
                        exc,
                    )

                    if not self._is_retryable(exc):
                        # Non-retryable — break out of retries for this
                        # model and go to the next fallback.
                        break

                    # Exponential backoff
                    delay = self._base_delay * (2 ** (attempt - 1))
                    logger.info("Backoff %.1fs before retry", delay)
                    await asyncio.sleep(delay)

        raise ModelRouterError(
            message=(
                f"All models failed for capability {capability!r} "
                f"after {attempt} attempt(s). "
                f"Models tried: {models_tried}"
            ),
            capability=capability,
            models_tried=models_tried,
            original_exception=last_exception,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return ``True`` if *exc* is a transient error worth retrying."""
        exc_name = type(exc).__name__
        return exc_name in {
            "RateLimitError",
            "ModelTimeoutError",
        }
