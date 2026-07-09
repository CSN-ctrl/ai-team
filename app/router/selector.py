"""
Model selector — picks the best available model for a capability.

Maintains an in-memory health cache.  Models marked as *degraded* are
skipped during selection until they are restored or the health-check
interval expires.
"""

import asyncio
import time
from typing import Optional

from app.router.registry import (
    ROUTING_TABLE,
    get_model_for_capability,
    list_available_models,
)

_DEFAULT_DEGRADED_TTL_S: float = 300.0  # 5 minutes


class ModelSelector:
    """Selects the healthiest available model for a given capability."""

    def __init__(
        self,
        health_check_interval: float = _DEFAULT_DEGRADED_TTL_S,
    ) -> None:
        self._health_check_interval = health_check_interval
        # model_name -> epoch when it was marked degraded (0 = healthy)
        self._degraded_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, capability: str) -> str:
        """Return the best available model name for *capability*.

        Raises ``KeyError`` if the capability is unknown, or
        ``RuntimeError`` if every candidate model for this capability
        is currently degraded.
        """
        if capability not in ROUTING_TABLE:
            raise KeyError(f"Unknown capability: {capability!r}")

        primary, fallback = get_model_for_capability(capability)

        for candidate in (primary, fallback):
            if candidate is None:
                continue
            if not self._is_degraded(candidate):
                return candidate

        raise RuntimeError(
            f"All models degraded for capability {capability!r} "
            f"(primary={primary!r}, fallback={fallback!r})"
        )

    def mark_degraded(self, model: str) -> None:
        """Temporarily mark *model* as degraded."""
        self._degraded_until[model] = time.monotonic() + self._health_check_interval

    def mark_healthy(self, model: str) -> None:
        """Restore *model* to healthy status."""
        self._degraded_until.pop(model, None)

    def get_health_status(self) -> dict[str, bool]:
        """Return ``{model_name: is_healthy}`` for every known model."""
        status: dict[str, bool] = {}
        for name in list_available_models():
            status[name] = not self._is_degraded(name)
        # Also include any models that aren't in the registry but were
        # marked degraded externally (e.g. ad-hoc models).
        for name in self._degraded_until:
            if name not in status:
                status[name] = not self._is_degraded(name)
        return status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_degraded(self, model: str) -> bool:
        expiry = self._degraded_until.get(model, 0.0)
        if expiry == 0.0:
            return False
        if time.monotonic() >= expiry:
            # TTL expired — auto-restore
            self._degraded_until.pop(model, None)
            return False
        return True
