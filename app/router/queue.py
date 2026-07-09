"""
Rate-limit-aware request queue — sliding-window RPM tracking per model.

Thread-safe (uses ``asyncio.Lock``) and non-blocking — ``acquire``
returns ``False`` immediately when the limit would be exceeded rather
than blocking the caller.
"""

import asyncio
import time
from collections import defaultdict
from typing import Optional

DEFAULT_RPM_LIMIT: int = 40


class RateLimitQueue:
    """Tracks requests-per-minute per model (and globally) via sliding windows.

    Each model has its own window of timestamps.  Additionally a *global*
    window tracks total requests across all models.  When *either* the
    per-model or the global limit would be exceeded, the call is rejected.

    The 40 RPM default is a *global* NVIDIA NIM free-tier constraint,
    shared across all models.  Per-model limits are set to the same value
    so users can tune individual models downward if one is too aggressive.
    """

    def __init__(self, rpm_limit: int = DEFAULT_RPM_LIMIT) -> None:
        self._rpm_limit = rpm_limit
        # model_name -> list[monotonic timestamp]
        self._windows: dict[str, list[float]] = defaultdict(list)
        # global window — one extra slot shared across all models
        self._global_window: list[float] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, model: str) -> bool:
        """Try to acquire a rate-limit slot for *model*.

        Checks both the per-model and the global window.  Returns ``True``
        if allowed (the request is recorded), or ``False`` immediately if
        *either* limit would be exceeded.
        """
        async with self._lock:
            self._evict_expired(model)
            win = self._windows[model]
            now = time.monotonic()
            # Prune global window
            cutoff = now - 60.0
            self._global_window = [t for t in self._global_window if t > cutoff]

            if len(win) >= self._rpm_limit:
                return False
            if len(self._global_window) >= self._rpm_limit:
                return False
            ts = time.monotonic()
            win.append(ts)
            self._global_window.append(ts)
            return True

    def release(self, model: str) -> None:
        """Remove the **last** recorded call for *model* (and from global).

        Call this after a failed request so the slot is reclaimed for
        future retries / fallbacks.
        """
        asyncio.ensure_future(self._async_release(model))

    async def get_usage_stats(self) -> dict:
        """Return current per-model and global usage statistics.

        Return shape::

            {
                "global": {"current_rpm": <int>, "limit": <int>, "available": <int>},
                "models": {
                    "<model>": {"current_rpm": <int>, "limit": <int>, "available": <int>},
                    ...
                },
            }
        """
        async with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self._global_window = [t for t in self._global_window if t > cutoff]

            model_stats: dict[str, dict] = {}
            for model in list(self._windows.keys()):
                self._evict_expired(model)
                win = self._windows.get(model)
                if win is None:
                    continue
                current = len(win)
                model_stats[model] = {
                    "current_rpm": current,
                    "limit": self._rpm_limit,
                    "available": max(0, self._rpm_limit - current),
                }

            return {
                "global": {
                    "current_rpm": len(self._global_window),
                    "limit": self._rpm_limit,
                    "available": max(0, self._rpm_limit - len(self._global_window)),
                },
                "models": model_stats,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self, model: str) -> None:
        """Remove timestamps older than 60 seconds for *model*."""
        cutoff = time.monotonic() - 60.0
        win = self._windows[model]
        self._windows[model] = [t for t in win if t > cutoff]

    async def _async_release(self, model: str) -> None:
        """Async version of ``release`` — pops the newest entry."""
        async with self._lock:
            self._evict_expired(model)
            win = self._windows.get(model)
            if win:
                win.pop()
                # Also pop from global window (oldest-first is fine since
                # we just need to free one global slot)
                if self._global_window:
                    self._global_window.pop()
            if win is not None and not win:
                del self._windows[model]
