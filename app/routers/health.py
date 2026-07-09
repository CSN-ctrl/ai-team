"""Health check endpoint — /v1/health."""

import os
import time

from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1")

PROCESS_START = time.time()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Return API and model health status.

    No authentication required (internal health check).
    """
    # Gather model health from the ModelRouter if available
    models_health: list[dict] = []
    router_instance = getattr(request.app.state, "router", None)
    if router_instance is not None:
        try:
            raw = await router_instance.check_health()
            models_health = [
                {"model": name, "healthy": status}
                for name, status in raw.items()
            ]
        except Exception:
            models_health = [{"error": "health check failed"}]

    uptime_seconds = int(time.time() - PROCESS_START)
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"

    return {
        "status": "ok",
        "uptime": uptime_str,
        "uptime_seconds": uptime_seconds,
        "version": "0.1.0",
        "models": models_health,
        "hostname": os.uname().nodename,
    }
