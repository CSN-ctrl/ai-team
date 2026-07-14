"""Activity log router — exposes agent interaction events."""

from __future__ import annotations

from fastapi import APIRouter

from app.activity import get_activity_log

router = APIRouter(tags=["activity"])


@router.get("/v1/activity")
async def list_activity(limit: int = 50):
    log = get_activity_log()
    events = log.recent(limit=limit)
    return {
        "events": [
            {
                "kind": e.kind,
                "timestamp": e.timestamp,
                "actor": e.actor,
                "task_id": e.task_id,
                "task_title": e.task_title,
                "detail": e.detail,
                "from_status": e.from_status,
                "to_status": e.to_status,
            }
            for e in events
        ]
    }
