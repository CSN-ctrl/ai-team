"""In-memory activity event log — tracks agent interactions and task lifecycle."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ActivityEvent:
    kind: str  # task_created, task_assigned, task_status, agent_busy, agent_idle, assigner_sweep
    timestamp: float = field(default_factory=time.time)
    actor: str = ""
    task_id: str = ""
    task_title: str = ""
    detail: str = ""
    from_status: str = ""
    to_status: str = ""


class ActivityLog:
    """Thread-safe ring buffer of recent activity events."""

    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque[ActivityEvent] = deque(maxlen=maxlen)

    def push(self, event: ActivityEvent) -> None:
        self._events.appendleft(event)

    def recent(self, limit: int = 50) -> list[ActivityEvent]:
        return list(self._events)[:limit]

    def by_task(self, task_id: str, limit: int = 20) -> list[ActivityEvent]:
        return [e for e in self._events if e.task_id == task_id][:limit]


# Module-level singleton
_activity_log: Optional[ActivityLog] = None


def get_activity_log() -> ActivityLog:
    global _activity_log
    if _activity_log is None:
        _activity_log = ActivityLog()
    return _activity_log


def emit(
    kind: str,
    actor: str = "",
    task_id: str = "",
    task_title: str = "",
    detail: str = "",
    from_status: str = "",
    to_status: str = "",
) -> None:
    get_activity_log().push(
        ActivityEvent(
            kind=kind,
            actor=actor,
            task_id=task_id,
            task_title=task_title,
            detail=detail,
            from_status=from_status,
            to_status=to_status,
        )
    )
