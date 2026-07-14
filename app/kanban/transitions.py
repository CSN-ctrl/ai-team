"""Kanban state machine — governs valid task-status transitions.

Every status change must go through ``is_valid_transition()`` or the
``AsyncKanbanBoard.update_task()`` method which calls it internally.
"""

from __future__ import annotations

from typing import Dict, List

# ── Transition Map ────────────────────────────────────────────────────────────
#
# Each key is a source status; the list contains all allowed target statuses.
# Transitions to "cancelled" are permitted from backlog and ready only.

VALID_TRANSITIONS: Dict[str, List[str]] = {
    # forward flow
    "backlog": ["ready", "cancelled"],
    "ready": ["planning", "cancelled"],
    "planning": ["in_progress"],
    "in_progress": ["review", "done"],
    "review": ["qa", "in_progress"],          # in_progress ← QA/security failed
    "qa": ["security", "in_progress"],         # in_progress ← QA itself failed
    "security": ["approval", "in_progress"],   # in_progress ← security failed
    "approval": ["release", "in_progress"],    # in_progress ← rejected
    "release": ["done", "in_progress"],        # in_progress ← rollback
    # terminal
    "done": [],
    "cancelled": [],
}


def is_valid_transition(from_status: str, to_status: str) -> bool:
    """Return ``True`` if moving from *from_status* to *to_status* is allowed.

    Both arguments are compared case-sensitively against the lower-case status
    strings used in ``TaskStatus``.
    """
    allowed = VALID_TRANSITIONS.get(from_status)
    if allowed is None:
        return False
    return to_status in allowed


def get_allowed_transitions(from_status: str) -> list[str]:
    """Return the list of statuses reachable from *from_status*.

    Returns an empty list for terminal states (``done``, ``cancelled``) or
    unknown statuses.
    """
    return list(VALID_TRANSITIONS.get(from_status, []))
