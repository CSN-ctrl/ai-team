"""Tests for the Kanban state machine (transitions module).

Covers every valid/invalid transition, terminal states, and the
``get_allowed_transitions`` helper.
"""

from __future__ import annotations

import pytest

from app.kanban.transitions import (
    VALID_TRANSITIONS,
    get_allowed_transitions,
    is_valid_transition,
)

# ── All valid transitions (explicitly enumerated) ────────────────────────────

VALID_CASES = [
    ("backlog", "ready"),
    ("backlog", "cancelled"),
    ("ready", "planning"),
    ("ready", "cancelled"),
    ("planning", "in_progress"),
    ("in_progress", "review"),
    ("review", "qa"),
    ("review", "in_progress"),
    ("qa", "security"),
    ("qa", "in_progress"),
    ("security", "approval"),
    ("security", "in_progress"),
    ("approval", "release"),
    ("approval", "in_progress"),
    ("release", "done"),
    ("release", "in_progress"),
]


@pytest.mark.parametrize("from_status,to_status", VALID_CASES)
def test_valid_transition(from_status: str, to_status: str) -> None:
    assert is_valid_transition(from_status, to_status)


# ── Invalid transitions (sampled exhaustively) ───────────────────────────────

INVALID_CASES = [
    # backward skips
    ("in_progress", "backlog"),
    ("review", "ready"),
    ("qa", "planning"),
    ("security", "review"),
    ("approval", "qa"),
    ("release", "security"),
    # skip forward
    ("backlog", "in_progress"),
    ("ready", "review"),
    ("planning", "qa"),
    ("in_progress", "approval"),
    ("review", "release"),
    ("qa", "done"),
    # self-transitions
    ("backlog", "backlog"),
    ("in_progress", "in_progress"),
    ("done", "done"),
    ("cancelled", "cancelled"),
    # cancelled from non-terminal
    ("in_progress", "cancelled"),
    ("review", "cancelled"),
    ("done", "cancelled"),
    # unknown statuses
    ("backlog", "unknown"),
    ("unknown", "backlog"),
    ("unknown", "unknown"),
]


@pytest.mark.parametrize("from_status,to_status", INVALID_CASES)
def test_invalid_transition(from_status: str, to_status: str) -> None:
    assert not is_valid_transition(from_status, to_status)


# ── Terminal states ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("terminal", ["done", "cancelled"])
def test_terminal_state_rejects_all(terminal: str) -> None:
    """Terminal states should reject every possible transition."""
    all_statuses = list(VALID_TRANSITIONS.keys())
    for target in all_statuses:
        assert not is_valid_transition(terminal, target), (
            f"Terminal state {terminal!r} should not allow transition to {target!r}"
        )


# ── get_allowed_transitions ──────────────────────────────────────────────────


def test_get_allowed_transitions_backlog() -> None:
    assert get_allowed_transitions("backlog") == ["ready", "cancelled"]


def test_get_allowed_transitions_ready() -> None:
    assert get_allowed_transitions("ready") == ["planning", "cancelled"]


def test_get_allowed_transitions_planning() -> None:
    assert get_allowed_transitions("planning") == ["in_progress"]


def test_get_allowed_transitions_in_progress() -> None:
    assert get_allowed_transitions("in_progress") == ["review"]


def test_get_allowed_transitions_review() -> None:
    assert get_allowed_transitions("review") == ["qa", "in_progress"]


def test_get_allowed_transitions_qa() -> None:
    assert get_allowed_transitions("qa") == ["security", "in_progress"]


def test_get_allowed_transitions_security() -> None:
    assert get_allowed_transitions("security") == ["approval", "in_progress"]


def test_get_allowed_transitions_approval() -> None:
    assert get_allowed_transitions("approval") == ["release", "in_progress"]


def test_get_allowed_transitions_release() -> None:
    assert get_allowed_transitions("release") == ["done", "in_progress"]


def test_get_allowed_transitions_done() -> None:
    assert get_allowed_transitions("done") == []


def test_get_allowed_transitions_cancelled() -> None:
    assert get_allowed_transitions("cancelled") == []


def test_get_allowed_transitions_unknown() -> None:
    assert get_allowed_transitions("nonexistent") == []


# ── VALID_TRANSITIONS dict integrity ─────────────────────────────────────────


def test_all_statuses_have_entry() -> None:
    """Every status in the transition map should be a key."""
    from app.models.task import TaskStatus

    for status in TaskStatus:
        assert status.value in VALID_TRANSITIONS, (
            f"TaskStatus.{status.name} ({status.value!r}) missing from VALID_TRANSITIONS"
        )


def test_all_targets_are_valid_statuses() -> None:
    """Every target in the transition map should be a known status."""
    from app.models.task import TaskStatus

    valid_values = {s.value for s in TaskStatus}
    for source, targets in VALID_TRANSITIONS.items():
        for t in targets:
            assert t in valid_values, (
                f"Transition {source!r} -> {t!r}: {t!r} is not a valid TaskStatus"
            )


def test_no_duplicate_targets() -> None:
    for source, targets in VALID_TRANSITIONS.items():
        assert len(targets) == len(set(targets)), (
            f"Duplicate targets in transition from {source!r}"
        )