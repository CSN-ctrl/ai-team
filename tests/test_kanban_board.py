"""Tests for the AsyncKanbanBoard persistence layer.

All tests use an in-memory SQLite database via the ``kanban_board`` fixture.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.kanban.board import AsyncKanbanBoard
from app.models.approval import ApprovalRequest
from app.models.release import Release
from app.models.task import Epic, Goal, Task, TaskStatus


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(**overrides: object) -> Task:
    defaults: dict = {
        "id": "task-001",
        "epic_id": "epic-001",
        "title": "Implement login",
        "description": "OAuth2 login flow",
    }
    defaults.update(overrides)
    return Task(**defaults)


def _make_goal(**overrides: object) -> Goal:
    defaults: dict = {
        "id": "goal-001",
        "text": "Build authentication system",
    }
    defaults.update(overrides)
    return Goal(**defaults)


def _make_epic(**overrides: object) -> Epic:
    defaults: dict = {
        "id": "epic-001",
        "name": "Auth Epic",
        "description": "All auth-related work",
        "goal_id": "goal-001",
    }
    defaults.update(overrides)
    return Epic(**defaults)


# ── Task CRUD ────────────────────────────────────────────────────────────────


class TestTaskCRUD:
    async def test_create_task(self, kanban_board: AsyncKanbanBoard) -> None:
        task = _make_task()
        result = await kanban_board.create_task(task)
        assert result.id == task.id
        assert result.status == TaskStatus.BACKLOG
        assert result.created_at is not None
        assert result.updated_at is not None

    async def test_get_task(self, kanban_board: AsyncKanbanBoard) -> None:
        task = _make_task()
        await kanban_board.create_task(task)
        fetched = await kanban_board.get_task(task.id)
        assert fetched is not None
        assert fetched.id == task.id
        assert fetched.title == task.title

    async def test_get_task_not_found(self, kanban_board: AsyncKanbanBoard) -> None:
        fetched = await kanban_board.get_task("nonexistent")
        assert fetched is None

    async def test_list_tasks_empty(self, kanban_board: AsyncKanbanBoard) -> None:
        tasks = await kanban_board.list_tasks()
        assert tasks == []

    async def test_list_tasks_all(self, kanban_board: AsyncKanbanBoard) -> None:
        t1 = _make_task(id="t1", epic_id="e1", title="Task 1")
        t2 = _make_task(id="t2", epic_id="e1", title="Task 2")
        await kanban_board.create_task(t1)
        await kanban_board.create_task(t2)
        tasks = await kanban_board.list_tasks()
        assert len(tasks) == 2

    async def test_list_tasks_filter_by_status(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        t1 = _make_task(id="t1", epic_id="e1", title="Backlog task")
        t2 = _make_task(id="t2", epic_id="e1", title="Ready task", status=TaskStatus.READY)
        await kanban_board.create_task(t1)
        await kanban_board.create_task(t2)
        backlog_tasks = await kanban_board.list_tasks(status="backlog")
        assert len(backlog_tasks) == 1
        assert backlog_tasks[0].id == "t1"

    async def test_list_tasks_filter_by_assignee(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        t1 = _make_task(id="t1", epic_id="e1", title="Mine", assignee="alice")
        t2 = _make_task(id="t2", epic_id="e1", title="Not mine", assignee="bob")
        await kanban_board.create_task(t1)
        await kanban_board.create_task(t2)
        alice_tasks = await kanban_board.list_tasks(assignee="alice")
        assert len(alice_tasks) == 1
        assert alice_tasks[0].id == "t1"


# ── Task Updates & Transitions ───────────────────────────────────────────────


class TestTaskUpdates:
    async def test_update_task_title(self, kanban_board: AsyncKanbanBoard) -> None:
        task = _make_task()
        await kanban_board.create_task(task)
        updated = await kanban_board.update_task(task.id, {"title": "New title"})
        assert updated is not None
        assert updated.title == "New title"

    async def test_update_task_valid_transition(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        task = _make_task()
        await kanban_board.create_task(task)
        updated = await kanban_board.update_task(task.id, {"status": "ready"})
        assert updated is not None
        assert updated.status == TaskStatus.READY

    async def test_update_task_invalid_transition(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        task = _make_task()
        await kanban_board.create_task(task)
        with pytest.raises(ValueError, match="Invalid status transition"):
            await kanban_board.update_task(task.id, {"status": "done"})

    async def test_update_task_not_found(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        result = await kanban_board.update_task("nonexistent", {"title": "Nope"})
        assert result is None

    async def test_update_task_empty_updates(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        task = _make_task()
        await kanban_board.create_task(task)
        result = await kanban_board.update_task(task.id, {})
        assert result is not None
        assert result.id == task.id


# ── Goal CRUD ────────────────────────────────────────────────────────────────


class TestGoalCRUD:
    async def test_create_goal(self, kanban_board: AsyncKanbanBoard) -> None:
        goal = _make_goal()
        result = await kanban_board.create_goal(goal)
        assert result.id == goal.id
        assert result.status == "pending"
        assert result.created_at is not None

    async def test_get_goal(self, kanban_board: AsyncKanbanBoard) -> None:
        goal = _make_goal()
        await kanban_board.create_goal(goal)
        fetched = await kanban_board.get_goal(goal.id)
        assert fetched is not None
        assert fetched.text == goal.text

    async def test_get_goal_not_found(self, kanban_board: AsyncKanbanBoard) -> None:
        fetched = await kanban_board.get_goal("nonexistent")
        assert fetched is None

    async def test_update_goal_status(self, kanban_board: AsyncKanbanBoard) -> None:
        goal = _make_goal()
        await kanban_board.create_goal(goal)
        await kanban_board.update_goal_status(goal.id, "active")
        fetched = await kanban_board.get_goal(goal.id)
        assert fetched is not None
        assert fetched.status == "active"


# ── Epic CRUD ─────────────────────────────────────────────────────────────────


class TestEpicCRUD:
    async def test_create_epic(self, kanban_board: AsyncKanbanBoard) -> None:
        epic = _make_epic()
        result = await kanban_board.create_epic(epic)
        assert result.id == epic.id
        assert result.created_at is not None

    async def test_get_epics(self, kanban_board: AsyncKanbanBoard) -> None:
        epic = _make_epic()
        await kanban_board.create_epic(epic)
        epics = await kanban_board.get_epics(goal_id="goal-001")
        assert len(epics) == 1
        assert epics[0].id == epic.id

    async def test_get_epics_empty(self, kanban_board: AsyncKanbanBoard) -> None:
        epics = await kanban_board.get_epics(goal_id="nonexistent")
        assert epics == []


# ── Approval CRUD ─────────────────────────────────────────────────────────────


class TestApprovalCRUD:
    async def test_create_approval(self, kanban_board: AsyncKanbanBoard) -> None:
        req = ApprovalRequest(
            id="apr-001",
            task_id="task-001",
            requested_by="alice",
            summary="Please approve this change",
        )
        result = await kanban_board.create_approval(req)
        assert result.id == "apr-001"
        assert result.status == "pending"
        assert result.created_at is not None

    async def test_list_approvals(self, kanban_board: AsyncKanbanBoard) -> None:
        req1 = ApprovalRequest(id="a1", task_id="t1", requested_by="alice", summary="S1")
        req2 = ApprovalRequest(id="a2", task_id="t2", requested_by="bob", summary="S2")
        await kanban_board.create_approval(req1)
        await kanban_board.create_approval(req2)
        all_reqs = await kanban_board.list_approvals()
        assert len(all_reqs) == 2

    async def test_list_approvals_filter_by_status(
        self, kanban_board: AsyncKanbanBoard
    ) -> None:
        req = ApprovalRequest(id="a1", task_id="t1", requested_by="alice", summary="S1")
        await kanban_board.create_approval(req)
        pending = await kanban_board.list_approvals(status="pending")
        assert len(pending) == 1
        resolved = await kanban_board.list_approvals(status="approved")
        assert resolved == []

    async def test_resolve_approval(self, kanban_board: AsyncKanbanBoard) -> None:
        req = ApprovalRequest(id="a1", task_id="t1", requested_by="alice", summary="S1")
        await kanban_board.create_approval(req)
        await kanban_board.resolve_approval("a1", "approved", comment="Looks good")
        approvals = await kanban_board.list_approvals(status="approved")
        assert len(approvals) == 1
        assert approvals[0].resolution_comment == "Looks good"
        assert approvals[0].resolved_at is not None


# ── Release CRUD ───────────────────────────────────────────────────────────


class TestReleaseCRUD:
    async def test_create_release(self, kanban_board: AsyncKanbanBoard) -> None:
        release = Release(id="rel-001", goal_id="goal-001")
        result = await kanban_board.create_release(release)
        assert result.id == "rel-001"
        assert result.gate == "G0"
        assert result.status == "active"
        assert result.created_at is not None

    async def test_get_release(self, kanban_board: AsyncKanbanBoard) -> None:
        release = Release(id="rel-001", goal_id="goal-001")
        await kanban_board.create_release(release)
        fetched = await kanban_board.get_release("goal-001")
        assert fetched is not None
        assert fetched.id == "rel-001"

    async def test_get_release_not_found(self, kanban_board: AsyncKanbanBoard) -> None:
        fetched = await kanban_board.get_release("nonexistent")
        assert fetched is None

    async def test_update_release_gate(self, kanban_board: AsyncKanbanBoard) -> None:
        release = Release(id="rel-001", goal_id="goal-001")
        await kanban_board.create_release(release)
        await kanban_board.update_release_gate("rel-001", "G1")
        fetched = await kanban_board.get_release("goal-001")
        assert fetched is not None
        assert fetched.gate == "G1"


# ── Integration: full lifecycle ───────────────────────────────────────────────


class TestLifecycle:
    async def test_goal_to_release_flow(self, kanban_board: AsyncKanbanBoard) -> None:
        """Create a goal, epic, task, advance it through the board, approve, release."""
        # Goal
        goal = _make_goal()
        await kanban_board.create_goal(goal)

        # Epic
        epic = _make_epic()
        await kanban_board.create_epic(epic)

        # Task
        task = _make_task()
        await kanban_board.create_task(task)

        # Advance through pipeline
        transitions = ["ready", "planning", "in_progress", "review", "qa", "security"]
        for status in transitions:
            updated = await kanban_board.update_task(task.id, {"status": status})
            assert updated is not None
            assert updated.status.value == status

        # Approval
        req = ApprovalRequest(
            id="apr-lifecycle",
            task_id=task.id,
            requested_by="alice",
            summary="Final approval",
        )
        await kanban_board.create_approval(req)
        await kanban_board.resolve_approval(req.id, "approved", comment="Ship it")

        # Advance to release
        await kanban_board.update_task(task.id, {"status": "approval"})
        await kanban_board.update_task(task.id, {"status": "release"})

        # Release
        release = Release(id="rel-lifecycle", goal_id=goal.id)
        await kanban_board.create_release(release)

        # Done
        await kanban_board.update_task(task.id, {"status": "done"})
        final = await kanban_board.get_task(task.id)
        assert final is not None
        assert final.status == TaskStatus.DONE