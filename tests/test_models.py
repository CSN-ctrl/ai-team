"""Tests for Pydantic model creation, defaults, and serialization.

Covers Task, Agent, ApprovalRequest, and Release models.
"""

from __future__ import annotations

from datetime import datetime

from app.models.agent import Agent
from app.models.approval import ApprovalRequest
from app.models.release import Release
from app.models.task import Epic, Goal, Task, TaskStatus


class TestTaskModel:
    def test_create_with_defaults(self) -> None:
        task = Task(id="t1", epic_id="e1", title="Test", description="Desc")
        assert task.status == TaskStatus.BACKLOG
        assert task.priority == 3
        assert task.assignee is None
        assert task.acceptance_criteria == []
        assert task.dependencies == []

    def test_create_with_explicit_status(self) -> None:
        task = Task(
            id="t1",
            epic_id="e1",
            title="Test",
            description="Desc",
            status=TaskStatus.IN_PROGRESS,
        )
        assert task.status == TaskStatus.IN_PROGRESS

    def test_create_with_assignee(self) -> None:
        task = Task(id="t1", epic_id="e1", title="Test", description="Desc", assignee="alice")
        assert task.assignee == "alice"

    def test_create_with_acceptance_criteria(self) -> None:
        task = Task(
            id="t1",
            epic_id="e1",
            title="Test",
            description="Desc",
            acceptance_criteria=["must pass tests", "must be documented"],
        )
        assert len(task.acceptance_criteria) == 2

    def test_serialization_round_trip(self) -> None:
        task = Task(
            id="t1",
            epic_id="e1",
            title="Test",
            description="Desc",
            status=TaskStatus.REVIEW,
            assignee="bob",
            priority=1,
            acceptance_criteria=["c1"],
            dependencies=["dep1"],
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 2),
        )
        data = task.model_dump()
        restored = Task(**data)
        assert restored.id == task.id
        assert restored.status == task.status
        assert restored.priority == task.priority
        assert restored.acceptance_criteria == task.acceptance_criteria
        assert restored.dependencies == task.dependencies
        assert restored.created_at == task.created_at
        assert restored.updated_at == task.updated_at

    def test_json_serialization(self) -> None:
        task = Task(id="t1", epic_id="e1", title="Test", description="Desc")
        json_str = task.model_dump_json()
        restored = Task.model_validate_json(json_str)
        assert restored.id == task.id
        assert restored.status == task.status


class TestEpicModel:
    def test_create_with_defaults(self) -> None:
        epic = Epic(id="e1", name="Epic 1", description="Desc", goal_id="g1")
        assert epic.created_at is None

    def test_serialization_round_trip(self) -> None:
        epic = Epic(
            id="e1",
            name="Epic 1",
            description="Desc",
            goal_id="g1",
            created_at=datetime(2026, 6, 1),
        )
        data = epic.model_dump()
        restored = Epic(**data)
        assert restored.id == epic.id
        assert restored.name == epic.name
        assert restored.created_at == epic.created_at


class TestGoalModel:
    def test_create_with_defaults(self) -> None:
        goal = Goal(id="g1", text="Build something")
        assert goal.status == "pending"
        assert goal.created_at is None
        assert goal.updated_at is None

    def test_create_with_explicit_status(self) -> None:
        goal = Goal(id="g1", text="Build", status="active")
        assert goal.status == "active"

    def test_serialization_round_trip(self) -> None:
        goal = Goal(
            id="g1",
            text="Build",
            status="completed",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 6, 1),
        )
        data = goal.model_dump()
        restored = Goal(**data)
        assert restored.text == goal.text
        assert restored.status == goal.status


class TestAgentModel:
    def test_create_with_defaults(self) -> None:
        agent = Agent(
            id="a1",
            name="coder-agent",
            role="developer",
            model="qwen3-coder",
            capabilities=["coding"],
        )
        assert agent.status == "idle"
        assert agent.current_task_id is None

    def test_create_with_busy_status(self) -> None:
        agent = Agent(
            id="a1",
            name="test",
            role="tester",
            model="nemotron-super",
            capabilities=["qa", "security"],
            status="busy",
            current_task_id="task-001",
        )
        assert agent.status == "busy"
        assert agent.current_task_id == "task-001"

    def test_serialization_round_trip(self) -> None:
        agent = Agent(
            id="a1",
            name="test",
            role="dev",
            model="deepseek-v4-flash",
            capabilities=["coding", "debugging", "refactoring"],
            status="idle",
        )
        data = agent.model_dump()
        restored = Agent(**data)
        assert restored.id == agent.id
        assert restored.capabilities == agent.capabilities
        assert restored.status == agent.status


class TestApprovalRequestModel:
    def test_create_with_defaults(self) -> None:
        req = ApprovalRequest(
            id="apr1",
            task_id="t1",
            requested_by="alice",
            summary="Please approve",
        )
        assert req.status == "pending"
        assert req.qa_report is None
        assert req.security_report is None
        assert req.final_review is None
        assert req.rollback_plan is None
        assert req.created_at is None
        assert req.resolved_at is None
        assert req.resolution_comment is None

    def test_create_with_all_fields(self) -> None:
        req = ApprovalRequest(
            id="apr1",
            task_id="t1",
            requested_by="alice",
            status="approved",
            summary="Looks good",
            qa_report="All tests pass",
            security_report="No vulnerabilities",
            final_review="Ready to ship",
            rollback_plan="Revert commit abc123",
            created_at=datetime(2026, 1, 1),
            resolved_at=datetime(2026, 1, 2),
            resolution_comment="Approved by Bob",
        )
        assert req.status == "approved"
        assert req.qa_report == "All tests pass"

    def test_serialization_round_trip(self) -> None:
        req = ApprovalRequest(
            id="apr1",
            task_id="t1",
            requested_by="alice",
            summary="Test",
        )
        data = req.model_dump()
        restored = ApprovalRequest(**data)
        assert restored.id == req.id
        assert restored.status == req.status


class TestReleaseModel:
    def test_create_with_defaults(self) -> None:
        release = Release(id="r1", goal_id="g1")
        assert release.gate == "G0"
        assert release.status == "active"
        assert release.artifacts == {}
        assert release.created_at is None
        assert release.completed_at is None

    def test_create_with_explicit_gate(self) -> None:
        release = Release(id="r1", goal_id="g1", gate="G3")
        assert release.gate == "G3"

    def test_serialization_round_trip(self) -> None:
        release = Release(
            id="r1",
            goal_id="g1",
            gate="G5",
            status="completed",
            artifacts={"docker_image": "sha256:abc123"},
            created_at=datetime(2026, 1, 1),
            completed_at=datetime(2026, 1, 2),
        )
        data = release.model_dump()
        restored = Release(**data)
        assert restored.id == release.id
        assert restored.gate == release.gate
        assert restored.artifacts == release.artifacts
        assert restored.completed_at == release.completed_at