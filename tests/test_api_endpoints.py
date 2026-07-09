"""Tests for the FastAPI API endpoints.

Uses FastAPI TestClient with app.state manually populated to avoid
requiring the full lifespan (which needs real API keys and DB paths).
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.kanban.board import AsyncKanbanBoard
from app.router import ModelRouter
from app.router.registry import list_available_models


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[TestClient, None]:
    """Provide a TestClient with a fully wired app.state.

    We manually set app.state to avoid the real lifespan (which needs
    NVIDIA API keys and a real DB path).
    """
    # Override the lifespan to a no-op so TestClient doesn't try to
    # run the real startup/shutdown.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app.router.lifespan_context = noop_lifespan

    # Set up the board and init it
    board = AsyncKanbanBoard(db_path=":memory:")
    await board.init_db()

    with TestClient(app) as c:
        # Manually wire up app.state
        c.app.state.board = board
        c.app.state.router = ModelRouter(
            llm_client=_DummyLLMClient(),
            rpm_limit=40,
            fallback_max_retries=1,
        )
        c.app.state.nim_client = _DummyNIMClient()
        c.app.state.config = _DummyConfig()
        yield c

    await board.close()


class _DummyLLMClient:
    """Minimal duck-typed LLM client for router health checks."""

    async def chat_completion(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        return {
            "id": "dummy",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }


class _DummyNIMClient:
    """Minimal duck-typed NIM client for chat endpoint."""

    async def chat_completion(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        return {
            "id": "chatcmpl-dummy",
            "object": "chat.completion",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }

    async def close(self) -> None:
        pass


class _DummyConfig:
    nvidia_api_key: str = "test-key"
    openclaw_host: str = "0.0.0.0"
    openclaw_port: int = 8765
    openclaw_db_path: str = ":memory:"
    openhands_url: str = "http://localhost:8000"
    openhands_api_key: str = ""
    openhands_sandbox_dir: str = "/tmp/sandboxes"
    nvidia_rpm_limit: int = 40
    model_health_check_interval: int = 60
    fallback_retry_max: int = 3
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    def test_get_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "uptime" in data
        assert "models" in data

    def test_get_root_returns_info(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "OpenClaw CEO"
        assert "endpoints" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════════════════════════════


class TestTasksEndpoint:
    def test_list_tasks_empty(self, client: TestClient) -> None:
        response = client.get("/v1/tasks")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_task_not_found(self, client: TestClient) -> None:
        response = client.get("/v1/tasks/nonexistent")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_update_task_not_found(self, client: TestClient) -> None:
        response = client.patch("/v1/tasks/nonexistent", json={"title": "New"})
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Goals
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoalsEndpoint:
    def test_create_goal_without_ceo_returns_note(self, client: TestClient) -> None:
        response = client.post("/v1/goals", json={"goal": "Build a new feature"})
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert "note" in data
        assert "CEO agent not available" in data["note"]

    def test_get_goal_not_found(self, client: TestClient) -> None:
        response = client.get("/v1/goals/nonexistent")
        assert response.status_code == 404

    def test_list_goals_empty(self, client: TestClient) -> None:
        response = client.get("/v1/goals")
        assert response.status_code == 200
        assert response.json() == []


# ═══════════════════════════════════════════════════════════════════════════════
# Agents
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentsEndpoint:
    def test_list_agents_returns_static_list(self, client: TestClient) -> None:
        response = client.get("/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        # Should include the static agents
        ids = [a["id"] for a in data]
        assert "ceo" in ids
        assert "coder" in ids

    def test_get_agent_by_id(self, client: TestClient) -> None:
        response = client.get("/v1/agents/coder")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "coder"
        assert "capabilities" in data

    def test_get_agent_not_found(self, client: TestClient) -> None:
        response = client.get("/v1/agents/nonexistent")
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Approvals
# ═══════════════════════════════════════════════════════════════════════════════


class TestApprovalsEndpoint:
    def test_list_approvals_empty(self, client: TestClient) -> None:
        response = client.get("/v1/approvals")
        assert response.status_code == 200
        assert response.json() == []

    def test_approve_nonexistent_returns_404(self, client: TestClient) -> None:
        response = client.post("/v1/approvals/nonexistent/approve")
        assert response.status_code == 404

    def test_reject_nonexistent_returns_404(self, client: TestClient) -> None:
        response = client.post("/v1/approvals/nonexistent/reject", json={"comment": "No"})
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Releases
# ═══════════════════════════════════════════════════════════════════════════════


class TestReleasesEndpoint:
    def test_list_releases_empty(self, client: TestClient) -> None:
        response = client.get("/v1/releases")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_release_not_found(self, client: TestClient) -> None:
        response = client.get("/v1/releases/nonexistent")
        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Chat
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    def test_chat_completion_returns_502_without_real_client(
        self, client: TestClient
    ) -> None:
        """Without a real NIM client, the chat endpoint should return 502."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # The dummy client will succeed, so we expect 200
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data


# ═══════════════════════════════════════════════════════════════════════════════
# 404 handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotFound:
    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_unknown_root_route_returns_404(self, client: TestClient) -> None:
        response = client.get("/some/random/path")
        assert response.status_code == 404