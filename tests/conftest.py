"""Shared fixtures for the OpenClaw CEO test suite."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.kanban.board import AsyncKanbanBoard
from app.kanban.db import DatabaseManager
from app.llm.client import NIMClient
from app.router import ModelRouter
from app.router.fallback import FallbackHandler
from app.router.queue import RateLimitQueue
from app.router.selector import ModelSelector


# ── Kanban fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def kanban_board() -> AsyncGenerator[AsyncKanbanBoard, None]:
    """Provide an AsyncKanbanBoard backed by an in-memory SQLite database."""
    board = AsyncKanbanBoard(db_path=":memory:")
    await board.init_db()
    try:
        yield board
    finally:
        await board.close()


# ── Model Router fixtures ────────────────────────────────────────────────────


@pytest.fixture
def model_selector() -> ModelSelector:
    return ModelSelector(health_check_interval=300.0)


@pytest.fixture
def rate_limit_queue() -> RateLimitQueue:
    return RateLimitQueue(rpm_limit=40)


@pytest.fixture
def fallback_handler() -> FallbackHandler:
    return FallbackHandler(base_delay=0.01, max_retries=1)


class DummyLLMClient:
    """A minimal duck-typed LLM client that returns a canned response."""

    async def chat_completion(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {
            "id": "dummy",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }


@pytest.fixture
def dummy_llm_client() -> DummyLLMClient:
    return DummyLLMClient()


@pytest.fixture
def model_router(dummy_llm_client: DummyLLMClient) -> ModelRouter:
    return ModelRouter(llm_client=dummy_llm_client, rpm_limit=40, fallback_max_retries=1)


# ── NIM Client fixture (no real HTTP calls) ──────────────────────────────────


@pytest.fixture
def nim_client() -> NIMClient:
    return NIMClient(api_key="test-key", base_url="http://localhost:0", timeout=0.1)