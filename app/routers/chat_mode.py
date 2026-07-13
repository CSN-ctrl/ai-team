"""Mode routing helpers for the chat completions endpoint.

Extracted from ``chat.py`` to keep the streaming/endpoint logic focused.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Optional

from fastapi import Request

from app.kanban.board import AsyncKanbanBoard
from app.llm.prompts import ASK_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from app.models.mode import ConversationManager, Mode, ModeRoute
from app.models.task import Epic, Task, TaskStatus

logger = logging.getLogger(__name__)


def resolve_mode(
    request: Request,
) -> tuple[Mode, Optional[str]]:
    """Determine the mode from the ``X-AI-Mode`` header; fallback to AUTO."""
    raw = request.headers.get("X-AI-Mode", "").strip().lower()
    try:
        return (Mode(raw), None)
    except ValueError:
        return (Mode.AUTO, None)


def get_or_create_conv_id(request: Request) -> str:
    """Return the conversation ID from the header, or generate one."""
    conv_id = request.headers.get("X-Conversation-Id", "").strip()
    if not conv_id:
        conv_id = f"conv-{uuid.uuid4().hex[:12]}"
    return conv_id


def prepare_messages(
    messages: list[dict],
    system_prompt: str,
    *,
    strip_tools: bool = True,
) -> list[dict]:
    """Prepend (or replace) a system prompt and optionally strip tool messages."""
    filtered: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        if strip_tools and role in ("tool", "tool_call"):
            continue
        filtered.append(msg)

    result: list[dict] = []
    system_replaced = False
    for msg in filtered:
        if msg.get("role") == "system":
            result.append({"role": "system", "content": system_prompt})
            system_replaced = True
        else:
            result.append(msg)
    if not system_replaced:
        result.insert(0, {"role": "system", "content": system_prompt})

    return result


async def classify_and_route(
    request: Request,
    messages: list[dict],
) -> ModeRoute:
    """Run the intent classifier, falling back to AUTO on any error."""
    classifier = getattr(request.app.state, "intent_classifier", None)
    if classifier is None:
        return ModeRoute(mode=Mode.AUTO, confidence=0.0)
    try:
        return await classifier.classify(messages)
    except Exception:
        logger.warning("Classifier error; falling back to AUTO", exc_info=True)
        return ModeRoute(mode=Mode.AUTO, confidence=0.0)


async def process_plan_text(
    request: Request,
    plan_text: str,
    conv_id: str,
) -> list[str]:
    """Parse a plan into steps and create kanban epic + tasks.

    Returns a list of created task IDs.
    """
    board: AsyncKanbanBoard = request.app.state.board

    title_match = re.search(r"^##\s+Plan:\s*(.+)$", plan_text, re.MULTILINE)
    plan_title = title_match.group(1).strip() if title_match else "Implementation Plan"

    step_pattern = re.compile(
        r"^###\s+Step\s+\d+\s*:\s*(.+?)(?:\s*\[(\w+)\])?\s*$",
        re.MULTILINE,
    )
    steps = step_pattern.findall(plan_text)

    epic_id = board._new_id()
    epic = Epic(
        id=epic_id,
        name=plan_title,
        description=plan_text[:500],
        goal_id="",
    )
    await board.create_epic(epic)

    task_ids: list[str] = []
    for step_name, effort in steps:
        step_name = step_name.strip()
        task_id = board._new_id()
        desc = f"[{effort or 'medium'}] {step_name}"
        task = Task(
            id=task_id,
            epic_id=epic_id,
            title=step_name,
            description=desc,
            status=TaskStatus.BACKLOG,
            priority=3,
        )
        await board.create_task(task)
        task_ids.append(task_id)

    conv_manager: ConversationManager = getattr(
        request.app.state, "conv_manager", None
    )
    if conv_manager is not None:
        conv_manager.set_plan(conv_id, plan_text, task_ids)

    logger.info(
        "Created plan epic=%s with %d tasks for conv=%s",
        epic_id,
        len(task_ids),
        conv_id,
    )
    return task_ids


def make_plan_complete_callback(
    request: Request,
    conv_id: str,
) -> Callable[[str], Any]:
    """Return an async callback that processes plan text after a stream."""
    async def _on_plan_complete(full_text: str) -> None:
        if full_text.strip():
            await process_plan_text(request, full_text, conv_id)

    return _on_plan_complete
