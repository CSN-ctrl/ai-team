"""Task management endpoints — /v1/tasks."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.kanban.board import AsyncKanbanBoard
from app.models.task import Task, TaskStatus

router = APIRouter(prefix="/v1")


class CreateTaskRequest(BaseModel):
    epic_id: Optional[str] = None
    title: str
    description: str = ""
    priority: int = 3


class AssignTaskRequest(BaseModel):
    task_id: str
    assignee: str


@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
) -> list[dict]:
    """List tasks with optional status / assignee filtering."""
    board: AsyncKanbanBoard = request.app.state.board
    tasks = await board.list_tasks(status=status, assignee=assignee)
    return [t.dict() for t in tasks]


@router.post("/tasks")
async def create_task(body: CreateTaskRequest, request: Request) -> dict:
    """Create a new task."""
    board: AsyncKanbanBoard = request.app.state.board
    task = Task(
        id=uuid.uuid4().hex[:12],
        epic_id=body.epic_id or "",
        title=body.title,
        description=body.description,
        status=TaskStatus.BACKLOG,
        priority=body.priority,
    )
    created = await board.create_task(task)
    return created.dict()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    """Return a single task by ID."""
    board: AsyncKanbanBoard = request.app.state.board
    task = await board.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return task.dict()


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, body: dict, request: Request) -> dict:
    """Apply partial updates to a task."""
    board: AsyncKanbanBoard = request.app.state.board

    body.pop("id", None)
    body.pop("created_at", None)
    body.pop("updated_at", None)

    try:
        updated = await board.update_task(task_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if updated is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    return updated.dict()


@router.post("/tasks/assign")
async def assign_task(body: AssignTaskRequest, request: Request) -> dict:
    """Assign a task to an agent (by assignee name)."""
    board: AsyncKanbanBoard = request.app.state.board
    updated = await board.update_task(
        body.task_id,
        {"assignee": body.assignee},
    )
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task {body.task_id!r} not found",
        )
    return updated.dict()
