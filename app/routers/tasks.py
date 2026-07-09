"""Task management endpoints — /v1/tasks."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.kanban.board import AsyncKanbanBoard

router = APIRouter(prefix="/v1")


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
    """Apply partial updates to a task.

    The request body may contain any subset of updatable fields.
    Status transitions are validated against the kanban state machine;
    invalid transitions return ``400 Bad Request``.
    """
    board: AsyncKanbanBoard = request.app.state.board

    # Strip read-only keys from the payload
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
