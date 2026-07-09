"""Goal management endpoints — /v1/goals."""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.kanban.board import AsyncKanbanBoard
from app.models.task import Goal

router = APIRouter(prefix="/v1")


class GoalCreateRequest(BaseModel):
    goal: str


class GoalCreateResponse(BaseModel):
    id: str
    status: str
    note: str | None = None


class GoalDetailResponse(BaseModel):
    goal: Goal
    epics: list[dict] = []
    tasks: list[dict] = []


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@router.post("/goals", status_code=201)
async def create_goal(body: GoalCreateRequest, request: Request) -> GoalCreateResponse:
    """Submit a new goal.

    Creates a Goal record in the kanban board.  If a CEO agent is
    available (via ``app.state.ceo_agent``) the goal is delegated for
    decomposition; otherwise the goal is recorded as ``pending``.
    """
    board: AsyncKanbanBoard = request.app.state.board

    goal = Goal(
        id=_new_id(),
        text=body.goal,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    await board.create_goal(goal)

    # Delegate to CEO agent if available
    ceo = getattr(request.app.state, "ceo_agent", None)
    if ceo is not None:
        try:
            result = await ceo.decompose_goal(body.goal)
            if result and result.get("epic"):
                goal.status = "processing"
                await board.update_goal_status(goal.id, "processing")
        except Exception:
            pass
        return GoalCreateResponse(id=goal.id, status="delegated")

    return GoalCreateResponse(
        id=goal.id,
        status="pending",
        note="CEO agent not available — goal recorded",
    )


@router.get("/goals/{goal_id}")
async def get_goal(goal_id: str, request: Request) -> GoalDetailResponse:
    """Return goal status, its epics, and all tasks belonging to those epics."""
    board: AsyncKanbanBoard = request.app.state.board

    goal = await board.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail=f"Goal {goal_id!r} not found")

    epics = await board.get_epics(goal_id)
    epic_ids = {e.id for e in epics}

    # Filter tasks belonging to this goal's epics
    all_tasks = await board.list_tasks()
    goal_tasks = [t for t in all_tasks if t.epic_id in epic_ids]

    return GoalDetailResponse(
        goal=goal,
        epics=[e.dict() for e in epics],
        tasks=[t.dict() for t in goal_tasks],
    )


@router.get("/goals")
async def list_goals(request: Request) -> list[dict]:
    """Return all goals."""
    board: AsyncKanbanBoard = request.app.state.board
    goals = await board.list_goals()
    return [g.dict() for g in goals]
