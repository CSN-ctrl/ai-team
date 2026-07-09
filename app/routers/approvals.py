"""Approval endpoints — /v1/approvals."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.kanban.board import AsyncKanbanBoard

router = APIRouter(prefix="/v1")


class RejectBody(BaseModel):
    comment: str = ""


@router.get("/approvals")
async def list_approvals(
    request: Request,
    status: Optional[str] = None,
) -> list[dict]:
    """List approval requests, optionally filtered by status.

    When *status* is omitted, all requests are returned (newest first).
    Pass ``status=pending`` to see only unresolved requests.
    """
    board: AsyncKanbanBoard = request.app.state.board
    approvals = await board.list_approvals(status=status)
    return [a.dict() for a in approvals]


@router.post("/approvals/{approval_id}/approve")
async def approve_approval(approval_id: str, request: Request) -> dict:
    """Approve a pending approval request."""
    board: AsyncKanbanBoard = request.app.state.board

    # Verify it exists first
    approvals = await board.list_approvals()
    if not any(a.id == approval_id for a in approvals):
        raise HTTPException(
            status_code=404, detail=f"Approval {approval_id!r} not found"
        )

    await board.resolve_approval(approval_id, status="approved")
    return {"id": approval_id, "status": "approved"}


@router.post("/approvals/{approval_id}/reject")
async def reject_approval(
    approval_id: str,
    body: RejectBody = RejectBody(),
    request: Request = None,
) -> dict:
    """Reject a pending approval request with an optional comment."""
    if request is None:
        raise HTTPException(status_code=500, detail="Internal error")
    board: AsyncKanbanBoard = request.app.state.board

    approvals = await board.list_approvals()
    if not any(a.id == approval_id for a in approvals):
        raise HTTPException(
            status_code=404, detail=f"Approval {approval_id!r} not found"
        )

    await board.resolve_approval(approval_id, status="rejected", comment=body.comment)
    return {"id": approval_id, "status": "rejected", "comment": body.comment}
