"""Release endpoints — /v1/releases."""

from fastapi import APIRouter, HTTPException, Request

from app.kanban.board import AsyncKanbanBoard

router = APIRouter(prefix="/v1")


@router.get("/releases")
async def list_releases(request: Request) -> list[dict]:
    """Return all releases."""
    board: AsyncKanbanBoard = request.app.state.board
    releases = await board.list_releases()
    return [r.dict() for r in releases]


@router.get("/releases/{release_id}")
async def get_release(release_id: str, request: Request) -> dict:
    """Return details for a specific release by its ID."""
    board: AsyncKanbanBoard = request.app.state.board
    release = await board.get_release_by_id(release_id)
    if release is None:
        raise HTTPException(
            status_code=404, detail=f"Release {release_id!r} not found"
        )
    return release.dict()
