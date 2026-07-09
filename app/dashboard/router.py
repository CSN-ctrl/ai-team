from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_303_SEE_OTHER

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(_DIR, "templates"))

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_PROCESS_START = time.time()


def _fmt_dt(dt: Any) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except (ValueError, TypeError):
            return dt[:10] if len(dt) >= 10 else dt
    if isinstance(dt, datetime):
        now = datetime.utcnow()
        diff = now - dt
        if diff.total_seconds() < 60:
            return "just now"
        if diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() / 60)}m ago"
        if diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() / 3600)}h ago"
        return dt.strftime("%b %d")
    return str(dt)


async def _build_health(request: Request) -> dict[str, Any]:
    router_instance = getattr(request.app.state, "router", None)
    if router_instance is None:
        return {"status": "unknown", "version": "?", "uptime": "?", "hostname": "?", "models": []}
    try:
        raw = await router_instance.check_health()
        models = [{"model": name, "healthy": status} for name, status in raw.items()]
        all_healthy = all(s for _, s in raw.items()) if raw else True
        status = "ok" if all_healthy else "degraded"
        uptime_seconds = int(time.time() - _PROCESS_START)
        days, rem = divmod(uptime_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
        return {
            "status": status,
            "version": "0.1.0",
            "uptime": uptime_str,
            "hostname": os.uname().nodename,
            "models": models,
        }
    except Exception:
        return {"status": "unknown", "version": "?", "uptime": "?", "hostname": "?", "models": []}


async def _build_activity_feed(request: Request) -> list[dict[str, str]]:
    activities: list[dict[str, str]] = []
    board = getattr(request.app.state, "board", None)
    if board is None:
        return activities

    try:
        for g in (await board.list_goals() or [])[:5]:
            activities.append({
                "type": "create",
                "message": f"Goal <strong>{g.text[:60]}</strong> created",
                "time": _fmt_dt(g.created_at),
            })
    except Exception:
        pass

    try:
        for t in (await board.list_tasks() or [])[:5]:
            activities.append({
                "type": "update",
                "message": f"Task <strong>{t.title[:60]}</strong> is now <em>{t.status.value}</em>",
                "time": _fmt_dt(t.updated_at),
            })
    except Exception:
        pass

    try:
        for a in (await board.list_approvals() or [])[:5]:
            if a.status == "approved":
                activities.append({
                    "type": "approve",
                    "message": f"Approval <strong>{a.summary[:60]}</strong> was approved",
                    "time": _fmt_dt(a.resolved_at),
                })
            elif a.status == "pending":
                activities.append({
                    "type": "update",
                    "message": f"Approval requested: <strong>{a.summary[:60]}</strong>",
                    "time": _fmt_dt(a.created_at),
                })
    except Exception:
        pass

    activities.sort(key=lambda x: x["time"] or "", reverse=True)
    return activities[:20]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("")
async def dashboard_index(request: Request):
    """Main dashboard overview page."""
    board = getattr(request.app.state, "board", None)

    # Stats
    stats = {"total_goals": 0, "active_tasks": 0, "pending_approvals": 0, "agents_online": 0}
    task_distribution: list[tuple[str, int]] = []

    if board:
        try:
            goals = await board.list_goals()
            stats["total_goals"] = len(goals or [])
        except Exception:
            pass

        try:
            all_tasks = await board.list_tasks()
            active = [t for t in (all_tasks or []) if t.status.value not in ("done", "cancelled")]
            stats["active_tasks"] = len(active)
            # Distribution
            dist: dict[str, int] = {}
            for t in (all_tasks or []):
                s = t.status.value
                dist[s] = dist.get(s, 0) + 1
            task_distribution = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        except Exception:
            pass

        try:
            approvals = await board.list_approvals(status="pending")
            stats["pending_approvals"] = len(approvals or [])
        except Exception:
            pass

    # Agents online
    registry = getattr(request.app.state, "agent_registry", {})
    stats["agents_online"] = len(registry) if registry else 0

    activities = await _build_activity_feed(request)
    health = await _build_health(request)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "stats": stats,
            "activities": activities,
            "task_distribution": task_distribution,
            "health_status": health.get("status", "unknown"),
        },
    )


@router.get("/kanban")
async def dashboard_kanban(request: Request):
    board = getattr(request.app.state, "board", None)
    col_keys = ["backlog", "ready", "in_progress", "review", "qa", "security", "approval", "done"]
    tasks_by_column: dict[str, list[dict[str, Any]]] = {k: [] for k in col_keys}

    if board:
        try:
            for t in (await board.list_tasks() or []):
                key = t.status.value
                if key not in tasks_by_column:
                    continue
                tasks_by_column[key].append({
                    "id": t.id,
                    "title": t.title,
                    "assignee": t.assignee,
                    "priority": t.priority,
                    "status": t.status.value,
                })
        except Exception as exc:
            logger.warning("Failed to load tasks for kanban: %s", exc)

    health = await _build_health(request)

    return templates.TemplateResponse(
        request,
        "kanban.html",
        {
            "tasks_by_column": tasks_by_column,
            "health_status": health.get("status", "unknown"),
        },
    )


@router.get("/approvals")
async def dashboard_approvals(
    request: Request,
    status: Optional[str] = Query(None),
):
    board = getattr(request.app.state, "board", None)
    approvals: list[dict[str, Any]] = []
    filter_status = status

    if board:
        try:
            for a in (await board.list_approvals(status=status) or []):
                approvals.append({
                    "id": a.id,
                    "task_id": a.task_id,
                    "requested_by": a.requested_by,
                    "status": a.status,
                    "summary": a.summary,
                    "qa_report": a.qa_report,
                    "security_report": a.security_report,
                    "final_review": a.final_review,
                    "rollback_plan": a.rollback_plan,
                    "created_at": _fmt_dt(a.created_at),
                    "resolved_at": _fmt_dt(a.resolved_at),
                    "resolution_comment": a.resolution_comment,
                })
        except Exception as exc:
            logger.warning("Failed to load approvals: %s", exc)

    health = await _build_health(request)

    return templates.TemplateResponse(
        request,
        "approvals.html",
        {
            "approvals": approvals,
            "filter_status": filter_status,
            "health_status": health.get("status", "unknown"),
        },
    )


@router.post("/approvals/{approval_id}/approve")
async def dashboard_approve(approval_id: str, request: Request):
    """Approve an approval request via form POST, then redirect back."""
    board = getattr(request.app.state, "board", None)
    if board:
        try:
            await board.resolve_approval(approval_id, status="approved")
            logger.info("Approval %s approved via dashboard", approval_id)
        except Exception as exc:
            logger.warning("Failed to approve %s: %s", approval_id, exc)
    return RedirectResponse(url="/dashboard/approvals?status=pending", status_code=HTTP_303_SEE_OTHER)


@router.post("/approvals/{approval_id}/reject")
async def dashboard_reject(
    approval_id: str,
    request: Request,
    comment: str = Form(""),
):
    """Reject an approval request via form POST, then redirect back."""
    board = getattr(request.app.state, "board", None)
    if board:
        try:
            await board.resolve_approval(approval_id, status="rejected", comment=comment)
            logger.info("Approval %s rejected via dashboard (comment=%r)", approval_id, comment)
        except Exception as exc:
            logger.warning("Failed to reject %s: %s", approval_id, exc)
    return RedirectResponse(url="/dashboard/approvals", status_code=HTTP_303_SEE_OTHER)


@router.get("/agents")
async def dashboard_agents(request: Request):
    """Agent activity overview."""
    registry = getattr(request.app.state, "agent_registry", None)
    agents: list[dict[str, Any]] = []

    if registry:
        agents = [
            {
                "id": a.id,
                "name": a.name,
                "role": a.__class__.__name__,
                "model": _model_for_agent(a),
                "capabilities": [a.capability] if hasattr(a, "capability") and a.capability else [],
                "status": getattr(a, "status", "idle"),
                "current_task_id": getattr(a, "current_task_id", None),
            }
            for a in registry.values()
        ]
    else:
        # Static fallback list matching the agents router
        agents = [
            {"id": "ceo", "name": "CEO Agent", "role": "CEOAgent", "model": "z-ai/glm-5.2", "capabilities": ["planning", "architecture", "final_review"], "status": "idle", "current_task_id": None},
            {"id": "planner", "name": "Planner Agent", "role": "PlannerAgent", "model": "qwen/qwen3-next-80b-a3b-instruct", "capabilities": ["planning"], "status": "idle", "current_task_id": None},
            {"id": "eng-a", "name": "Engineer A", "role": "EngineerAgent", "model": "qwen/qwen3-next-80b-a3b-instruct", "capabilities": ["coding"], "status": "idle", "current_task_id": None},
            {"id": "eng-b", "name": "Engineer B", "role": "EngineerAgent", "model": "qwen/qwen3-next-80b-a3b-instruct", "capabilities": ["coding"], "status": "idle", "current_task_id": None},
            {"id": "dbg", "name": "Debugger", "role": "DebuggerAgent", "model": "qwen/qwen3.5-397b-a17b", "capabilities": ["debugging"], "status": "idle", "current_task_id": None},
            {"id": "qa", "name": "QA Lead", "role": "QAAgent", "model": "nvidia/llama-3.3-nemotron-super-49b-v1", "capabilities": ["qa", "security"], "status": "idle", "current_task_id": None},
            {"id": "sec", "name": "Security Reviewer", "role": "SecurityAgent", "model": "nvidia/llama-3.3-nemotron-super-49b-v1", "capabilities": ["security"], "status": "idle", "current_task_id": None},
            {"id": "rev", "name": "Final Reviewer", "role": "FinalReviewerAgent", "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "capabilities": ["final_review"], "status": "idle", "current_task_id": None},
        ]

    health = await _build_health(request)

    return templates.TemplateResponse(
        request,
        "agents.html",
        {
            "agents": agents,
            "health_status": health.get("status", "unknown"),
        },
    )


def _model_for_agent(agent_obj) -> str:
    """Try to extract model from an agent object."""
    router = getattr(agent_obj, "_router", None)
    if router is None:
        return "unknown"
    from app.router.registry import ROUTING_TABLE
    capability = getattr(agent_obj, "capability", "")
    entry = ROUTING_TABLE.get(capability)
    if entry:
        return entry.get("primary", "unknown")
    return "unknown"


@router.get("/health")
async def dashboard_health(request: Request):
    """System health page."""
    health = await _build_health(request)

    # Also try to get RPM info from router
    router_instance = getattr(request.app.state, "router", None)
    rpm_info: list[dict] = []
    if router_instance:
        try:
            queue = router_instance.queue
            rpm_info = [
                {"model": m, "rpm_used": q.rpm_used, "rpm_limit": q.rpm_limit}
                for m, q in queue._queues.items()
            ]
        except Exception:
            pass

    if rpm_info:
        for model in health.get("models", []):
            for r in rpm_info:
                if r["model"] == model["model"]:
                    model["rpm_used"] = r["rpm_used"]
                    model["rpm_limit"] = r["rpm_limit"]

    return templates.TemplateResponse(
        request,
        "health.html",
        {
            "health": health,
            "health_status": health.get("status", "unknown"),
        },
    )


@router.get("/releases")
async def dashboard_releases(request: Request):
    board = getattr(request.app.state, "board", None)
    releases: list[dict[str, Any]] = []

    if board:
        try:
            for r in (await board.list_releases() or []):
                releases.append({
                    "id": r.id,
                    "goal_id": r.goal_id,
                    "gate": r.gate,
                    "status": r.status,
                    "artifacts": r.artifacts or {},
                    "created_at": _fmt_dt(r.created_at),
                    "completed_at": _fmt_dt(r.completed_at),
                })
        except Exception as exc:
            logger.warning("Failed to load releases: %s", exc)

    health = await _build_health(request)

    return templates.TemplateResponse(
        request,
        "releases.html",
        {
            "releases": releases,
            "health_status": health.get("status", "unknown"),
        },
    )
