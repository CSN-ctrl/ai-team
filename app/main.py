"""
OpenClaw CEO — FastAPI application entry point.

Wires together the Kanban engine, Model Router, and NVIDIA NIM LLM client
into a unified orchestration API.  Start with::

    uvicorn app.main:app
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import Config
from app.agents import CEOAgent, PlannerAgent, EngineerAgent, DebuggerAgent, QAAgent, SecurityAgent, FinalReviewerAgent, DeveloperAgent, HermesAgent
from app.kanban.board import AsyncKanbanBoard
from app.llm.client import NIMClient
from app.llm.intent import IntentClassifier
from app.models.mode import ConversationManager
from app.router import ModelRouter
from app.workers.auto_assigner import AutoAssigner
from app.workflows import WorkflowEngine

# ── Routers ────────────────────────────────────────────────────────────────

from app.routers import (
    activity,
    agents,
    approvals,
    chat,
    goals,
    health,
    releases,
    tasks,
)
from app.dashboard.router import router as dashboard_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenClaw CEO",
    description=(
        "Orchestration server for AI software development agents. "
        "Manages goals, epics, tasks, approvals, and releases through "
        "a kanban engine backed by an NVIDIA NIM LLM router."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Startup / Shutdown  (using on_event for compatibility with FastAPI 0.92)
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    """Initialise Kanban board, NIM client, and Model Router."""
    cfg = Config()
    app.state.config = cfg

    # ── Board ──────────────────────────────────────────────────────────
    db_path = os.path.expanduser(cfg.openclaw_db_path)
    board = AsyncKanbanBoard(db_path=db_path)
    await board.init_db()
    app.state.board = board

    # ── NIM Client ─────────────────────────────────────────────────────
    nim_client = NIMClient(
        api_key=cfg.nvidia_api_key,
        base_url=cfg.nvidia_base_url,
    )
    app.state.nim_client = nim_client

    # ── Conversation Manager & Intent Classifier ───────────────────────
    conv_manager = ConversationManager()
    app.state.conv_manager = conv_manager

    intent_classifier = IntentClassifier(
        llm_client=nim_client,
    )
    app.state.intent_classifier = intent_classifier

    # ── Model Router ───────────────────────────────────────────────────
    router = ModelRouter(
        llm_client=nim_client,
        rpm_limit=cfg.nvidia_rpm_limit,
        fallback_max_retries=cfg.fallback_retry_max,
    )
    app.state.router = router

    # ── Specialist Agents ──────────────────────────────────────────────
    agents = {
        "ceo": CEOAgent("ceo", "CEO Agent", router),
        "planner": PlannerAgent("planner", "Planner Agent", router),
        "engineer-a": EngineerAgent("eng-a", "Engineer A", router),
        "engineer-b": EngineerAgent("eng-b", "Engineer B", router),
        "debugger": DebuggerAgent("dbg", "Debugger", router),
        "qa": QAAgent("qa", "QA Lead", router),
        "security": SecurityAgent("sec", "Security Reviewer", router),
        "reviewer": FinalReviewerAgent("rev", "Final Reviewer", router),
        "hermes": HermesAgent("hermes", "Hermes", router),
        "dev-exp": DeveloperAgent("dev-exp", "Dev Expert", router),
    }
    app.state.agent_registry = agents
    app.state.ceo_agent = agents["ceo"]

    # ── Workflow Engine ────────────────────────────────────────────────
    workflow_engine = WorkflowEngine()
    app.state.workflow_engine = workflow_engine

    # ── AutoAssigner ───────────────────────────────────────────────────
    assigner = AutoAssigner(
        board=board,
        agent_registry=agents,
        workflow_engine=workflow_engine,
        poll_interval=5.0,
        max_assign_per_cycle=3,
    )
    assigner.start()
    app.state.auto_assigner = assigner

    logger.info(
        "OpenClaw CEO started — board=%s, router.rpm=%d, agents=%d, workflows=%d, assigner=enabled",
        db_path,
        cfg.nvidia_rpm_limit,
        len(agents),
        len(workflow_engine.list_workflows()),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    """Close DB connection and HTTP client gracefully."""
    board = getattr(app.state, "board", None)
    if board is not None:
        await board.close()

    nim_client = getattr(app.state, "nim_client", None)
    if nim_client is not None:
        await nim_client.close()

    assigner = getattr(app.state, "auto_assigner", None)
    if assigner is not None:
        await assigner.stop()

    logger.info("OpenClaw CEO shut down")


# ---------------------------------------------------------------------------
# CORS (development: allow all origins)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Internal server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(activity.router)
app.include_router(health.router)
app.include_router(goals.router)
app.include_router(tasks.router)
app.include_router(agents.router)
app.include_router(approvals.router)
app.include_router(releases.router)
app.include_router(chat.router)
app.include_router(dashboard_router)

# ── Discord-like dashboard at root ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DASHBOARD_HTML = os.path.join(_HERE, "static", "dashboard.html")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_DASHBOARD_HTML)


@app.get("/api.json")
async def api_index() -> dict:
    return {
        "name": "OpenClaw CEO",
        "version": "0.1.0",
        "description": "Orchestration server for AI software development agents",
        "endpoints": {
            "activity": "/v1/activity",
            "health": "/v1/health",
            "goals": "/v1/goals",
            "tasks": "/v1/tasks",
            "agents": "/v1/agents",
            "approvals": "/v1/approvals",
            "releases": "/v1/releases",
            "chat": "/v1/chat/completions",
        },
    }


# ---------------------------------------------------------------------------
# Entry point (direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    cfg = Config()
    uvicorn.run(
        "app.main:app",
        host=cfg.openclaw_host,
        port=cfg.openclaw_port,
        reload=False,
    )
