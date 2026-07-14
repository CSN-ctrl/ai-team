"""Auto-assignment worker — polls kanban, assigns backlog tasks to idle agents.

Runs as an asyncio background task inside the CEO FastAPI process.
Scans for unassigned ``backlog`` and ``ready`` tasks every N seconds,
matches them to idle agents by capability, advances their status,
and monitors completion through the kanban state machine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.activity import emit as activity_emit
from app.agents.base import BaseAgent
from app.kanban.board import AsyncKanbanBoard
from app.models.task import Task, TaskStatus

logger = logging.getLogger(__name__)

# ── Capability mapping: task type → agent capability ────────────────────
# The CEO agent outputs tasks with a "type" field (coding, architecture,
# qa, security, planning).  We map those to agent capabilities.
_TASK_TYPE_TO_CAPABILITY: dict[str, str] = {
    "coding": "coding",
    "architecture": "planning",
    "qa": "qa",
    "security": "security",
    "planning": "planning",
    "debugging": "debugging",
    "refactoring": "refactoring",
    "final_review": "final_review",
    "research": "research",
    "development": "development",
}

# Agent capability → which agents can handle it
_CAPABILITY_TO_AGENTS: dict[str, list[str]] = {
    "planning": ["ceo", "planner"],
    "coding": ["eng-a", "eng-b"],
    "debugging": ["dbg"],
    "qa": ["qa"],
    "security": ["sec"],
    "final_review": ["rev"],
    "refactoring": ["eng-a", "eng-b"],
    "research": ["researcher"],
    "development": ["dev-exp"],
}


class AutoAssigner:
    """Background worker that auto-assigns tasks to agents.

    Call ``start()`` during application startup.  The worker runs until
    ``stop()`` is called (or the event loop shuts down).
    """

    def __init__(
        self,
        board: AsyncKanbanBoard,
        agent_registry: dict[str, BaseAgent],
        poll_interval: float = 5.0,
        max_assign_per_cycle: int = 3,
    ) -> None:
        self._board = board
        self._agents = agent_registry
        self._poll_interval = poll_interval
        self._max_per_cycle = max_assign_per_cycle
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        # Internal agent busy tracking (agent_id → task_id)
        self._busy_agents: dict[str, str] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background polling loop (non-blocking)."""
        if self._task is not None and not self._task.done():
            logger.warning("AutoAssigner already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("AutoAssigner started (poll=%ss, max/cycle=%d)", self._poll_interval, self._max_per_cycle)

    async def stop(self) -> None:
        """Signal the worker to shut down and wait for it."""
        if self._task is None or self._task.done():
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("AutoAssigner stop timed out, cancelling")
            self._task.cancel()
        self._task = None
        logger.info("AutoAssigner stopped")

    # ── Main loop ────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        logger.debug("AutoAssigner loop started")
        sweep_counter = 0
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("AutoAssigner tick error")

            # Sweep completed tasks every 6 ticks (~30s)
            sweep_counter += 1
            if sweep_counter >= 6:
                sweep_counter = 0
                try:
                    released = await self.sweep_completed()
                    if released:
                        logger.info("AutoAssigner released %d agents", len(released))
                except Exception:
                    logger.exception("AutoAssigner sweep error")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        idle_agents = self._get_idle_agents()
        if not idle_agents:
            return

        # 2. Find unassigned tasks to work on
        candidates = await self._find_candidates()
        if not candidates:
            return

        # 3. Assign up to max_per_cycle
        assigned = 0
        for task in candidates:
            if assigned >= self._max_per_cycle:
                break
            if not idle_agents:
                break

            agent_id = self._pick_agent(task, idle_agents)
            if agent_id is None:
                continue

            try:
                await self._assign_task(task, agent_id)
                idle_agents.discard(agent_id)
                assigned += 1
                logger.info("Assigned task %s (%s) → agent %s", task.id, task.title, agent_id)
            except Exception:
                logger.exception("Failed to assign task %s", task.id)

    # ── Candidate search ─────────────────────────────────────────────────

    async def _find_candidates(self) -> list[Task]:
        """Return backlog or ready tasks that are unassigned."""
        backlog = await self._board.list_tasks(status="backlog")
        ready = await self._board.list_tasks(status="ready")
        candidates = [t for t in backlog + ready if not t.assignee]
        # Sort by priority (lower number = higher priority)
        candidates.sort(key=lambda t: t.priority if t.priority else 3)
        return candidates

    # ── Agent selection ──────────────────────────────────────────────────

    def _get_idle_agents(self) -> set[str]:
        """Return set of agent IDs that are not currently busy."""
        idle: set[str] = set()
        for aid in self._agents:
            if aid not in self._busy_agents:
                idle.add(aid)
        return idle

    def _pick_agent(self, task: Task, idle_agents: set[str]) -> Optional[str]:
        """Pick the best idle agent for *task*.

        Attempts capability matching first; falls back to any idle agent.
        """
        # Try capability-based routing
        task_type = self._infer_task_type(task)
        capability = _TASK_TYPE_TO_CAPABILITY.get(task_type)
        if capability:
            preferred = _CAPABILITY_TO_AGENTS.get(capability, [])
            for aid in preferred:
                if aid in idle_agents:
                    return aid

        # Fallback: any idle agent
        if idle_agents:
            return next(iter(idle_agents))
        return None

    @staticmethod
    def _infer_task_type(task: Task) -> str:
        """Guess the task type from its title, description, or epic context.

        Returns one of: coding, architecture, qa, security, planning, debugging.
        """
        text = (task.title + " " + (task.description or "")).lower()
        if any(kw in text for kw in ("security", "vulnerability", "audit", "auth")):
            return "security"
        if any(kw in text for kw in ("qa", "test", "coverage", "regression")):
            return "qa"
        if any(kw in text for kw in ("bug", "error", "crash", "fix", "broken")):
            return "debugging"
        if any(kw in text for kw in ("plan", "design", "architect", "refactor")):
            return "planning"
        if any(kw in text for kw in ("review", "validate", "approve")):
            return "final_review"
        return "coding"  # default

    # ── Assignment ───────────────────────────────────────────────────────

    async def _assign_task(self, task: Task, agent_id: str) -> None:
        current = task.status
        if isinstance(current, TaskStatus):
            current = current.value

        # Walk through valid transitions: backlog → ready → planning
        if current == "backlog":
            await self._board.update_task(task.id, {"assignee": agent_id, "status": "ready"})
            await self._board.update_task(task.id, {"status": "planning"})
            activity_emit("task_assigned", actor=agent_id, task_id=task.id, task_title=task.title,
                          detail=f"Assigned {agent_id} (backlog→planning)")
            activity_emit("task_status", actor="assigner", task_id=task.id, task_title=task.title,
                          from_status="backlog", to_status="planning",
                          detail=f"Auto-progressed to planning for {agent_id}")
        elif current == "ready":
            await self._board.update_task(task.id, {"assignee": agent_id, "status": "planning"})
            activity_emit("task_assigned", actor=agent_id, task_id=task.id, task_title=task.title,
                          detail=f"Assigned {agent_id} (ready→planning)")
            activity_emit("task_status", actor="assigner", task_id=task.id, task_title=task.title,
                          from_status="ready", to_status="planning",
                          detail=f"Auto-progressed to planning for {agent_id}")
        else:
            await self._board.update_task(task.id, {"assignee": agent_id})
            activity_emit("task_assigned", actor=agent_id, task_id=task.id, task_title=task.title,
                          detail=f"Assigned {agent_id}")

        self._busy_agents[agent_id] = task.id
        activity_emit("agent_busy", actor=agent_id, task_id=task.id, task_title=task.title,
                      detail=f"{agent_id} started working on {task.title}")

    # ── Completion monitoring ────────────────────────────────────────────

    async def sweep_completed(self) -> list[Task]:
        done_tasks = await self._board.list_tasks(status="done")
        released: list[Task] = []
        for task in done_tasks:
            if task.assignee and task.assignee in self._busy_agents:
                del self._busy_agents[task.assignee]
                released.append(task)
                activity_emit("agent_idle", actor=task.assignee, task_id=task.id, task_title=task.title,
                              detail=f"{task.assignee} completed {task.title}")

        cancelled = await self._board.list_tasks(status="cancelled")
        for task in cancelled:
            if task.assignee and task.assignee in self._busy_agents:
                del self._busy_agents[task.assignee]
                activity_emit("agent_idle", actor=task.assignee, task_id=task.id, task_title=task.title,
                              detail=f"{task.assignee} released from cancelled {task.title}")

        return released
