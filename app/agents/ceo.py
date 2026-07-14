from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.agents.base import BaseAgent
from app.router import ModelRouter

logger = logging.getLogger(__name__)


class CEOAgent(BaseAgent):
    """CEO agent — orchestrates the whole team.

    Decomposes goals, delegates to specialists via the execution loop,
    collects outputs, and advances the pipeline. Acts as the executor
    that makes every agent actually run.
    """

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "planning"
        self.system_prompt = (
            "You are the OpenClaw CEO. Your role is to decompose software goals into "
            "epics, milestones, and tasks. You assign specialists to each task and "
            "evaluate their output. You never write code yourself."
        )
        # Injected references (set after construction)
        self._board: Any = None
        self._agent_registry: dict[str, BaseAgent] = {}
        self._workflows: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def inject(self, board, agent_registry: dict[str, BaseAgent], workflow_engine) -> None:
        """Inject board, registry, and workflow engine after construction."""
        self._board = board
        self._agent_registry = agent_registry
        self._workflows = workflow_engine

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background execution loop."""
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._execution_loop())
        logger.warning("CEO executor started (task=%s)", id(self._task))

    async def stop(self) -> None:
        """Stop the execution loop gracefully."""
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("CEO executor stop timed out, cancelling")
            self._task.cancel()
        self._task = None
        logger.info("CEO executor stopped")

    # ── Execution loop ─────────────────────────────────────────────────

    async def _execution_loop(self) -> None:
        """Background loop: pick up in_progress tasks and execute them."""
        poll_interval = 3.0
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("CEO executor tick error")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=poll_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        """One execution cycle: find in_progress tasks and run their agent."""
        if self._board is None:
            return
        in_progress = await self._board.list_tasks(status="in_progress")
        for task in in_progress:
            if not task.assignee:
                continue
            agent = self._agent_registry.get(task.assignee)
            if agent is None:
                logger.warning("No agent %r found for task %s", task.assignee, task.id)
                continue
            # Skip if this agent is already busy executing another task
            # (we check by seeing if the task is still in_progress with the same assignee)
            try:
                await self._execute_task_step(task, agent)
            except Exception as exc:
                logger.exception("CEO failed to execute task %s via %s: %s", task.id, agent.id, exc)

    async def _execute_task_step(self, task: Any, agent: BaseAgent) -> None:
        """Run the agent on the current workflow step and mark done on success."""
        step = self._workflows.get_current_step(task) if self._workflows else None
        step_label = step.label if step else "work"

        logger.info("CEO executing task %s → %s (%s)", task.id, agent.id, step_label)

        # Gather input context for the agent
        input_data = {
            "title": task.title,
            "description": task.description,
            "workflow_step": step_label,
            "previous_output": task.workflow_output or "",
        }

        try:
            # Call the specialist agent with a timeout
            result = await asyncio.wait_for(
                agent.execute(input_data),
                timeout=90.0,
            )
            logger.info("Agent %s completed step %s on task %s", agent.id, step_label, task.id)
        except asyncio.TimeoutError:
            logger.error("Agent %s timed out on task %s", agent.id, task.id)
            await self._board.update_task(task.id, {"status": "backlog", "assignee": None})
            from app.activity import emit as activity_emit
            activity_emit("task_status", actor=agent.id, task_id=task.id,
                          task_title=task.title,
                          from_status="in_progress", to_status="backlog",
                          detail=f"{agent.id} timed out on {step_label}, retrying")
            return
        except Exception as exc:
            logger.error("Agent %s failed on task %s: %s", agent.id, task.id, exc)
            await self._board.update_task(task.id, {"status": "backlog", "assignee": None})
            from app.activity import emit as activity_emit
            activity_emit("task_status", actor=agent.id, task_id=task.id,
                          task_title=task.title,
                          from_status="in_progress", to_status="backlog",
                          detail=f"{agent.id} failed on {step_label}, retrying")
            return

        # Store agent output as workflow context
        import json
        output_json = json.dumps(result, ensure_ascii=False, default=str)

        from app.activity import emit as activity_emit
        activity_emit("task_status", actor=agent.id, task_id=task.id,
                      task_title=task.title,
                      from_status="in_progress", to_status="done",
                      detail=f"{agent.id} completed {step_label}")

        # Mark done — auto_assigner sweep will advance the pipeline
        await self._board.update_task(task.id, {
            "status": "done",
            "workflow_output": output_json,
        })

    # ── Original CEO logic ─────────────────────────────────────────────

    async def execute(self, input_data: dict) -> dict:
        goal_text = input_data.get("goal", "")
        if not goal_text:
            return {"error": "No goal provided", "epic": None, "milestones": [], "tasks": []}

        schema = {
            "type": "object",
            "properties": {
                "epic": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "tasks": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "description", "tasks"],
                    },
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {"type": "string", "enum": ["coding", "architecture", "qa", "security", "planning"]},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "description", "type", "acceptance_criteria"],
                    },
                },
            },
            "required": ["epic", "milestones", "tasks"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Decompose this goal into an epic, milestones, and tasks:\n\n{goal_text}"},
        ]
        return await self.call_structured(messages, schema)

    async def decompose_goal(self, goal_text: str) -> dict:
        return await self.execute({"goal": goal_text})
