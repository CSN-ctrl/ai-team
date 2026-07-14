"""
WorkflowEngine — multi-agent team pipelines.

Each task can follow a *workflow* — a sequence of steps where each step
is handled by a different specialist agent. When a step completes, the
engine advances the task to the next step and assigns the next agent.
This is how the team works together like a real dev team.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.models.task import Task

logger = logging.getLogger(__name__)


# ── Workflow definitions ──────────────────────────────────────────────


@dataclass
class WorkflowStep:
    """A single stage in a multi-agent workflow pipeline."""

    name: str  # capability name (research, coding, qa, etc.)
    agent_id: str  # which agent handles this step
    label: str  # human-readable name shown in UI
    description: str  # what happens in this step


@dataclass
class WorkflowDef:
    """Complete workflow pipeline definition."""

    label: str  # display name (e.g. "R&D Pipeline")
    steps: list[WorkflowStep] = field(default_factory=list)


# ── Built-in pipelines ────────────────────────────────────────────────
# Each workflow defines how a team of agents collaborates on a task.
# Steps run in sequence — when one agent finishes, the next picks up.

WORKFLOWS: dict[str, WorkflowDef] = {
    "research_dev": WorkflowDef(
        label="R&D Pipeline",
        steps=[
            WorkflowStep("research", "hermes", "Research",
                         "Internet research & intel gathering"),
            WorkflowStep("development", "dev-exp", "Develop",
                         "Build solution from research"),
            WorkflowStep("qa", "qa", "QA",
                         "Quality verification & testing"),
            WorkflowStep("final_review", "rev", "Review",
                         "Final code review"),
            WorkflowStep("evaluation", "ceo", "CEO Eval",
                         "CEO evaluation & distribution"),
        ],
    ),
    "feature": WorkflowDef(
        label="Feature Pipeline",
        steps=[
            WorkflowStep("planning", "planner", "Plan",
                         "Architecture & implementation plan"),
            WorkflowStep("coding", "eng-a", "Implement",
                         "Feature implementation"),
            WorkflowStep("debugging", "dbg", "Debug",
                         "Bug fixing & edge cases"),
            WorkflowStep("qa", "qa", "QA",
                         "Test coverage & verification"),
            WorkflowStep("security", "sec", "Security",
                         "Security audit"),
            WorkflowStep("final_review", "rev", "Review",
                         "Final review & approval"),
            WorkflowStep("evaluation", "ceo", "CEO Eval",
                         "CEO evaluation & sign-off"),
        ],
    ),
    "bugfix": WorkflowDef(
        label="Bugfix Pipeline",
        steps=[
            WorkflowStep("debugging", "dbg", "Diagnose",
                         "Root cause analysis"),
            WorkflowStep("coding", "eng-b", "Fix",
                         "Implement fix"),
            WorkflowStep("qa", "qa", "Verify",
                         "Verify fix & regression check"),
            WorkflowStep("final_review", "rev", "Review",
                         "Quick final review"),
        ],
    ),
    "security_review": WorkflowDef(
        label="Security Pipeline",
        steps=[
            WorkflowStep("security", "sec", "Audit",
                         "Vulnerability scan & audit"),
            WorkflowStep("coding", "eng-a", "Remediate",
                         "Fix security findings"),
            WorkflowStep("security", "sec", "Re-audit",
                         "Verify remediations"),
            WorkflowStep("final_review", "rev", "Review",
                         "Final security sign-off"),
        ],
    ),
    "quick": WorkflowDef(
        label="Quick Pipeline",
        steps=[
            WorkflowStep("coding", "eng-a", "Implement",
                         "Direct implementation"),
            WorkflowStep("final_review", "rev", "Review",
                         "Quick review & close"),
        ],
    ),
}


# ── Workflow Engine ───────────────────────────────────────────────────


class WorkflowEngine:
    """Manages per-task pipeline state and step transitions.

    Usage::

        engine = WorkflowEngine()
        next_agent = engine.get_next_agent(task)  # who's up next?
        engine.advance_task(task)                  # move to next step
    """

    def __init__(self) -> None:
        self._workflows = WORKFLOWS

    # ── Public API ────────────────────────────────────────────────────

    def get_workflow(self, wf_type: str) -> Optional[WorkflowDef]:
        """Return the workflow definition for *wf_type*, or None."""
        return self._workflows.get(wf_type)

    def list_workflows(self) -> dict[str, str]:
        """Return {workflow_type: label} for all registered workflows."""
        return {k: v.label for k, v in self._workflows.items()}

    def get_current_step(self, task: Task) -> Optional[WorkflowStep]:
        """Return the current step for *task* based on its pipeline position."""
        wf = self.get_workflow(task.workflow_type)
        if wf is None:
            return None
        step_idx = task.workflow_step
        if step_idx < 0 or step_idx >= len(wf.steps):
            return None
        return wf.steps[step_idx]

    def get_next_step(self, task: Task) -> Optional[WorkflowStep]:
        """Return the *next* step after the current one (or None if done)."""
        wf = self.get_workflow(task.workflow_type)
        if wf is None:
            return None
        next_idx = task.workflow_step + 1
        if next_idx >= len(wf.steps):
            return None  # pipeline complete
        return wf.steps[next_idx]

    def get_next_agent(self, task: Task) -> Optional[str]:
        """Return the agent ID that should handle the current step."""
        step = self.get_current_step(task)
        if step is None:
            return None
        return step.agent_id

    def get_step_count(self, task: Task) -> int:
        """Total number of steps in this task's workflow."""
        wf = self.get_workflow(task.workflow_type)
        if wf is None:
            return 0
        return len(wf.steps)

    def advance(self, task: Task) -> Task:
        """Move *task* to the next workflow step.

        Returns the updated Task with incremented step index.
        The caller is responsible for persisting the change.
        """
        wf = self.get_workflow(task.workflow_type)
        if wf is None:
            logger.warning("advance: no workflow %r for task %s", task.workflow_type, task.id)
            return task

        next_idx = task.workflow_step + 1
        if next_idx >= len(wf.steps):
            logger.info("Task %s — pipeline complete (step %d/%d)", task.id, task.workflow_step, len(wf.steps))
            return task

        task.workflow_step = next_idx
        task.assignee = None  # unassign — next tick will re-assign to next agent
        task.status = "backlog"  # recycle for next-stage assignment

        next_step = wf.steps[next_idx]
        logger.info(
            "Task %s advanced to step %d/%d → %s (%s)",
            task.id, next_idx + 1, len(wf.steps),
            next_step.label, next_step.agent_id,
        )
        return task

    def is_pipeline_done(self, task: Task) -> bool:
        """Check if *task* has completed all workflow steps."""
        wf = self.get_workflow(task.workflow_type)
        if wf is None:
            return True  # no workflow = done
        return task.workflow_step >= len(wf.steps)
