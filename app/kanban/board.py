"""AsyncKanbanBoard — the core persistence API for the OpenClaw CEO engine.

All CRUD operations for goals, epics, tasks, approvals, and releases.
JSON serialization/deserialization is handled transparently for TEXT columns.
Timestamp conversion between Pydantic ``datetime`` objects and ISO strings
is handled at the DB boundary.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

import aiosqlite

from app.kanban.db import DatabaseManager, _now as _iso_now
from app.kanban.transitions import is_valid_transition
from app.models.approval import ApprovalRequest
from app.models.release import Release
from app.models.task import Epic, Goal, Task, TaskStatus


def _now_dt() -> datetime:
    """Current UTC timestamp as a datetime (matching Pydantic model types)."""
    return datetime.utcnow()


def _fmt(dt: Optional[datetime]) -> str:
    """Format a datetime (or None) to an ISO string for SQLite binding."""
    if dt is None:
        return ""
    return dt.isoformat()


class AsyncKanbanBoard:
    """High-level board abstraction over the kanban SQLite store."""

    def __init__(self, db_path: str = "~/.openclaw/kanban.db") -> None:
        self._db = DatabaseManager(db_path)

    async def init_db(self) -> None:
        """Initialise the database schema (safe to call multiple times)."""
        await self._db.init_db()

    async def close(self) -> None:
        await self._db.close()

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _now() -> datetime:
        return _now_dt()

    async def _execute(self, sql: str, params: tuple = ()) -> None:
        await self._db.conn.execute(sql, params)

    async def _fetchone(self, sql: str, params: tuple = ()) -> Optional[aiosqlite.Row]:
        cur = await self._db.conn.execute(sql, params)
        return await cur.fetchone()

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cur = await self._db.conn.execute(sql, params)
        return await cur.fetchall()

    # ── Goals ─────────────────────────────────────────────────────────────

    async def create_goal(self, goal: Goal) -> Goal:
        """Persist a new goal. Fills in missing timestamps."""
        now = self._now()
        if not goal.created_at:
            goal.created_at = now
        if not goal.updated_at:
            goal.updated_at = now
        await self._execute(
            "INSERT OR REPLACE INTO goals (id, text, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (goal.id, goal.text, goal.status, _fmt(goal.created_at), _fmt(goal.updated_at)),
        )
        await self._db.conn.commit()
        return goal

    async def get_goal(self, goal_id: str) -> Optional[Goal]:
        row = await self._fetchone("SELECT * FROM goals WHERE id = ?", (goal_id,))
        if row is None:
            return None
        return DatabaseManager.row_to_goal(row)

    async def update_goal_status(self, goal_id: str, status: str) -> None:
        await self._execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
            (status, _iso_now(), goal_id),
        )
        await self._db.conn.commit()

    async def list_goals(self) -> list[Goal]:
        """Return all goals ordered by creation time (newest first)."""
        rows = await self._fetchall(
            "SELECT * FROM goals ORDER BY created_at DESC"
        )
        return [DatabaseManager.row_to_goal(r) for r in rows]

    # ── Epics ─────────────────────────────────────────────────────────────

    async def create_epic(self, epic: Epic) -> Epic:
        if not epic.created_at:
            epic.created_at = self._now()
        await self._execute(
            "INSERT OR REPLACE INTO epics (id, name, description, goal_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (epic.id, epic.name, epic.description, epic.goal_id, _fmt(epic.created_at)),
        )
        await self._db.conn.commit()
        return epic

    async def get_epics(self, goal_id: str) -> list[Epic]:
        rows = await self._fetchall(
            "SELECT * FROM epics WHERE goal_id = ? ORDER BY created_at", (goal_id,)
        )
        return [DatabaseManager.row_to_epic(r) for r in rows]

    # ── Tasks ─────────────────────────────────────────────────────────────

    async def create_task(self, task: Task) -> Task:
        now = self._now()
        if not task.created_at:
            task.created_at = now
        if not task.updated_at:
            task.updated_at = now
        await self._execute(
            "INSERT OR REPLACE INTO tasks "
            "(id, epic_id, title, description, status, assignee, priority, "
            " acceptance_criteria, dependencies, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task.id,
                task.epic_id,
                task.title,
                task.description,
                task.status.value if isinstance(task.status, TaskStatus) else task.status,
                task.assignee,
                task.priority,
                DatabaseManager.serialize(task.acceptance_criteria),
                DatabaseManager.serialize(task.dependencies),
                _fmt(task.created_at),
                _fmt(task.updated_at),
            ),
        )
        await self._db.conn.commit()
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        row = await self._fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            return None
        return DatabaseManager.row_to_task(row)

    async def list_tasks(
        self,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
    ) -> list[Task]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if assignee is not None:
            clauses.append("assignee = ?")
            params.append(assignee)
        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        rows = await self._fetchall(
            f"SELECT * FROM tasks {where} ORDER BY priority DESC, created_at",
            tuple(params),
        )
        return [DatabaseManager.row_to_task(r) for r in rows]

    async def update_task(self, task_id: str, updates: dict) -> Optional[Task]:
        """Apply partial updates to a task. Enforces valid status transitions.

        If ``updates`` contains a ``status`` key the transition is validated
        against ``is_valid_transition`` — raises ``ValueError`` on invalid move.
        """
        current = await self.get_task(task_id)
        if current is None:
            return None

        new_status = updates.get("status")
        if new_status is not None:
            from_val = (
                current.status.value
                if isinstance(current.status, TaskStatus)
                else str(current.status)
            )
            to_val = (
                new_status.value
                if isinstance(new_status, TaskStatus)
                else str(new_status)
            )
            if not is_valid_transition(from_val, to_val):
                raise ValueError(
                    f"Invalid status transition: {from_val!r} → {to_val!r}"
                )
            updates["status"] = to_val

        if not updates:
            return current

        updates["updated_at"] = _iso_now()
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        await self._execute(
            f"UPDATE tasks SET {set_clauses} WHERE id = ?",
            tuple(values),
        )
        await self._db.conn.commit()
        return await self.get_task(task_id)

    # ── Approvals ─────────────────────────────────────────────────────────

    async def create_approval(self, req: ApprovalRequest) -> ApprovalRequest:
        if not req.created_at:
            req.created_at = _now_dt()
        await self._execute(
            "INSERT OR REPLACE INTO approval_requests "
            "(id, task_id, requested_by, status, summary, qa_report, security_report, "
            " final_review, rollback_plan, created_at, resolved_at, resolution_comment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                req.id,
                req.task_id,
                req.requested_by,
                req.status,
                req.summary,
                req.qa_report,
                req.security_report,
                req.final_review,
                req.rollback_plan,
                _fmt(req.created_at),
                _fmt(req.resolved_at),
                req.resolution_comment,
            ),
        )
        await self._db.conn.commit()
        return req

    async def list_approvals(
        self, status: Optional[str] = None
    ) -> list[ApprovalRequest]:
        if status:
            rows = await self._fetchall(
                "SELECT * FROM approval_requests WHERE status = ? ORDER BY created_at",
                (status,),
            )
        else:
            rows = await self._fetchall(
                "SELECT * FROM approval_requests ORDER BY created_at"
            )
        return [DatabaseManager.row_to_approval(r) for r in rows]

    async def resolve_approval(
        self,
        approval_id: str,
        status: str,
        comment: Optional[str] = None,
    ) -> None:
        await self._execute(
            "UPDATE approval_requests SET status = ?, resolved_at = ?, resolution_comment = ? "
            "WHERE id = ?",
            (status, _iso_now(), comment, approval_id),
        )
        await self._db.conn.commit()

    # ── Releases ──────────────────────────────────────────────────────────

    async def create_release(self, release: Release) -> Release:
        if not release.created_at:
            release.created_at = _now_dt()
        await self._execute(
            "INSERT OR REPLACE INTO releases "
            "(id, goal_id, gate, status, artifacts, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                release.id,
                release.goal_id,
                release.gate,
                release.status,
                DatabaseManager.serialize(release.artifacts),
                _fmt(release.created_at),
                _fmt(release.completed_at),
            ),
        )
        await self._db.conn.commit()
        return release

    async def get_release(self, goal_id: str) -> Optional[Release]:
        row = await self._fetchone(
            "SELECT * FROM releases WHERE goal_id = ? ORDER BY created_at DESC LIMIT 1",
            (goal_id,),
        )
        if row is None:
            return None
        return DatabaseManager.row_to_release(row)

    async def update_release_gate(self, release_id: str, gate: str) -> None:
        await self._execute(
            "UPDATE releases SET gate = ? WHERE id = ?",
            (gate, release_id),
        )
        await self._db.conn.commit()

    async def get_release_by_id(self, release_id: str) -> Optional[Release]:
        """Return a release by its own ID."""
        row = await self._fetchone(
            "SELECT * FROM releases WHERE id = ?",
            (release_id,),
        )
        if row is None:
            return None
        return DatabaseManager.row_to_release(row)

    async def list_releases(self) -> list[Release]:
        """Return all releases ordered by creation time (newest first)."""
        rows = await self._fetchall(
            "SELECT * FROM releases ORDER BY created_at DESC"
        )
        return [DatabaseManager.row_to_release(r) for r in rows]
