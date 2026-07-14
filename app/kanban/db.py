"""Database manager for the OpenClaw Kanban engine.

Uses aiosqlite exclusively — no synchronous sqlite3 calls.
Schema defined here; all table creation happens via init_db().
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import aiosqlite

from app.models.approval import ApprovalRequest
from app.models.release import Release
from app.models.task import Epic, Goal, Task, TaskStatus


def _now() -> str:
    """ISO-8601 timestamp string (UTC)."""
    return datetime.utcnow().isoformat()


def _serialize(value: Any) -> str:
    """Serialize a Python value to a JSON string for TEXT column storage."""
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, default=str)


def _deserialize_list(raw: Optional[str]) -> list[str]:
    """Deserialize a TEXT column storing a JSON array of strings."""
    if raw is None or raw == "null":
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(v) for v in val]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _deserialize_dict(raw: Optional[str]) -> dict[str, Any]:
    """Deserialize a TEXT column storing a JSON object."""
    if raw is None or raw == "null":
        return {}
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _ts(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string from the DB into a datetime object."""
    if raw is None or raw == "null" or raw == "":
        return None
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ── Row → model helpers ──────────────────────────────────────────────────────


def _row_to_goal(row: aiosqlite.Row) -> Goal:
    return Goal(
        id=row["id"],
        text=row["text"],
        status=row["status"],
        created_at=_ts(row["created_at"]),
        updated_at=_ts(row["updated_at"]),
    )


def _row_to_epic(row: aiosqlite.Row) -> Epic:
    return Epic(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        goal_id=row["goal_id"],
        created_at=_ts(row["created_at"]),
    )


def _row_to_task(row: aiosqlite.Row) -> Task:
    return Task(
        id=row["id"],
        epic_id=row["epic_id"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]) if row["status"] else TaskStatus.backlog,
        assignee=row["assignee"],
        priority=row["priority"] if row["priority"] is not None else 0,
        acceptance_criteria=_deserialize_list(row["acceptance_criteria"]),
        dependencies=_deserialize_list(row["dependencies"]),
        created_at=_ts(row["created_at"]),
        updated_at=_ts(row["updated_at"]),
        workflow_type=row["workflow_type"] if row["workflow_type"] else None,
        workflow_step=row["workflow_step"] if row["workflow_step"] is not None else 0,
        workflow_output=row["workflow_output"] or "",
    )


def _row_to_approval(row: aiosqlite.Row) -> ApprovalRequest:
    return ApprovalRequest(
        id=row["id"],
        task_id=row["task_id"],
        requested_by=row["requested_by"],
        status=row["status"],
        summary=row["summary"],
        qa_report=row["qa_report"],
        security_report=row["security_report"],
        final_review=row["final_review"],
        rollback_plan=row["rollback_plan"],
        created_at=_ts(row["created_at"]),
        resolved_at=_ts(row["resolved_at"]),
        resolution_comment=row["resolution_comment"],
    )


def _row_to_release(row: aiosqlite.Row) -> Release:
    return Release(
        id=row["id"],
        goal_id=row["goal_id"],
        gate=row["gate"] or "",
        status=row["status"] or "pending",
        artifacts=_deserialize_dict(row["artifacts"]),
        created_at=_ts(row["created_at"]),
        completed_at=_ts(row["completed_at"]),
    )


# ── Schema DDL ────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    goal_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    epic_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'backlog',
    assignee TEXT,
    priority INTEGER NOT NULL DEFAULT 0,
    acceptance_criteria TEXT DEFAULT '[]',
    dependencies TEXT DEFAULT '[]',
    workflow_type TEXT,
    workflow_step INTEGER NOT NULL DEFAULT 0,
    workflow_output TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_requests (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    summary TEXT,
    qa_report TEXT,
    security_report TEXT,
    final_review TEXT,
    rollback_plan TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_comment TEXT
);

CREATE TABLE IF NOT EXISTS releases (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    gate TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    artifacts TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    completed_at TEXT
);
"""


# ── Database Manager ─────────────────────────────────────────────────────────


class DatabaseManager:
    """Manages the aiosqlite connection and provides row-to-model helpers."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def init_db(self) -> None:
        """Open connection, enable WAL, and run CREATE TABLE statements."""
        expanded = os.path.expanduser(self.db_path)
        if expanded != ":memory:":
            parent = os.path.dirname(expanded)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = await aiosqlite.connect(expanded)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(SCHEMA_SQL)
        # ── Migrations for columns added after v1 ─────────────────────
        for col, col_def in [("workflow_type", "TEXT"), ("workflow_step", "INTEGER NOT NULL DEFAULT 0"), ("workflow_output", "TEXT DEFAULT ''")]:
            try:
                await self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_def}")
            except aiosqlite.OperationalError:
                pass  # column already exists
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError(
                "Database not initialized. Call init_db() first."
            )
        return self._conn

    # ── Serialisation helpers (exposed for board.py) ──────────────────────

    @staticmethod
    def serialize(value: Any) -> str:
        return _serialize(value)

    @staticmethod
    def deserialize_list(raw: Optional[str]) -> list[str]:
        return _deserialize_list(raw)

    @staticmethod
    def now() -> str:
        return _now()

    # ── Row → model ───────────────────────────────────────────────────────

    @staticmethod
    def row_to_goal(row: aiosqlite.Row) -> Goal:
        return _row_to_goal(row)

    @staticmethod
    def row_to_epic(row: aiosqlite.Row) -> Epic:
        return _row_to_epic(row)

    @staticmethod
    def row_to_task(row: aiosqlite.Row) -> Task:
        return _row_to_task(row)

    @staticmethod
    def row_to_approval(row: aiosqlite.Row) -> ApprovalRequest:
        return _row_to_approval(row)

    @staticmethod
    def row_to_release(row: aiosqlite.Row) -> Release:
        return _row_to_release(row)


# ── Context manager dependency ────────────────────────────────────────────────


@asynccontextmanager
async def get_db(db_path: str = "~/.openclaw/kanban.db") -> AsyncGenerator[DatabaseManager, None]:
    """Context manager that yields a connected DatabaseManager.

    Usage::

        async with get_db(":memory:") as db:
            await db.init_db()
            # ... work with db.conn
    """
    mgr = DatabaseManager(db_path)
    try:
        await mgr.init_db()
        yield mgr
    finally:
        await mgr.close()
