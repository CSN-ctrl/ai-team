from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional


class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    READY = "ready"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    QA = "qa"
    SECURITY = "security"
    APPROVAL = "approval"
    RELEASE = "release"
    DONE = "done"
    CANCELLED = "cancelled"


class Task(BaseModel):
    id: str
    epic_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.BACKLOG
    assignee: Optional[str] = None
    priority: int = 3
    acceptance_criteria: list[str] = []
    dependencies: list[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Epic(BaseModel):
    id: str
    name: str
    description: str
    goal_id: str
    created_at: Optional[datetime] = None


class Goal(BaseModel):
    id: str
    text: str
    status: str = "pending"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
