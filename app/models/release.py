from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class Release(BaseModel):
    id: str
    goal_id: str
    gate: str = "G0"
    status: str = "active"
    artifacts: dict = {}
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
