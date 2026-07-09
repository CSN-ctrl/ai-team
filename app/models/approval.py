from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ApprovalRequest(BaseModel):
    id: str
    task_id: str
    requested_by: str
    status: str = "pending"
    summary: str
    qa_report: Optional[str] = None
    security_report: Optional[str] = None
    final_review: Optional[str] = None
    rollback_plan: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_comment: Optional[str] = None
