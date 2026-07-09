from pydantic import BaseModel
from typing import Optional, Literal

AgentStatus = Literal["idle", "busy", "error"]
AgentCapability = Literal[
    "planning",
    "architecture",
    "coding",
    "debugging",
    "refactoring",
    "qa",
    "security",
    "final_review",
    "vision",
]


class Agent(BaseModel):
    id: str
    name: str
    role: str
    model: str
    capabilities: list[AgentCapability]
    status: AgentStatus = "idle"
    current_task_id: Optional[str] = None
