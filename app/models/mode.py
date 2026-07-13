"""Mode routing models for the OpenClaw CEO.

Defines the conversation modes (Ask/Plan/Code/Auto), mode routing
results, and an in-memory conversation state store.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Mode(str, Enum):
    """The four conversation modes of the CEO router."""

    ASK = "ask"
    PLAN = "plan"
    CODE = "code"
    AUTO = "auto"


class ModeRoute(BaseModel):
    """Result of intent classification — a mode plus optional sub-mode."""

    mode: Mode
    sub_mode: Optional[str] = None  # quick, debug, inline, batch for code
    confidence: float = 1.0


class ConversationState(BaseModel):
    """Per-conversation state tracked in memory."""

    conversation_id: str
    mode: Mode = Mode.AUTO
    plan: Optional[str] = None
    task_ids: list[str] = []
    created_at: Optional[datetime] = None


class ConversationManager:
    """In-memory conversation state store.

    Each conversation is identified by a ``X-Conversation-Id`` header
    (or an auto-generated UUID if omitted).  State lives only in memory
    and is lost on process restart.
    """

    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}

    def get_or_create(self, conv_id: str) -> ConversationState:
        """Return the state for *conv_id*, creating a new one if absent."""
        if conv_id not in self._store:
            self._store[conv_id] = ConversationState(
                conversation_id=conv_id,
                created_at=datetime.utcnow(),
            )
        return self._store[conv_id]

    def update_mode(self, conv_id: str, mode: Mode) -> None:
        """Update the mode for *conv_id*."""
        state = self.get_or_create(conv_id)
        state.mode = mode

    def set_plan(self, conv_id: str, plan: str, task_ids: list[str]) -> None:
        """Store a plan and its associated task IDs for *conv_id*."""
        state = self.get_or_create(conv_id)
        state.plan = plan
        state.task_ids = task_ids
