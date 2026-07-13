"""Intent classifier — analyses user messages and determines the
appropriate conversation mode (Ask / Plan / Code / Auto).

Uses the NIMClient to run a lightweight classification prompt against
an NVIDIA NIM model.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.llm.client import NIMClient
from app.models.mode import Mode, ModeRoute

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for an AI coding assistant. Analyze the user's request and classify it into exactly one of these modes:

- ask: Simple Q&A, explanation requests, how-to questions, conceptual discussions. User wants information, not action.
- plan: Multi-step tasks, feature requests, "build X", "create Y", complex projects that need decomposition. User wants a structured plan first.
- code: Direct implementation requests, bug fixes, code changes, specific file edits. User wants immediate action with tools.

Also detect sub-modes for code:
- quick: Trivial one-line changes, simple edits
- debug: Error traces, "this is broken", unexpected behavior
- inline: "Change line X", "fix this function" (targeted, file-local)
- batch: "Do all of these", list of tasks
- null: Default code mode

Return ONLY valid JSON with NO markdown formatting, NO code blocks:
{"mode": "ask"|"plan"|"code", "sub_mode": null|"quick"|"debug"|"inline"|"batch", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


class IntentClassifier:
    """Uses an LLM to classify user messages into conversation modes.

    Parameters
    ----------
    llm_client:
        An NIMClient (or compatible async client) used to call the
        classification model.
    classify_model:
        Model name to use for classification.  Defaults to the same
        fast model used for general completions.
    """

    def __init__(
        self,
        llm_client: NIMClient,
        classify_model: Optional[str] = None,
    ) -> None:
        self.client = llm_client
        self.classify_model = classify_model or "deepseek-ai/deepseek-v4-flash"

    async def classify(self, messages: list[dict]) -> ModeRoute:
        """Classify the last user message in *messages* into a mode route.

        Falls back to ``Mode.AUTO`` with confidence ``0.0`` on any error
        (network failure, parse failure, etc.).
        """
        # Get last user message
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        classify_messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": last_user_msg[:2000]},  # Truncate long messages
        ]

        try:
            response = await self.client.chat_completion(
                model=self.classify_model,
                messages=classify_messages,
                temperature=0.1,
                max_tokens=300,
            )
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            # Strip code blocks if present
            text = text.strip()
            if text.startswith("```"):
                first_nl = text.find("\n")
                if first_nl != -1:
                    text = text[first_nl + 1:]
                closing = text.rfind("```")
                if closing != -1:
                    text = text[:closing]
            text = text.strip()

            data = json.loads(text)
            mode = Mode(data.get("mode", "auto"))
            sub_mode = data.get("sub_mode")
            confidence = data.get("confidence", 0.5)
            return ModeRoute(mode=mode, sub_mode=sub_mode, confidence=confidence)
        except Exception:
            logger.warning("Intent classification failed; falling back to auto", exc_info=True)
            return ModeRoute(mode=Mode.AUTO, confidence=0.0)
