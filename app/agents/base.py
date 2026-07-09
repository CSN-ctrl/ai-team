import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.router import ModelRouter

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all OpenClaw specialist agents."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        self.id = agent_id
        self.name = name
        self.router = router
        self.capability: str = ""
        self.system_prompt: str = ""

    async def call(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096) -> dict:
        return await self.router.route(self.capability, messages)

    async def call_structured(self, messages: list[dict], output_schema: dict, temperature: float = 0.3) -> dict:
        schema_instruction = (
            f"\n\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(output_schema, indent=2)}\n\n"
            f"Output ONLY valid JSON. No markdown, no explanation."
        )
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            enhanced_messages[-1]["content"] += schema_instruction
        else:
            enhanced_messages.append({"role": "user", "content": schema_instruction})

        for attempt in range(3):
            try:
                response = await self.router.route(self.capability, enhanced_messages)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                return json.loads(content)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Structured output parse failed (attempt %d): %s", attempt + 1, e)
                if attempt == 2:
                    raise ValueError(f"Failed to parse structured output after 3 attempts") from e
        return {}

    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        ...
