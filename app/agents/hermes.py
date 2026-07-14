from app.agents.base import BaseAgent
from app.router import ModelRouter


class HermesAgent(BaseAgent):
    """Hermes — communication & coordination agent. Routes messages, synthesizes insights, keeps everyone aligned."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "coordination"
        self.system_prompt = (
            "You are Hermes, the OpenClaw Messenger & Coordinator. You synthesize "
            "information from multiple specialists, keep conversations coherent, "
            "escalate blockers, and ensure every agent has what they need. "
            "You are the nervous system of the AI team."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "action_items": {"type": "array", "items": {"type": "string"}},
                "blockers": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "action_items", "next_steps"],
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Coordinate the following:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
