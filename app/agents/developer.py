from app.agents.base import BaseAgent
from app.router import ModelRouter


class DeveloperAgent(BaseAgent):
    """Development expert — implements features, fixes bugs, writes tests."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "development"
        self.system_prompt = (
            "You are the OpenClaw Development Expert. You implement features, "
            "fix bugs, write tests, and produce production-quality code. "
            "You follow best practices, handle edge cases, and document your work."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "plan": {"type": "string"},
                "implementation": {"type": "string"},
                "tests": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["plan", "implementation"],
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Implement the following:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
