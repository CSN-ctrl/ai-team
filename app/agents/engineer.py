from app.agents.base import BaseAgent
from app.router import ModelRouter


class EngineerAgent(BaseAgent):
    """Engineer agent — produces implementation plans for coding tasks."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "coding"
        self.system_prompt = (
            "You are an OpenClaw Engineer. You produce implementation plans "
            "for software tasks: approach, files to modify, risks. "
            "You do not write code directly — the OpenHands bridge executes it."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "approach": {"type": "string"},
                "files_to_modify": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "change_type": {"type": "string", "enum": ["create", "modify", "delete"]},
                            "description": {"type": "string"},
                        },
                        "required": ["path", "change_type", "description"],
                    },
                },
                "risks": {"type": "array", "items": {"type": "string"}},
                "test_strategy": {"type": "string"},
            },
            "required": ["approach", "files_to_modify", "risks", "test_strategy"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Create an implementation plan for this task:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
