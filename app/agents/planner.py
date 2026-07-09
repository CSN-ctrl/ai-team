from app.agents.base import BaseAgent
from app.router import ModelRouter


class PlannerAgent(BaseAgent):
    """Planner agent — validates dependencies, suggests ordering, adds acceptance criteria."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "planning"
        self.system_prompt = (
            "You are the OpenClaw Planner. You validate task dependencies, "
            "suggest optimal milestone ordering, and ensure every task has "
            "clear, testable acceptance criteria."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "milestone_order": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "refined_tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "estimated_effort": {"type": "string"},
                        },
                        "required": ["title", "acceptance_criteria", "dependencies"],
                    },
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["milestone_order", "refined_tasks", "risks"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Validate and refine this plan:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
