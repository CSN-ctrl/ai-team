from app.agents.base import BaseAgent
from app.router import ModelRouter


class QAAgent(BaseAgent):
    """QA agent — generates test cases and validates patches."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "qa"
        self.system_prompt = (
            "You are the OpenClaw QA Lead. You generate test cases from "
            "acceptance criteria, review patches for correctness, and assess "
            "test coverage. You never approve your own work."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "test_cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {"type": "string", "enum": ["unit", "integration", "e2e"]},
                            "expected_result": {"type": "string"},
                        },
                        "required": ["name", "description", "type", "expected_result"],
                    },
                },
                "coverage_assessment": {"type": "string"},
                "risks": {"type": "array", "items": {"type": "string"}},
                "pass": {"type": "boolean"},
            },
            "required": ["test_cases", "coverage_assessment", "risks", "pass"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Review this task and patch:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
