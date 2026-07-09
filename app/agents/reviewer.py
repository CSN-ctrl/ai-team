from app.agents.base import BaseAgent
from app.router import ModelRouter


class FinalReviewerAgent(BaseAgent):
    """Final reviewer — architecture consistency, code quality, rollback feasibility."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "final_review"
        self.system_prompt = (
            "You are the OpenClaw Final Reviewer. You perform the last review "
            "before human approval. You check architecture consistency, code "
            "quality, test adequacy, and rollback feasibility. "
            "Your verdict determines whether the release proceeds to human approval."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "revision", "block"]},
                "comments": {"type": "string"},
                "rollback_plan": {"type": "string"},
                "architecture_notes": {"type": "string"},
            },
            "required": ["verdict", "comments", "rollback_plan", "architecture_notes"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Perform final review:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
