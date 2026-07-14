from app.agents.base import BaseAgent
from app.router import ModelRouter


class ResearcherAgent(BaseAgent):
    """Research agent — investigates codebases, docs, and finds optimal solutions."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "research"
        self.system_prompt = (
            "You are the OpenClaw Research Expert. You analyze codebases, "
            "search documentation, evaluate library options, and produce "
            "evidence-backed recommendations. You never guess — you cite sources."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "findings": {"type": "array", "items": {"type": "string"}},
                "recommendation": {"type": "string"},
                "alternatives": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["findings", "recommendation", "confidence"],
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Research the following:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
