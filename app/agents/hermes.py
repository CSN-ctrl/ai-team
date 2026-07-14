from app.agents.base import BaseAgent
from app.router import ModelRouter


class HermesAgent(BaseAgent):
    """Hermes — R&D research agent. Gathers intel from the internet,
    analyzes findings, and produces actionable research for the dev team."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "research"
        self.system_prompt = (
            "You are Hermes, the OpenClaw Research & Intelligence agent. "
            "You search the internet, analyze documentation, evaluate libraries "
            "and tools, gather community knowledge, and produce structured "
            "research reports. Every finding includes sources and reasoning. "
            "Your output feeds directly into the development team."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "detail": {"type": "string"},
                            "sources": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["topic", "detail", "sources"],
                    },
                },
                "recommendation": {"type": "string"},
                "next_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "findings", "recommendation"],
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Research the following thoroughly using internet sources and produce a structured report:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
