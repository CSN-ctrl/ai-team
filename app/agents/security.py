from app.agents.base import BaseAgent
from app.router import ModelRouter


class SecurityAgent(BaseAgent):
    """Security reviewer — scans patches for vulnerabilities and secrets."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "security"
        self.system_prompt = (
            "You are the OpenClaw Security Reviewer. You review code changes "
            "for security vulnerabilities: hardcoded secrets, SQL injection, "
            "authentication flaws, dependency risks. You block releases with "
            "critical issues."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "file": {"type": "string"},
                        },
                        "required": ["severity", "title", "description"],
                    },
                },
                "pass": {"type": "boolean"},
                "critical_issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["findings", "pass", "critical_issues"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Review this patch for security issues:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
