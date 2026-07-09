from app.agents.base import BaseAgent
from app.router import ModelRouter


class DebuggerAgent(BaseAgent):
    """Debugger agent — root cause analysis and fix strategy."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "debugging"
        self.system_prompt = (
            "You are the OpenClaw Debugger. You analyze failures, identify "
            "root causes, and propose fix strategies. You do not implement "
            "fixes yourself — you provide analysis for engineers."
        )

    async def execute(self, input_data: dict) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string"},
                "fix_strategy": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "files_to_inspect": {"type": "array", "items": {"type": "string"}},
                "suggested_fixes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["file", "description"],
                    },
                },
            },
            "required": ["root_cause", "fix_strategy", "confidence"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Analyze this failure:\n\n{input_data}"},
        ]
        return await self.call_structured(messages, schema)
