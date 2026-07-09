from app.agents.base import BaseAgent
from app.router import ModelRouter


class CEOAgent(BaseAgent):
    """CEO agent — decomposes goals into epics, milestones, and tasks."""

    def __init__(self, agent_id: str, name: str, router: ModelRouter):
        super().__init__(agent_id, name, router)
        self.capability = "planning"
        self.system_prompt = (
            "You are the OpenClaw CEO. Your role is to decompose software goals into "
            "epics, milestones, and tasks. You must never write code yourself. "
            "You must never approve your own work. You assign specialists to each task. "
            "Output structured plans only."
        )

    async def execute(self, input_data: dict) -> dict:
        goal_text = input_data.get("goal", "")
        if not goal_text:
            return {"error": "No goal provided", "epic": None, "milestones": [], "tasks": []}

        schema = {
            "type": "object",
            "properties": {
                "epic": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "tasks": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "description", "tasks"],
                    },
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {"type": "string", "enum": ["coding", "architecture", "qa", "security", "planning"]},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "description", "type", "acceptance_criteria"],
                    },
                },
            },
            "required": ["epic", "milestones", "tasks"],
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Decompose this goal into an epic, milestones, and tasks:\n\n{goal_text}"},
        ]
        return await self.call_structured(messages, schema)

    async def decompose_goal(self, goal_text: str) -> dict:
        return await self.execute({"goal": goal_text})
