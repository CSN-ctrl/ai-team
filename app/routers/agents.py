"""Agent registry endpoints — /v1/agents."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/v1")

# Static fallback registry used when no agent module is loaded.
_STATIC_AGENTS = [
    {
        "id": "ceo",
        "name": "CEO Agent",
        "role": "CEO / Orchestrator",
        "model": "z-ai/glm-5.2",
        "capabilities": ["planning", "architecture", "final_review"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "planner",
        "name": "Planner Agent",
        "role": "Planner",
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "capabilities": ["planning"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "architect",
        "name": "Architect Agent",
        "role": "Architect",
        "model": "z-ai/glm-5.2",
        "capabilities": ["architecture"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "coder",
        "name": "Coder Agent",
        "role": "Developer",
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "capabilities": ["coding", "refactoring"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "debugger",
        "name": "Debugger Agent",
        "role": "Debugger",
        "model": "qwen/qwen3.5-397b-a17b",
        "capabilities": ["debugging"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "qa",
        "name": "QA Agent",
        "role": "Quality Assurance",
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "capabilities": ["qa", "security"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "reviewer",
        "name": "Reviewer Agent",
        "role": "Final Reviewer",
        "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "capabilities": ["final_review"],
        "status": "idle",
        "current_task_id": None,
    },
    {
        "id": "vision",
        "name": "Vision Agent",
        "role": "Vision Analyst",
        "model": "minimaxai/minimax-m3",
        "capabilities": ["vision"],
        "status": "idle",
        "current_task_id": None,
    },
]


def _get_agents(app_state) -> list[dict]:
    """Return agents from app.state if available, otherwise static list."""
    registry = getattr(app_state, "agent_registry", None)
    if registry is not None:
        return [
            {
                "id": a.id,
                "name": a.name,
                "role": a.__class__.__name__,
                "model": _model_for_capability(getattr(a, "capability", "")),
                "capabilities": [a.capability] if a.capability else [],
                "status": "idle",
                "current_task_id": None,
            }
            for a in registry.values()
        ]
    return _STATIC_AGENTS


def _model_for_capability(capability: str) -> str:
    from app.router.registry import ROUTING_TABLE
    entry = ROUTING_TABLE.get(capability)
    if entry:
        return entry.get("primary", "unknown")
    return "unknown"


@router.get("/agents")
async def list_agents(request: Request) -> list[dict]:
    """List all available agents."""
    return _get_agents(request.app.state)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, request: Request) -> dict:
    """Return details for a specific agent."""
    agents = _get_agents(request.app.state)
    for agent in agents:
        if agent.get("id") == agent_id:
            return agent
    raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
