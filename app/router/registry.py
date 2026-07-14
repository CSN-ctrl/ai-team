"""
Model registry — routing table and metadata for NVIDIA NIM models.

Maps task capabilities to primary/fallback model pairs and stores
per-model metadata (provider, token limits, feature support).
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

# Actual NVIDIA NIM model IDs verified via GET /v1/models on 2026-07-09
# Using the user's API key against https://integrate.api.nvidia.com/v1

ROUTING_TABLE: dict[str, dict[str, str | None]] = {
    "planning":      {"primary": "z-ai/glm-5.2",                           "fallback": "qwen/qwen3-next-80b-a3b-instruct"},
    "architecture":  {"primary": "z-ai/glm-5.2",                           "fallback": "deepseek-ai/deepseek-v4-flash"},
    "coding":        {"primary": "qwen/qwen3-next-80b-a3b-instruct",        "fallback": "deepseek-ai/deepseek-v4-flash"},
    "debugging":     {"primary": "qwen/qwen3.5-397b-a17b",                  "fallback": "z-ai/glm-5.2"},
    "refactoring":   {"primary": "deepseek-ai/deepseek-v4-flash",           "fallback": "qwen/qwen3-next-80b-a3b-instruct"},
    "qa":            {"primary": "nvidia/llama-3.3-nemotron-super-49b-v1",  "fallback": "qwen/qwen3-next-80b-a3b-instruct"},
    "security":      {"primary": "nvidia/llama-3.3-nemotron-super-49b-v1",  "fallback": "z-ai/glm-5.2"},
    "final_review":  {"primary": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "fallback": "z-ai/glm-5.2"},
    "vision":        {"primary": "minimaxai/minimax-m3",                    "fallback": None},
    "research":      {"primary": "qwen/qwen3-next-80b-a3b-instruct",        "fallback": "deepseek-ai/deepseek-v4-flash"},
    "development":   {"primary": "qwen/qwen3-next-80b-a3b-instruct",        "fallback": "deepseek-ai/deepseek-v4-flash"},
}

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

MODEL_META: dict[str, dict] = {
    "glm-5.2": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "GLM-5.2 — strong planning & architecture reasoning",
    },
    "qwen3-coder": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "Qwen3-Coder — production coding & refactoring",
    },
    "deepseek-v4-flash": {
        "provider": "nvidia",
        "max_tokens": 65_536,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "DeepSeek-V4-Flash — fast refactoring & fallback coding",
    },
    "qwen3-thinking": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "Qwen3-Thinking — deep debugging & analysis",
    },
    "nemotron-super": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "Nemotron-Super — QA, security audits",
    },
    "nemotron-ultra": {
        "provider": "nvidia",
        "max_tokens": 262_144,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "Nemotron-Ultra — final review, maximum quality",
    },
    "minimax-vision": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": False,
        "supports_streaming": True,
        "description": "MiniMax-Vision — image understanding",
    },
}

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_model_for_capability(capability: str) -> tuple[str, str | None]:
    """Return (primary_model, fallback_model) for a given capability.

    Raises ``KeyError`` when the capability is not registered.
    """
    entry = ROUTING_TABLE[capability]
    return entry["primary"], entry["fallback"]


def get_fallback(model_name: str) -> str | None:
    """Return the fallback model for *model_name*, or *None*.

    Scans ``ROUTING_TABLE`` for the primary that matches *model_name*
    and returns its configured fallback.
    """
    for entry in ROUTING_TABLE.values():
        if entry["primary"] == model_name:
            return entry["fallback"]
    return None


def list_available_models() -> list[str]:
    """Return a sorted list of all known model names."""
    return sorted(MODEL_META.keys())
