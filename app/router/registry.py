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
    "planning":      {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "architecture":  {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "coding":        {"primary": "deepseek-ai/deepseek-v4-flash",           "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "debugging":     {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "deepseek-ai/deepseek-v4-flash"},
    "refactoring":   {"primary": "deepseek-ai/deepseek-v4-flash",           "fallback": "nvidia/nemotron-3-ultra-550b-a55b"},
    "qa":            {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "security":      {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "final_review":  {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "vision":        {"primary": "minimaxai/minimax-m3",                    "fallback": "nvidia/nemotron-3-ultra-550b-a55b"},
    "research":      {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
    "development":   {"primary": "nvidia/nemotron-3-ultra-550b-a55b",       "fallback": "nvidia/llama-3.3-nemotron-super-49b-v1"},
}

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

MODEL_META: dict[str, dict] = {
    "nemotron-super": {
        "provider": "nvidia",
        "max_tokens": 131_072,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "Llama-3.3-Nemotron-Super-49B — QA, security, planning",
    },
    "deepseek-v4-flash": {
        "provider": "nvidia",
        "max_tokens": 65_536,
        "supports_tool_calling": True,
        "supports_streaming": True,
        "description": "DeepSeek-V4-Flash — fast coding & refactoring",
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
        "description": "MiniMax-M3 — image understanding",
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
