"""
Workflow classifier — auto-routes tasks to the right pipeline
based on title/description keywords. No LLM needed.
"""

# Keyword → workflow mapping (checked in priority order)
_WORKFLOW_RULES: list[tuple[list[str], str]] = [
    # Research — highest priority (explicit research intent)
    (["research", "investigate", "explore", "find", "learn",
      "study", "analyze", "hermes", "intel", "gather", "lookup",
      "search", "documentation", "discover", "compare", "evaluate"], "research_dev"),
    # Security
    (["security", "vulnerability", "audit", "exploit",
      "penetration", "cve", "owasp", "xss", "sqli", "csrf"], "security_review"),
    # Quick / trivial (before bugfix — "typo" could also contain "fix")
    (["quick", "simple", "tiny", "small", "minor", "trivial",
      "typo", "rename", "reword", "cosmetic"], "quick"),
    # Bugs & debugging
    (["bug", "error", "crash", "fix", "broken", "fail",
      "regression", "hotfix", "glitch", "malfunction"], "bugfix"),
    # Features & enhancements
    (["feature", "implement", "build", "create", "add",
      "new", "enhance", "improve", "develop", "pipeline",
      "integration", "setup", "support for", "auth"], "feature"),
    # Planning / architecture
    (["plan", "design", "architect", "schema", "blueprint"], "feature"),
    # QA / testing
    (["qa", "test", "coverage", "regression", "verify",
      "validate", "assert"], "bugfix"),
]


def classify_workflow(title: str, description: str = "") -> str:
    """Auto-detect the best workflow pipeline for a task.

    Scans the combined title + description for keywords and returns
    the most specific matching workflow type.
    """
    text = (title + " " + (description or "")).lower()

    for keywords, workflow in _WORKFLOW_RULES:
        if any(kw in text for kw in keywords):
            return workflow

    # Default: feature pipeline
    return "feature"


def describe_workflow(workflow_type: str) -> str:
    """Return a short human-readable description of a workflow."""
    descriptions = {
        "security_review": "Security audit pipeline",
        "bugfix": "Bugfix pipeline",
        "research_dev": "R&D pipeline",
        "feature": "Feature pipeline",
        "quick": "Quick pipeline",
    }
    return descriptions.get(workflow_type, workflow_type)
