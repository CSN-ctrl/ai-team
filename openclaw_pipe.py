"""
OpenClaw CEO Pipe — register OpenClaw as a model provider in Open WebUI.

Installation:
1. Copy this file to Open WebUI's functions directory
2. Or paste into Admin Panel → Functions → Create Function
3. Select "OpenClaw CEO" from the model dropdown to chat with the CEO
"""
import asyncio
import json
import time
from typing import List, Dict, Optional, Any

import requests


class Pipe:
    """Registers OpenClaw CEO as a model in Open WebUI."""

    class Valves:
        def __init__(self):
            self.openclaw_url: str = "http://localhost:8282"
            self.openclaw_api_key: str = "openclaw-ceo-admin-key"
            self.poll_interval_seconds: float = 5.0
            self.max_poll_time_seconds: float = 300.0

    def __init__(self):
        self.type = "manifold"
        self.id = "openclaw"
        self.name = "OpenClaw"
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        """Return available models — currently just the CEO."""
        return [
            {"id": "ceo", "name": "OpenClaw CEO"},
        ]

    async def pipe(self, body: Dict) -> str:
        """Process a user message through OpenClaw CEO.

        Sends the last user message as a goal to OpenClaw, polls for
        completion, and returns a formatted summary.
        """
        messages = body.get("messages", [])
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m["content"]
                break

        if not user_msg:
            return "No user message found."

        # Submit goal to OpenClaw CEO
        url = f"{self.valves.openclaw_url}/v1/goals"
        headers = {"Content-Type": "application/json"}
        if self.valves.openclaw_api_key:
            headers["X-API-Key"] = self.valves.openclaw_api_key

        try:
            resp = requests.post(url, json={"goal": user_msg}, headers=headers, timeout=120)
            resp.raise_for_status()
            goal_data = resp.json()
        except requests.RequestException as e:
            return f"Failed to submit goal to OpenClaw: {e}"

        goal_id = goal_data.get("id")
        if not goal_id:
            return f"OpenClaw did not return a goal ID: {goal_data}"

        # Poll for completion
        start = time.time()
        status_url = f"{self.valves.openclaw_url}/v1/goals/{goal_id}"

        while time.time() - start < self.valves.max_poll_time_seconds:
            await asyncio.sleep(self.valves.poll_interval_seconds)

            try:
                status_resp = requests.get(status_url, headers=headers, timeout=30)
                status_resp.raise_for_status()
                status = status_resp.json()
            except requests.RequestException:
                continue

            goal = status.get("goal", {})
            goal_status = goal.get("status", "unknown")

            if goal_status in ("done", "failed", "approved"):
                return self._format_summary(status)

            if goal_status == "delegated":
                # Check if tasks were created
                tasks = status.get("tasks", [])
                if tasks:
                    return self._format_summary(status)

        return f"Goal {goal_id} is still processing. Check the OpenClaw dashboard for updates."

    def _format_summary(self, data: Dict) -> str:
        """Format goal status into a readable summary."""
        goal = data.get("goal", {})
        epics = data.get("epics", [])
        tasks = data.get("tasks", [])

        lines = [
            f"## Goal: {goal.get('text', 'Untitled')}",
            f"**Status:** {goal.get('status', 'unknown')}",
            f"**ID:** {goal.get('id', 'N/A')}",
            "",
        ]

        if epics:
            lines.append(f"### Epics ({len(epics)})")
            for e in epics:
                lines.append(f"- **{e.get('name', 'Unnamed')}**: {e.get('description', '')}")
            lines.append("")

        if tasks:
            lines.append(f"### Tasks ({len(tasks)})")
            for t in tasks:
                lines.append(
                    f"- [{t.get('status', '?').upper()}] **{t.get('title', 'Untitled')}** "
                    f"— *{t.get('assignee', 'unassigned')}*"
                )
            lines.append("")

        lines.append("---")
        lines.append("Use the OpenClaw dashboard at `/dashboard` for detailed status.")

        return "\n".join(lines)
