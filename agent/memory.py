from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (session_id or "default").strip())
    return cleaned or "default"


class AgentMemoryStore:
    def __init__(self, memory_dir: str | None = None):
        self.memory_dir = Path(memory_dir or settings.AGENT_MEMORY_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            now = _utc_now()
            return {
                "session_id": _safe_session_id(session_id),
                "created_at": now,
                "updated_at": now,
                "history": [],
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            now = _utc_now()
            return {
                "session_id": _safe_session_id(session_id),
                "created_at": now,
                "updated_at": now,
                "history": [],
            }

    def save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _utc_now()
        path = self._session_path(str(session.get("session_id", "default")))
        path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_user_message(self, session_id: str, content: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        session["history"].append({
            "kind": "user",
            "timestamp": _utc_now(),
            "content": content,
        })
        self.save_session(session)
        return session

    def add_tool_step(
        self,
        session_id: str,
        thought: str,
        tool_name: str,
        tool_input: dict[str, Any],
        observation: str,
    ) -> dict[str, Any]:
        session = self.load_session(session_id)
        session["history"].append({
            "kind": "tool",
            "timestamp": _utc_now(),
            "thought": thought,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "observation": observation,
        })
        self.save_session(session)
        return session

    def add_assistant_message(self, session_id: str, content: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        session["history"].append({
            "kind": "assistant",
            "timestamp": _utc_now(),
            "content": content,
        })
        self.save_session(session)
        return session

    def build_planner_context(self, session_id: str, limit: int = 8) -> str:
        session = self.load_session(session_id)
        history = session.get("history", [])[-limit:]
        if not history:
            return "No prior session history."

        lines: list[str] = []
        for entry in history:
            kind = entry.get("kind", "unknown")
            if kind == "user":
                lines.append(f"User: {entry.get('content', '')}")
            elif kind == "assistant":
                lines.append(f"Assistant: {entry.get('content', '')}")
            elif kind == "tool":
                lines.append(
                    "Tool step: "
                    f"{entry.get('tool_name', '')} "
                    f"input={entry.get('tool_input', {})} "
                    f"observation={entry.get('observation', '')}"
                )
        return "\n".join(lines)

    def _session_path(self, session_id: str) -> Path:
        return self.memory_dir / f"{_safe_session_id(session_id)}.json"
