"""
Tests for agent session memory (agent/memory.py).
"""
import json
import tempfile
from pathlib import Path

import pytest

from agent.memory import AgentMemoryStore, _MAX_HISTORY_ENTRIES


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    return AgentMemoryStore(memory_dir=tmp)


class TestSessionLifecycle:
    def test_load_new_session(self, store):
        session = store.load_session("test-session")
        assert session["session_id"] == "test-session"
        assert session["history"] == []
        assert "created_at" in session

    def test_add_user_message(self, store):
        session = store.add_user_message("s1", "Hello!")
        assert len(session["history"]) == 1
        assert session["history"][0]["kind"] == "user"
        assert session["history"][0]["content"] == "Hello!"

    def test_add_assistant_message(self, store):
        store.add_user_message("s1", "Hello")
        session = store.add_assistant_message("s1", "Hi there!")
        assert len(session["history"]) == 2
        assert session["history"][-1]["kind"] == "assistant"

    def test_add_tool_step(self, store):
        session = store.add_tool_step(
            "s1", "thinking...", "search", {"query": "test"}, "found 3 results"
        )
        assert len(session["history"]) == 1
        assert session["history"][0]["kind"] == "tool"
        assert session["history"][0]["tool_name"] == "search"

    def test_persistence(self, store):
        store.add_user_message("s2", "persisted message")
        # Load again from disk
        session = store.load_session("s2")
        assert len(session["history"]) == 1
        assert session["history"][0]["content"] == "persisted message"


class TestHistoryTruncation:
    def test_truncation_at_limit(self, store):
        session = store.load_session("trunc")
        # Add more entries than the limit
        for i in range(_MAX_HISTORY_ENTRIES + 20):
            session["history"].append({
                "kind": "user",
                "timestamp": "2025-01-01T00:00:00",
                "content": f"message {i}",
            })
        store.save_session(session)

        reloaded = store.load_session("trunc")
        assert len(reloaded["history"]) == _MAX_HISTORY_ENTRIES
        # Most recent messages should be preserved
        assert reloaded["history"][-1]["content"] == f"message {_MAX_HISTORY_ENTRIES + 19}"

    def test_no_truncation_under_limit(self, store):
        store.add_user_message("small", "msg1")
        store.add_user_message("small", "msg2")
        session = store.load_session("small")
        assert len(session["history"]) == 2


class TestPlannerContext:
    def test_build_context(self, store):
        store.add_user_message("ctx", "Hello")
        store.add_assistant_message("ctx", "Hi!")
        ctx = store.build_planner_context("ctx")
        assert "User: Hello" in ctx
        assert "Assistant: Hi!" in ctx

    def test_empty_context(self, store):
        ctx = store.build_planner_context("empty")
        assert ctx == "No prior session history."

    def test_context_limit(self, store):
        for i in range(20):
            store.add_user_message("limited", f"msg {i}")
        ctx = store.build_planner_context("limited", limit=3)
        # Should only have last 3 entries
        lines = [l for l in ctx.split("\n") if l.strip()]
        assert len(lines) == 3


class TestSessionIdSanitization:
    def test_special_characters(self, store):
        session = store.add_user_message("../../../etc/passwd", "attack")
        # Sanitized ID should be safe
        assert ".." not in session["session_id"]

    def test_empty_session_id(self, store):
        session = store.load_session("")
        assert session["session_id"] == "default"
