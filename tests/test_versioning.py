"""
Tests for document versioning (ingestion/versioning.py).
"""
import tempfile
from pathlib import Path

import pytest

from ingestion.versioning import (
    stamp_chunks,
    recency_score,
    apply_recency_boost,
    _file_hash,
)


class TestStampChunks:
    def test_adds_metadata(self):
        chunks = [{"text": "hello", "metadata": {}}]
        stamped = stamp_chunks(chunks, "/tmp/test.pdf")
        meta = stamped[0]["metadata"]
        assert "ingested_at" in meta
        assert "doc_version" in meta
        assert meta["source_file"] == "/tmp/test.pdf"
        assert meta["is_deprecated"] is False

    def test_sets_filename(self):
        chunks = [{"text": "hello", "metadata": {}}]
        stamped = stamp_chunks(chunks, "/some/path/paper.pdf")
        assert stamped[0]["metadata"]["filename"] == "paper.pdf"

    def test_preserves_existing_filename(self):
        chunks = [{"text": "hello", "metadata": {"filename": "original.pdf"}}]
        stamped = stamp_chunks(chunks, "/new/path.pdf")
        assert stamped[0]["metadata"]["filename"] == "original.pdf"


class TestRecencyScore:
    def test_recent_is_high(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        score = recency_score(now)
        assert score >= 0.95

    def test_none_returns_default(self):
        score = recency_score(None)
        assert score == 0.75

    def test_returns_in_range(self):
        score = recency_score("2020-01-01T00:00:00+00:00")
        assert 0.5 <= score <= 1.0


class TestApplyRecencyBoost:
    def test_applies_boost(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        results = [
            {"score": 0.5, "metadata": {"ingested_at": now}},
            {"score": 0.5, "metadata": {"ingested_at": "2020-01-01T00:00:00+00:00"}},
        ]
        boosted = apply_recency_boost(results)
        # Recent document should have higher score
        assert boosted[0]["score"] >= boosted[1]["score"]

    def test_sorted_descending(self):
        results = [
            {"score": 0.3, "metadata": {}},
            {"score": 0.8, "metadata": {}},
        ]
        boosted = apply_recency_boost(results)
        assert boosted[0]["score"] >= boosted[1]["score"]


class TestFileHash:
    def test_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("test content")
            f.flush()
            h = _file_hash(f.name)
        assert len(h) == 32  # MD5 hex digest length

    def test_missing_file(self):
        h = _file_hash("/nonexistent/file.pdf")
        assert len(h) == 32  # Falls back to hashing the path string
