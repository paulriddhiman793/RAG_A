"""
Tests for the BM25 keyword search index (indexing/bm25_index.py).
"""
import json
import tempfile
from pathlib import Path

import pytest

from indexing.bm25_index import BM25Index


@pytest.fixture
def sample_chunks():
    return [
        {"chunk_id": "c1", "text": "machine learning algorithms for prediction", "metadata": {}, "score": 0.0},
        {"chunk_id": "c2", "text": "deep neural network architecture design", "metadata": {}, "score": 0.0},
        {"chunk_id": "c3", "text": "random forest regression model evaluation", "metadata": {}, "score": 0.0},
        {"chunk_id": "c4", "text": "support vector machine classification accuracy", "metadata": {}, "score": 0.0},
    ]


@pytest.fixture
def bm25(sample_chunks, monkeypatch):
    """Build a BM25 index with sample data, using a temp cache path."""
    tmp = Path(tempfile.mkdtemp()) / "bm25_test.json"
    monkeypatch.setattr("indexing.bm25_index._BM25_CACHE_PATH", tmp)
    idx = BM25Index()
    idx.build(sample_chunks)
    return idx


class TestBM25Build:
    def test_build_creates_index(self, bm25):
        assert bm25._bm25 is not None
        assert len(bm25._chunks) == 4

    def test_build_saves_json(self, bm25, monkeypatch):
        import indexing.bm25_index as mod
        cache_path = mod._BM25_CACHE_PATH
        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "chunks" in data
        assert len(data["chunks"]) == 4


class TestBM25Search:
    def test_search_returns_results(self, bm25):
        results = bm25.search("random forest")
        assert len(results) > 0

    def test_search_top_result_is_relevant(self, bm25):
        results = bm25.search("random forest")
        top = results[0]
        assert "random" in top["text"].lower() or "forest" in top["text"].lower()

    def test_search_scores_normalized(self, bm25):
        results = bm25.search("machine learning")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_search_respects_top_k(self, bm25):
        results = bm25.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []


class TestBM25Persistence:
    def test_save_and_load(self, sample_chunks, monkeypatch):
        tmp = Path(tempfile.mkdtemp()) / "bm25_roundtrip.json"
        monkeypatch.setattr("indexing.bm25_index._BM25_CACHE_PATH", tmp)

        idx = BM25Index()
        idx.build(sample_chunks)

        # Create a fresh index and load from disk
        idx2 = BM25Index()
        loaded = idx2.load()
        assert loaded is True
        assert len(idx2._chunks) == 4

        # Loaded index should produce same search results
        r1 = idx.search("neural network")
        r2 = idx2.search("neural network")
        assert [r["chunk_id"] for r in r1] == [r["chunk_id"] for r in r2]

    def test_load_missing_file(self, monkeypatch):
        tmp = Path(tempfile.mkdtemp()) / "does_not_exist.json"
        monkeypatch.setattr("indexing.bm25_index._BM25_CACHE_PATH", tmp)
        idx = BM25Index()
        assert idx.load() is False

    def test_no_pickle_file_created(self, sample_chunks, monkeypatch):
        """Ensure we never revert to pickle serialization."""
        tmp = Path(tempfile.mkdtemp())
        monkeypatch.setattr("indexing.bm25_index._BM25_CACHE_PATH", tmp / "bm25.json")
        monkeypatch.setattr("indexing.bm25_index._LEGACY_PICKLE_PATH", tmp / "bm25.pkl")
        idx = BM25Index()
        idx.build(sample_chunks)
        assert not (tmp / "bm25.pkl").exists()
        assert (tmp / "bm25.json").exists()
