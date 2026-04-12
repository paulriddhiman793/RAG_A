"""
BM25 keyword search — complements vector search for:
  - Exact term matching (model names, IDs, rare words)
  - Cases where semantic similarity fails (specific numbers, codes)

Built fresh from all stored chunks on startup, rebuilt after ingestion.

Persistence uses JSON for the chunk data and rebuilds the BM25Okapi
object from the stored corpus on load.  This avoids the arbitrary-code-
execution risk of Python pickle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from utils.logger import logger


_BM25_CACHE_PATH = Path("./bm25_index.json")
# Keep the old pickle path around so we can clean it up on first save.
_LEGACY_PICKLE_PATH = Path("./bm25_index.pkl")


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of chunk dicts."""
        self._chunks = chunks
        corpus = [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(chunks)} documents")
        self._save()

    def search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        """
        Search BM25 index. Returns list of result dicts with 'score' key.
        Scores are normalized to [0, 1].
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty — skipping keyword search")
            return []

        tokens = _tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)

        # Normalize
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalized = [s / max_score for s in raw_scores]

        scored = [
            {**self._chunks[i], "score": round(float(normalized[i]), 4)}
            for i in range(len(self._chunks))
            if normalized[i] > 0
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        self._save()

    def load(self) -> bool:
        """Load a cached index from disk. Returns True on success."""
        if not _BM25_CACHE_PATH.exists():
            return False
        try:
            raw = json.loads(_BM25_CACHE_PATH.read_text(encoding="utf-8"))
            self._chunks = raw["chunks"]
            corpus = [_tokenize(c.get("text", "")) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)
            logger.info(f"BM25 index loaded ({len(self._chunks)} docs)")
            return True
        except Exception as e:
            logger.warning(f"BM25 load failed: {e}")
            return False

    def _save(self) -> None:
        try:
            payload = {"chunks": self._chunks}
            _BM25_CACHE_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            # Remove legacy pickle file if it still exists
            if _LEGACY_PICKLE_PATH.exists():
                _LEGACY_PICKLE_PATH.unlink(missing_ok=True)
                logger.info("Removed legacy pickle BM25 cache")
        except Exception as e:
            logger.warning(f"BM25 save failed: {e}")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()