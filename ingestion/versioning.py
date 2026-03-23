"""
Document versioning — manages:
  - Ingestion timestamps on every chunk
  - Soft-delete (mark old chunks deprecated without removing them)
  - Recency decay scoring used during retrieval
  - Document version registry
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import logger
from config import settings


# ── Version registry (lightweight JSON file) ─────────────────────────

_REGISTRY_PATH = Path("./version_registry.json")


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_registry(reg: dict) -> None:
    _REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))


# ── Public API ────────────────────────────────────────────────────────

def stamp_chunks(
    chunks: list[dict],
    source_file: str,
    version: str | None = None,
) -> list[dict]:
    """
    Add versioning metadata to every chunk dict:
        ingested_at  : ISO timestamp (UTC)
        doc_version  : version string (hash of file content or provided)
        source_file  : original file path
        is_deprecated: False (set to True when superseded)
    """
    now = datetime.now(timezone.utc).isoformat()
    ver = version or _file_hash(source_file)

    reg = _load_registry()
    previous_version = reg.get(source_file, {}).get("current_version")
    reg[source_file] = {
        "current_version": ver,
        "previous_version": previous_version,
        "last_ingested": now,
    }
    _save_registry(reg)

    stamped = []
    for ch in chunks:
        meta = ch.get("metadata", {}) if isinstance(ch, dict) else ch.metadata
        # Always set filename — ensures ParentChunks (built before stamping)
        # get the correct filename for the sources table in query results.
        _fname = Path(source_file).name
        # Only overwrite filename if it's missing or blank
        if not meta.get("filename"):
            meta["filename"] = _fname
        meta.update({
            "ingested_at":  now,
            "doc_version":  ver,
            "source_file":  source_file,
            "is_deprecated": False,
        })
        if isinstance(ch, dict):
            stamped.append({**ch, "metadata": meta})
        else:
            ch.metadata = meta
            stamped.append(ch)

    logger.info(f"Stamped {len(stamped)} chunks for {source_file} (version {ver[:8]})")
    return stamped


def get_deprecated_ids(source_file: str, new_version: str) -> list[str]:
    """
    Return chunk_ids of all chunks from a previous version of source_file
    that should be soft-deleted when a new version is ingested.
    Called by the ingestion pipeline before adding new chunks.
    """
    reg = _load_registry()
    prev = reg.get(source_file, {}).get("previous_version")
    if not prev or prev == new_version:
        return []
    logger.info(f"Will soft-delete chunks from {source_file} version {prev[:8]}")
    return [prev]   # vector store uses doc_version to filter


def recency_score(ingested_at: str | None) -> float:
    """
    Return a recency multiplier in [0.5, 1.0].
    Documents ingested today → 1.0
    Documents older than RECENCY_DECAY_DAYS → 0.5
    Uses linear decay.
    """
    if not ingested_at:
        return 0.75

    try:
        dt = datetime.fromisoformat(ingested_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).days
        decay = max(0.0, 1.0 - age_days / settings.RECENCY_DECAY_DAYS)
        return 0.5 + 0.5 * decay       # range [0.5, 1.0]
    except Exception:
        return 0.75


def apply_recency_boost(
    results: list[dict], score_key: str = "score"
) -> list[dict]:
    """
    Apply a small additive recency bonus to each result score.
    Bonus ranges from 0.0 (very old) to 0.05 (ingested today).
    Using additive rather than multiplicative so old-but-relevant
    chunks are not penalised below the relevance threshold.
    Results are re-sorted descending after boosting.
    """
    for r in results:
        ingested_at = r.get("metadata", {}).get("ingested_at")
        # recency_score returns [0.5, 1.0] — convert to small [0, 0.05] bonus
        bonus = (recency_score(ingested_at) - 0.5) * 0.1
        r[score_key] = r.get(score_key, 0.0) + bonus
    return sorted(results, key=lambda x: x.get(score_key, 0.0), reverse=True)


# ── Helpers ──────────────────────────────────────────────────────────

def _file_hash(file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        return hashlib.md5(file_path.encode()).hexdigest()
    return hashlib.md5(p.read_bytes()).hexdigest()