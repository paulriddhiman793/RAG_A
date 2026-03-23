"""
Security guards — two layers:
  1. Input: detect prompt injection in retrieved chunks before sending to LLM
  2. Output: detect signs of successful injection in LLM responses
"""
from __future__ import annotations

import re
from typing import Any

from config import settings
from utils.logger import logger


# ── Input guard ───────────────────────────────────────────────────────

def scan_chunks_for_injection(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Scan retrieved chunks for prompt injection patterns.
    Flagged chunks are either removed or sanitized.
    Returns a cleaned list of chunks.
    """
    clean: list[dict[str, Any]] = []
    flagged_count = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        if _is_injection(text):
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            logger.warning(
                f"Prompt injection detected in chunk from {source}. "
                f"Chunk suppressed. Preview: {text[:100]}"
            )
            flagged_count += 1
            # Sanitize rather than drop — preserves citation numbers
            clean.append({
                **chunk,
                "text": "[CONTENT REDACTED: Potential prompt injection detected]",
                "context_text": "[CONTENT REDACTED: Potential prompt injection detected]",
                "flagged": True,
            })
        else:
            clean.append(chunk)

    if flagged_count:
        logger.warning(f"Injection scan: {flagged_count}/{len(chunks)} chunks flagged")

    return clean


def scan_user_query(query: str) -> tuple[bool, str]:
    """
    Scan the user's own query for injection attempts.
    Returns (is_safe, reason).
    """
    if _is_injection(query):
        return False, "Query contains instruction-override patterns"
    return True, ""


# ── Output guard ──────────────────────────────────────────────────────

def scan_output(response: str) -> tuple[bool, str]:
    """
    Scan LLM output for signs of successful injection.
    Returns (is_safe, reason).
    """
    suspicious_patterns = [
        r"ignore\s+previous\s+instructions",
        r"system\s+prompt\s*:",
        r"my\s+instructions\s+are",
        r"i\s+am\s+now\s+(acting|behaving|operating)\s+as",
        r"new\s+persona",
        r"DAN\s+mode",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False, f"Output matches injection pattern: {pattern}"
    return True, ""


# ── Helpers ──────────────────────────────────────────────────────────

def _is_injection(text: str) -> bool:
    """Return True if text contains prompt injection patterns."""
    lower = text.lower()
    for pattern in settings.INJECTION_PATTERNS:
        if pattern.lower() in lower:
            return True
    # Also check for structural injection (all-caps instruction blocks)
    if re.search(r"\bINSTRUCTION[S]?\s*:", text):
        return True
    if re.search(r"\bSYSTEM\s*:", text):
        return True
    return False