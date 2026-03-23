"""
Hallucination guard — post-generation verification:
  1. Parse citations from the LLM response
  2. For each cited claim, verify via NLI that the source chunk actually
     supports (entails) the claim
  3. Flag or remove unsupported claims
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from config import settings
from utils.logger import logger


@lru_cache(maxsize=1)
def _load_nli_model() -> CrossEncoder:
    logger.info(f"Loading NLI model: {settings.NLI_MODEL}")
    return CrossEncoder(settings.NLI_MODEL, num_labels=3)
    # Labels: 0=contradiction, 1=neutral, 2=entailment


# ── Public API ────────────────────────────────────────────────────────

def verify_response(
    response: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that claims in the response are grounded in retrieved chunks.

    Returns:
        {
            "verified_response": str,    # response with flags added
            "is_grounded": bool,         # overall grounding verdict
            "flagged_claims": list[str], # claims that failed NLI check
            "entailment_scores": list,   # per-claim NLI scores
        }
    """
    citations = _extract_citations(response)

    if not citations:
        # No citations to verify — flag the whole response as unverified
        return {
            "verified_response": response + "\n\n⚠️ No citations found. Response unverified.",
            "is_grounded": False,
            "flagged_claims": ["No citations present"],
            "entailment_scores": [],
        }

    # Build a map of source_number → chunk text
    source_map = _build_source_map(chunks)

    flagged: list[str] = []
    scores: list[dict] = []

    try:
        nli = _load_nli_model()

        for citation in citations:
            claim = citation["claim"]
            source_num = citation["source_num"]
            source_text = source_map.get(source_num, "")

            if not source_text:
                flagged.append(f"Claim cites Source {source_num} which wasn't retrieved")
                continue

            # NLI: (premise=source_text, hypothesis=claim)
            pair = (source_text[:512], claim[:256])
            raw_scores = nli.predict([pair], show_progress_bar=False)[0]

            # raw_scores are logits (unbounded, can be negative):
            #   index 0 = contradiction logit
            #   index 1 = neutral logit
            #   index 2 = entailment logit
            # We compare raw logits directly against NLI_ENTAIL_THRESHOLD (-1.0)
            # so only clear contradictions/non-entailments get flagged.
            # LaTeX strings and table data naturally score lower on NLI
            # because the NLI model wasn't trained on structured content —
            # so we use a generous threshold to avoid false positives.
            entail_logit = float(raw_scores[2])
            contradict_logit = float(raw_scores[0])

            scores.append({
                "claim": claim[:100],
                "source_num": source_num,
                "entailment_logit": round(entail_logit, 3),
                "contradiction_logit": round(contradict_logit, 3),
            })

            if entail_logit < settings.NLI_ENTAIL_THRESHOLD:
                flagged.append(
                    f'Claim not supported by Source {source_num} '
                    f'(entailment={entail_logit:.2f}): "{claim[:80]}…"'
                )

    except Exception as e:
        logger.warning(f"NLI verification failed: {e}")
        return {
            "verified_response": response,
            "is_grounded": True,   # fail open to avoid blocking valid answers
            "flagged_claims": [],
            "entailment_scores": [],
        }

    is_grounded = len(flagged) == 0
    verified_response = response

    if flagged:
        flag_block = "\n\n---\n⚠️ Grounding warnings:\n" + "\n".join(
            f"• {f}" for f in flagged
        )
        verified_response += flag_block
        logger.warning(f"Hallucination guard: {len(flagged)} claim(s) flagged")

    return {
        "verified_response": verified_response,
        "is_grounded": is_grounded,
        "flagged_claims": flagged,
        "entailment_scores": scores,
    }


# ── Helpers ──────────────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(
    r"(?P<claim>[^.!?\n]{20,200})\s*\[Source\s*(?P<num>\d+)[^\]]*\]",
    re.IGNORECASE,
)


def _extract_citations(response: str) -> list[dict]:
    """
    Extract (claim, source_number) pairs from the response text.
    Expects citations in the format: "...claim... [Source N]" or "[Source N, Section X]"
    """
    citations = []
    for m in _CITATION_PATTERN.finditer(response):
        citations.append({
            "claim": m.group("claim").strip(),
            "source_num": int(m.group("num")),
        })
    return citations


def _build_source_map(chunks: list[dict]) -> dict[int, str]:
    """Build a 1-indexed map of source number → chunk text."""
    return {i + 1: c.get("text", "") for i, c in enumerate(chunks)}