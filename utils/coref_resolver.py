"""
Coreference resolution — replaces pronouns and vague references
with their actual referents so every chunk is self-contained.

Two modes:
  - FULL mode   : uses fastcoref (lingmess-coref, 2.3 GB model)
                  Set USE_FASTCOREF=true in .env to enable.
  - LIGHT mode  : rule-based heuristics, zero extra dependencies (default)
                  Handles the most common cases without any model download.
"""
from __future__ import annotations

import os
import re

from utils.logger import logger

# ── Toggle ────────────────────────────────────────────────────────────
# Set USE_FASTCOREF=true in .env ONLY if you have ~3 GB free on C:
# or have set HF_HOME to a drive with enough space.
_USE_FASTCOREF = os.getenv("USE_FASTCOREF", "false").lower() == "true"

_nlp = None  # lazy-loaded only if USE_FASTCOREF=true


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        from fastcoref import spacy_component  # noqa: F401
        logger.info("Loading fastcoref pipeline (first run downloads ~2.3 GB)…")
        _nlp = spacy.load("en_core_web_sm", exclude=["ner"])
        _nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": "cpu",
            },
        )
        return _nlp
    except Exception as e:
        logger.warning(f"fastcoref load failed: {e}. Falling back to light mode.")
        return None


# ── Public API ────────────────────────────────────────────────────────

def resolve_coreferences(text: str) -> str:
    """
    Resolve coreferences in text.
    Uses fastcoref if USE_FASTCOREF=true, otherwise light heuristics.
    """
    if not text or len(text.strip()) < 30:
        return text

    if _USE_FASTCOREF:
        return _resolve_fastcoref(text)
    return _resolve_light(text)


def resolve_batch(texts: list[str]) -> list[str]:
    return [resolve_coreferences(t) for t in texts]


# ── Full mode (fastcoref) ─────────────────────────────────────────────

def _resolve_fastcoref(text: str) -> str:
    try:
        nlp = _get_nlp()
        if nlp is None:
            return _resolve_light(text)
        doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
        return doc._.resolved_text or text
    except Exception as e:
        logger.warning(f"fastcoref resolution failed: {e}")
        return _resolve_light(text)


# ── Light mode (rule-based heuristics) ───────────────────────────────
#
# Handles the most common RAG coreference problem:
# a chunk starts with "It", "They", "This", "These" after being split
# from the sentence that defined the referent.
#
# Strategy: if a chunk's first sentence starts with a pronoun/demonstrative,
# prepend the last noun phrase from the preceding sentence when available.
# This is enough to make most standalone chunks interpretable.

_PRONOUN_STARTS = re.compile(
    r"^(It|They|This|These|That|Those|He|She|Its|Their|His|Her)\b",
    re.IGNORECASE,
)

_LAST_NP = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|the\s+[a-z]+(?:\s+[a-z]+){0,3})\s*[.,]?\s*$",
    re.IGNORECASE,
)


def _resolve_light(text: str) -> str:
    """
    Lightweight heuristic: ensure each sentence is reasonably self-contained
    by carrying the last meaningful noun phrase forward when a sentence starts
    with a pronoun.

    This is not full coref — it's a cheap approximation that helps the
    embedding model and LLM understand isolated chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text

    resolved = [sentences[0]]
    for i in range(1, len(sentences)):
        sent = sentences[i]
        if _PRONOUN_STARTS.match(sent):
            prev = sentences[i - 1]
            np_match = _LAST_NP.search(prev)
            if np_match:
                antecedent = np_match.group(1).strip().rstrip(".,")
                # Replace the leading pronoun with the antecedent
                sent = _PRONOUN_STARTS.sub(antecedent, sent, count=1)
        resolved.append(sent)

    return " ".join(resolved)