"""
Query understanding — transforms a raw user query into optimal retrieval queries.

Techniques implemented:
  1. Query rewriting          — conversational → retrieval-optimised keywords
  2. HyDE                     — generate hypothetical answer, embed that
  3. Multi-query expansion    — 3-5 alternative phrasings
  4. Multi-hop decomposition  — split compound questions into sub-queries
  5. Vocab expansion          — add synonyms / domain terms
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.logger import logger
from utils.language_detector import detect_language


@dataclass
class ExpandedQuery:
    original: str
    rewritten: str
    hypothetical_answer: str
    alternatives: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    language: str = "en"

    def all_queries(self) -> list[str]:
        """All query strings to retrieve with, deduplicated."""
        seen: set[str] = set()
        result: list[str] = []
        for q in [self.rewritten, self.original] + self.alternatives + self.sub_queries:
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result


def expand_query(query: str, llm_client=None) -> ExpandedQuery:
    """
    Full query expansion pipeline.
    Falls back gracefully when LLM is unavailable.
    """
    lang = detect_language(query)
    rewritten = _rewrite_query(query, llm_client)
    hyde = _generate_hyde(query, llm_client)
    alternatives = _generate_alternatives(query, llm_client)
    sub_queries = _decompose_if_compound(query, llm_client)

    return ExpandedQuery(
        original=query,
        rewritten=rewritten,
        hypothetical_answer=hyde,
        alternatives=alternatives,
        sub_queries=sub_queries,
        language=lang,
    )


# ── Individual techniques ─────────────────────────────────────────────

def _rewrite_query(query: str, llm_client=None) -> str:
    if llm_client is None:
        return _keyword_fallback(query)
    prompt = (
        "Rewrite the following question into a concise retrieval query "
        "optimised for searching a scientific/technical document collection. "
        "Remove conversational filler. Keep domain-specific terms. "
        "Output only the rewritten query, nothing else.\n\n"
        f"Question: {query}\nRewritten query:"
    )
    try:
        return llm_client.complete(prompt, max_tokens=80).strip()
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return _keyword_fallback(query)


def _generate_hyde(query: str, llm_client=None) -> str:
    """
    HyDE — generate a hypothetical answer paragraph and use its
    embedding instead of the question embedding.
    This dramatically improves retrieval for knowledge-intensive queries.
    """
    if llm_client is None:
        return query   # fallback: just embed the query itself

    prompt = (
        "Write a concise, factual paragraph (3-5 sentences) that would be "
        "the ideal answer to the following question, as if it were found in "
        "a scientific paper or technical document. Be specific.\n\n"
        f"Question: {query}\n\nHypothetical answer paragraph:"
    )
    try:
        return llm_client.complete(prompt, max_tokens=200).strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


def _generate_alternatives(query: str, llm_client=None, n: int = 3) -> list[str]:
    """Generate N alternative phrasings of the query."""
    if llm_client is None:
        return []

    prompt = (
        f"Generate {n} alternative phrasings of the following question. "
        "Each should ask the same thing but use different vocabulary. "
        f"Output exactly {n} lines, one query per line, no numbering.\n\n"
        f"Question: {query}\n\nAlternatives:"
    )
    try:
        raw = llm_client.complete(prompt, max_tokens=200).strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[:n]
    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}")
        return []


def _decompose_if_compound(query: str, llm_client=None) -> list[str]:
    """
    Detect multi-hop questions and decompose them into atomic sub-queries.
    E.g. "What was the method in Study A and how does it compare to Study B?"
    → ["Study A method", "Study B method", "compare Study A Study B methods"]
    """
    # Quick heuristic: compound question indicators
    compound_indicators = [
        " and how ", " and what ", " compared to ", " vs ", " versus ",
        " while also ", " as well as ", " in addition to ", " but also ",
        "; ", " differ from ",
    ]
    is_compound = any(ind in query.lower() for ind in compound_indicators)
    if not is_compound or llm_client is None:
        return []

    prompt = (
        "The following question is compound and requires multiple retrievals. "
        "Break it into 2-4 simple atomic sub-questions, one per line, no numbering. "
        "Each sub-question should be independently answerable.\n\n"
        f"Compound question: {query}\n\nSub-questions:"
    )
    try:
        raw = llm_client.complete(prompt, max_tokens=200).strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        return lines[:4]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return []


def _keyword_fallback(query: str) -> str:
    """Simple keyword extraction when LLM is unavailable."""
    stopwords = {
        "what", "how", "why", "when", "where", "who", "which", "did",
        "does", "do", "is", "are", "was", "were", "the", "a", "an",
        "in", "on", "of", "for", "to", "with", "about", "can", "could",
        "would", "should", "and", "or", "but", "if", "that", "this",
        "it", "its", "be", "been", "being", "have", "has", "had",
    }
    words = re.findall(r"\b\w+\b", query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(keywords) if keywords else query