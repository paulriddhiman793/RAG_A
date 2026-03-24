"""
Hybrid retrieval pipeline:
  Stage 1 — Vector search (semantic similarity, top-K)
  Stage 2 — BM25 keyword search (exact terms, top-K)
  Stage 3 — Reciprocal Rank Fusion (merge & re-score)
  Stage 4 — Cross-encoder reranking (precision pass)
  Stage 5 — Relevance threshold gate (no-answer path)
  Stage 6 — Recency boost (penalise stale results)
  Stage 7 — Context ordering (fix lost-in-the-middle)
  Stage 8 — Parent chunk expansion
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from config import settings
from indexing.vector_store import VectorStore
from indexing.bm25_index import BM25Index
from indexing.embeddings import embed_query
from ingestion.versioning import apply_recency_boost
from utils.logger import logger


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
    return CrossEncoder(settings.RERANKER_MODEL)


# ── Public API ────────────────────────────────────────────────────────

def retrieve(
    queries: list[str],
    vector_store: VectorStore,
    bm25_index: BM25Index,
    domain: str = "general",
    rerank_query: str | None = None,
    top_k_final: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Full hybrid retrieval for a list of query strings (original + alternatives).

    Returns:
        (results, top_score)
        results   : list of chunk dicts, ordered for LLM context
        top_score : highest relevance score (used for threshold check)
    """
    top_k_final = top_k_final or settings.RERANK_TOP_K

    # Stage 1 & 2 — gather candidates from all query variants
    candidates: dict[str, dict] = {}
    for query in queries:
        q_emb = embed_query(query, domain=domain)

        vec_results = vector_store.query(q_emb, top_k=settings.VECTOR_SEARCH_TOP_K)
        bm25_results = bm25_index.search(query, top_k=settings.BM25_TOP_K)

        for r in vec_results + bm25_results:
            cid = r["chunk_id"]
            if cid not in candidates or r["score"] > candidates[cid]["score"]:
                candidates[cid] = r

    if not candidates:
        logger.warning("No candidates found in retrieval")
        return [], 0.0

    candidate_list = list(candidates.values())
    logger.debug(f"Retrieval: {len(candidate_list)} unique candidates before reranking")

    # Stage 3 — RRF fusion across query variants
    fused = _reciprocal_rank_fusion(candidate_list)

    # Stage 4 — Cross-encoder reranking
    # Use the real user query when available; HyDE text is useful for retrieval
    # expansion but often too generic for precise reranking.
    primary_query = rerank_query or (queries[0] if queries else "")
    reranked = _rerank(primary_query, fused, top_k=top_k_final * 3)

    # Stage 5 — Query-intent boost for structured chunks
    reranked = _boost_for_query_intent(primary_query, reranked[:top_k_final * 2])

    # Stage 6 — Relevance threshold
    if not reranked or reranked[0].get("score", 0.0) < settings.RELEVANCE_THRESHOLD:
        top_score = reranked[0].get("score", 0.0) if reranked else 0.0
        logger.info(f"Top score {top_score:.3f} below threshold {settings.RELEVANCE_THRESHOLD}")
        return [], top_score

    # Stage 7 — Structured type boost
    # Formula / Table / Image chunks carry structured data the LLM needs.
    # Boost their scores so they survive reranking even with weak NL descriptions.
    boosted_types = _boost_structured_types(reranked[:top_k_final * 2])

    # Stage 8 — Recency boost
    boosted = apply_recency_boost(boosted_types)

    # Stage 9 — Context ordering (best first + best last, middle = weakest)
    ordered = _order_for_context(boosted[:top_k_final])

    top_score = ordered[0].get("score", 0.0) if ordered else 0.0

    # Stage 10 — Replace text child chunks with parent chunks for richer context
    # Structured chunks (Formula/Table/Image) are passed through unchanged.
    expanded = _expand_to_parents(ordered, vector_store)

    return expanded, top_score


def check_relevance(top_score: float) -> bool:
    """Return True if retrieval found sufficiently relevant content."""
    return top_score >= settings.RELEVANCE_THRESHOLD


# ── Internal stages ───────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    candidates: list[dict], k: int = 60
) -> list[dict]:
    """
    RRF score = Σ 1/(k + rank_i) across all retrieval lists.
    Promotes chunks that rank well in both vector AND keyword search.
    """
    # Sort by score descending to get rank
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)

    rrf_scores: dict[str, float] = {}
    for rank, item in enumerate(sorted_candidates, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    for item in candidates:
        item["rrf_score"] = rrf_scores.get(item["chunk_id"], 0.0)

    return sorted(candidates, key=lambda x: x.get("rrf_score", 0.0), reverse=True)


def _rerank(
    query: str, candidates: list[dict], top_k: int
) -> list[dict]:
    """
    Cross-encoder reranking — reads query + chunk together for
    much more accurate relevance scoring than bi-encoder vectors.
    """
    if not candidates or not query:
        return candidates[:top_k]

    try:
        reranker = _load_reranker()
        pairs = [(query, _build_rerank_text(c)) for c in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)

        for i, item in enumerate(candidates):
            # sigmoid normalises raw cross-encoder logits to [0, 1]
            item["score"] = float(1.0 / (1.0 + math.exp(-scores[i])))

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed, using RRF order: {e}")
        return candidates[:top_k]


def _build_rerank_text(candidate: dict[str, Any]) -> str:
    meta = candidate.get("metadata", {})
    el_type = meta.get("element_type", "")
    text = candidate.get("text", "")

    if el_type == "Formula":
        parts = []
        latex = meta.get("latex", "")
        variables = meta.get("variables", [])
        if latex:
            parts.append(f"Equation: {latex}")
        if variables:
            parts.append(f"Variables: {', '.join(str(v) for v in variables[:12])}")
        if text:
            parts.append(f"Description: {text}")
        return "\n".join(parts) if parts else text

    if el_type == "Table":
        table_json = meta.get("table_json", {})
        headers = table_json.get("headers", [])
        rows = table_json.get("rows", [])
        parts = []
        if headers:
            parts.append(f"Headers: {headers}")
        if rows:
            parts.append(f"Rows: {rows[:3]}")
        if text:
            parts.append(f"Summary: {text}")
        return "\n".join(parts) if parts else text

    if el_type == "Image":
        alt_text = meta.get("alt_text", "")
        caption = meta.get("caption", "")
        parts = [p for p in [alt_text, caption, text] if p]
        return "\n".join(parts) if parts else text

    return text


# Chunk types that carry structured data — never replace with parent text
_STRUCTURED_TYPES = {"Formula", "Table", "Image", "CodeSnippet"}


def _boost_structured_types(
    chunks: list[dict], boost: float = 0.25
) -> list[dict]:
    """
    Apply a score boost to Formula / Table / Image chunks.

    These chunks carry LaTeX / JSON / alt-text that the LLM needs to answer
    precisely. Without the boost, the cross-encoder tends to rank prose
    descriptions higher than the raw structured chunks because their NL
    embed_text is shorter and less descriptive.

    boost=0.25 means a Formula chunk scoring 0.10 becomes 0.35,
    putting it ahead of most prose chunks.
    """
    for chunk in chunks:
        el_type = chunk.get("metadata", {}).get("element_type", "")
        if el_type in _STRUCTURED_TYPES:
            chunk["score"] = min(1.0, chunk.get("score", 0.0) + boost)
    return sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)


def _boost_for_query_intent(query: str, chunks: list[dict]) -> list[dict]:
    q = (query or "").lower()
    if not q:
        return chunks

    wants_formula = any(term in q for term in [
        "equation", "equations", "formula", "formulas", "latex", "loss function"
    ])
    wants_table = any(term in q for term in [
        "table", "metric", "metrics", "value", "values", "ssim", "mse", "rmse", "r-squared"
    ])

    for chunk in chunks:
        el_type = chunk.get("metadata", {}).get("element_type", "")
        score = chunk.get("score", 0.0)
        if wants_formula and el_type == "Formula":
            chunk["score"] = min(1.0, score + 0.35)
        elif wants_table and el_type == "Table":
            chunk["score"] = min(1.0, score + 0.30)

    return sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)


def _order_for_context(chunks: list[dict]) -> list[dict]:
    """
    Fix 'lost in the middle' bias.
    Place highest-score chunk FIRST, second-highest LAST,
    and the weakest chunks in the middle.
    This ensures the LLM sees the most relevant content at the boundaries.
    """
    if len(chunks) <= 2:
        return chunks

    ordered = list(chunks)
    # Interleave: best=0, second=last, rest fill middle
    result = [ordered[0]]
    middle = ordered[1:-1]
    last = ordered[-1]

    # Sort middle by reading_order to preserve document flow
    middle.sort(key=lambda x: x.get("metadata", {}).get("reading_order", 0))
    result.extend(middle)
    result.append(last)

    return result


def _expand_to_parents(
    chunks: list[dict], vector_store: VectorStore
) -> list[dict]:
    """
    Replace text child chunks with their parent chunks for richer context.
    Structured chunks (Formula / Table / Image) are NEVER replaced — they
    must be sent to the LLM directly so their LaTeX / JSON is visible.
    Deduplicates: if two text children share a parent, include parent once.
    """
    seen_parents: set[str] = set()
    expanded: list[dict] = []

    for chunk in chunks:
        el_type = chunk.get("metadata", {}).get("element_type", "")

        # Structured chunks go through unchanged — preserve their metadata
        if el_type in _STRUCTURED_TYPES:
            expanded.append(chunk)
            continue

        parent_id = chunk.get("metadata", {}).get("parent_id") or chunk.get("parent_id", "")
        if parent_id and parent_id not in seen_parents:
            parent = vector_store.get_parent(parent_id)
            if parent:
                parent["score"] = chunk.get("score", 0.0)
                parent["source_chunk_id"] = chunk["chunk_id"]
                expanded.append(parent)
                seen_parents.add(parent_id)
                continue
        # No parent found or already included — use child directly
        if chunk["chunk_id"] not in seen_parents:
            expanded.append(chunk)

    return expanded
