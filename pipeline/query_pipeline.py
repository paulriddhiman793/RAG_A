"""
Query pipeline — end-to-end query answering:

  1. Security scan on user query
  2. Query expansion (rewrite, HyDE, multi-query, decomposition)
  3. Hybrid retrieval (vector + BM25 + RRF + rerank)
  4. Relevance threshold check (no-answer path)
  5. Prompt injection scan on retrieved chunks
  6. Context compression (token budget management)
  7. Answer generation with citation grounding
  8. Hallucination verification (NLI)
  9. Return structured result
"""
from __future__ import annotations

from typing import Any

from retrieval.query_expander import expand_query
from retrieval.hybrid_retriever import retrieve, check_relevance
from retrieval.context_compressor import compress_context
from generation.answer_generator import generate_answer
from generation.security import scan_user_query, scan_chunks_for_injection
from generation.llm_client import LLMClient
from config import settings
from indexing.vector_store import VectorStore
from indexing.bm25_index import BM25Index
from indexing.embeddings import detect_domain
from utils.intent import (
    has_figure_intent as _has_figure_intent,
    has_formula_intent as _has_formula_intent,
    has_metric_lookup_intent as _has_metric_lookup_intent,
    has_summary_intent as _has_summary_intent,
    looks_table_like_image as _looks_table_like_image,
)
from utils.logger import logger


def query(
    user_query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    llm_client: LLMClient,
    domain: str | None = None,
    verify_hallucinations: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Full RAG query pipeline.

    Returns:
        {
            "query": str,
            "answer": str,
            "is_grounded": bool,
            "no_answer": bool,
            "flagged_claims": list,
            "sources_used": list,
            "expanded_queries": list,
            "top_score": float,
            "chunks_retrieved": int,
        }
    """
    logger.info(f"Query: {user_query[:100]}")

    # ── Step 1: Input security scan ───────────────────────────────────
    is_safe, reason = scan_user_query(user_query)
    if not is_safe:
        logger.warning(f"Query blocked by security scan: {reason}")
        return _blocked_response(user_query, reason)

    # ── Step 2: Query expansion ───────────────────────────────────────
    expanded = expand_query(user_query, llm_client)
    all_queries = expanded.all_queries()

    if verbose:
        logger.info(f"Expanded to {len(all_queries)} queries: {all_queries}")

    # ── Step 3: Domain detection ──────────────────────────────────────
    if domain is None:
        detected_domain = detect_domain([user_query])
        if settings.AUTO_DETECT_EMBEDDING_DOMAIN:
            domain = detected_domain
        else:
            domain = "general"
            if detected_domain != "general":
                logger.info(
                    "Domain-specific embedding auto-selection is disabled; "
                    f"detected '{detected_domain}' but using '{domain}' "
                    "for retrieval consistency"
                )

    # ── Step 4: Hybrid retrieval ──────────────────────────────────────
    # Use HyDE embedding as primary query for vector search
    hyde_queries = [expanded.hypothetical_answer] + all_queries
    retrieved_chunks, top_score = retrieve(
        queries=hyde_queries,
        vector_store=vector_store,
        bm25_index=bm25_index,
        domain=domain,
        rerank_query=user_query,
    )

    if _has_formula_intent(user_query):
        retrieved_chunks = _ensure_formula_context(retrieved_chunks, vector_store)
        if retrieved_chunks:
            top_score = max(top_score, retrieved_chunks[0].get("score", top_score))
    if _has_figure_intent(user_query):
        retrieved_chunks, top_score = _ensure_figure_context(
            retrieved_chunks,
            vector_store,
            user_query,
            top_score,
        )
    if _has_metric_lookup_intent(user_query):
        retrieved_chunks, top_score = _ensure_metric_context(
            retrieved_chunks,
            vector_store,
            user_query,
            top_score,
        )
    if _has_summary_intent(user_query):
        retrieved_chunks, top_score = _ensure_summary_context(
            retrieved_chunks,
            vector_store,
            top_score,
        )

    if verbose:
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, top score: {top_score:.3f}")

    # ── Step 5: Relevance threshold ───────────────────────────────────
    if not check_relevance(top_score):
        # Before giving up, try a broad-context fallback if we have
        # indexed content — general questions often score low with the
        # cross-encoder but can be answered from the full document.
        if vector_store.count() > 0:
            logger.info(
                f"Low relevance score ({top_score:.3f}) but DB has content — "
                "using broad-context fallback"
            )
            retrieved_chunks, top_score = _ensure_summary_context(
                retrieved_chunks, vector_store, top_score
            )
        if not retrieved_chunks:
            logger.info(f"No relevant content found (score={top_score:.3f})")
            return {
                "query": user_query,
                "answer": (
                    "I cannot find relevant information about this in the "
                    "provided documents. The query may be outside the scope "
                    "of the indexed content."
                ),
                "is_grounded": True,
                "no_answer": True,
                "flagged_claims": [],
                "sources_used": [],
                "expanded_queries": all_queries,
                "top_score": top_score,
                "chunks_retrieved": 0,
            }

    # ── Step 6: Injection scan on retrieved content ───────────────────
    safe_chunks = scan_chunks_for_injection(retrieved_chunks)

    # ── Step 7: Context compression ───────────────────────────────────
    compressed_chunks = compress_context(safe_chunks, user_query)

    # ── Step 8: Generate answer ───────────────────────────────────────
    generation_result = generate_answer(
        query=user_query,
        chunks=compressed_chunks,
        llm_client=llm_client,
        verify=verify_hallucinations,
    )

    return {
        "query": user_query,
        "answer": generation_result["answer"],
        "is_grounded": generation_result["is_grounded"],
        "no_answer": generation_result["no_answer"],
        "flagged_claims": generation_result["flagged_claims"],
        "sources_used": generation_result["sources_used"],
        "expanded_queries": all_queries,
        "top_score": top_score,
        "chunks_retrieved": len(compressed_chunks),
    }


def _blocked_response(user_query: str, reason: str) -> dict[str, Any]:
    return {
        "query": user_query,
        "answer": f"Query could not be processed: {reason}",
        "is_grounded": False,
        "no_answer": False,
        "flagged_claims": [reason],
        "sources_used": [],
        "expanded_queries": [],
        "top_score": 0.0,
        "chunks_retrieved": 0,
    }


def _ensure_formula_context(
    chunks: list[dict[str, Any]],
    vector_store: VectorStore,
) -> list[dict[str, Any]]:
    if any(c.get("metadata", {}).get("element_type") == "Formula" for c in chunks):
        return chunks

    all_chunks = vector_store.get_all_chunks()
    formula_chunks = [
        c for c in all_chunks
        if c.get("metadata", {}).get("element_type") == "Formula"
    ]
    if not formula_chunks:
        return chunks

    formula_chunks.sort(
        key=lambda c: (
            c.get("metadata", {}).get("page_number", 0),
            c.get("metadata", {}).get("reading_order", 0),
        )
    )

    existing_ids = {c.get("chunk_id") for c in chunks}
    injected = []
    for chunk in formula_chunks:
        if chunk.get("chunk_id") in existing_ids:
            continue
        copy = dict(chunk)
        # Score just above RELEVANCE_THRESHOLD so it passes the gate
        # without overshadowing truly relevant reranked chunks.
        copy["score"] = max(copy.get("score", 0.0), settings.RELEVANCE_THRESHOLD + 0.15)
        injected.append(copy)

    if not injected:
        return chunks

    logger.info(
        f"Formula-intent fallback injected {len(injected)} formula chunk(s) into context"
    )
    return injected + chunks


def _ensure_figure_context(
    chunks: list[dict[str, Any]],
    vector_store: VectorStore,
    query: str,
    top_score: float,
) -> tuple[list[dict[str, Any]], float]:
    q = (query or "").lower()
    wants_all = any(term in q for term in [
        "how many", "count", "number of", "all figures", "all graphs",
        "all charts", "all plots", "list"
    ])

    current_figure_chunks = [
        c for c in chunks
        if c.get("metadata", {}).get("element_type") == "Image" and not _looks_table_like_image(c)
    ]
    if current_figure_chunks and not wants_all:
        return chunks, top_score

    all_chunks = vector_store.get_all_chunks()
    figure_chunks = [
        c for c in all_chunks
        if c.get("metadata", {}).get("element_type") == "Image" and not _looks_table_like_image(c)
    ]
    if not figure_chunks:
        return chunks, top_score

    figure_chunks.sort(
        key=lambda c: (
            c.get("metadata", {}).get("page_number", 0),
            c.get("metadata", {}).get("reading_order", 0),
        )
    )

    if not wants_all:
        figure_chunks = figure_chunks[: min(3, len(figure_chunks))]

    existing_ids = {c.get("chunk_id") for c in chunks}
    injected = []
    for chunk in figure_chunks:
        if chunk.get("chunk_id") in existing_ids:
            continue
        copy = dict(chunk)
        copy["score"] = max(copy.get("score", 0.0), settings.RELEVANCE_THRESHOLD + 0.10)
        injected.append(copy)

    if not injected:
        return chunks, top_score

    logger.info(
        f"Figure-intent fallback injected {len(injected)} image chunk(s) into context"
    )
    return injected + chunks, max(top_score, injected[0].get("score", top_score))


def _ensure_summary_context(
    chunks: list[dict[str, Any]],
    vector_store: VectorStore,
    top_score: float,
) -> tuple[list[dict[str, Any]], float]:
    parent_chunks = vector_store.get_all_parent_chunks()
    if not parent_chunks:
        fallback_children = vector_store.get_all_chunks()
        if not fallback_children:
            return chunks, top_score
        fallback_children.sort(
            key=lambda c: (
                c.get("metadata", {}).get("page_number", 0),
                c.get("metadata", {}).get("reading_order", 0),
            )
        )
        parent_chunks = fallback_children[:8]

    parent_chunks.sort(
        key=lambda c: (
            c.get("metadata", {}).get("page_number", 0),
            c.get("metadata", {}).get("reading_order", 0),
        )
    )

    existing_ids = {c.get("chunk_id") for c in chunks}
    injected = []
    for chunk in parent_chunks:
        if chunk.get("chunk_id") in existing_ids:
            continue
        copy = dict(chunk)
        copy["score"] = max(copy.get("score", 0.0), settings.RELEVANCE_THRESHOLD + 0.10)
        injected.append(copy)

    if not injected:
        return chunks, max(top_score, (settings.RELEVANCE_THRESHOLD + 0.10) if chunks else top_score)

    logger.info(
        f"Summary-intent fallback injected {len(injected)} parent chunk(s) into context"
    )
    return injected + chunks, max(top_score, injected[0].get("score", top_score))


def _ensure_metric_context(
    chunks: list[dict[str, Any]],
    vector_store: VectorStore,
    query: str,
    top_score: float,
) -> tuple[list[dict[str, Any]], float]:
    q = (query or "").lower()
    all_chunks = vector_store.get_all_chunks()
    if not all_chunks:
        return chunks, top_score

    metric_terms = [t for t in ["r2", "r-squared", "rmse", "mae", "mse", "score"] if t in q]
    model_terms = [t for t in [
        "random forest", "randomforest", "regressor", "xgboost", "ridge",
        "lasso", "linear regression", "mlr", "stacking", "ensemble", "poly",
    ] if t in q]

    ranked = []
    for chunk in all_chunks:
        meta = chunk.get("metadata", {})
        blob = " ".join([
            str(chunk.get("text", "")),
            str(meta.get("caption", "")),
            str(meta.get("alt_text", "")),
            str(meta.get("section", "")),
        ]).lower()
        score = 0
        for term in metric_terms:
            if term in blob:
                score += 2
        for term in model_terms:
            if term in blob:
                score += 3
        if "r?" in blob and any(term in q for term in ["r2", "r-squared", "score"]):
            score += 2
        if score <= 0:
            continue
        copy = dict(chunk)
        copy["score"] = max(copy.get("score", 0.0), min(settings.RELEVANCE_THRESHOLD + 0.15, 0.40 + (score * 0.04)))
        ranked.append(copy)

    if not ranked:
        return chunks, top_score

    ranked.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    existing_ids = {c.get("chunk_id") for c in chunks}
    injected = [c for c in ranked if c.get("chunk_id") not in existing_ids][:6]
    if not injected:
        return chunks, max(top_score, ranked[0].get("score", top_score))

    logger.info(
        f"Metric-intent fallback injected {len(injected)} chunk(s) into context"
    )
    return injected + chunks, max(top_score, injected[0].get("score", top_score))


# _looks_table_like_image is imported from utils.intent
