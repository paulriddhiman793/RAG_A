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
    )

    if verbose:
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, top score: {top_score:.3f}")

    # ── Step 5: Relevance threshold ───────────────────────────────────
    if not check_relevance(top_score):
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
