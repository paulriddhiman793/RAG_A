"""
Ingestion pipeline — end-to-end document processing:

  1. Load document via Unstructured.io API
  2. Attach captions to figures
  3. Parse special elements (math, tables, figures)
  4. Resolve coreferences in text chunks
  5. Chunk with overlap + parent-child hierarchy
  6. Stamp with version/timestamp metadata
  7. Detect domain for embedding model selection
  8. Embed all chunks
  9. Soft-delete old version in vector store
 10. Upsert new chunks into vector store + BM25
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.document_loader import load_document
from ingestion.parsers import (
    attach_captions,
    parse_formula_batch,
    parse_table_batch,
    parse_figure_batch,
)
from ingestion.chunking import chunk_elements, build_parent_child, Chunk
from ingestion.versioning import stamp_chunks, get_deprecated_ids
from indexing.embeddings import embed_chunks, detect_domain
from indexing.vector_store import VectorStore
from indexing.bm25_index import BM25Index
from generation.llm_client import LLMClient
from utils.language_detector import detect_language
from utils.logger import logger


def ingest_document(
    file_path: str | Path,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    llm_client: LLMClient | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """
    Full ingestion pipeline for a single document.

    Returns a summary dict with counts and detected domain/language.
    """
    path = Path(file_path)
    logger.info(f"=== Ingesting: {path.name} ===")

    # ── Step 1: Load via Unstructured API ─────────────────────────────
    elements = load_document(path)
    if not elements:
        logger.warning(f"No elements extracted from {path.name}")
        return {"file": str(path), "status": "empty", "chunks": 0}

    # ── Step 2: Attach figure captions ────────────────────────────────
    elements = attach_captions(elements)

    # ── Step 3: Nougat disabled — using regex/HTML fallback parsers ──────
    # To re-enable: set USE_NOUGAT=true in .env and pip install nougat-ocr
    nougat_data: dict | None = None
    logger.info("Nougat disabled — using regex + HTML parsers for math/tables")

    # ── Step 4: Parse special element types ───────────────────────────
    # Math — passes Nougat equations for accurate LaTeX matching
    formula_els = [e for e in elements if e["type"] == "Formula"]
    parsed_formulas = {
        e["text"]: parse_formula_batch([e], llm_client, nougat_data)[0]
        for e in formula_els
    }

    # Tables — passes Nougat tables for clean JSON extraction
    table_els = [e for e in elements if e["type"] == "Table"]
    parsed_tables = {
        e.get("table_html", e["text"]): parse_table_batch([e], llm_client, nougat_data)[0]
        for e in table_els
    }

    # Figures
    image_els = [e for e in elements if e["type"] == "Image"]
    parsed_figures = parse_figure_batch(image_els, llm_client)

    # Merge enriched elements back
    enriched: list[dict] = []
    img_idx = 0
    for el in elements:
        if el["type"] == "Formula" and el["text"] in parsed_formulas:
            enriched.append(parsed_formulas[el["text"]])
        elif el["type"] == "Table":
            key = el.get("table_html", el["text"])
            enriched.append(parsed_tables.get(key, el))
        elif el["type"] == "Image" and img_idx < len(parsed_figures):
            enriched.append(parsed_figures[img_idx])
            img_idx += 1
        else:
            enriched.append(el)

    # ── Step 4+5: Chunk with coreference resolution ────────────────────
    chunks_obj: list[Chunk] = chunk_elements(enriched, resolve_coref=True)
    parent_chunks, child_chunks = build_parent_child(chunks_obj)

    chunks_dicts = [c.to_dict() for c in child_chunks]
    parent_dicts = [c.to_dict() for c in parent_chunks]

    if not chunks_dicts:
        logger.warning(f"No chunks produced for {path.name}")
        return {"file": str(path), "status": "no_chunks", "chunks": 0}

    # ── Step 6: Language detection per chunk ─────────────────────────
    for ch in chunks_dicts:
        ch["metadata"]["language"] = detect_language(ch.get("text", ""))

    # ── Step 7: Domain detection ──────────────────────────────────────
    sample_texts = [c.get("text", "") for c in chunks_dicts[:10]]
    detected_domain = domain or detect_domain(sample_texts)
    logger.info(f"Domain: {detected_domain}")

    # ── Step 8: Version stamping ──────────────────────────────────────
    from ingestion.versioning import _file_hash
    new_version = _file_hash(str(path))
    old_version_ids = get_deprecated_ids(str(path), new_version)

    chunks_dicts = stamp_chunks(chunks_dicts, str(path), new_version)
    parent_dicts = stamp_chunks(parent_dicts, str(path), new_version)

    # ── Step 9: Soft-delete old version ──────────────────────────────
    for old_ver in old_version_ids:
        vector_store.soft_delete_version(str(path), old_ver)

    # ── Step 10: Embed + upsert ───────────────────────────────────────
    _, child_embeddings = embed_chunks(chunks_dicts, domain=detected_domain)
    vector_store.add_chunks(chunks_dicts, child_embeddings)

    _, parent_embeddings = embed_chunks(parent_dicts, domain=detected_domain)
    vector_store.add_parent_chunks(parent_dicts, parent_embeddings)

    # ── Step 11: Rebuild BM25 ─────────────────────────────────────────
    all_chunks = vector_store.get_all_chunks()
    bm25_index.build(all_chunks)

    summary = {
        "file": str(path),
        "status": "success",
        "domain": detected_domain,
        "elements": len(enriched),
        "child_chunks": len(chunks_dicts),
        "parent_chunks": len(parent_dicts),
        "formulas": len(formula_els),
        "tables": len(table_els),
        "figures": len(image_els),
        "total_indexed": vector_store.count(),
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


def ingest_documents(
    file_paths: list[str | Path],
    vector_store: VectorStore,
    bm25_index: BM25Index,
    llm_client: LLMClient | None = None,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Ingest multiple documents."""
    results = []
    for fp in file_paths:
        try:
            result = ingest_document(fp, vector_store, bm25_index, llm_client, domain)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to ingest {fp}: {e}")
            results.append({"file": str(fp), "status": "error", "error": str(e)})
    return results