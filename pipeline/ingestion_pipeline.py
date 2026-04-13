"""
Ingestion pipeline - end-to-end document processing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings
from generation.llm_client import LLMClient
from indexing.bm25_index import BM25Index
from indexing.embeddings import detect_domain, embed_chunks
from indexing.vector_store import VectorStore
from ingestion.chunking import Chunk, build_parent_child, chunk_elements
from ingestion.document_loader import load_document
from ingestion.parsers import (
    attach_captions,
    parse_figure_batch,
    parse_formula_batch,
    parse_table_batch,
)
from ingestion.versioning import get_deprecated_ids, stamp_chunks
from utils.language_detector import detect_language
from utils.logger import logger
import concurrent.futures


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

    elements = load_document(path)
    if not elements:
        logger.warning(f"No elements extracted from {path.name}")
        return {"file": str(path), "status": "empty", "chunks": 0}

    elements = attach_captions(elements)

    formula_els = [e for e in elements if e["type"] == "Formula"]
    table_els = [e for e in elements if e["type"] == "Table"]
    
    parsed_formulas = {}
    parsed_tables = {}
    
    # Process formulas and tables in parallel using up to 10 threads as per constraints
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        form_futures = {
            executor.submit(lambda e=e: parse_formula_batch([e], llm_client)[0]): e 
            for e in formula_els
        }
        for future in concurrent.futures.as_completed(form_futures):
            e = form_futures[future]
            try:
                parsed_formulas[e["text"]] = future.result()
            except Exception as exc:
                logger.error(f"Formula parsing failed for {e['text'][:30]}: {exc}")

        tab_futures = {
            executor.submit(lambda e=e: parse_table_batch([e], llm_client)[0]): e 
            for e in table_els
        }
        for future in concurrent.futures.as_completed(tab_futures):
            e = tab_futures[future]
            try:
                parsed_tables[e.get("table_html", e["text"])] = future.result()
            except Exception as exc:
                logger.error(f"Table parsing failed: {exc}")

    image_els = [e for e in elements if e["type"] == "Image"]
    parsed_figures = parse_figure_batch(image_els, llm_client)

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

    chunks_obj: list[Chunk] = chunk_elements(enriched, resolve_coref=True)
    parent_chunks, child_chunks = build_parent_child(chunks_obj)

    chunks_dicts = [c.to_dict() for c in child_chunks]
    parent_dicts = [c.to_dict() for c in parent_chunks]

    if not chunks_dicts:
        logger.warning(f"No chunks produced for {path.name}")
        return {"file": str(path), "status": "no_chunks", "chunks": 0}

    for ch in chunks_dicts:
        ch["metadata"]["language"] = detect_language(ch.get("text", ""))

    sample_texts = [c.get("text", "") for c in chunks_dicts[:10]]
    detected_domain = detect_domain(sample_texts)
    if domain is not None:
        embedding_domain = domain
        logger.info(f"Embedding domain forced: {embedding_domain}")
    elif settings.AUTO_DETECT_EMBEDDING_DOMAIN:
        embedding_domain = detected_domain
        logger.info(f"Embedding domain auto-selected: {embedding_domain}")
    else:
        embedding_domain = "general"
        if detected_domain != "general":
            logger.info(
                "Domain-specific embedding auto-selection is disabled; "
                f"detected '{detected_domain}' but using '{embedding_domain}' "
                "for collection consistency"
            )
    logger.info(f"Detected domain: {detected_domain}")

    from ingestion.versioning import _file_hash

    new_version = _file_hash(str(path))
    old_version_ids = get_deprecated_ids(str(path), new_version)

    chunks_dicts = stamp_chunks(chunks_dicts, str(path), new_version)
    parent_dicts = stamp_chunks(parent_dicts, str(path), new_version)

    for old_ver in old_version_ids:
        vector_store.soft_delete_version(str(path), old_ver)

    _, child_embeddings = embed_chunks(chunks_dicts, domain=embedding_domain)
    vector_store.add_chunks(chunks_dicts, child_embeddings)

    _, parent_embeddings = embed_chunks(parent_dicts, domain=embedding_domain)
    vector_store.add_parent_chunks(parent_dicts, parent_embeddings)

    all_chunks = vector_store.get_all_chunks()
    bm25_index.build(all_chunks)

    summary = {
        "file": str(path),
        "status": "success",
        "domain": embedding_domain,
        "detected_domain": detected_domain,
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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_fp = {
            executor.submit(ingest_document, fp, vector_store, bm25_index, llm_client, domain): fp 
            for fp in file_paths
        }
        for future in concurrent.futures.as_completed(future_to_fp):
            fp = future_to_fp[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {fp}: {e}")
                results.append({"file": str(fp), "status": "error", "error": str(e)})
    return results
