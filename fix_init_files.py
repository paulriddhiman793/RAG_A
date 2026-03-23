"""
fix_init_files.py — recreates all __init__.py files in the project.
Run once from your project root:
    python fix_init_files.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INIT_FILES = {
    "config/__init__.py": "from .settings import settings\n\n__all__ = ['settings']\n",

    "utils/__init__.py": """\
from .logger import logger
from .language_detector import detect_language, is_multilingual
from .coref_resolver import resolve_coreferences, resolve_batch

__all__ = [
    "logger",
    "detect_language",
    "is_multilingual",
    "resolve_coreferences",
    "resolve_batch",
]
""",

    "ingestion/__init__.py": """\
from .document_loader import load_document, load_documents
from .chunking import chunk_elements, build_parent_child, Chunk
from .versioning import stamp_chunks, get_deprecated_ids, recency_score

__all__ = [
    "load_document",
    "load_documents",
    "chunk_elements",
    "build_parent_child",
    "Chunk",
    "stamp_chunks",
    "get_deprecated_ids",
    "recency_score",
]
""",

    "ingestion/parsers/__init__.py": """\
from .math_parser import parse_formula, parse_formula_batch
from .table_parser import parse_table, parse_table_batch, query_dataframe
from .figure_parser import parse_figure, parse_figure_batch, attach_captions

__all__ = [
    "parse_formula", "parse_formula_batch",
    "parse_table", "parse_table_batch", "query_dataframe",
    "parse_figure", "parse_figure_batch", "attach_captions",
]
""",

    "indexing/__init__.py": """\
from .vector_store import VectorStore
from .bm25_index import BM25Index
from .embeddings import embed_query, embed_chunks, detect_domain
from .metadata import build_metadata, restore_metadata

__all__ = [
    "VectorStore",
    "BM25Index",
    "embed_query",
    "embed_chunks",
    "detect_domain",
    "build_metadata",
    "restore_metadata",
]
""",

    "retrieval/__init__.py": """\
from .query_expander import expand_query, ExpandedQuery
from .hybrid_retriever import retrieve, check_relevance
from .context_compressor import compress_context, build_context_string

__all__ = [
    "expand_query",
    "ExpandedQuery",
    "retrieve",
    "check_relevance",
    "compress_context",
    "build_context_string",
]
""",

    "generation/__init__.py": """\
from .llm_client import LLMClient
from .answer_generator import generate_answer
from .hallucination_guard import verify_response
from .security import scan_chunks_for_injection, scan_user_query, scan_output

__all__ = [
    "LLMClient",
    "generate_answer",
    "verify_response",
    "scan_chunks_for_injection",
    "scan_user_query",
    "scan_output",
]
""",

    "pipeline/__init__.py": """\
from .ingestion_pipeline import ingest_document, ingest_documents
from .query_pipeline import query

__all__ = ["ingest_document", "ingest_documents", "query"]
""",
}


def main():
    print(f"Project root: {ROOT}\n")
    for rel_path, content in INIT_FILES.items():
        full_path = ROOT / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        print(f"  ✓  {rel_path}")

    print("\nAll __init__.py files recreated.")
    print("Now run:  python check_env.py  to verify, then  python main.py ingest <path>")


if __name__ == "__main__":
    main()