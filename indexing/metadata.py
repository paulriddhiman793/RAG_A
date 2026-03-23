"""
Metadata schema — defines the shape of metadata attached to every chunk.
ChromaDB stores metadata as flat key-value pairs (no nested dicts).
We serialize complex fields (table_json, variables) as JSON strings.
"""
from __future__ import annotations

import json
from typing import Any


# All keys stored in ChromaDB metadata
METADATA_KEYS = (
    # Provenance
    "source_file",
    "filename",
    "page_number",
    "section",
    "element_type",
    "parent_id",
    "chunk_id",
    "reading_order",
    "token_count",
    # Versioning
    "ingested_at",
    "doc_version",
    "is_deprecated",
    # Language
    "language",
    # Type-specific (serialized as JSON strings)
    "latex",
    "variables_json",
    "table_json_str",
    "caption",
    "alt_text",
    "image_base64",
    "image_mime_type",
)


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a chunk dict into a ChromaDB-compatible metadata dict.
    All values must be str, int, float, or bool.
    """
    raw_meta = chunk.get("metadata", {})
    el_type = chunk.get("element_type", raw_meta.get("element_type", "NarrativeText"))

    meta: dict[str, Any] = {
        "source_file":    str(raw_meta.get("source_file", "")),
        "filename":       str(raw_meta.get("filename", "")),
        "page_number":    int(raw_meta.get("page_number", 0)),
        "section":        str(raw_meta.get("section", "")),
        "element_type":   str(el_type),
        "parent_id":      str(chunk.get("parent_id") or ""),
        "chunk_id":       str(chunk.get("chunk_id", "")),
        "reading_order":  int(chunk.get("reading_order", 0)),
        "token_count":    int(chunk.get("token_count", 0)),
        "ingested_at":    str(raw_meta.get("ingested_at", "")),
        "doc_version":    str(raw_meta.get("doc_version", "")),
        "is_deprecated":  bool(raw_meta.get("is_deprecated", False)),
        "language":       str(raw_meta.get("language", "en")),
        # Type-specific
        "latex":          str(raw_meta.get("latex", "")),
        "variables_json": json.dumps(raw_meta.get("variables", [])),
        "table_json_str": json.dumps(raw_meta.get("table_json", {})),
        "caption":        str(raw_meta.get("caption", "")),
        "alt_text":       str(raw_meta.get("alt_text", "")),
        "image_base64":   str(raw_meta.get("image_base64", "")),
        "image_mime_type":str(raw_meta.get("image_mime_type", "")),
    }
    return meta


def restore_metadata(flat_meta: dict[str, Any]) -> dict[str, Any]:
    """Inverse of build_metadata — deserialize JSON fields back to Python objects."""
    meta = dict(flat_meta)
    try:
        meta["variables"] = json.loads(meta.pop("variables_json", "[]"))
    except Exception:
        meta["variables"] = []
    try:
        meta["table_json"] = json.loads(meta.pop("table_json_str", "{}"))
    except Exception:
        meta["table_json"] = {}
    return meta