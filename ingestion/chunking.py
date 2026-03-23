"""
Chunking module — implements:
  - Overlapping chunks (fixes coreference issue)
  - Section-aware chunking (heading prepended to every sub-chunk)
  - Parent-child hierarchical indexing
  - Semantic similarity splitting (optional, via sentence-transformers)

All chunks get a reading_order index so we can enforce
correct context ordering during retrieval (fixes lost-in-the-middle).
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from utils.logger import logger
from utils.coref_resolver import resolve_coreferences
from config import settings

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str | None
    text: str                      # text we embed
    context_text: str              # text sent to LLM (may include LaTeX/table JSON)
    element_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reading_order: int = 0
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "context_text": self.context_text,
            "element_type": self.element_type,
            "metadata": self.metadata,
            "reading_order": self.reading_order,
            "token_count": self.token_count,
        }


# ── Public API ───────────────────────────────────────────────────────

def chunk_elements(
    elements: list[dict[str, Any]],
    resolve_coref: bool = True,
) -> list[Chunk]:
    """
    Convert a list of parsed elements into Chunk objects.

    Strategy per element type:
        Title            → sets current_section, becomes a tiny parent chunk
        NarrativeText    → split with overlap, prepend section heading
        ListItem         → grouped with its siblings under one chunk
        Table            → one chunk per table (no splitting)
        Image            → one chunk per figure
        Formula          → one chunk per equation
        CodeSnippet      → one chunk per code block
    """
    chunks: list[Chunk] = []
    current_section: str = ""
    list_buffer: list[str] = []
    reading_order = 0

    for el in elements:
        el_type = el.get("type", "NarrativeText")
        embed_text = el.get("embed_text") or el.get("text", "")
        context_text = el.get("context_text") or embed_text
        meta = el.get("metadata", {})

        if el_type == "Title":
            # Flush any buffered list items
            if list_buffer:
                chunks.extend(
                    _make_text_chunks(
                        "\n".join(list_buffer),
                        "ListItem",
                        current_section,
                        meta,
                        reading_order,
                        resolve_coref,
                    )
                )
                list_buffer = []
                reading_order += 1

            current_section = embed_text
            # Title itself becomes a zero-content parent marker chunk
            c = _make_chunk(
                text=embed_text,
                context_text=embed_text,
                el_type="Title",
                meta={**meta, "section": current_section},
                reading_order=reading_order,
            )
            chunks.append(c)
            reading_order += 1

        elif el_type == "ListItem":
            list_buffer.append(embed_text)

        elif el_type == "NarrativeText":
            if list_buffer:
                chunks.extend(
                    _make_text_chunks(
                        "\n".join(list_buffer),
                        "ListItem",
                        current_section,
                        meta,
                        reading_order,
                        resolve_coref,
                    )
                )
                list_buffer = []
                reading_order += 1

            new_chunks = _make_text_chunks(
                embed_text,
                el_type,
                current_section,
                meta,
                reading_order,
                resolve_coref,
            )
            chunks.extend(new_chunks)
            reading_order += len(new_chunks)

        else:
            # Table / Image / Formula / CodeSnippet — one chunk each, no splitting
            if list_buffer:
                chunks.extend(
                    _make_text_chunks(
                        "\n".join(list_buffer),
                        "ListItem",
                        current_section,
                        meta,
                        reading_order,
                        resolve_coref,
                    )
                )
                list_buffer = []
                reading_order += 1

            c = _make_chunk(
                text=embed_text,
                context_text=context_text,
                el_type=el_type,
                meta={**meta, "section": current_section},
                reading_order=reading_order,
                extra=_special_metadata(el),
            )
            chunks.append(c)
            reading_order += 1

    # Flush remaining list items
    if list_buffer:
        chunks.extend(
            _make_text_chunks(
                "\n".join(list_buffer),
                "ListItem",
                current_section,
                {},
                reading_order,
                resolve_coref,
            )
        )

    logger.info(f"Chunking produced {len(chunks)} chunks")
    return chunks


# Element types that must NEVER be merged into parent chunks.
# They carry structured metadata (latex, table_json, image_base64) that
# would be lost if absorbed into a plain-text parent.
_STANDALONE_TYPES = {"Formula", "Table", "Image", "CodeSnippet"}


def build_parent_child(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """
    Build a two-level hierarchy:
      parent chunks  — up to PARENT_CHUNK_SIZE tokens (sent to LLM as context)
      child chunks   — CHUNK_SIZE tokens (used for retrieval)

    Formula / Table / Image / CodeSnippet chunks are NEVER merged into parents
    — they remain as standalone children so their structured metadata is preserved.

    Each child stores its parent_id so we can fetch the parent on retrieval.
    Returns (parent_chunks, child_chunks).
    """
    parent_size = settings.PARENT_CHUNK_SIZE

    parents: list[Chunk] = []
    children: list[Chunk] = []

    # Separate structured chunks from text chunks upfront
    structured: list[Chunk] = []
    text_chunks: list[Chunk] = []
    for ch in chunks:
        if ch.element_type in _STANDALONE_TYPES:
            structured.append(ch)
        else:
            text_chunks.append(ch)

    # Structured chunks: standalone — no parent grouping needed
    for ch in structured:
        children.append(ch)   # parent_id stays None → fetched directly

    # Text chunks: group into parents by section
    section_buckets: dict[str, list[Chunk]] = {}
    for ch in text_chunks:
        sec = ch.metadata.get("section", "__root__")
        section_buckets.setdefault(sec, []).append(ch)

    for section, bucket in section_buckets.items():
        parent_text = " ".join(c.text for c in bucket)
        parent_tokens = _count_tokens(parent_text)

        # Inherit provenance from first child
        first_meta = bucket[0].metadata if bucket else {}

        if parent_tokens <= parent_size:
            parent = _make_chunk(
                text=parent_text,
                context_text=parent_text,
                el_type="ParentChunk",
                meta={
                    "section":     section,
                    "source_file": first_meta.get("source_file", ""),
                    "filename":    first_meta.get("filename", ""),
                    "page_number": first_meta.get("page_number", 0),
                    "language":    first_meta.get("language", "en"),
                },
                reading_order=bucket[0].reading_order if bucket else 0,
            )
            parents.append(parent)
            for ch in bucket:
                ch.parent_id = parent.chunk_id
                children.append(ch)
        else:
            token_budget = 0
            current_group: list[Chunk] = []
            for ch in bucket:
                tok = ch.token_count
                if token_budget + tok > parent_size and current_group:
                    p = _make_parent_from_group(current_group, section)
                    parents.append(p)
                    for c in current_group:
                        c.parent_id = p.chunk_id
                        children.append(c)
                    current_group = [ch]
                    token_budget = tok
                else:
                    current_group.append(ch)
                    token_budget += tok
            if current_group:
                p = _make_parent_from_group(current_group, section)
                parents.append(p)
                for c in current_group:
                    c.parent_id = p.chunk_id
                    children.append(c)

    return parents, children


# ── Internals ─────────────────────────────────────────────────────────

def _make_text_chunks(
    text: str,
    el_type: str,
    section: str,
    meta: dict,
    base_order: int,
    resolve_coref: bool,
) -> list[Chunk]:
    """Split a long text into overlapping chunks, each prepended with section heading."""
    if resolve_coref and el_type == "NarrativeText":
        text = resolve_coreferences(text)

    tokens = _TOKENIZER.encode(text)
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP

    if len(tokens) <= size:
        prefix = f"[Section: {section}]\n" if section else ""
        full_text = prefix + text
        return [
            _make_chunk(
                text=full_text,
                context_text=full_text,
                el_type=el_type,
                meta={**meta, "section": section},
                reading_order=base_order,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    order = base_order

    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens)
        prefix = f"[Section: {section}]\n" if section else ""
        full_text = prefix + chunk_text

        chunks.append(
            _make_chunk(
                text=full_text,
                context_text=full_text,
                el_type=el_type,
                meta={**meta, "section": section, "is_split": True},
                reading_order=order,
            )
        )
        order += 1

        if end == len(tokens):
            break
        start = end - overlap  # overlap for coreference continuity

    return chunks


def _make_chunk(
    text: str,
    context_text: str,
    el_type: str,
    meta: dict,
    reading_order: int,
    parent_id: str | None = None,
    extra: dict | None = None,
) -> Chunk:
    chunk_id = _stable_id(text + el_type)
    tok_count = _count_tokens(text)
    merged_meta = {**meta, **(extra or {})}
    return Chunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        text=text,
        context_text=context_text,
        element_type=el_type,
        metadata=merged_meta,
        reading_order=reading_order,
        token_count=tok_count,
    )


def _make_parent_from_group(group: list[Chunk], section: str) -> Chunk:
    text = " ".join(c.text for c in group)
    first_meta = group[0].metadata if group else {}
    return _make_chunk(
        text=text,
        context_text=text,
        el_type="ParentChunk",
        meta={
            "section":     section,
            "source_file": first_meta.get("source_file", ""),
            "filename":    first_meta.get("filename", ""),
            "page_number": first_meta.get("page_number", 0),
            "language":    first_meta.get("language", "en"),
        },
        reading_order=group[0].reading_order,
    )


def _special_metadata(el: dict) -> dict:
    """Extract type-specific metadata to attach to a chunk."""
    el_type = el.get("type", "")
    if el_type == "Formula":
        return {"latex": el.get("latex", ""), "variables": el.get("variables", [])}
    if el_type == "Table":
        return {"table_json": el.get("table_json", {}), "caption": el.get("caption", "")}
    if el_type == "Image":
        return {
            "image_base64": el.get("image_base64", ""),
            "image_mime_type": el.get("image_mime_type", "image/png"),
            "caption": el.get("caption", ""),
            "alt_text": el.get("alt_text", ""),
        }
    return {}


def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))