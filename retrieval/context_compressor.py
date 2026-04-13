"""
Context compression — reduces token count before sending to LLM:
  1. Token budget check — trim to MAX_CONTEXT_TOKENS
  2. Sentence-level extraction — keep only query-relevant sentences
  3. Smart selection — if still over budget, pick best chunks, not first N

Fixes: context overflow, lost-in-the-middle (by keeping fewer, better chunks).
"""
from __future__ import annotations

from typing import Any

import tiktoken

from config import settings
from utils.logger import logger

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def compress_context(
    chunks: list[dict[str, Any]],
    query: str,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Compress retrieved chunks to fit within the token budget.

    Steps:
      1. Try LLMLingua sentence-level compression if available
      2. Fall back to smart selection (highest score chunks)
      3. Ensure context ordering is preserved
    """
    max_tokens = max_tokens or settings.MAX_CONTEXT_TOKENS

    total_tokens = _total_tokens(chunks)
    if total_tokens <= max_tokens:
        logger.debug(f"Context fits ({total_tokens} tokens), no compression needed")
        return chunks

    logger.info(
        f"Context too large ({total_tokens} tokens > {max_tokens}). Compressing…"
    )

    # Try LLMLingua first (sentence-level)
    compressed = _try_llmlingua(chunks, query, max_tokens)
    if compressed:
        return compressed

    # Fallback: smart chunk selection
    return _smart_select(chunks, max_tokens)


def build_context_string(chunks: list[dict[str, Any]]) -> str:
    """
    Build the final context string sent to the LLM.
    Each chunk is labelled with its source, type, and reading order.

    IMPORTANT: context_text is NOT stored in ChromaDB (only text/NL summary is).
    We reconstruct the full structured content from stored metadata here so the
    LLM sees actual LaTeX equations and table JSON, not just their descriptions.
    """
    parts: list[str] = []

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        el_type = meta.get("element_type", "text")
        source = meta.get("filename") or meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section", "")

        # Header line for citation grounding
        header = f"[Source {i+1} | {source} | Page {page}"
        if section:
            header += f" | {section}"
        header += f" | type:{el_type}]"

        # Reconstruct structured body from stored metadata
        body = _reconstruct_body(chunk, el_type, meta)

        parts.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(parts)


def _reconstruct_body(chunk: dict, el_type: str, meta: dict) -> str:
    """
    Reconstruct the LLM-facing content from stored metadata.

    For Formula  → show LaTeX equation + description
    For Table    → show structured JSON + caption
    For Image    → show alt-text description + caption
    For all else → use stored text directly
    """
    plain_text = chunk.get("text", "")

    if el_type == "Formula":
        latex = meta.get("latex", "").strip()
        raw_formula_text = meta.get("raw_formula_text", "").strip()
        if latex:
            lines = [f"LaTeX equation: ${latex}$"]
            if raw_formula_text and raw_formula_text != latex:
                lines.append(f"Raw extracted text: {raw_formula_text}")
            # Also include the NL description for context
            if plain_text and plain_text != latex:
                lines.append(f"Description: {plain_text}")
            return "\n".join(lines)
        return plain_text

    if el_type == "Table":
        import json as _json
        caption = meta.get("caption", "")
        table_json = meta.get("table_json")
        table_json_str = meta.get("table_json_str", "")
        lines = []
        if caption:
            lines.append(f"Caption: {caption}")
        if table_json is None and table_json_str:
            try:
                table_json = _json.loads(table_json_str)
            except Exception:
                table_json = None
                lines.append(table_json_str)
        if table_json:
            headers = [_clean_table_cell(h) for h in table_json.get("headers", [])]
            rows = table_json.get("rows", [])
            if headers:
                lines.append(f"Headers: {headers}")
            if rows:
                lines.append("Data:")
                for row in rows:
                    cleaned_row = {
                        _clean_table_cell(k): _clean_table_cell(v)
                        for k, v in row.items()
                    }
                    lines.append(f"  {cleaned_row}")
        if not lines:
            return plain_text
        return "\n".join(lines)

    if el_type == "Image":
        alt_text = meta.get("alt_text", "").strip()
        caption = meta.get("caption", "").strip()
        lines = []
        if alt_text:
            lines.append(f"Figure description: {alt_text}")
        if caption and caption not in alt_text:
            lines.append(f"Caption: {caption}")
        return "\n".join(lines) if lines else plain_text

    # NarrativeText, Title, ListItem, ParentChunk, CodeSnippet, etc.
    return plain_text


def _clean_table_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("\n", " ").strip().strip("|").strip()


# ── Helpers ──────────────────────────────────────────────────────────

def _total_tokens(chunks: list[dict]) -> int:
    return sum(
        len(_TOKENIZER.encode(chunk.get("text", "")))
        for chunk in chunks
    )


def _try_llmlingua(
    chunks: list[dict], query: str, max_tokens: int
) -> list[dict] | None:
    """
    Attempt sentence-level compression using LLMLingua.

    Each chunk is compressed individually so that per-chunk provenance
    (chunk_id, metadata, source_file) is preserved for citation grounding.

    Long chunks are split into ≤ 450-token segments before compression to
    avoid exceeding the xlm-roberta model's 512-token sequence limit.

    Returns None if LLMLingua is not installed or fails.
    """
    try:
        from llmlingua import PromptCompressor  # type: ignore

        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )

        current_tokens = _total_tokens(chunks)
        if current_tokens <= 0:
            return None

        # Max tokens per segment — leave room for special tokens within 512 limit
        _MAX_SEGMENT_TOKENS = 450

        # Per-chunk compression preserving provenance
        compressed: list[dict] = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_tokens = len(_TOKENIZER.encode(chunk_text))
            if chunk_tokens <= 0:
                compressed.append(chunk)
                continue

            # Allocate a proportional slice of the total budget
            chunk_budget = max(20, int(max_tokens * chunk_tokens / current_tokens))

            try:
                # Split into segments that fit within model's 512-token limit
                if chunk_tokens > _MAX_SEGMENT_TOKENS:
                    segments = _split_text_to_segments(chunk_text, _MAX_SEGMENT_TOKENS)
                    segment_budget = max(10, chunk_budget // len(segments))
                    compressed_parts = []
                    for segment in segments:
                        seg_result = compressor.compress_prompt(
                            context=[segment],
                            instruction="",
                            question=query,
                            target_token=segment_budget,
                            iterative_size=200,
                        )
                        compressed_parts.append(
                            seg_result.get("compressed_prompt", segment)
                        )
                    compressed_text = " ".join(p for p in compressed_parts if p)
                else:
                    result = compressor.compress_prompt(
                        context=[chunk_text],
                        instruction="",
                        question=query,
                        target_token=chunk_budget,
                        iterative_size=200,
                    )
                    compressed_text = result.get("compressed_prompt", "")

                if compressed_text:
                    compressed.append({
                        **chunk,
                        "text": compressed_text,
                        "context_text": compressed_text,
                    })
                else:
                    compressed.append(chunk)
            except Exception:
                # If individual chunk compression fails, keep original
                compressed.append(chunk)

        return compressed if compressed else None

    except ImportError:
        logger.debug("LLMLingua not available, using smart selection")
        return None
    except Exception as e:
        logger.warning(f"LLMLingua compression failed: {e}")
        return None


def _split_text_to_segments(text: str, max_tokens: int) -> list[str]:
    """Split text into segments of at most max_tokens on sentence boundaries."""
    import re as _re

    sentences = _re.split(r'(?<=[.!?])\s+', text)
    segments: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        sent_tokens = len(_TOKENIZER.encode(sentence))
        if current_count + sent_tokens > max_tokens and current:
            segments.append(" ".join(current))
            current = [sentence]
            current_count = sent_tokens
        else:
            current.append(sentence)
            current_count += sent_tokens

    if current:
        segments.append(" ".join(current))

    return segments if segments else [text]


def _smart_select(
    chunks: list[dict], max_tokens: int
) -> list[dict]:
    """
    Greedily select highest-scoring chunks that fit within the token budget.
    Preserves reading_order after selection for correct context flow.
    """
    # Sort by score descending
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)

    selected: list[dict] = []
    token_count = 0

    for chunk in sorted_chunks:
        chunk_tokens = len(_TOKENIZER.encode(chunk.get("text", "")))
        if token_count + chunk_tokens <= max_tokens:
            selected.append(chunk)
            token_count += chunk_tokens
        else:
            logger.debug(f"Dropped chunk {chunk.get('chunk_id')} to fit token budget")

    # Re-sort by reading_order for correct flow
    selected.sort(
        key=lambda x: x.get("metadata", {}).get("reading_order", 0)
    )

    logger.info(f"Smart select: {len(selected)}/{len(chunks)} chunks, {token_count} tokens")
    return selected
