"""
Figure parser — converts Image elements into searchable objects by:
  1. Associating the image with its nearest caption
  2. Generating a rich alt-text description via a vision LLM
  3. Producing both an embed_text (NL description) and the raw image

The embed_text is what goes into the vector store.
The raw base64 image is sent to the LLM at retrieval time for visual reasoning.
"""
from __future__ import annotations

import base64
from typing import Any

from utils.logger import logger


# ── Public API ───────────────────────────────────────────────────────

def parse_figure(element: dict[str, Any], llm_client=None) -> dict[str, Any]:
    """
    Enrich an Image element with a searchable description.

    Returns enriched element with:
        alt_text     : rich description from vision LLM
        caption      : nearest figure caption text
        embed_text   : alt_text + caption (what we embed)
        context_text : formatted figure reference for LLM context
        image_base64 : the raw image (kept for multimodal LLM calls)
    """
    caption = element.get("text", "").strip() or _extract_metadata_caption(element)
    image_b64 = element.get("image_base64", "")
    mime = element.get("image_mime_type", "image/png")

    alt_text = _generate_alt_text(image_b64, mime, caption, llm_client)
    embed_text = _build_embed_text(alt_text, caption)

    return {
        **element,
        "alt_text": alt_text,
        "caption": caption,
        "embed_text": embed_text,
        "context_text": _format_for_context(alt_text, caption, image_b64, mime, element),
    }


def parse_figure_batch(
    elements: list[dict[str, Any]], llm_client=None
) -> list[dict[str, Any]]:
    return [
        parse_figure(el, llm_client)
        for el in elements
        if el.get("type") == "Image"
    ]


def attach_captions(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Associate FigureCaption / Caption elements with the nearest Image.
    Must be called before parse_figure_batch to enrich captions.
    """
    result = list(elements)
    caption_buffer: str = ""

    for i, el in enumerate(result):
        el_type = el.get("type", "")
        if el_type in ("FigureCaption", "Caption"):
            caption_buffer = el.get("text", "").strip()
        elif el_type == "Image" and caption_buffer:
            result[i] = {**el, "text": caption_buffer}
            caption_buffer = ""

    return result


# ── Helpers ──────────────────────────────────────────────────────────

_ALT_TEXT_PROMPT = """\
Describe this scientific diagram or figure in detail for a search index.
Your description should include:
1. What TYPE of diagram this is (e.g., line chart, bar chart, circuit diagram, geometric construction, flowchart, microscopy image, etc.)
2. What the diagram SHOWS — axes, labels, key values, trends, relationships
3. What CONCEPT or result it illustrates
4. Any important annotations, legends, or scale markers

Be specific and thorough. 2-4 sentences."""


def _generate_alt_text(
    image_b64: str, mime: str, caption: str, llm_client=None
) -> str:
    """Call a vision LLM to describe the image."""
    if not image_b64 or llm_client is None:
        return caption or "Figure — no description available."

    try:
        prompt_text = _ALT_TEXT_PROMPT
        if caption:
            prompt_text += f"\n\nFigure caption (for context): {caption}"

        description = llm_client.complete_vision(
            text=prompt_text,
            image_base64=image_b64,
            image_mime_type=mime,
            max_tokens=300,
        )
        return description.strip()
    except Exception as e:
        logger.warning(f"Vision LLM alt-text generation failed: {e}")
        return caption or "Figure — description generation failed."


def _build_embed_text(alt_text: str, caption: str) -> str:
    """Combine alt-text and caption into the string we embed."""
    parts = []
    if alt_text:
        parts.append(alt_text)
    if caption and caption not in alt_text:
        parts.append(f"Caption: {caption}")
    return " ".join(parts)


def _extract_metadata_caption(element: dict) -> str:
    return element.get("metadata", {}).get("caption", "")


def _format_for_context(
    alt_text: str, caption: str, image_b64: str, mime: str, element: dict
) -> str:
    """Format figure reference for LLM context (text + optional image)."""
    page = element.get("metadata", {}).get("page_number", "?")
    parts = [f"[Figure — Page {page}]"]
    if caption:
        parts.append(f"Caption: {caption}")
    if alt_text:
        parts.append(f"Description: {alt_text}")
    # The actual image_base64 is preserved in element for multimodal calls
    return "\n".join(parts)