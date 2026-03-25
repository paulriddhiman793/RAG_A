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
import re
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
    result = [dict(el) for el in elements]
    pending_caption_by_page: dict[int, str] = {}
    last_image_by_page: dict[int, int] = {}

    for i, el in enumerate(result):
        el_type = el.get("type", "")
        page = int(el.get("metadata", {}).get("page_number", 0) or 0)

        if el_type == "Image":
            pending_caption = pending_caption_by_page.pop(page, "")
            existing_text = (el.get("text", "") or "").strip()
            if pending_caption and _should_replace_image_text(existing_text, pending_caption):
                result[i] = {**el, "text": pending_caption}
            last_image_by_page[page] = i
            continue

        if el_type not in ("FigureCaption", "Caption"):
            continue

        caption = (el.get("text", "") or "").strip()
        if not caption or _looks_like_table_caption(caption):
            continue

        previous_image_idx = last_image_by_page.get(page)
        if previous_image_idx is not None:
            previous_image = result[previous_image_idx]
            existing_text = (
                (previous_image.get("text", "") or "").strip()
                or _extract_metadata_caption(previous_image)
            )
            if _should_replace_image_text(existing_text, caption):
                result[previous_image_idx] = {**previous_image, "text": caption}
                last_image_by_page.pop(page, None)
                continue

        pending_caption_by_page[page] = caption

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
    parts = ["Figure"]
    label = _extract_figure_label(caption)
    if label:
        parts.append(f"Label: {label}")
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


_FIGURE_LABEL_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)\b", re.IGNORECASE)


def _extract_figure_label(text: str) -> str:
    match = _FIGURE_LABEL_RE.search(text or "")
    if not match:
        return ""
    return f"Fig. {match.group(1)}"


def _looks_like_figure_caption(text: str) -> bool:
    return bool(_FIGURE_LABEL_RE.search(text or ""))


def _looks_like_table_caption(text: str) -> bool:
    return bool(re.match(r"^\s*table\b", text or "", flags=re.IGNORECASE))


def _looks_like_ocr_noise(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    if _looks_like_figure_caption(sample) or _looks_like_table_caption(sample):
        return False
    letters = sum(ch.isalpha() for ch in sample)
    digits = sum(ch.isdigit() for ch in sample)
    punctuation = sum(ch in "|[]()/:;,-" for ch in sample)
    return punctuation + digits > letters or len(sample.split()) > 18


def _should_replace_image_text(existing_text: str, new_caption: str) -> bool:
    existing = (existing_text or "").strip()
    incoming = (new_caption or "").strip()
    if not incoming:
        return False
    if not existing:
        return True
    if _looks_like_figure_caption(incoming) and not _looks_like_figure_caption(existing):
        return True
    if _looks_like_ocr_noise(existing) and not _looks_like_ocr_noise(incoming):
        return True
    return False
