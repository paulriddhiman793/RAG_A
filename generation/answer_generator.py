"""
Answer generator — builds the final LLM prompt and produces a
citation-grounded, verified response.
"""
from __future__ import annotations

from typing import Any

from generation.llm_client import LLMClient
from generation.hallucination_guard import verify_response
from generation.security import scan_output
from retrieval.context_compressor import build_context_string
from utils.logger import logger


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise research assistant. You answer questions strictly based on \
the provided source documents.

Rules you MUST follow:
1. Answer ONLY using information present in the sources below.
2. After EVERY factual claim, cite the source as [Source N] where N is the \
   source number shown in the header.
3. If multiple sources support a claim, cite all of them: [Source 1, Source 3].
4. If the answer is not in the sources, respond exactly with: \
   "I cannot find this information in the provided documents."
5. Do NOT use your general knowledge to fill gaps — only the sources.
6. For mathematical equations, reproduce the LaTeX exactly as shown in the source.
7. If an equation or OCR text looks corrupted, garbled, or unreadable, say that explicitly and do not infer a cleaner equation.
8. When asked to list equations, include only equations explicitly present in Formula sources, not narrative mentions of models or physics.
9. For tables, summarise the data and cite the source — do not reproduce large tables.
10. Never fabricate citations or claim a source says something it does not.
"""

NO_ANSWER_RESPONSE = (
    "I cannot find relevant information about this in the provided documents. "
    "The query may be outside the scope of the indexed content, or the documents "
    "may not contain this information."
)


# ── Public API ────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    llm_client: LLMClient,
    verify: bool = True,
    include_images: bool = True,
) -> dict[str, Any]:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Returns:
        {
            "answer": str,
            "is_grounded": bool,
            "flagged_claims": list,
            "sources_used": list,
            "no_answer": bool,
        }
    """
    if not chunks:
        return {
            "answer": NO_ANSWER_RESPONSE,
            "is_grounded": True,
            "flagged_claims": [],
            "sources_used": [],
            "no_answer": True,
        }

    # Build context string with source labels
    context = build_context_string(chunks)

    # Construct user prompt
    user_prompt = (
        f"Sources:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite every claim with [Source N]):"
    )

    # Generate — optionally include images for multimodal chunks
    image_chunks = _extract_image_chunks(chunks) if include_images else []

    try:
        if image_chunks:
            raw_answer = _generate_with_images(
                user_prompt, image_chunks, llm_client
            )
        else:
            raw_answer = llm_client.complete(
                user_prompt,
                system=SYSTEM_PROMPT,
                max_tokens=settings_max_tokens(),
            )
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            "answer": f"Generation failed: {e}",
            "is_grounded": False,
            "flagged_claims": ["Generation error"],
            "sources_used": [],
            "no_answer": False,
        }

    # Output injection scan
    output_safe, reason = scan_output(raw_answer)
    if not output_safe:
        logger.warning(f"Output injection detected: {reason}")
        raw_answer = "[Response suppressed due to security policy]"

    # Hallucination verification
    if verify:
        verification = verify_response(raw_answer, chunks)
        final_answer = verification["verified_response"]
        is_grounded = verification["is_grounded"]
        flagged = verification["flagged_claims"]
    else:
        final_answer = raw_answer
        is_grounded = True
        flagged = []

    sources_used = _summarise_sources(chunks)

    return {
        "answer": final_answer,
        "is_grounded": is_grounded,
        "flagged_claims": flagged,
        "sources_used": sources_used,
        "no_answer": False,
    }


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_image_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks that contain image data for multimodal prompting."""
    return [
        c for c in chunks
        if c.get("metadata", {}).get("image_base64")
        or c.get("metadata", {}).get("element_type") == "Image"
    ]


def _generate_with_images(
    prompt: str,
    image_chunks: list[dict],
    llm_client: LLMClient,
) -> str:
    """
    Generate answer using the first available image chunk as visual context.
    Falls back to text-only if vision call fails.
    """
    img_chunk = image_chunks[0]
    img_b64 = img_chunk.get("metadata", {}).get("image_base64", "")
    img_mime = img_chunk.get("metadata", {}).get("image_mime_type", "image/png")

    if not img_b64:
        return llm_client.complete(prompt, system=SYSTEM_PROMPT,
                                   max_tokens=settings_max_tokens())
    try:
        return llm_client.complete_vision(
            text=f"{SYSTEM_PROMPT}\n\n{prompt}",
            image_base64=img_b64,
            image_mime_type=img_mime,
            max_tokens=settings_max_tokens(),
        )
    except Exception as e:
        logger.warning(f"Vision generation failed, falling back to text: {e}")
        return llm_client.complete(prompt, system=SYSTEM_PROMPT,
                                   max_tokens=settings_max_tokens())


def _summarise_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source_num": i + 1,
            "file": c.get("metadata", {}).get("filename", "unknown"),
            "page": c.get("metadata", {}).get("page_number", "?"),
            "section": c.get("metadata", {}).get("section", ""),
            "type": c.get("metadata", {}).get("element_type", "text"),
        }
        for i, c in enumerate(chunks)
    ]


def settings_max_tokens() -> int:
    from config import settings
    return settings.MAX_TOKENS
