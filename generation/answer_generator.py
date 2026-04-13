"""
Answer generator — builds the final LLM prompt and produces a
citation-grounded, verified response.
"""
from __future__ import annotations

import base64
from io import BytesIO
import re
from typing import Any

from generation.llm_client import LLMClient
from generation.hallucination_guard import verify_response
from generation.security import scan_output
from retrieval.context_compressor import build_context_string
from utils.intent import looks_table_like_visual as _looks_table_like_visual
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

    formula_answer = _maybe_answer_formula_query(query, chunks)
    if formula_answer is not None:
        return {
            "answer": formula_answer,
            "is_grounded": True,
            "flagged_claims": [],
            "sources_used": _summarise_sources(chunks),
            "no_answer": False,
        }

    summary_answer = _maybe_answer_summary_query(query, chunks, llm_client)
    if summary_answer is not None:
        return {
            "answer": summary_answer,
            "is_grounded": True,
            "flagged_claims": [],
            "sources_used": _summarise_sources(chunks),
            "no_answer": False,
        }

    figure_answer = _maybe_answer_figure_query(query, chunks, llm_client)
    if figure_answer is not None:
        return {
            "answer": figure_answer,
            "is_grounded": True,
            "flagged_claims": [],
            "sources_used": _summarise_sources(chunks),
            "no_answer": False,
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
    image_chunks = (
        _extract_image_chunks(chunks)
        if include_images and _should_include_images(query)
        else []
    )

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


def _should_include_images(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "figure", "figures", "image", "images", "diagram", "plot", "plots",
        "chart", "charts", "graph", "graphs", "heatmap", "visual"
    ])


def _maybe_answer_summary_query(
    query: str,
    chunks: list[dict[str, Any]],
    llm_client: LLMClient,
) -> str | None:
    q = (query or "").lower().strip()
    if not any(term in q for term in [
        "summary", "summarize", "summarise", "overview", "gist",
        "what is this pdf about", "what is this document about",
        "explain this pdf", "explain this document",
    ]):
        return None

    context = build_context_string(chunks)
    prompt = (
        f"Sources:\n\n{context}\n\n"
        "---\n\n"
        "Question: Provide a concise but well-structured summary of the document. "
        "Cover the main objective, methodology, important results, and conclusion. "
        "Cite every factual claim with [Source N]."
    )
    try:
        return llm_client.complete(
            prompt,
            system=SYSTEM_PROMPT,
            max_tokens=settings_max_tokens(),
        ).strip()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return None


def _maybe_answer_formula_query(query: str, chunks: list[dict[str, Any]]) -> str | None:
    q = (query or "").lower()
    if not any(term in q for term in ["equation", "equations", "formula", "formulas", "latex", "loss function"]):
        return None

    formula_entries = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        if meta.get("element_type") != "Formula":
            continue
        latex = (meta.get("latex", "") or "").strip()
        raw = (meta.get("raw_formula_text", "") or "").strip()
        corrupted = _looks_corrupted_formula_text(latex, raw or chunk.get("text", ""))
        formula_entries.append({
            "source_num": i,
            "page": meta.get("page_number", "?"),
            "latex": latex,
            "raw": raw,
            "corrupted": corrupted,
            "vision_source": bool(meta.get("vision_source")),
        })

    if not formula_entries:
        return None

    if "loss function" in q:
        ranked = sorted(
            formula_entries,
            key=lambda e: _formula_relevance_score(q, e),
            reverse=True,
        )
        best = ranked[0]
        ref = f"[Source {best['source_num']}]"
        if not best["corrupted"] and best["latex"]:
            return f"The loss function equation is `${best['latex']}$` {ref}."

        corrupted = [e for e in ranked if e["corrupted"]]
        if corrupted:
            refs = ", ".join(f"[Source {e['source_num']}]" for e in corrupted)
            pages = ", ".join(str(e["page"]) for e in corrupted)
            return (
                "I found formula chunks that are likely relevant, but the extracted equation text "
                f"for the page {pages} formula(s) appears OCR-corrupted, so I cannot reproduce the "
                f"loss function equation reliably {refs}."
            )

    if "all the equations" in q or "all equations" in q or "equations" in q:
        ranked = sorted(
            formula_entries,
            key=lambda e: (
                int(e["corrupted"]),
                -int(e.get("page", 10**6) if str(e.get("page", "")).isdigit() else 10**6),
                -int(e.get("vision_source", False)),
            ),
        )
        clean = [e for e in ranked if not e["corrupted"] and e["latex"]]
        corrupted = [e for e in ranked if e["corrupted"]]

        parts = []
        if clean:
            clean_bits = []
            for e in clean:
                ref = f"[Source {e['source_num']}]"
                clean_bits.append(f"Page {e['page']}: `${e['latex']}$` {ref}")
            parts.append("I could reliably extract these equations: " + "; ".join(clean_bits) + ".")
        if corrupted:
            bad_bits = []
            for e in corrupted:
                ref = f"[Source {e['source_num']}]"
                text = e["raw"] or e["latex"] or "unreadable equation text"
                bad_bits.append(f"Page {e['page']}: `{text}` {ref}")
            parts.append(
                "One or more additional equation chunks are still OCR-corrupted, so I am listing them as unresolved extractions: "
                + "; ".join(bad_bits)
                + "."
            )
        return " ".join(parts)

    best = formula_entries[0]
    ref = f"[Source {best['source_num']}]"
    if best["corrupted"]:
        text = best["raw"] or best["latex"] or "unreadable equation text"
        return (
            "The retrieved formula text appears OCR-corrupted, so I cannot reproduce it reliably. "
            f"Extracted text: `{text}` {ref}"
        )
    return f"The equation is `${best['latex']}$` {ref}."


def _maybe_answer_figure_query(
    query: str,
    chunks: list[dict[str, Any]],
    llm_client: LLMClient | None = None,
) -> str | None:
    q = (query or "").lower()
    if not any(term in q for term in [
        "figure", "figures", "image", "images", "diagram", "plot", "plots",
        "chart", "charts", "graph", "graphs", "heatmap", "visual"
    ]):
        return None

    requested_num = _extract_requested_figure_number(query)
    figure_entries = []
    seen: set[tuple[Any, ...]] = set()
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        if meta.get("element_type") != "Image":
            continue
        caption = (meta.get("caption", "") or "").strip()
        alt_text = (meta.get("alt_text", "") or "").strip()
        text = (chunk.get("text", "") or "").strip()
        if _looks_table_like_visual(caption, alt_text, text):
            continue
        page = meta.get("page_number", "?")
        label = _extract_figure_label(" ".join([caption, alt_text, text]))
        dedupe_key = (page, label or caption or alt_text[:80] or text[:80])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        figure_entries.append({
            "source_num": i,
            "page": page,
            "label": label,
            "label_num": _extract_requested_figure_number(" ".join([caption, alt_text, text])),
            "caption": caption,
            "alt_text": alt_text,
            "image_base64": meta.get("image_base64", ""),
            "image_mime_type": meta.get("image_mime_type", "image/png"),
            "source_file": meta.get("source_file", ""),
        })

    if not figure_entries:
        return None

    figure_entries.sort(
        key=lambda e: (
            int(e.get("page", 10**6) if str(e.get("page", "")).isdigit() else 10**6),
            e.get("label", ""),
        )
    )

    requested_nums = _extract_requested_figure_numbers(query)
    if len(requested_nums) >= 2:
        targets = []
        for num in requested_nums[:2]:
            target = _select_figure_entry(num, figure_entries)
            if target is not None:
                targets.append((num, target))
        if len(targets) == 2:
            return _compare_requested_figures(query, targets, chunks, llm_client)

    if requested_num is not None:
        target = _select_figure_entry(requested_num, figure_entries)
        if target is not None:
            support_chunks = _collect_supporting_figure_chunks(requested_num, target, chunks)
            ref = f"[Source {target['source_num']}]"
            support_refs = ", ".join(
                f"[Source {c['source_num']}]"
                for c in support_chunks
                if c["source_num"] != target["source_num"]
            )
            support_summary = "\n".join(
                f"[Source {c['source_num']}] {c['text']}"
                for c in support_chunks
                if c["source_num"] != target["source_num"]
            )
            caption = target["caption"] or target["alt_text"] or f"Figure {requested_num}"

            image_base64 = target.get("image_base64") or _render_page_image(
                target.get("source_file", ""),
                target.get("page"),
            )
            if llm_client is not None and image_base64:
                prompt = (
                    "Explain the requested scientific figure using only the provided evidence.\n"
                    f"Requested figure: Figure {requested_num}\n"
                    f"Primary figure source: {ref}\n"
                    f"Primary figure caption: {caption}\n\n"
                    f"Supporting text:\n{support_summary or 'None'}\n\n"
                    "Write 2-4 sentences. Cite every factual claim with [Source N]. "
                    "Use the image itself to describe the layout or components conservatively when visible. "
                    "Do not say the figure is unavailable if a figure image is present."
                )
                try:
                    return llm_client.complete_vision(
                        text=prompt,
                        image_base64=image_base64,
                        image_mime_type=target.get("image_mime_type", "image/png"),
                        max_tokens=settings_max_tokens(),
                    ).strip()
                except Exception as e:
                    logger.warning(f"Figure-specific vision explanation failed: {e}")

            if support_summary:
                ref_bits = ref if not support_refs else f"{ref}, {support_refs}"
                # Build a concise summary from actual supporting text instead of
                # hardcoded domain-specific language.
                first_support = support_lines[0] if (support_lines := [
                    c['text'] for c in support_chunks
                    if c.get('source_num') != target['source_num'] and c.get('text')
                ]) else ""
                detail = first_support[:200] if first_support else "related document content"
                return (
                    f"Figure {requested_num} presents {caption.lower()} {ref}. "
                    f"The surrounding text describes: {detail} {ref_bits}."
                )
            return f"Figure {requested_num} is captioned as {caption} {ref}."

    if any(term in q for term in ["how many", "count", "number of"]):
        refs = ", ".join(f"[Source {e['source_num']}]" for e in figure_entries)
        pages = ", ".join(str(e["page"]) for e in figure_entries)
        labels = [e["label"] for e in figure_entries if e["label"]]
        if labels and len(labels) == len(figure_entries):
            return (
                f"I found {len(figure_entries)} figures in the indexed document content: "
                f"{', '.join(labels)} on pages {pages} {refs}."
            )
        return (
            f"I found {len(figure_entries)} figure entries in the indexed document content, "
            f"on pages {pages} {refs}."
        )

    if any(term in q for term in [
        "all figures", "all graphs", "all charts", "all plots",
        "list", "which figures", "explain all the graphs", "explain all graphs"
    ]):
        if llm_client is not None:
            selected = figure_entries[: min(4, len(figure_entries))]
            image_payloads = []
            for entry in selected:
                image_payload = entry.get("image_base64") or _render_page_image(
                    entry.get("source_file", ""),
                    entry.get("page"),
                )
                if image_payload:
                    image_payloads.append(image_payload)
            composite = _compose_images_side_by_side(image_payloads) if len(image_payloads) >= 2 else (image_payloads[0] if image_payloads else "")
            if composite:
                figure_lines = []
                for entry in selected:
                    ref = f"[Source {entry['source_num']}]"
                    label = entry["label"] or f"Figure on page {entry['page']}"
                    caption = entry["caption"] or entry["alt_text"] or "No caption available"
                    figure_lines.append(f"{label}: {caption} {ref}")
                prompt = (
                    "Explain the figures shown in the combined image using only the provided captions and visual evidence.\n"
                    "Give a compact summary of each graph/figure and mention how they differ where helpful.\n"
                    "Cite every factual claim with [Source N].\n\n"
                    + "\n".join(figure_lines)
                )
                try:
                    return llm_client.complete_vision(
                        text=prompt,
                        image_base64=composite,
                        image_mime_type="image/png",
                        max_tokens=settings_max_tokens(),
                    ).strip()
                except Exception as e:
                    logger.warning(f"All-figures vision summary failed: {e}")

        parts = []
        for entry in figure_entries:
            ref = f"[Source {entry['source_num']}]"
            label = entry["label"] or f"Figure on page {entry['page']}"
            detail = entry["caption"] or entry["alt_text"] or "No caption available"
            parts.append(f"{label}: {detail} {ref}")
        return "The figures identified in the indexed document content are: " + "; ".join(parts) + "."

    return None


def _formula_relevance_score(query: str, entry: dict[str, Any]) -> tuple[int, int, int, int]:
    text = f"{entry.get('latex', '')} {entry.get('raw', '')}".lower()
    q = (query or "").lower()

    score = 0
    if "loss function" in q:
        if "l_" in text or "l{" in text:
            score += 6
        if "custom" in text:
            score += 4
        if "mse" in text:
            score += 3
        if "vec" in text or "grad" in text or "proj" in text or "div" in text:
            score += 3
    if "equation" in q or "formula" in q or "latex" in q:
        if "=" in text:
            score += 1

    clean_bonus = 1 if not entry.get("corrupted") else 0
    vision_bonus = 1 if entry.get("vision_source") else 0
    page_rank = -int(entry.get("page", 10**6) if str(entry.get("page", "")).isdigit() else 10**6)
    return (score, clean_bonus, vision_bonus, page_rank)


_FIGURE_LABEL_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)\b", re.IGNORECASE)


def _extract_figure_label(text: str) -> str:
    match = _FIGURE_LABEL_RE.search(text or "")
    if not match:
        return ""
    return f"Fig. {match.group(1)}"


def _extract_requested_figure_number(text: str) -> int | None:
    match = _FIGURE_LABEL_RE.search(text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _extract_requested_figure_numbers(text: str) -> list[int]:
    found: list[int] = []
    q = text or ""

    for match in _FIGURE_LABEL_RE.finditer(q):
        try:
            num = int(match.group(1))
        except Exception:
            continue
        if num not in found:
            found.append(num)

    paired_pattern = re.compile(
        r"\bfig(?:ure)?s?\.?\s*(\d+)\s*(?:,|and|&|vs\.?|versus)\s*(\d+)\b",
        re.IGNORECASE,
    )
    for match in paired_pattern.finditer(q):
        for group in (1, 2):
            try:
                num = int(match.group(group))
            except Exception:
                continue
            if num not in found:
                found.append(num)
    return found


def _select_figure_entry(requested_num: int, figure_entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    exact = [e for e in figure_entries if e.get("label_num") == requested_num]
    if exact:
        exact.sort(key=lambda e: int(e.get("page", 10**6) if str(e.get("page", "")).isdigit() else 10**6))
        return exact[0]
    if requested_num == 1 and figure_entries:
        return figure_entries[0]
    return None


def _collect_supporting_figure_chunks(
    requested_num: int,
    target: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_page = int(target.get("page", 0) or 0)
    collected = [target]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        el_type = meta.get("element_type", "")
        if el_type in {"Image", "Table", "Formula"}:
            continue
        text = (chunk.get("text", "") or "").strip()
        page = int(meta.get("page_number", 0) or 0)
        if not text:
            continue
        mentions_target = f"figure {requested_num}" in text.lower() or f"fig. {requested_num}" in text.lower()
        nearby = target_page and abs(page - target_page) <= 1
        if mentions_target or nearby:
            collected.append({
                "source_num": i,
                "text": text,
            })
    return collected[:4]


def _compare_requested_figures(
    query: str,
    targets: list[tuple[int, dict[str, Any]]],
    chunks: list[dict[str, Any]],
    llm_client: LLMClient | None,
) -> str:
    (left_num, left), (right_num, right) = targets
    left_ref = f"[Source {left['source_num']}]"
    right_ref = f"[Source {right['source_num']}]"

    support_lines = []
    for requested_num, target in targets:
        for chunk in _collect_supporting_figure_chunks(requested_num, target, chunks):
            if chunk["source_num"] == target["source_num"]:
                continue
            support_lines.append(f"[Source {chunk['source_num']}] {chunk['text']}")
    support_lines = list(dict.fromkeys(support_lines))
    support_summary = "\n".join(support_lines[:6])

    left_image = left.get("image_base64") or _render_page_image(left.get("source_file", ""), left.get("page"))
    right_image = right.get("image_base64") or _render_page_image(right.get("source_file", ""), right.get("page"))
    composite = _compose_images_side_by_side([left_image, right_image]) if left_image and right_image else ""

    if llm_client is not None and composite:
        prompt = (
            "Compare the two scientific figures shown in the combined image.\n"
            f"Left image is Figure {left_num} from {left_ref}.\n"
            f"Left caption: {left.get('caption') or left.get('alt_text') or f'Figure {left_num}'}\n"
            f"Right image is Figure {right_num} from {right_ref}.\n"
            f"Right caption: {right.get('caption') or right.get('alt_text') or f'Figure {right_num}'}\n\n"
            f"Supporting text:\n{support_summary or 'None'}\n\n"
            "Answer the user query directly. If the figures are not the same, clearly state the main differences. "
            "Use only the provided evidence and the composite image. Cite every factual claim with [Source N]."
        )
        try:
            return llm_client.complete_vision(
                text=prompt,
                image_base64=composite,
                image_mime_type="image/png",
                max_tokens=settings_max_tokens(),
            ).strip()
        except Exception as e:
            logger.warning(f"Figure-comparison vision explanation failed: {e}")

    left_caption = left.get("caption") or left.get("alt_text") or f"Figure {left_num}"
    right_caption = right.get("caption") or right.get("alt_text") or f"Figure {right_num}"
    if "same" in (query or "").lower():
        return (
            f"Figure {left_num} and Figure {right_num} are not the same. "
            f"Figure {left_num} is described as {left_caption.lower()} {left_ref}, while "
            f"Figure {right_num} is described as {right_caption.lower()} {right_ref}."
        )
    return (
        f"Figure {left_num} and Figure {right_num} are different. "
        f"Figure {left_num} is described as {left_caption.lower()} {left_ref}, while "
        f"Figure {right_num} is described as {right_caption.lower()} {right_ref}."
    )


# _looks_table_like_visual is imported from utils.intent


def _render_page_image(source_file: str, page_number: Any) -> str:
    if not source_file:
        return ""
    try:
        import fitz
    except Exception:
        return ""

    try:
        page_index = max(0, int(page_number or 1) - 1)
    except Exception:
        page_index = 0

    try:
        doc = fitz.open(source_file)
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")
    except Exception as e:
        logger.warning(f"Figure page render fallback failed: {e}")
        return ""


def _compose_images_side_by_side(images_b64: list[str]) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""

    try:
        images = [
            Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
            for img in images_b64
            if img
        ]
        if not images:
            return ""
        height = max(img.height for img in images)
        resized = []
        for img in images:
            if img.height == height:
                resized.append(img)
                continue
            width = max(1, int(img.width * (height / img.height)))
            resized.append(img.resize((width, height)))
        total_width = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_width, height), color="white")
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += img.width
        buffer = BytesIO()
        canvas.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning(f"Figure composite render failed: {e}")
        return ""


def _looks_corrupted_formula_text(latex: str, raw_text: str) -> bool:
    text = (latex or raw_text or "").strip()
    if not text:
        return True

    suspicious_tokens = {"lyse", "lewstom", "lyag", "lyee", "las", "lz"}
    lowered = text.lower()
    if any(tok in lowered for tok in suspicious_tokens):
        return True
    if "|" in text or "—" in text or ";" in text:
        return True
    return False


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
