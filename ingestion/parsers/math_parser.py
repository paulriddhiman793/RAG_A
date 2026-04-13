"""
Math parser — converts Formula elements into dual representation:
  1. LaTeX  — from regex fallback
  2. Natural language — for embedding and semantic search
"""
from __future__ import annotations

import base64
import re
from typing import Any

from utils.logger import logger


def parse_formula(
    element: dict[str, Any],
    llm_client=None,
) -> dict[str, Any]:
    raw_text = element.get("text", "")
    page = element.get("metadata", {}).get("page_number", 0)

    latex, source = _get_latex(raw_text, page)
    vision_latex = _get_latex_from_vision(element, llm_client)
    if vision_latex and (_looks_corrupted_formula(latex, raw_text) or not latex):
        latex, source = vision_latex, "vision"
    variables = _extract_variables(latex or raw_text)
    nl = _generate_nl_description(latex or raw_text, raw_text, llm_client)
    embed_text = (
        f"equation formula raw: {raw_text} "
        f"latex: {latex} "
        f"description: {nl}"
    ).strip()

    return {
        **element,
        "latex": latex,
        "raw_formula_text": raw_text,
        "nl_description": nl,
        "variables": variables,
        "source": source,
        "embed_text": embed_text,
        "context_text": _format_for_context(latex, raw_text, nl, element),
        "metadata": {
            **element.get("metadata", {}),
            "latex": latex,
            "raw_formula_text": raw_text,
            "nl_description": nl,
            "variables": variables,
            "vision_source": source == "vision",
        },
    }


def parse_formula_batch(
    elements: list[dict[str, Any]],
    llm_client=None,
) -> list[dict[str, Any]]:
    return [
        parse_formula(el, llm_client)
        for el in elements
        if el.get("type") == "Formula"
    ]


def _get_latex(raw_text, page):
    latex = _extract_latex_regex(raw_text)
    return latex, "regex"


def _get_latex_from_vision(element: dict[str, Any], llm_client=None) -> str:
    from config import settings

    if not settings.USE_FORMULA_VISION_FALLBACK or llm_client is None:
        return ""

    image_b64 = _crop_formula_region(element)
    if not image_b64:
        return ""

    prompt = (
        "Transcribe the mathematical equation shown in this crop.\n"
        "Return only plain LaTeX for the equation on a single line.\n"
        "Do not add dollar signs, prose, labels, or explanations.\n"
        "If the equation is unreadable, return exactly: UNREADABLE"
    )
    try:
        response = llm_client.complete_vision(
            text=prompt,
            image_base64=image_b64,
            image_mime_type="image/png",
            max_tokens=120,
        ).strip()
    except Exception as e:
        logger.warning(f"Formula vision transcription failed: {e}")
        return ""

    response = response.strip().strip("$").strip()
    response = response.splitlines()[0].strip() if response else ""
    if not response or response.upper() == "UNREADABLE":
        return ""
    return response


def _extract_latex_regex(text: str) -> str:
    m = re.search(r"\$\$(.+?)\$\$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\\[(.+?)\\\]", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\\((.+?)\\\)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"[^\w\s\+\-\=\*\/\^\(\)\[\]\{\}\.\,\|\\]", "", text)
    return cleaned.strip()


def _crop_formula_region(element: dict[str, Any]) -> str:
    try:
        import fitz
    except Exception:
        return ""

    meta = element.get("metadata", {})
    source_file = meta.get("source_file", "")
    page_number = int(meta.get("page_number", 1) or 1)
    coords = meta.get("coordinates")
    if not source_file or not coords:
        return ""

    rect = _coords_to_rect(coords)
    if rect is None:
        return ""

    try:
        doc = fitz.open(source_file)
        page = doc[page_number - 1]
        page_rect = page.rect
        clip = _scale_rect_to_page(rect, coords, page_rect)
        clip = clip + (-12, -12, 12, 12)
        clip &= page_rect
        if clip.is_empty:
            return ""
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")
    except Exception as e:
        logger.warning(f"Formula crop failed: {e}")
        return ""


def _coords_to_rect(coords: Any):
    try:
        import fitz
    except Exception:
        return None

    if isinstance(coords, dict):
        points = coords.get("points") or coords.get("layout_points") or coords.get("bbox")
    else:
        points = None

    if isinstance(points, dict):
        x0 = points.get("x1", points.get("left", 0))
        y0 = points.get("y1", points.get("top", 0))
        x1 = points.get("x2", points.get("right", 0))
        y1 = points.get("y2", points.get("bottom", 0))
        return fitz.Rect(float(x0), float(y0), float(x1), float(y1))

    if isinstance(points, (list, tuple)) and points:
        if len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
            x0, y0, x1, y1 = points
            return fitz.Rect(float(x0), float(y0), float(x1), float(y1))

        xs: list[float] = []
        ys: list[float] = []
        for pt in points:
            if isinstance(pt, dict):
                xs.append(float(pt.get("x", 0)))
                ys.append(float(pt.get("y", 0)))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
        if xs and ys:
            return fitz.Rect(min(xs), min(ys), max(xs), max(ys))
    return None


def _scale_rect_to_page(rect, coords: dict[str, Any], page_rect):
    try:
        import fitz
    except Exception:
        return rect

    layout_w = (
        coords.get("layout_width")
        or coords.get("width")
        or coords.get("system", {}).get("width")
        if isinstance(coords, dict)
        else None
    )
    layout_h = (
        coords.get("layout_height")
        or coords.get("height")
        or coords.get("system", {}).get("height")
        if isinstance(coords, dict)
        else None
    )

    if not layout_w or not layout_h:
        return rect

    sx = float(page_rect.width) / float(layout_w)
    sy = float(page_rect.height) / float(layout_h)
    return fitz.Rect(rect.x0 * sx, rect.y0 * sy, rect.x1 * sx, rect.y1 * sy)


def _generate_nl_description(latex, raw_text, llm_client=None):
    if llm_client is None or _looks_corrupted_formula(latex, raw_text):
        return _rule_based_description(latex or raw_text)
    prompt = (
        "Describe this mathematical equation in 1-2 plain-English sentences "
        "for a scientific search index. Include what it defines and what "
        "physical concept it represents.\n\n"
        f"Equation (LaTeX): {latex}\nRaw text: {raw_text}\n\nDescription:"
    )
    try:
        return llm_client.complete(prompt, max_tokens=120).strip()
    except Exception as e:
        logger.warning(f"LLM NL description failed: {e}")
        return _rule_based_description(latex or raw_text)


def _rule_based_description(text: str) -> str:
    patterns = [
        (r"\\nabla|nabla|gradient",            "gradient/differential operator equation"),
        (r"\\frac|\\partial|partial",           "differential equation"),
        (r"\\sum|summation",                    "summation equation"),
        (r"\\int|integral",                     "integral equation"),
        (r"[Ll]oss|MSE|Lvec|Lc\b|Lproj|Ldiv|Lgrad|Lcustom", "loss function equation"),
        (r"velocity|\\hat.*[Uu]|Ux|Uy",        "velocity field equation"),
        (r"LBM|Boltzmann|relaxation",           "Lattice Boltzmann equation"),
        (r"Navier|Stokes",                      "Navier-Stokes fluid dynamics equation"),
        (r"Buckley|Leverett",                   "Buckley-Leverett transport equation"),
        (r"permeab|porosity|Kozeny|Carman",     "porous media permeability equation"),
    ]
    for pattern, description in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return f"Mathematical {description}: {text[:100]}"
    return f"Mathematical formula: {text[:100]}"


def _looks_corrupted_formula(latex: str, raw_text: str) -> bool:
    text = (latex or raw_text or "").strip()
    if not text:
        return True

    math_markers = sum(text.count(ch) for ch in "=+-/*^_\\()[]{}")
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    weird_words = re.findall(r"[A-Za-z]{5,}", text)
    suspicious_tokens = {"lyse", "lewstom", "lyag", "lyee", "las", "lz"}

    if any(tok.lower() in suspicious_tokens for tok in weird_words):
        return True
    if math_markers == 0 and digits > 0:
        return True
    if letters > 0 and math_markers <= 1 and len(weird_words) >= 2:
        return True
    return False


_VARIABLE_PATTERN = re.compile(
    r"(?<![a-zA-Z])([A-Za-z](?:_\{[^}]+\}|_[a-zA-Z0-9])?)(?![a-zA-Z])"
)
_COMMON_OPERATORS = {
    "frac", "sum", "int", "prod", "sqrt", "lim", "log", "exp",
    "sin", "cos", "tan", "nabla", "partial", "infty", "alpha", "beta",
    "gamma", "delta", "epsilon", "theta", "lambda", "mu", "sigma",
    "phi", "psi", "omega", "rho", "tau", "pi", "hat", "vec", "bar",
    "dot", "text", "mathbf", "mathrm", "left", "right",
}


def _extract_variables(text: str) -> list[str]:
    candidates = _VARIABLE_PATTERN.findall(text)
    return list({v for v in candidates if v.lower() not in _COMMON_OPERATORS})


def _format_for_context(latex: str, raw_text: str, nl: str, element: dict) -> str:
    page = element.get("metadata", {}).get("page_number", "?")
    section = element.get("metadata", {}).get("section", "")
    lines = [f"[Equation — Page {page}{f' | {section}' if section else ''}]"]
    if latex:
        lines.append(f"LaTeX: ${latex}$")
    if raw_text and raw_text != latex:
        lines.append(f"Raw extracted text: {raw_text}")
    if _looks_corrupted_formula(latex, raw_text):
        lines.append("Extraction quality note: this equation text appears OCR-corrupted.")
    lines.append(f"Description: {nl}")
    return "\n".join(lines)


def _clean_for_comparison(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _trigrams(text: str) -> list[str]:
    return [text[i:i+3] for i in range(len(text)-2)] if len(text) >= 3 else [text]
