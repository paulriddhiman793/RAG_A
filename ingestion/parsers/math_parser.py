"""
Math parser — converts Formula elements into dual representation:
  1. LaTeX  — from Nougat (clean, accurate) or regex fallback
  2. Natural language — for embedding and semantic search

Priority:
  - If nougat_data is provided (from NougatProcessor), use its equations
  - Otherwise fall back to regex extraction from Unstructured's text
"""
from __future__ import annotations

import re
from typing import Any

from utils.logger import logger


def parse_formula(
    element: dict[str, Any],
    llm_client=None,
    nougat_equations: list[dict] | None = None,
) -> dict[str, Any]:
    raw_text = element.get("text", "")
    page = element.get("metadata", {}).get("page_number", 0)

    latex, source = _get_latex(raw_text, page, nougat_equations)
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
            "nougat_source": source == "nougat",
        },
    }


def parse_formula_batch(
    elements: list[dict[str, Any]],
    llm_client=None,
    nougat_data: dict | None = None,
) -> list[dict[str, Any]]:
    nougat_equations = (nougat_data or {}).get("equations", [])
    return [
        parse_formula(el, llm_client, nougat_equations)
        for el in elements
        if el.get("type") == "Formula"
    ]


def _get_latex(raw_text, page, nougat_equations):
    if nougat_equations:
        match = _match_nougat_equation(raw_text, page, nougat_equations)
        if match:
            logger.debug(f"Formula matched via Nougat: {match[:60]}")
            return match, "nougat"
    latex = _extract_latex_regex(raw_text)
    return latex, "regex"


def _match_nougat_equation(raw_text, page, nougat_equations):
    candidates = [eq for eq in nougat_equations if abs(eq.get("page", 0) - page) <= 1]
    if not candidates:
        candidates = nougat_equations[:10]
    if not candidates:
        return ""

    raw_clean = _clean_for_comparison(raw_text)

    def score(eq):
        latex_clean = _clean_for_comparison(eq.get("latex", ""))
        if not latex_clean or not raw_clean:
            return 0.0
        raw_tri = set(_trigrams(raw_clean))
        lat_tri = set(_trigrams(latex_clean))
        if not raw_tri or not lat_tri:
            return 0.0
        intersection = len(raw_tri & lat_tri)
        union = len(raw_tri | lat_tri)
        boost = 1.2 if eq.get("display") else 1.0
        return (intersection / union) * boost if union else 0.0

    scored = [(score(eq), eq) for eq in candidates]
    best_score, best_eq = max(scored, key=lambda x: x[0])
    if best_score > 0.15:
        return best_eq.get("latex", "")
    return ""


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
