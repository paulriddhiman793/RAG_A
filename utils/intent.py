"""
Shared intent detection utilities.

Centralises all question-intent heuristics so every module uses the same
keyword lists.  Previously these functions were duplicated across
query_pipeline, hybrid_retriever, answer_generator, and agent/router.
"""
from __future__ import annotations

import re
from typing import Any


# ── Public helpers ────────────────────────────────────────────────────

def has_formula_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "equation", "equations", "formula", "formulas", "latex", "loss function",
    ])


def has_figure_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "figure", "figures", "image", "images", "diagram", "plot", "plots",
        "chart", "charts", "graph", "graphs", "heatmap", "visual",
    ])


def has_table_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "table", "metric", "metrics", "value", "values", "ssim", "mse",
        "rmse", "r-squared", "r2", "r^2", "mae", "score",
    ])


def has_summary_intent(query: str) -> bool:
    q = (query or "").lower().strip()
    return any(term in q for term in [
        "summary", "summarize", "summarise", "overview", "gist",
        "what is this pdf about", "what is this document about",
        "what is the pdf about", "what is the document about",
        "what is pdf about", "explain this pdf", "explain this document",
    ])


def has_metric_lookup_intent(query: str) -> bool:
    q = (query or "").lower()
    metric_terms = [
        "r2", "r^2", "r-squared", "r squared", "rmse", "mae", "mse",
        "score", "cv r2", "test r2", "train r2",
    ]
    model_terms = [
        "random forest", "randomforest", "regressor", "xgboost", "ridge",
        "lasso", "linear regression", "mlr", "stacking", "ensemble", "poly",
    ]
    return any(t in q for t in metric_terms) and any(t in q for t in model_terms)


def has_compare_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "compare", "difference", "different", "same", "vs", "versus",
    ])


def looks_table_like_visual(caption: str, alt_text: str, text: str) -> bool:
    """Return True when an Image chunk is actually a table rendered as a picture."""
    sample = " ".join([caption, alt_text, text]).strip().lower()
    return (
        sample.startswith("table ")
        or sample.startswith("table:")
        or "table 2" in sample[:40]
    )


def looks_table_like_image(chunk: dict[str, Any]) -> bool:
    """Convenience wrapper that reads metadata from a chunk dict."""
    meta = chunk.get("metadata", {})
    return looks_table_like_visual(
        str(meta.get("caption", "")),
        str(meta.get("alt_text", "")),
        str(chunk.get("text", "")),
    )
