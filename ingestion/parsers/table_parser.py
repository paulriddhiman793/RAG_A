"""
Table parser — converts Unstructured Table elements into:
  1. Structured JSON     — preserves row/column relationships
  2. Natural language summary — embedded for semantic retrieval
  3. Pandas DataFrame     — for numeric/data questions

The NL summary is what gets embedded.
The full JSON table is what gets sent to the LLM.

Nougat integration: if nougat_data is provided and has a table on the
same page, we use its cleaner Markdown-based extraction.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from utils.logger import logger


def parse_table(
    element: dict[str, Any],
    llm_client=None,
    nougat_tables: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Enrich a Table element with structured data and a searchable summary.
    """
    page = element.get("metadata", {}).get("page_number", 0)

    # Try Nougat table first (cleaner parsing)
    table_json, source = _get_table_json(element, page, nougat_tables)
    caption = _find_caption(element, table_json)
    df = _json_to_dataframe(table_json)

    # Generate NL summary — this is what gets embedded
    nl_summary = _generate_summary(table_json, caption, df, llm_client)

    # embed_text includes "table data results" for BM25 retrieval
    embed_text = f"table data results: {nl_summary}"

    return {
        **element,
        "table_json": table_json,
        "dataframe": df,
        "nl_summary": nl_summary,
        "caption": caption,
        "source": source,
        "embed_text": embed_text,
        "context_text": _format_for_context(table_json, caption, nl_summary, element),
        "metadata": {
            **element.get("metadata", {}),
            "table_json": table_json,
            "caption": caption,
            "nl_summary": nl_summary,
        },
    }


def parse_table_batch(
    elements: list[dict[str, Any]],
    llm_client=None,
    nougat_data: dict | None = None,
) -> list[dict[str, Any]]:
    nougat_tables = (nougat_data or {}).get("tables", [])
    return [
        parse_table(el, llm_client, nougat_tables)
        for el in elements
        if el.get("type") == "Table"
    ]


def query_dataframe(df: pd.DataFrame, question: str, llm_client=None) -> str:
    """Answer a numeric question directly from a DataFrame."""
    if llm_client is None or df is None or df.empty:
        return ""
    schema = f"Columns: {list(df.columns)}\nSample:\n{df.head(3).to_string()}"
    prompt = (
        f"Given a pandas DataFrame `df`:\n{schema}\n\n"
        f"Write ONE Python expression (no assignment, no print) to answer: '{question}'\n"
        f"Return ONLY the expression."
    )
    try:
        code = llm_client.complete(prompt, max_tokens=80).strip()
        if any(kw in code for kw in ["import", "exec", "eval", "open", "os.", "__"]):
            return ""
        result = eval(code, {"df": df, "pd": pd})  # noqa: S307
        return str(result)
    except Exception as e:
        logger.warning(f"DataFrame query failed: {e}")
        return ""


# ── Table JSON extraction ─────────────────────────────────────────────

def _get_table_json(
    element: dict,
    page: int,
    nougat_tables: list[dict] | None,
) -> tuple[dict, str]:
    """Try Nougat table first, fall back to HTML parsing."""
    if nougat_tables:
        match = _match_nougat_table(element, page, nougat_tables)
        if match:
            logger.debug(f"Table matched via Nougat on page {page}")
            return match, "nougat"

    html = element.get("table_html") or element.get("text", "")
    return _html_to_json(html), "html"


def _match_nougat_table(element: dict, page: int, nougat_tables: list[dict]) -> dict | None:
    """Match a Nougat table to this element by page proximity."""
    candidates = [t for t in nougat_tables if abs(t.get("page", 0) - page) <= 1]
    if not candidates:
        return None
    # Use the first match on the same page
    return candidates[0].get("json")


def _html_to_json(html: str) -> dict:
    """Parse HTML table → {headers: [...], rows: [...]}"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return {"headers": [], "rows": [], "raw": html[:200]}

        rows = table.find_all("tr")
        headers: list[str] = []
        data_rows: list[list[str]] = []

        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            if i == 0 or all(c.name == "th" for c in cells):
                if not headers:
                    headers = cell_texts
                    continue
            data_rows.append(cell_texts)

        n = len(headers) or (len(data_rows[0]) if data_rows else 0)
        padded = [r + [""] * max(0, n - len(r)) for r in data_rows]

        if headers:
            structured_rows = [dict(zip(headers, r)) for r in padded]
        else:
            structured_rows = [{"col_" + str(i): v for i, v in enumerate(r)} for r in padded]

        return {"headers": headers, "rows": structured_rows}
    except Exception as e:
        logger.warning(f"Table HTML parsing failed: {e}")
        return {"headers": [], "rows": [], "raw": html[:200]}


def _json_to_dataframe(table_json: dict) -> pd.DataFrame | None:
    rows = table_json.get("rows")
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
    except Exception:
        return None


# ── NL summary generation ─────────────────────────────────────────────

def _generate_summary(
    table_json: dict,
    caption: str,
    df: pd.DataFrame | None,
    llm_client=None,
) -> str:
    """
    Generate a rich NL summary for embedding.
    Uses LLM if available, otherwise builds a descriptive rule-based summary.
    """
    headers = table_json.get("headers", [])
    rows = table_json.get("rows", [])

    if not headers and not rows:
        return caption or "Table with no extractable data."

    row_count = len(rows)

    if llm_client is None:
        return _rule_based_summary(headers, rows, row_count, caption, df)

    # Rich LLM summary
    table_preview = json.dumps(rows[:6], ensure_ascii=False)
    prompt = (
        f"Summarize this data table in 2-3 sentences for a scientific search index.\n"
        f"Include: what it compares or measures, key values or trends, and what it shows.\n"
        f"Be specific about numbers and column names.\n\n"
        f"Caption: {caption or 'None'}\n"
        f"Headers: {headers}\n"
        f"Data rows (up to 6): {table_preview}\n\n"
        f"Summary:"
    )
    try:
        return llm_client.complete(prompt, max_tokens=200).strip()
    except Exception as e:
        logger.warning(f"Table summary LLM call failed: {e}")
        return _rule_based_summary(headers, rows, row_count, caption, df)


def _rule_based_summary(
    headers: list,
    rows: list,
    row_count: int,
    caption: str,
    df: pd.DataFrame | None,
) -> str:
    """Build a descriptive summary without LLM."""
    parts = []

    if caption:
        parts.append(caption)

    col_str = ", ".join(str(h) for h in headers[:6])
    parts.append(f"Table with {row_count} rows and columns: {col_str}.")

    # Add numeric stats if we have a DataFrame
    if df is not None and not df.empty:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols[:3]:
            try:
                min_v = round(float(df[col].min()), 4)
                max_v = round(float(df[col].max()), 4)
                parts.append(f"{col} ranges from {min_v} to {max_v}.")
            except Exception:
                pass

    # Sample first row values
    if rows:
        first_row_str = ", ".join(f"{k}={v}" for k, v in list(rows[0].items())[:4])
        parts.append(f"Sample row: {first_row_str}.")

    return " ".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────

def _find_caption(element: dict, table_json: dict) -> str:
    text = element.get("text", "")
    if re.match(r"(?i)table\s+\d+", text):
        return text[:200]
    return element.get("metadata", {}).get("caption", "")


def _format_for_context(
    table_json: dict,
    caption: str,
    nl_summary: str,
    element: dict,
) -> str:
    page = element.get("metadata", {}).get("page_number", "?")
    section = element.get("metadata", {}).get("section", "")
    parts = [f"[Table — Page {page}{f' | {section}' if section else ''}]"]
    if caption:
        parts.append(f"Caption: {caption}")
    parts.append(f"Summary: {nl_summary}")
    parts.append(json.dumps(table_json, ensure_ascii=False, indent=2))
    return "\n".join(parts)