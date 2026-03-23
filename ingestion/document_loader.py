"""
Document loader using the Unstructured.io hosted API (unstructuredapp.io).

Handles:
  - PDF (digital + scanned via built-in OCR)
  - DOCX, PPTX, XLSX
  - HTML, Markdown
  - Images

Returns a list of typed element dicts ready for downstream parsers.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from utils.logger import logger


# Element types we keep — everything else is discarded
KEPT_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",
    "Image",
    "Formula",
    "CodeSnippet",
    "Header",
    "Footer",
    "Caption",
    "FigureCaption",
    "PageBreak",
}

STRIP_TYPES = {"Header", "Footer", "PageBreak"}

# Correct production URL (unstructuredapp.io, NOT unstructured.io)
_API_URL = "https://api.unstructuredapp.io/general/v0/general"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20))
def _call_unstructured_api(file_bytes: bytes, filename: str) -> list[dict]:
    """
    POST a file to the Unstructured.io API and return raw element list.
    Retries up to 3 times with exponential backoff on transient failures.
    """
    headers = {
        "unstructured-api-key": settings.UNSTRUCTURED_API_KEY,
        "accept": "application/json",
    }

    # NOTE: all form fields must be plain strings.
    # Do NOT pass JSON strings for list fields — send them as repeated keys
    # or as plain comma-separated values depending on the field.
    files = {
        "files": (filename, io.BytesIO(file_bytes), _guess_mime(filename)),
    }
    data = {
        "strategy": "hi_res",
        "infer_table_structure": "true",
        "coordinates": "true",
    }

    logger.debug(f"POSTing to {_API_URL} — file={filename}, size={len(file_bytes)} bytes")

    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            _API_URL,
            headers=headers,
            data=data,
            files=files,
        )

    if response.status_code == 422:
        logger.error(
            f"Unstructured API 422 — request rejected.\n"
            f"Response: {response.text[:500]}"
        )
        raise RuntimeError(f"422 Unprocessable Entity: {response.text[:300]}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Unstructured API error {response.status_code}: {response.text[:300]}"
        )

    return response.json()


def load_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load any supported document and return a list of cleaned element dicts.

    Each element dict has at minimum:
        type        : str   — element type (Title, NarrativeText, Table, …)
        text        : str   — extracted text (or HTML for tables)
        metadata    : dict  — page_number, filename, coordinates, languages, …

    Extra keys for special types:
        image_base64: str   — base64-encoded image crop (Image elements)
        table_html  : str   — raw HTML (Table elements, before JSON conversion)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    logger.info(f"Loading document via Unstructured API: {path.name}")
    file_bytes = path.read_bytes()

    raw_elements = _call_unstructured_api(file_bytes, path.name)
    logger.info(f"  API returned {len(raw_elements)} raw elements")

    elements = _clean_elements(raw_elements, source_file=str(path))
    logger.info(f"  After cleaning: {len(elements)} usable elements")

    return elements


def load_documents(file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load multiple documents, tagging each element with its source file."""
    all_elements: list[dict] = []
    for fp in file_paths:
        try:
            all_elements.extend(load_document(fp))
        except Exception as e:
            logger.error(f"Failed to load {fp}: {e}")
    return all_elements


# ── Helpers ──────────────────────────────────────────────────────────

def _clean_elements(raw: list[dict], source_file: str) -> list[dict]:
    cleaned = []
    for el in raw:
        el_type = el.get("type", "Unknown")

        if el_type in STRIP_TYPES:
            continue
        if el_type not in KEPT_TYPES:
            continue

        text = (el.get("text") or "").strip()
        meta = el.get("metadata", {})

        element: dict[str, Any] = {
            "type": el_type,
            "text": text,
            "metadata": {
                "source_file":  source_file,
                "page_number":  meta.get("page_number", 1),
                "languages":    meta.get("languages", ["eng"]),
                "coordinates":  meta.get("coordinates"),
                "parent_id":    meta.get("parent_id"),
                "filename":     meta.get("filename", Path(source_file).name),
            },
        }

        # Preserve raw table HTML for the table parser
        if el_type == "Table":
            element["table_html"] = meta.get("text_as_html", text)

        # Preserve image data for figure parser
        if el_type == "Image":
            element["image_base64"]   = meta.get("image_base64", "")
            element["image_mime_type"] = meta.get("image_mime_type", "image/png")

        cleaned.append(element)

    return cleaned


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".html": "text/html",
        ".md":   "text/markdown",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return mime_map.get(ext, "application/octet-stream")