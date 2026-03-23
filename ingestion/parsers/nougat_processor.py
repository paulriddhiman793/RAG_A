"""
Nougat processor — uses Meta's Nougat model to convert academic PDF pages
into structured Markdown with proper LaTeX equations.

Nougat is a vision transformer trained on arXiv papers. It outputs:
  - Section headings
  - Paragraph text
  - Display equations as $$ ... $$ blocks
  - Inline equations as $ ... $
  - Tables as Markdown tables

Installation (run once):
    pip install nougat-ocr

Usage:
    from ingestion.parsers.nougat_processor import extract_with_nougat
    result = extract_with_nougat("paper.pdf")
    # result = {"pages": [...], "equations": [...], "tables": [...]}

Falls back gracefully to None if nougat is not installed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from utils.logger import logger


# ── Public API ────────────────────────────────────────────────────────

def is_nougat_available() -> bool:
    """
    Check if nougat-ocr is installed and runnable.

    nougat-ocr installs as a shell entry point (nougat / nougat.exe).
    It does NOT support `python -m nougat`.
    We check:
      1. nougat shell entry point via shutil.which
      2. Direct presence in venv Scripts/bin folder
      3. import nougat (package importable check)
    """
    import shutil

    # Method 1: shell entry point
    if shutil.which("nougat"):
        return True

    # Method 2: venv Scripts/bin folder directly
    scripts_dir = Path(sys.executable).parent
    for name in ("nougat", "nougat.exe", "nougat.cmd"):
        if (scripts_dir / name).exists():
            return True

    # Method 3: importable package
    try:
        import nougat  # noqa: F401
        return True
    except ImportError:
        pass

    return False


def extract_with_nougat(
    pdf_path: str | Path,
    pages: list[int] | None = None,
) -> dict[str, Any] | None:
    """
    Run Nougat on a PDF and return structured content.

    Parameters
    ----------
    pdf_path : path to the PDF file
    pages    : optional list of 1-indexed page numbers to process.
               None = process all pages.

    Returns
    -------
    dict with keys:
        raw_markdown : str   — full Nougat output
        equations    : list  — each item: {latex, page, display, label}
        tables       : list  — each item: {markdown, page, caption}
        pages        : list  — per-page text content
    or None if nougat is unavailable or fails.
    """
    if not is_nougat_available():
        logger.info("Nougat not installed — skipping. Install with: pip install nougat-ocr")
        return None

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return None

    logger.info(f"Running Nougat on {pdf_path.name} …")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build nougat CLI command
        # Try python -m nougat first (works in most venvs),
        # fall back to nougat shell command
        base_cmd = _get_nougat_command()
        if base_cmd is None:
            logger.warning("Cannot find nougat executable")
            return None

        # Auto-detect GPU — passes CUDA_VISIBLE_DEVICES and batchsize
        gpu_args = _get_gpu_args()
        logger.info(f"Nougat device: {gpu_args['device_label']}")

        cmd = base_cmd + [
            str(pdf_path),
            "--out", str(tmp),
            "--no-skipping",
            "--batchsize", str(gpu_args["batchsize"]),
        ]
        if pages:
            cmd += ["--pages", ",".join(str(p) for p in pages)]

        # Set CUDA env for subprocess
        env = {**os.environ, **gpu_args["env"]}

        # Timeout scales by device:
        #   GPU : 30 sec/page  (batchsize ≥ 4)
        #   CPU : 3 min/page   (batchsize 1)
        estimated_pages = len(pages) if pages else 15
        secs_per_page = 30 if gpu_args["batchsize"] >= 4 else 180
        timeout_secs = max(300, min(5400, estimated_pages * secs_per_page))
        logger.info(
            f"Nougat timeout: {timeout_secs//60} min "
            f"({secs_per_page}s/page × {estimated_pages} pages, "
            f"device: {gpu_args['device_label']})"
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_secs,
                env=env,
            )
            if result.returncode != 0:
                logger.warning(f"Nougat exited with code {result.returncode}: {result.stderr[:200]}")
                return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Nougat timed out after {timeout_secs//60} minutes")
            return None
        except Exception as e:
            logger.warning(f"Nougat subprocess failed: {e}")
            return None

        # Find the output .mmd file (Nougat outputs <stem>.mmd)
        mmd_files = list(tmp.glob("*.mmd"))
        if not mmd_files:
            logger.warning("Nougat produced no .mmd output file")
            return None

        raw_markdown = mmd_files[0].read_text(encoding="utf-8")

    return _parse_nougat_output(raw_markdown)


def _get_gpu_args() -> dict:
    """
    Detect CUDA GPU availability and return appropriate Nougat settings.

    Returns a dict with:
        device_label : str  — human-readable label for logging
        batchsize    : int  — pages per batch (larger = faster on GPU)
        env          : dict — extra env vars to pass to subprocess

    GPU speeds (approximate):
        CPU only      : 2-4 min/page
        RTX 3060/4060 : 8-15 sec/page  (~15x faster)
        RTX 3090/4090 : 4-8 sec/page   (~25x faster)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Larger batchsize = faster but needs more VRAM
            # 4 GB VRAM  → batchsize 4
            # 8 GB VRAM  → batchsize 8
            # 12+ GB     → batchsize 12
            batchsize = min(12, max(4, int(vram_gb / 1.5)))

            logger.info(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM) → batchsize {batchsize}")
            return {
                "device_label": f"CUDA ({gpu_name})",
                "batchsize": batchsize,
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
            }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")

    # CPU fallback
    return {
        "device_label": "CPU (no CUDA GPU found)",
        "batchsize": 1,   # batchsize 1 is safest on CPU
        "env": {"CUDA_VISIBLE_DEVICES": ""},  # disable CUDA
    }


def _get_nougat_command() -> list[str] | None:
    """
    Return the correct command list to invoke nougat.

    nougat-ocr installs a shell entry point (nougat.exe on Windows,
    nougat on Linux/Mac) but does NOT have a __main__.py so
    `python -m nougat` does not work.
    """
    import shutil

    # Primary: nougat shell entry point (nougat.exe on Windows)
    nougat_bin = shutil.which("nougat")
    if nougat_bin:
        return [nougat_bin]

    # Fallback: look directly in the venv Scripts / bin folder
    scripts_dir = Path(sys.executable).parent
    for name in ("nougat", "nougat.exe", "nougat.cmd"):
        candidate = scripts_dir / name
        if candidate.exists():
            return [str(candidate)]

    return None


# ── Parser ────────────────────────────────────────────────────────────

def _parse_nougat_output(raw: str) -> dict[str, Any]:
    """Parse Nougat's .mmd output into structured components."""
    equations = _extract_equations(raw)
    tables = _extract_tables(raw)
    pages = _split_pages(raw)

    logger.info(
        f"Nougat extracted: {len(equations)} equations, "
        f"{len(tables)} tables, {len(pages)} pages"
    )

    return {
        "raw_markdown": raw,
        "equations": equations,
        "tables": tables,
        "pages": pages,
    }


# Display equations: $$ ... $$ (multi-line)
_DISPLAY_EQ = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
# Inline equations: $ ... $ (single line, not inside $$)
_INLINE_EQ = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
# Equation labels: \tag{...} or \label{...} or (1), (2) etc.
_EQ_LABEL = re.compile(r"\\(?:tag|label)\{([^}]+)\}|\((\d+)\)")
# Page breaks in Nougat output
_PAGE_BREAK = re.compile(r"\[MISSING_PAGE[^\]]*\]|\n{3,}")


def _extract_equations(text: str) -> list[dict]:
    """Extract all equations with their LaTeX and position."""
    equations = []
    # Track approximate page number (Nougat uses ## headings for sections)
    approx_page = 1

    # Display equations
    for m in _DISPLAY_EQ.finditer(text):
        latex = m.group(1).strip()
        if not latex or len(latex) < 3:
            continue

        # Find label near this equation
        context = text[max(0, m.start()-50):m.end()+50]
        label_m = _EQ_LABEL.search(context)
        label = label_m.group(1) or label_m.group(2) if label_m else ""

        equations.append({
            "latex": latex,
            "display": True,
            "label": label,
            "position": m.start(),
            "page": _estimate_page(text, m.start()),
        })

    # Inline equations (shorter, simpler)
    for m in _INLINE_EQ.finditer(text):
        latex = m.group(1).strip()
        # Skip trivial ones like $x$, $n$
        if not latex or len(latex) < 5:
            continue
        # Skip if it's inside a display equation region
        if any(e["position"] <= m.start() <= e["position"] + 100
               for e in equations if e["display"]):
            continue

        equations.append({
            "latex": latex,
            "display": False,
            "label": "",
            "position": m.start(),
            "page": _estimate_page(text, m.start()),
        })

    return equations


# Markdown table pattern: starts with | header | header |
_MD_TABLE = re.compile(
    r"(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n)*)",
    re.MULTILINE,
)
# Caption before or after a table
_TABLE_CAPTION = re.compile(
    r"(?:Table\s+\d+[.:]\s*[^\n]+)|(?:\*\*Table\s+\d+\*\*[^\n]*)",
    re.IGNORECASE,
)


def _extract_tables(text: str) -> list[dict]:
    """Extract Markdown tables from Nougat output."""
    tables = []
    for m in _MD_TABLE.finditer(text):
        md_table = m.group(1).strip()

        # Look for caption near the table
        context_before = text[max(0, m.start()-200):m.start()]
        context_after = text[m.end():min(len(text), m.end()+200)]
        cap_m = _TABLE_CAPTION.search(context_before) or _TABLE_CAPTION.search(context_after)
        caption = cap_m.group(0).strip() if cap_m else ""

        tables.append({
            "markdown": md_table,
            "caption": caption,
            "page": _estimate_page(text, m.start()),
            "json": _markdown_table_to_json(md_table),
        })

    return tables


def _split_pages(text: str) -> list[str]:
    """Split Nougat output into approximate per-page text blocks."""
    # Nougat doesn't always emit page markers — split on large gaps
    blocks = re.split(r"\n{4,}", text)
    return [b.strip() for b in blocks if b.strip()]


def _estimate_page(text: str, position: int) -> int:
    """Estimate page number from position in Nougat output."""
    # Count approximate page breaks before this position
    preceding = text[:position]
    breaks = len(re.findall(r"\[MISSING_PAGE[^\]]*\]", preceding))
    # Rough heuristic: ~3000 chars per page
    char_pages = position // 3000
    return max(1, breaks + char_pages // 2)


def _markdown_table_to_json(md: str) -> dict:
    """Convert Markdown table to JSON with headers and rows."""
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return {"headers": [], "rows": []}

    # Parse header row
    headers = [c.strip() for c in lines[0].split("|") if c.strip()]
    # Skip separator line (index 1)
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            rows.append(dict(zip(headers, cells + [""] * (len(headers) - len(cells)))))

    return {"headers": headers, "rows": rows}