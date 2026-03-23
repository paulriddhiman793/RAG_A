"""
CLI entry point for the Advanced RAG system.

Usage:
    # Ingest documents
    python main.py ingest path/to/paper.pdf
    python main.py ingest path/to/docs/

    # Ask a question
    python main.py query "What is the Kozeny-Carman equation?"

    # Interactive mode
    python main.py interactive

    # Show stats
    python main.py stats
"""
import sys
import os

# ── Load .env into os.environ FIRST ──────────────────────────────────
# Must happen before ANY other import so HuggingFace, torch, and other
# libraries that read os.environ at import time see the correct values.
from pathlib import Path as _Path
_env_file = _Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val
# ─────────────────────────────────────────────────────────────────────

import json
import argparse
from pathlib import Path

# ── Ensure the project root is always first on sys.path ──────────────
# This fixes "unknown location" / import errors on Windows when Python
# picks up a cached or installed package instead of the local one.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from rag_system import RAGSystem

console = Console()


def cmd_ingest(args, rag: RAGSystem) -> None:
    path = Path(args.path)
    with console.status(f"[bold green]Ingesting {path.name}…"):
        if path.is_dir():
            results = rag.ingest_directory(path)
            for r in results:
                _print_ingest_result(r)
        else:
            result = rag.ingest(path)
            _print_ingest_result(result)


def cmd_query(args, rag: RAGSystem) -> None:
    question = args.question
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Retrieving and generating answer…"):
        result = rag.query(question, verify=not args.no_verify, verbose=args.verbose)

    # Print answer
    answer_panel = Panel(
        Markdown(result["answer"]),
        title="[bold green]Answer",
        border_style="green" if result["is_grounded"] else "yellow",
    )
    console.print(answer_panel)

    # Print sources
    if result["sources_used"]:
        t = Table(title="Sources used", show_header=True)
        t.add_column("No.", style="dim", width=4)
        t.add_column("File")
        t.add_column("Page", justify="right")
        t.add_column("Section")
        t.add_column("Type")
        for s in result["sources_used"]:
            t.add_row(
                str(s["source_num"]),
                s["file"],
                str(s["page"]),
                s["section"][:40],
                s["type"],
            )
        console.print(t)

    # Grounding status
    if result["flagged_claims"]:
        console.print("\n[yellow]⚠ Grounding warnings:[/yellow]")
        for claim in result["flagged_claims"]:
            console.print(f"  • {claim}")

    if result["no_answer"]:
        console.print("\n[dim]No relevant content found in indexed documents.[/dim]")

    console.print(
        f"\n[dim]Score: {result['top_score']:.3f} | "
        f"Chunks: {result['chunks_retrieved']} | "
        f"Queries expanded: {len(result['expanded_queries'])}[/dim]"
    )


def cmd_interactive(rag: RAGSystem) -> None:
    console.print(Panel(
        "[bold green]Advanced RAG — Interactive Mode[/bold green]\n"
        "Type your question and press Enter. Type 'exit' to quit.",
        border_style="green",
    ))
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        args_mock = argparse.Namespace(
            question=question, no_verify=False, verbose=False
        )
        cmd_query(args_mock, rag)


def cmd_stats(rag: RAGSystem) -> None:
    stats = rag.stats()
    t = Table(title="RAG System Stats")
    t.add_column("Setting")
    t.add_column("Value")
    for k, v in stats.items():
        t.add_row(k, str(v))
    console.print(t)


def _print_ingest_result(result: dict) -> None:
    status = result.get("status", "unknown")
    color = "green" if status == "success" else "red"
    console.print(
        f"[{color}]{status}[/{color}] {result['file']} | "
        f"chunks: {result.get('child_chunks', 0)} | "
        f"domain: {result.get('domain', '-')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced RAG System")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest document(s)")
    p_ingest.add_argument("path", help="Path to file or directory")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--no-verify", action="store_true",
                         help="Skip NLI hallucination verification")
    p_query.add_argument("--verbose", action="store_true",
                         help="Show expanded queries and retrieval details")

    # interactive
    sub.add_parser("interactive", help="Interactive question-answering mode")

    # stats
    sub.add_parser("stats", help="Show system statistics")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    rag = RAGSystem()

    if args.command == "ingest":
        cmd_ingest(args, rag)
    elif args.command == "query":
        cmd_query(args, rag)
    elif args.command == "interactive":
        cmd_interactive(rag)
    elif args.command == "stats":
        cmd_stats(rag)


if __name__ == "__main__":
    main()