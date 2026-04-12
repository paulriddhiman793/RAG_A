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
    if args.direct and args.agent:
        console.print("[red]Choose only one of --direct or --agent.[/red]")
        sys.exit(1)
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Retrieving and generating answer..."):
        if args.direct:
            result = rag.query(question, verify=not args.no_verify, verbose=args.verbose)
            result["mode"] = "direct"
            result["route_reason"] = "forced direct mode by CLI flag"
        elif args.agent:
            result = rag.agent_query(
                question,
                session_id=args.session,
                max_steps=args.max_steps,
            )
            result["mode"] = "agent"
            result["route_reason"] = "forced agent mode by CLI flag"
        else:
            result = rag.auto_query(
                question,
                session_id=args.session,
                max_steps=args.max_steps,
                verify=not args.no_verify,
                verbose=args.verbose,
            )

    console.print(
        f"[dim]Mode: {result.get('mode', 'direct')} | "
        f"Reason: {result.get('route_reason', '-')}"
        f"[/dim]\n"
    )

    if result.get("mode") == "agent":
        answer_panel = Panel(
            Markdown(result["answer"]),
            title="[bold cyan]Agent Answer",
            border_style="cyan",
        )
        console.print(answer_panel)

        if args.show_trace and result.get("steps"):
            t = Table(title=f"Agent Trace | session={result['session_id']}", show_header=True)
            t.add_column("Step", width=4)
            t.add_column("Tool")
            t.add_column("Thought")
            t.add_column("Observation")
            for step in result["steps"]:
                t.add_row(
                    str(step.get("step", "")),
                    step.get("tool_name", ""),
                    str(step.get("thought", ""))[:80],
                    str(step.get("observation", ""))[:120],
                )
            console.print(t)

        console.print(
            f"\n[dim]Session: {result['session_id']} | "
            f"Steps: {len(result.get('steps', []))} | "
            f"Halt reason: {result.get('halt_reason', 'unknown')}[/dim]"
        )
        return

    _render_direct_result(result)


def cmd_agent_query(args, rag: RAGSystem) -> None:
    question = args.question
    console.print(f"\n[bold]Agent Question:[/bold] {question}\n")

    with console.status("[bold cyan]Planning, using tools, and answering..."):
        result = rag.agent_query(
            question,
            session_id=args.session,
            max_steps=args.max_steps,
        )

    answer_panel = Panel(
        Markdown(result["answer"]),
        title="[bold cyan]Agent Answer",
        border_style="cyan",
    )
    console.print(answer_panel)

    if args.show_trace and result.get("steps"):
        t = Table(title=f"Agent Trace | session={result['session_id']}", show_header=True)
        t.add_column("Step", width=4)
        t.add_column("Tool")
        t.add_column("Thought")
        t.add_column("Observation")
        for step in result["steps"]:
            t.add_row(
                str(step.get("step", "")),
                step.get("tool_name", ""),
                str(step.get("thought", ""))[:80],
                str(step.get("observation", ""))[:120],
            )
        console.print(t)

    console.print(
        f"\n[dim]Session: {result['session_id']} | "
        f"Steps: {len(result.get('steps', []))} | "
        f"Halt reason: {result.get('halt_reason', 'unknown')}[/dim]"
    )


def _render_direct_result(result: dict) -> None:
    answer_panel = Panel(
        Markdown(result["answer"]),
        title="[bold green]Answer",
        border_style="green" if result["is_grounded"] else "yellow",
    )
    console.print(answer_panel)

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
            question=question,
            no_verify=False,
            verbose=False,
            direct=False,
            agent=False,
            session="default",
            max_steps=None,
            show_trace=False,
        )
        cmd_query(args_mock, rag)


def cmd_agent_interactive(args, rag: RAGSystem) -> None:
    console.print(Panel(
        "[bold cyan]Advanced RAG Agent - Interactive Mode[/bold cyan]\n"
        "Type your question and press Enter. Type 'exit' to quit.",
        border_style="cyan",
    ))
    while True:
        try:
            question = input("\nAgent> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        args_mock = argparse.Namespace(
            question=question,
            session=args.session,
            max_steps=args.max_steps,
            show_trace=args.show_trace,
        )
        cmd_agent_query(args_mock, rag)


def cmd_stats(rag: RAGSystem) -> None:
    stats = rag.stats()
    t = Table(title="RAG System Stats")
    t.add_column("Setting")
    t.add_column("Value")
    for k, v in stats.items():
        t.add_row(k, str(v))
    console.print(t)


def cmd_tools(rag: RAGSystem) -> None:
    tools = rag.list_tools()
    t = Table(title="Available Agent Tools")
    t.add_column("Tool")
    t.add_column("Description")
    t.add_column("Input Schema")
    for tool in tools:
        t.add_row(
            tool["name"],
            tool["description"],
            json.dumps(tool["input_schema"]),
        )
    console.print(t)


def cmd_route(args, rag: RAGSystem) -> None:
    result = rag.route_query(args.question)
    panel = Panel(
        Markdown(
            f"Mode: `{result['mode']}`\n\n"
            f"Reason: {result['reason']}\n\n"
            f"Suggested tool: `{result['suggested_tool']}`\n\n"
            f"Suggested input: `{json.dumps(result['suggested_input'])}`\n\n"
            f"Complexity: `{result['complexity']}`"
        ),
        title="[bold blue]Route Decision",
        border_style="blue",
    )
    console.print(panel)


def cmd_tool(args, rag: RAGSystem) -> None:
    try:
        payload = json.loads(args.input) if args.input else {}
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON for --input:[/red] {e}")
        sys.exit(1)

    result = rag.run_tool(args.name, payload)
    panel = Panel(
        Markdown(str(result.get("observation", result))),
        title=f"[bold magenta]Tool Result: {args.name}",
        border_style="magenta",
    )
    console.print(panel)
    if args.show_json:
        console.print_json(json.dumps(result))


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
    p_query.add_argument("--direct", action="store_true",
                         help="Force direct RAG mode")
    p_query.add_argument("--agent", action="store_true",
                         help="Force agent mode")
    p_query.add_argument("--session", default="default",
                         help="Session ID when agent mode is used")
    p_query.add_argument("--max-steps", type=int, default=None,
                         help="Maximum planner/tool steps when agent mode is used")
    p_query.add_argument("--show-trace", action="store_true",
                         help="Show tool trace when agent mode is used")

    # agent-query
    p_agent_query = sub.add_parser("agent-query", help="Ask a question via the agent loop")
    p_agent_query.add_argument("question", help="Question to ask")
    p_agent_query.add_argument("--session", default="default", help="Session ID for agent memory")
    p_agent_query.add_argument("--max-steps", type=int, default=None,
                               help="Maximum planning/execution steps")
    p_agent_query.add_argument("--show-trace", action="store_true",
                               help="Show tool-by-tool reasoning trace")

    # interactive
    sub.add_parser("interactive", help="Interactive question-answering mode")

    # agent interactive
    p_agent_interactive = sub.add_parser("agent-interactive", help="Interactive agent mode")
    p_agent_interactive.add_argument("--session", default="default",
                                     help="Session ID for agent memory")
    p_agent_interactive.add_argument("--max-steps", type=int, default=None,
                                     help="Maximum planning/execution steps")
    p_agent_interactive.add_argument("--show-trace", action="store_true",
                                     help="Show tool-by-tool reasoning trace")

    # stats
    sub.add_parser("stats", help="Show system statistics")

    # tools
    sub.add_parser("tools", help="List available agent tools")

    # route
    p_route = sub.add_parser("route", help="Inspect automatic routing for a question")
    p_route.add_argument("question", help="Question to inspect")

    # tool execution
    p_tool = sub.add_parser("tool", help="Execute one explicit agent tool")
    p_tool.add_argument("name", help="Tool name")
    p_tool.add_argument("--input", default="{}", help="JSON object for tool input")
    p_tool.add_argument("--show-json", action="store_true", help="Print full JSON result")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    rag = RAGSystem()

    if args.command == "ingest":
        cmd_ingest(args, rag)
    elif args.command == "query":
        cmd_query(args, rag)
    elif args.command == "agent-query":
        cmd_agent_query(args, rag)
    elif args.command == "interactive":
        cmd_interactive(rag)
    elif args.command == "agent-interactive":
        cmd_agent_interactive(args, rag)
    elif args.command == "stats":
        cmd_stats(rag)
    elif args.command == "tools":
        cmd_tools(rag)
    elif args.command == "route":
        cmd_route(args, rag)
    elif args.command == "tool":
        cmd_tool(args, rag)


if __name__ == "__main__":
    main()
