"""
example_usage.py — quick demonstration of the RAG system with Groq.

Before running:
  1. Get a free Groq API key at https://console.groq.com/keys
  2. Copy .env.example → .env and fill in:
       GROQ_API_KEY=your_key
       UNSTRUCTURED_API_KEY=your_key
  3. pip install -r requirements.txt
  4. Place any PDF / DOCX in the current directory
"""
from rag_system import RAGSystem
from rich.console import Console

console = Console()


def main():
    # ── 1. Initialise (uses LLM_MODEL from .env, default: llama-3.3-70b-versatile)
    console.print("[bold]Initialising RAG system (Groq backend)…[/bold]")
    rag = RAGSystem()

    # Print available Groq models
    models = rag.llm.list_models()
    console.print(f"[dim]Available Groq models: {', '.join(models[:6])}…[/dim]\n")
    console.print(f"[dim]Active model: {rag.llm.model} — {rag.llm.model_info}[/dim]\n")

    # ── 2. Ingest a document
    # result = rag.ingest("research_paper.pdf")
    # console.print(f"Ingested: {result}")
    console.print("[dim](Add your documents to ingest — skipping for demo)[/dim]\n")

    # ── 3. Query examples
    questions = [
        "What does the Kozeny-Carman equation describe?",
        "Which experimental condition had the highest hydraulic conductivity?",
        "What does Figure 1 show about porosity vs conductivity?",
        "How do baseline and high-porosity results compare?",
        "What is the GDP of France?",   # out-of-scope → no-answer path
    ]

    for q in questions:
        console.print(f"[bold cyan]Q: {q}[/bold cyan]")
        result = rag.query(q)

        answer_preview = result["answer"][:280]
        if len(result["answer"]) > 280:
            answer_preview += "…"
        console.print(f"[green]A:[/green] {answer_preview}")

        status = "[green]✓ grounded[/green]" if result["is_grounded"] else "[yellow]⚠ unverified[/yellow]"
        no_ans = " | [dim]no-answer[/dim]" if result["no_answer"] else ""
        console.print(
            f"  {status}{no_ans} | "
            f"score={result['top_score']:.3f} | "
            f"chunks={result['chunks_retrieved']} | "
            f"queries={len(result['expanded_queries'])}\n"
        )


if __name__ == "__main__":
    main()