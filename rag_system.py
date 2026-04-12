"""
RAGSystem — the single entry point for the entire pipeline.

Usage:
    rag = RAGSystem()
    rag.ingest("paper.pdf")
    result = rag.query("What is the Kozeny-Carman equation?")
    print(result["answer"])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent import AgentLoop, AgentMemoryStore, ToolRegistry
from config import settings
from generation.llm_client import LLMClient
from indexing.vector_store import VectorStore
from indexing.bm25_index import BM25Index
from pipeline.ingestion_pipeline import ingest_document, ingest_documents
from pipeline.query_pipeline import query as _query
from utils.logger import logger


class RAGSystem:
    """
    Fully wired advanced RAG system.

    Parameters
    ----------
    domain : str, optional
        Force a specific embedding domain ("general", "biomedical", "legal",
        "financial", "code"). Auto-detected per document if None.
    """

    def __init__(self, domain: str | None = None):
        os.makedirs("logs", exist_ok=True)
        self.domain = domain
        self.llm = LLMClient()
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self.memory_store = AgentMemoryStore()
        self.tool_registry = ToolRegistry(
            vector_store=self.vector_store,
            bm25_index=self.bm25,
            llm_client=self.llm,
            domain=self.domain,
        )
        self.agent = AgentLoop(
            tool_registry=self.tool_registry,
            llm_client=self.llm,
            memory_store=self.memory_store,
        )

        # Try to load cached BM25 index
        if not self.bm25.load():
            # Rebuild from existing vector store data
            all_chunks = self.vector_store.get_all_chunks()
            if all_chunks:
                logger.info(f"Rebuilding BM25 from {len(all_chunks)} existing chunks")
                self.bm25.build(all_chunks)

        logger.info(
            f"RAGSystem ready | "
            f"Chunks indexed: {self.vector_store.count()} | "
            f"LLM: groq/{settings.LLM_MODEL}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def ingest(self, file_path: str | Path) -> dict[str, Any]:
        """
        Ingest a single document into the RAG system.
        Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, images.
        """
        return ingest_document(
            file_path=file_path,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
            llm_client=self.llm,
            domain=self.domain,
        )

    def ingest_directory(
        self, directory: str | Path, glob: str = "**/*"
    ) -> list[dict[str, Any]]:
        """
        Ingest all supported documents in a directory.
        """
        supported = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".png", ".jpg"}
        dir_path = Path(directory)
        files = [f for f in dir_path.glob(glob) if f.suffix.lower() in supported]
        logger.info(f"Found {len(files)} documents in {directory}")
        return ingest_documents(
            file_paths=files,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
            llm_client=self.llm,
            domain=self.domain,
        )

    def query(
        self,
        question: str,
        verify: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Ask a question. Returns a dict with answer, citations, and metadata.

        Parameters
        ----------
        question : str
            The user's natural language question.
        verify   : bool
            Run NLI hallucination verification on the answer. Slightly slower.
        verbose  : bool
            Log expanded queries and retrieval details.
        """
        return _query(
            user_query=question,
            vector_store=self.vector_store,
            bm25_index=self.bm25,
            llm_client=self.llm,
            domain=self.domain,
            verify_hallucinations=verify,
            verbose=verbose,
        )

    def ask(self, question: str) -> str:
        """Convenience wrapper — returns just the answer string."""
        result = self.query(question)
        return result["answer"]

    def agent_query(
        self,
        question: str,
        session_id: str = "default",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run the agentic planner/tool loop for a question."""
        return self.agent.run(question, session_id=session_id, max_steps=max_steps)

    def run_tool(self, name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an explicit tool from the tool registry."""
        return self.tool_registry.run_tool(name, tool_input or {})

    def list_tools(self) -> list[dict[str, Any]]:
        """List explicit agent tools available in the current setup."""
        return self.tool_registry.list_tools()

    def stats(self) -> dict[str, Any]:
        """Return system statistics."""
        return {
            "chunks_indexed": self.vector_store.count(),
            "llm_provider": "groq",
            "llm_model": settings.LLM_MODEL,
            "vision_model": settings.VISION_MODEL,
            "embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
            "domain": self.domain or "auto",
            "agent_max_steps": settings.AGENT_MAX_STEPS,
            "tools_available": len(self.tool_registry.list_tools()),
        }
