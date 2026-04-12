from __future__ import annotations

import re
from typing import Any

from config import settings
from generation.answer_generator import generate_answer
from generation.llm_client import LLMClient
from indexing.bm25_index import BM25Index
from indexing.embeddings import detect_domain
from indexing.vector_store import VectorStore
from pipeline.query_pipeline import (
    _ensure_figure_context,
    _ensure_formula_context,
    _ensure_metric_context,
    _ensure_summary_context,
    _has_figure_intent,
    _has_formula_intent,
    _has_metric_lookup_intent,
    _has_summary_intent,
    query as run_query_pipeline,
)
from retrieval.context_compressor import compress_context
from retrieval.hybrid_retriever import retrieve
from retrieval.query_expander import expand_query
from utils.logger import logger


class ToolRegistry:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        llm_client: LLMClient,
        domain: str | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.llm_client = llm_client
        self.domain = domain
        self._tools: dict[str, dict[str, Any]] = {
            "search_docs": {
                "description": "Retrieve the most relevant chunks for a question without final synthesis.",
                "input_schema": {"query": "string", "top_k": "integer optional"},
                "handler": self.search_docs,
            },
            "answer_query": {
                "description": "Produce a grounded final answer with citations for a user question.",
                "input_schema": {"question": "string"},
                "handler": self.answer_query,
            },
            "summarize_document": {
                "description": "Generate a concise summary of the current indexed document set.",
                "input_schema": {},
                "handler": self.summarize_document,
            },
            "explain_figure": {
                "description": "Explain a specific figure number from the indexed documents.",
                "input_schema": {"figure_number": "integer"},
                "handler": self.explain_figure,
            },
            "compare_figures": {
                "description": "Compare two figure numbers from the indexed documents.",
                "input_schema": {"left_figure": "integer", "right_figure": "integer"},
                "handler": self.compare_figures,
            },
            "lookup_table": {
                "description": "Explain a specific table and its contents.",
                "input_schema": {"table_number": "integer"},
                "handler": self.lookup_table,
            },
            "compare_tables": {
                "description": "Compare two tables from the indexed documents.",
                "input_schema": {"left_table": "integer", "right_table": "integer"},
                "handler": self.compare_tables,
            },
            "lookup_formula": {
                "description": "Find or explain a formula, equation, or loss function.",
                "input_schema": {"question": "string optional", "formula_hint": "string optional"},
                "handler": self.lookup_formula,
            },
            "lookup_metric": {
                "description": "Find a model metric such as R2, RMSE, MAE, or MSE.",
                "input_schema": {"model_name": "string", "metric_name": "string"},
                "handler": self.lookup_metric,
            },
            "stats": {
                "description": "Return system stats about the current index and models.",
                "input_schema": {},
                "handler": self.stats,
            },
            "answer_from_retrieval": {
                "description": "Retrieve chunks first and then answer from that retrieved context.",
                "input_schema": {"question": "string"},
                "handler": self.answer_from_retrieval,
            },
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for name, spec in self._tools.items()
        ]

    def describe_tools(self) -> str:
        return "\n".join(
            f"- {tool['name']}: {tool['description']} | input={tool['input_schema']}"
            for tool in self.list_tools()
        )

    def run_tool(self, name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
        canonical_name, normalized_input = self._normalize_tool_input(name, tool_input or {})
        spec = self._tools.get(canonical_name)
        if not spec:
            return {
                "tool_name": canonical_name,
                "status": "error",
                "observation": f"Unknown tool: {canonical_name}",
            }
        try:
            result = spec["handler"](**normalized_input)
            if "tool_name" not in result:
                result["tool_name"] = canonical_name
            if "status" not in result:
                result["status"] = "success"
            if "observation" not in result:
                result["observation"] = self._default_observation(result)
            return result
        except TypeError as e:
            return {
                "tool_name": canonical_name,
                "status": "error",
                "observation": f"Invalid tool input for {canonical_name}: {e}",
            }
        except Exception as e:
            logger.error(f"Tool {canonical_name} failed: {e}")
            return {
                "tool_name": canonical_name,
                "status": "error",
                "observation": f"Tool {canonical_name} failed: {e}",
            }

    def search_docs(self, query: str, top_k: int = 5) -> dict[str, Any]:
        chunks, top_score, expanded_queries = self._retrieve_chunks(query)
        limited = chunks[: max(1, int(top_k))]
        hits = []
        for idx, chunk in enumerate(limited, start=1):
            meta = chunk.get("metadata", {})
            hits.append({
                "rank": idx,
                "file": meta.get("filename", "unknown"),
                "page": meta.get("page_number", "?"),
                "section": meta.get("section", ""),
                "type": meta.get("element_type", "text"),
                "score": round(float(chunk.get("score", 0.0)), 3),
                "snippet": (chunk.get("text", "") or "")[:260],
            })
        observation = (
            f"Retrieved {len(limited)} chunk(s) with top score {top_score:.3f}. "
            + " | ".join(
                f"#{hit['rank']} {hit['type']} p{hit['page']} score={hit['score']}: {hit['snippet']}"
                for hit in hits
            )
        )
        return {
            "tool_name": "search_docs",
            "status": "success",
            "observation": observation,
            "hits": hits,
            "top_score": top_score,
            "expanded_queries": expanded_queries,
        }

    def answer_query(self, question: str) -> dict[str, Any]:
        result = run_query_pipeline(
            user_query=question,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            llm_client=self.llm_client,
            domain=self.domain,
            verify_hallucinations=True,
            verbose=False,
        )
        return {
            "tool_name": "answer_query",
            "status": "success",
            "observation": result.get("answer", ""),
            "result": result,
            "answer": result.get("answer", ""),
            "sources_used": result.get("sources_used", []),
            "top_score": result.get("top_score", 0.0),
        }

    def summarize_document(self) -> dict[str, Any]:
        return self.answer_query(settings.TOOL_PROMPT_SUMMARIZE_DOCUMENT)

    def explain_figure(self, figure_number: int) -> dict[str, Any]:
        prompt = self._render_prompt(
            settings.TOOL_PROMPT_EXPLAIN_FIGURE,
            figure_number=int(figure_number),
        )
        return self.answer_query(prompt)

    def compare_figures(self, left_figure: int, right_figure: int) -> dict[str, Any]:
        prompt = self._render_prompt(
            settings.TOOL_PROMPT_COMPARE_FIGURES,
            left_figure=int(left_figure),
            right_figure=int(right_figure),
        )
        return self.answer_query(prompt)

    def lookup_table(self, table_number: int) -> dict[str, Any]:
        prompt = self._render_prompt(
            settings.TOOL_PROMPT_LOOKUP_TABLE,
            table_number=int(table_number),
        )
        return self.answer_query(prompt)

    def compare_tables(self, left_table: int, right_table: int) -> dict[str, Any]:
        prompt = self._render_prompt(
            settings.TOOL_PROMPT_COMPARE_TABLES,
            left_table=int(left_table),
            right_table=int(right_table),
        )
        return self.answer_query(prompt)

    def lookup_formula(
        self,
        question: str | None = None,
        formula_hint: str | None = None,
    ) -> dict[str, Any]:
        final_question = (question or "").strip()
        if not final_question and formula_hint:
            final_question = self._render_prompt(
                settings.TOOL_PROMPT_LOOKUP_FORMULA,
                formula_hint=formula_hint,
            )
        if not final_question:
            final_question = settings.TOOL_PROMPT_LOOKUP_FORMULA_ALL
        return self.answer_query(final_question)

    def lookup_metric(self, model_name: str, metric_name: str) -> dict[str, Any]:
        prompt = self._render_prompt(
            settings.TOOL_PROMPT_LOOKUP_METRIC,
            model_name=model_name,
            metric_name=metric_name,
        )
        return self.answer_query(prompt)

    def stats(self) -> dict[str, Any]:
        observation = (
            f"Child chunks: {self.vector_store.count()} | "
            f"Embedding model: {settings.DEFAULT_EMBEDDING_MODEL} | "
            f"Text model: {settings.LLM_MODEL} | Vision model: {settings.VISION_MODEL}"
        )
        return {
            "tool_name": "stats",
            "status": "success",
            "observation": observation,
            "stats": {
                "chunks_indexed": self.vector_store.count(),
                "embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
                "text_model": settings.LLM_MODEL,
                "vision_model": settings.VISION_MODEL,
            },
        }

    def answer_from_retrieval(self, question: str) -> dict[str, Any]:
        chunks, top_score, expanded_queries = self._retrieve_chunks(question)
        if not chunks:
            return {
                "tool_name": "answer_from_retrieval",
                "status": "success",
                "observation": "No relevant chunks were retrieved.",
                "answer": "No relevant chunks were retrieved.",
                "sources_used": [],
                "top_score": top_score,
                "expanded_queries": expanded_queries,
            }
        compressed = compress_context(chunks, question)
        result = generate_answer(question, compressed, self.llm_client, verify=True)
        return {
            "tool_name": "answer_from_retrieval",
            "status": "success",
            "observation": result.get("answer", ""),
            "answer": result.get("answer", ""),
            "sources_used": result.get("sources_used", []),
            "top_score": top_score,
            "expanded_queries": expanded_queries,
        }

    def _retrieve_chunks(self, question: str) -> tuple[list[dict[str, Any]], float, list[str]]:
        expanded = expand_query(question, self.llm_client)
        all_queries = expanded.all_queries()
        detected_domain = detect_domain([question]) if self.domain is None else self.domain
        domain = detected_domain if settings.AUTO_DETECT_EMBEDDING_DOMAIN else "general"
        hyde_queries = [expanded.hypothetical_answer] + all_queries
        chunks, top_score = retrieve(
            queries=hyde_queries,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            domain=domain,
            rerank_query=question,
        )
        if _has_formula_intent(question):
            chunks = _ensure_formula_context(chunks, self.vector_store)
            if chunks:
                top_score = max(top_score, chunks[0].get("score", top_score))
        if _has_figure_intent(question):
            chunks, top_score = _ensure_figure_context(
                chunks,
                self.vector_store,
                question,
                top_score,
            )
        if _has_metric_lookup_intent(question):
            chunks, top_score = _ensure_metric_context(
                chunks,
                self.vector_store,
                question,
                top_score,
            )
        if _has_summary_intent(question):
            chunks, top_score = _ensure_summary_context(
                chunks,
                self.vector_store,
                top_score,
            )
        return chunks, top_score, all_queries

    @staticmethod
    def _default_observation(result: dict[str, Any]) -> str:
        if "answer" in result:
            return str(result.get("answer", ""))
        if "hits" in result:
            return f"Retrieved {len(result.get('hits', []))} hits."
        if "stats" in result:
            return str(result.get("stats", {}))
        return "Tool completed."

    def _normalize_tool_input(
        self,
        name: str,
        tool_input: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        canonical_name = self._tool_alias(name)
        payload = dict(tool_input or {})

        if canonical_name in {"answer_query", "answer_from_retrieval", "search_docs", "lookup_formula"}:
            query_value = self._pick_first(payload, ["question", "query", "user_query", "prompt", "goal", "request"])
            if query_value:
                key = "query" if canonical_name == "search_docs" else "question"
                return canonical_name, {key: str(query_value)}

        if canonical_name == "lookup_metric":
            model_name = self._pick_first(payload, ["model_name", "model", "regressor", "algorithm"])
            metric_name = self._pick_first(payload, ["metric_name", "metric", "score_name"])
            normalized = {}
            if model_name:
                normalized["model_name"] = str(model_name)
            if metric_name:
                normalized["metric_name"] = str(metric_name)
            return canonical_name, normalized

        if canonical_name == "explain_figure":
            number = self._extract_first_number(payload, ["figure_number", "figure", "number", "id"])
            return canonical_name, {"figure_number": number} if number is not None else {}

        if canonical_name == "lookup_table":
            number = self._extract_first_number(payload, ["table_number", "table", "number", "id"])
            return canonical_name, {"table_number": number} if number is not None else {}

        if canonical_name == "compare_figures":
            left = self._extract_first_number(payload, ["left_figure", "left", "figure_a", "a"])
            right = self._extract_first_number(payload, ["right_figure", "right", "figure_b", "b"])
            return canonical_name, self._pair_payload("left_figure", left, "right_figure", right)

        if canonical_name == "compare_tables":
            left = self._extract_first_number(payload, ["left_table", "left", "table_a", "a"])
            right = self._extract_first_number(payload, ["right_table", "right", "table_b", "b"])
            return canonical_name, self._pair_payload("left_table", left, "right_table", right)

        return canonical_name, payload

    @staticmethod
    def _tool_alias(name: str) -> str:
        aliases = {
            "summary": "summarize_document",
            "summarise_document": "summarize_document",
            "summarize": "summarize_document",
            "figure": "explain_figure",
            "table": "lookup_table",
            "formula": "lookup_formula",
            "metric": "lookup_metric",
            "qa": "answer_query",
        }
        lowered = (name or "").strip().lower()
        return aliases.get(lowered, lowered)

    @staticmethod
    def _pick_first(payload: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _extract_first_number(cls, payload: dict[str, Any], keys: list[str]) -> int | None:
        value = cls._pick_first(payload, keys)
        if value is None:
            return None
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None

    @staticmethod
    def _pair_payload(left_key: str, left: int | None, right_key: str, right: int | None) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        if left is not None:
            normalized[left_key] = left
        if right is not None:
            normalized[right_key] = right
        return normalized

    @staticmethod
    def _render_prompt(template: str, **kwargs: Any) -> str:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
