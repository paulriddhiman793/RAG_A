from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from utils import intent as _intent


@dataclass
class RouteDecision:
    mode: str
    reason: str
    suggested_tool: str
    suggested_input: dict[str, Any]
    complexity: str = "simple"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryRouter:
    def decide(self, query: str) -> RouteDecision:
        q = (query or "").strip()
        lowered = q.lower()
        suggested_tool, suggested_input = self._suggest_tool(lowered, q)
        complex_task = self._is_complex_task(lowered)
        mixed_intents = self._intent_count(lowered) >= 2

        if complex_task or mixed_intents:
            reason_parts = []
            if complex_task:
                reason_parts.append("query appears multi-step or comparative")
            if mixed_intents:
                reason_parts.append("query mixes multiple intent types")
            return RouteDecision(
                mode="agent",
                reason=", ".join(reason_parts) or "complex query",
                suggested_tool=suggested_tool,
                suggested_input=suggested_input,
                complexity="complex",
            )

        if self._has_greeting_intent(lowered):
            return RouteDecision(
                mode="chat",
                reason="generic greeting or chit-chat detected",
                suggested_tool="chat",
                suggested_input={"question": q},
                complexity="trivial",
            )

        if suggested_tool in {
            "summarize_document",
            "explain_figure",
            "compare_figures",
            "lookup_table",
            "compare_tables",
            "lookup_formula",
            "lookup_metric",
        }:
            return RouteDecision(
                mode="direct",
                reason=f"single-intent query matched specialized capability: {suggested_tool}",
                suggested_tool=suggested_tool,
                suggested_input=suggested_input,
                complexity="simple",
            )

        return RouteDecision(
            mode="direct",
            reason="defaulting to direct RAG for a simple question",
            suggested_tool=suggested_tool or "answer_query",
            suggested_input=suggested_input or {"question": q},
            complexity="simple",
        )

    def _suggest_tool(self, lowered: str, original: str) -> tuple[str, dict[str, Any]]:
        figure_nums = self._extract_numbers_for_label(lowered, "fig(?:ure)?")
        table_nums = self._extract_numbers_for_label(lowered, "table")

        if self._has_summary_intent(lowered):
            return "summarize_document", {}
        if len(figure_nums) >= 2 and self._has_compare_intent(lowered):
            return "compare_figures", {
                "left_figure": figure_nums[0],
                "right_figure": figure_nums[1],
            }
        if len(table_nums) >= 2 and self._has_compare_intent(lowered):
            return "compare_tables", {
                "left_table": table_nums[0],
                "right_table": table_nums[1],
            }
        if len(figure_nums) >= 1:
            return "explain_figure", {"figure_number": figure_nums[0]}
        if len(table_nums) >= 1:
            return "lookup_table", {"table_number": table_nums[0]}
        if self._has_formula_intent(lowered):
            return "lookup_formula", {"question": original}
        if self._has_metric_intent(lowered):
            metric_name = self._extract_metric_name(lowered)
            model_name = self._extract_model_name(original)
            if metric_name and model_name:
                return "lookup_metric", {
                    "model_name": model_name,
                    "metric_name": metric_name,
                }
        return "answer_query", {"question": original}

    def _is_complex_task(self, lowered: str) -> bool:
        multi_step_markers = [
            " then ",
            " and then ",
            " after that ",
            " finally ",
            " first ",
            " next ",
            " also ",
            " along with ",
            " as well as ",
        ]
        if any(marker in lowered for marker in multi_step_markers):
            return True
        if self._has_compare_intent(lowered) and any(term in lowered for term in ["summary", "conclusion", "graph", "figure", "table", "metric"]):
            return True
        return False

    def _intent_count(self, lowered: str) -> int:
        intents = 0
        intents += int(self._has_summary_intent(lowered))
        intents += int(self._has_figure_intent(lowered))
        intents += int(self._has_table_intent(lowered))
        intents += int(self._has_formula_intent(lowered))
        intents += int(self._has_metric_intent(lowered))
        return intents

    @staticmethod
    def _has_compare_intent(lowered: str) -> bool:
        return any(term in lowered for term in [
            "compare", "difference", "different", "same", "vs", "versus",
        ])

    @staticmethod
    def _has_greeting_intent(lowered: str) -> bool:
        return _intent.has_greeting_intent(lowered)

    @staticmethod
    def _has_summary_intent(lowered: str) -> bool:
        return _intent.has_summary_intent(lowered)

    @staticmethod
    def _has_figure_intent(lowered: str) -> bool:
        return _intent.has_figure_intent(lowered)

    @staticmethod
    def _has_table_intent(lowered: str) -> bool:
        return _intent.has_table_intent(lowered)

    @staticmethod
    def _has_formula_intent(lowered: str) -> bool:
        return _intent.has_formula_intent(lowered)

    @staticmethod
    def _has_metric_intent(lowered: str) -> bool:
        return _intent.has_table_intent(lowered)  # metric terms overlap with table intent

    @staticmethod
    def _extract_numbers_for_label(lowered: str, label_pattern: str) -> list[int]:
        results: list[int] = []
        pattern = re.compile(
            rf"\b{label_pattern}s?\.?\s*(\d+)(?:\s*(?:,|and|&|vs\.?|versus)\s*(\d+))?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(lowered):
            for idx in (1, 2):
                raw = match.group(idx)
                if not raw:
                    continue
                num = int(raw)
                if num not in results:
                    results.append(num)
        return results

    @staticmethod
    def _extract_metric_name(lowered: str) -> str:
        for metric in ["R2", "R-squared", "RMSE", "MAE", "MSE"]:
            if metric.lower() in lowered or metric.replace("-", " ").lower() in lowered:
                return metric
        return "score"

    @staticmethod
    def _extract_model_name(original: str) -> str:
        known_models = [
            "Random Forest Regressor",
            "Random Forest",
            "RandomForest",
            "XGBoost",
            "Ridge Regression",
            "Lasso Regression",
            "Multiple Linear Regression",
            "Linear Regression",
            "Stacking",
            "Ensemble",
            "Polynomial Regression",
        ]
        lowered = original.lower()
        for model in known_models:
            if model.lower() in lowered:
                return model
        return original
