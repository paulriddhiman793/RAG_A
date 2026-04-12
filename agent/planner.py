from __future__ import annotations

from dataclasses import dataclass

from generation.llm_client import LLMClient
from utils.logger import logger


@dataclass
class PlannerDecision:
    thought: str
    tool_name: str
    tool_input: dict
    done: bool
    final_answer: str = ""


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def plan_next_action(
        self,
        user_goal: str,
        tool_descriptions: str,
        session_context: str,
        prior_steps: list[dict],
        max_steps: int,
    ) -> PlannerDecision:
        step_count = len(prior_steps)
        prompt = (
            "You are an orchestration planner for a document QA agent.\n"
            "Decide the next best tool to call, or finish if enough information is already available.\n\n"
            "Rules:\n"
            "1. Use only the listed tools.\n"
            "2. Prefer specialized tools for figures, tables, formulas, metrics, and summaries.\n"
            "3. Use search_docs when uncertain what evidence exists.\n"
            "4. Use answer_query when you are ready to produce the final grounded answer.\n"
            "5. If a tool observation already contains a good grounded answer, you may finish and reuse it.\n"
            "6. Keep tool_input as a JSON object.\n"
            "7. If you finish, set done=true and provide final_answer.\n\n"
            f"User goal:\n{user_goal}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Session context:\n{session_context}\n\n"
            f"Prior steps taken: {step_count} / {max_steps}\n"
            f"Prior step records:\n{prior_steps}\n\n"
            "Respond as JSON with keys:\n"
            '{"thought": string, "tool_name": string, "tool_input": object, "done": boolean, "final_answer": string}'
        )
        try:
            raw = self.llm_client.complete_json(prompt, max_tokens=600)
        except Exception as e:
            logger.warning(f"Planner failed, using fallback action: {e}")
            raw = self._fallback_plan(user_goal, prior_steps)

        decision = PlannerDecision(
            thought=str(raw.get("thought", "")).strip() or "Use the general QA tool.",
            tool_name=str(raw.get("tool_name", "answer_query")).strip() or "answer_query",
            tool_input=raw.get("tool_input", {}) if isinstance(raw.get("tool_input", {}), dict) else {},
            done=bool(raw.get("done", False)),
            final_answer=str(raw.get("final_answer", "")).strip(),
        )
        if not decision.done and decision.tool_name == "answer_query" and "question" not in decision.tool_input:
            decision.tool_input = {"question": user_goal}
        return decision

    @staticmethod
    def _fallback_plan(user_goal: str, prior_steps: list[dict]) -> dict:
        q = (user_goal or "").lower()
        if prior_steps:
            last_observation = str(prior_steps[-1].get("observation", "")).strip()
            if last_observation:
                return {
                    "thought": "The last tool already produced the best available answer, so finish with it.",
                    "tool_name": "",
                    "tool_input": {},
                    "done": True,
                    "final_answer": last_observation,
                }

        if any(term in q for term in ["summary", "summarize", "summarise", "overview", "what is the pdf about"]):
            return {
                "thought": "Use the dedicated document summary tool.",
                "tool_name": "summarize_document",
                "tool_input": {},
                "done": False,
                "final_answer": "",
            }

        return {
            "thought": "Use the general QA tool.",
            "tool_name": "answer_query",
            "tool_input": {"question": user_goal},
            "done": False,
            "final_answer": "",
        }
