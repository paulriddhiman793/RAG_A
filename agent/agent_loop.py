from __future__ import annotations

from typing import Any

from config import settings
from generation.llm_client import LLMClient
from utils.logger import logger

from .memory import AgentMemoryStore
from .planner import Planner
from .tool_registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        memory_store: AgentMemoryStore | None = None,
    ):
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.memory_store = memory_store or AgentMemoryStore()
        self.planner = Planner(llm_client)

    def run(
        self,
        user_query: str,
        session_id: str = "default",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        max_steps = max_steps or settings.AGENT_MAX_STEPS
        self.memory_store.add_user_message(session_id, user_query)

        steps: list[dict[str, Any]] = []
        last_tool_result: dict[str, Any] | None = None

        for _ in range(max_steps):
            session_context = self.memory_store.build_planner_context(session_id)
            decision = self.planner.plan_next_action(
                user_goal=user_query,
                tool_descriptions=self.tool_registry.describe_tools(),
                session_context=session_context,
                prior_steps=steps,
                max_steps=max_steps,
            )

            if decision.done:
                final_answer = decision.final_answer.strip()
                if not final_answer and last_tool_result:
                    final_answer = str(
                        last_tool_result.get("answer")
                        or last_tool_result.get("observation")
                        or ""
                    ).strip()
                if not final_answer:
                    final_answer = (
                        "I could not complete the task confidently with the current agent plan."
                    )
                self.memory_store.add_assistant_message(session_id, final_answer)
                return {
                    "answer": final_answer,
                    "session_id": session_id,
                    "steps": steps,
                    "halt_reason": "completed",
                    "final_tool_result": last_tool_result or {},
                }

            tool_result = self.tool_registry.run_tool(decision.tool_name, decision.tool_input)
            last_tool_result = tool_result
            step_record = {
                "step": len(steps) + 1,
                "thought": decision.thought,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
                "observation": tool_result.get("observation", ""),
                "status": tool_result.get("status", "success"),
            }
            steps.append(step_record)
            self.memory_store.add_tool_step(
                session_id=session_id,
                thought=decision.thought,
                tool_name=decision.tool_name,
                tool_input=decision.tool_input,
                observation=str(tool_result.get("observation", "")),
            )

        logger.info("Agent loop hit max steps; returning best available result")
        fallback_answer = "I reached the agent step limit before fully finishing."
        if last_tool_result:
            fallback_answer = str(
                last_tool_result.get("answer")
                or last_tool_result.get("observation")
                or fallback_answer
            )
        self.memory_store.add_assistant_message(session_id, fallback_answer)
        return {
            "answer": fallback_answer,
            "session_id": session_id,
            "steps": steps,
            "halt_reason": "max_steps",
            "final_tool_result": last_tool_result or {},
        }
