"""
Tests for the query router (agent/router.py).
"""
import pytest

from agent.router import QueryRouter


@pytest.fixture
def router():
    return QueryRouter()


class TestRouterDecision:
    def test_simple_query_goes_direct(self, router):
        decision = router.decide("What is the main finding?")
        assert decision.mode == "direct"

    def test_summary_query_goes_direct(self, router):
        decision = router.decide("Summarize the paper")
        assert decision.mode == "direct"

    def test_compound_query_goes_agent(self, router):
        decision = router.decide(
            "Compare the results of Model A and Model B, "
            "and also show me the equations they used"
        )
        assert decision.mode == "agent"

    def test_multi_intent_goes_agent(self, router):
        decision = router.decide(
            "Show me table 3 and also explain figure 2 and list all equations"
        )
        assert decision.mode == "agent"

    def test_empty_query_goes_direct(self, router):
        decision = router.decide("")
        assert decision.mode == "direct"


class TestRouterIntents:
    def test_formula_intent(self, router):
        decision = router.decide("What equations are used?")
        # The router should identify this as formula-related
        assert "formula" in decision.reason.lower() or "equation" in decision.reason.lower() or decision.suggested_tool != ""

    def test_figure_intent(self, router):
        decision = router.decide("Describe figure 1")
        assert "figure" in decision.reason.lower() or decision.suggested_tool != ""

    def test_table_intent(self, router):
        decision = router.decide("What does table 2 show?")
        assert "table" in decision.reason.lower() or decision.suggested_tool != ""

    def test_summary_intent(self, router):
        decision = router.decide("Give an overview of the paper")
        assert "summary" in decision.reason.lower() or "overview" in decision.reason.lower() or decision.mode == "direct"
