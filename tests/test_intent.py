"""
Tests for the shared intent detection utilities (utils/intent.py).
"""
import pytest

from utils.intent import (
    has_formula_intent,
    has_figure_intent,
    has_table_intent,
    has_summary_intent,
    has_metric_lookup_intent,
    has_compare_intent,
    looks_table_like_visual,
    looks_table_like_image,
)


# ── has_formula_intent ────────────────────────────────────────────────

class TestHasFormulaIntent:
    @pytest.mark.parametrize("query", [
        "What equations are used?",
        "Show me the loss function",
        "List all FORMULAS in the paper",
        "What is the LaTeX for the main equation?",
    ])
    def test_positive(self, query):
        assert has_formula_intent(query) is True

    @pytest.mark.parametrize("query", [
        "What is the main conclusion?",
        "How many pages does it have?",
        "Describe the methodology",
    ])
    def test_negative(self, query):
        assert has_formula_intent(query) is False

    def test_empty_and_none(self):
        assert has_formula_intent("") is False
        assert has_formula_intent(None) is False


# ── has_figure_intent ─────────────────────────────────────────────────

class TestHasFigureIntent:
    @pytest.mark.parametrize("query", [
        "How many figures are in the paper?",
        "Describe figure 3",
        "Show me the heatmap",
        "What does the diagram show?",
        "List all plots",
    ])
    def test_positive(self, query):
        assert has_figure_intent(query) is True

    @pytest.mark.parametrize("query", [
        "What is the abstract?",
        "Summarize the paper",
    ])
    def test_negative(self, query):
        assert has_figure_intent(query) is False


# ── has_table_intent ──────────────────────────────────────────────────

class TestHasTableIntent:
    @pytest.mark.parametrize("query", [
        "What does table 1 show?",
        "What are the RMSE values?",
        "Show me the metrics",
        "What is the R2 score?",
    ])
    def test_positive(self, query):
        assert has_table_intent(query) is True

    @pytest.mark.parametrize("query", [
        "What is the methodology?",
        "Explain the introduction",
    ])
    def test_negative(self, query):
        assert has_table_intent(query) is False


# ── has_summary_intent ────────────────────────────────────────────────

class TestHasSummaryIntent:
    @pytest.mark.parametrize("query", [
        "Give me a summary",
        "What is this PDF about?",
        "Provide an overview",
        "Summarize the document",
    ])
    def test_positive(self, query):
        assert has_summary_intent(query) is True

    @pytest.mark.parametrize("query", [
        "What is equation 3?",
        "Show figure 2",
    ])
    def test_negative(self, query):
        assert has_summary_intent(query) is False


# ── has_metric_lookup_intent ──────────────────────────────────────────

class TestHasMetricLookupIntent:
    @pytest.mark.parametrize("query", [
        "What is the R2 of the random forest model?",
        "Show RMSE for the ridge regressor",
        "What is the MAE of the XGBoost model?",
    ])
    def test_positive(self, query):
        assert has_metric_lookup_intent(query) is True

    @pytest.mark.parametrize("query", [
        "What is the R2?",             # metric but no model
        "Random forest results",       # model but no metric
        "What is the abstract?",
    ])
    def test_negative(self, query):
        assert has_metric_lookup_intent(query) is False


# ── has_compare_intent ────────────────────────────────────────────────

class TestHasCompareIntent:
    @pytest.mark.parametrize("query", [
        "Compare method A vs method B",
        "What is the difference between the two?",
        "Are results the same?",
    ])
    def test_positive(self, query):
        assert has_compare_intent(query) is True


# ── looks_table_like_visual / looks_table_like_image ──────────────────

class TestLooksTableLikeVisual:
    def test_table_prefix(self):
        assert looks_table_like_visual("Table 1: Results", "", "") is True

    def test_table_colon(self):
        assert looks_table_like_visual("table: comparison", "", "") is True

    def test_table_2_in_early_text(self):
        assert looks_table_like_visual("", "", "table 2 shows accuracy") is True

    def test_regular_figure(self):
        assert looks_table_like_visual("Architecture diagram", "Model overview", "") is False


class TestLooksTableLikeImage:
    def test_with_chunk_dict(self):
        chunk = {
            "text": "some figure",
            "metadata": {
                "caption": "Table 1: Final results",
                "alt_text": "",
            },
        }
        assert looks_table_like_image(chunk) is True

    def test_normal_image_chunk(self):
        chunk = {
            "text": "Neural network architecture",
            "metadata": {
                "caption": "Model architecture",
                "alt_text": "Overview of the proposed model",
            },
        }
        assert looks_table_like_image(chunk) is False
