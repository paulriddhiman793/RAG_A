"""
Tests for context compression (retrieval/context_compressor.py).
"""
import pytest

from retrieval.context_compressor import (
    _total_tokens,
    _smart_select,
    build_context_string,
)


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "a",
            "text": "This is a short chunk about neural networks.",
            "metadata": {
                "element_type": "NarrativeText",
                "filename": "paper.pdf",
                "source_file": "paper.pdf",
                "page_number": 1,
                "section": "Introduction",
                "reading_order": 0,
            },
            "score": 0.9,
        },
        {
            "chunk_id": "b",
            "text": "Random forest models achieve good accuracy on tabular data.",
            "metadata": {
                "element_type": "NarrativeText",
                "filename": "paper.pdf",
                "source_file": "paper.pdf",
                "page_number": 2,
                "section": "Methods",
                "reading_order": 1,
            },
            "score": 0.7,
        },
        {
            "chunk_id": "c",
            "text": "The results show significant improvement.",
            "metadata": {
                "element_type": "NarrativeText",
                "filename": "paper.pdf",
                "source_file": "paper.pdf",
                "page_number": 3,
                "section": "Results",
                "reading_order": 2,
            },
            "score": 0.5,
        },
    ]


class TestTotalTokens:
    def test_counts_tokens(self, sample_chunks):
        total = _total_tokens(sample_chunks)
        assert total > 0
        assert isinstance(total, int)

    def test_empty_list(self):
        assert _total_tokens([]) == 0


class TestSmartSelect:
    def test_selects_highest_score_first(self, sample_chunks):
        selected = _smart_select(sample_chunks, max_tokens=20)
        # Should pick the highest-scoring chunk(s) that fit
        assert len(selected) >= 1
        ids = {c["chunk_id"] for c in selected}
        assert "a" in ids  # highest score

    def test_respects_budget(self, sample_chunks):
        # Very tight budget — should not include all chunks
        selected = _smart_select(sample_chunks, max_tokens=15)
        assert len(selected) < len(sample_chunks)

    def test_preserves_reading_order(self, sample_chunks):
        selected = _smart_select(sample_chunks, max_tokens=10000)
        orders = [c["metadata"]["reading_order"] for c in selected]
        assert orders == sorted(orders)


class TestBuildContextString:
    def test_produces_source_headers(self, sample_chunks):
        ctx = build_context_string(sample_chunks)
        assert "[Source 1 |" in ctx
        assert "[Source 2 |" in ctx
        assert "paper.pdf" in ctx

    def test_separates_chunks(self, sample_chunks):
        ctx = build_context_string(sample_chunks)
        assert "---" in ctx

    def test_includes_section(self, sample_chunks):
        ctx = build_context_string(sample_chunks)
        assert "Introduction" in ctx

    def test_formula_reconstruction(self):
        formula = [{
            "chunk_id": "f1",
            "text": "Energy equals mass times speed of light squared",
            "metadata": {
                "element_type": "Formula",
                "filename": "physics.pdf",
                "page_number": 5,
                "section": "Theory",
                "latex": "E = mc^2",
                "raw_formula_text": "",
            },
            "score": 0.8,
        }]
        ctx = build_context_string(formula)
        assert "LaTeX equation: $E = mc^2$" in ctx
        assert "type:Formula" in ctx

    def test_table_reconstruction(self):
        table = [{
            "chunk_id": "t1",
            "text": "Model comparison table",
            "metadata": {
                "element_type": "Table",
                "filename": "results.pdf",
                "page_number": 10,
                "section": "Results",
                "caption": "Table 1: Model comparison",
                "table_json": {"headers": ["Model", "R2"], "rows": [{"Model": "RF", "R2": "0.95"}]},
            },
            "score": 0.85,
        }]
        ctx = build_context_string(table)
        assert "Caption: Table 1: Model comparison" in ctx
        assert "Headers:" in ctx

    def test_empty_chunks(self):
        assert build_context_string([]) == ""
