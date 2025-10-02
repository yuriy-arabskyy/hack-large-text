"""Tests for FAISS retriever."""

import pytest
from hack.retriever import FaissRetriever


class TestFaissRetriever:
    """Tests for FaissRetriever class."""

    @pytest.fixture
    def retriever(self):
        """Create a FaissRetriever instance for testing."""
        return FaissRetriever(
            faiss_index_path="experiments/chess_pdf.faiss",
            workspace_json_path="experiments/workspace_with_embeddings.json",
            k=5
        )

    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes correctly."""
        assert retriever is not None
        assert retriever.index is not None
        assert len(retriever.blocks) > 0
        assert retriever.model is not None

    def test_search_text_returns_results(self, retriever):
        """Test that search_text returns results."""
        results = retriever.search_text("chess opening moves")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_format(self, retriever):
        """Test that search results have the expected format."""
        results = retriever.search_text("endgame strategy", k=1)
        assert len(results) == 1

        result = results[0]
        assert "unit_id" in result
        assert "content" in result
        assert "page" in result
        assert "section_path" in result
        assert "bbox" in result
        assert "type" in result
        assert "similarity" in result

    def test_search_respects_k_parameter(self, retriever):
        """Test that k parameter limits the number of results."""
        results = retriever.search_text("chess", k=3)
        assert len(results) <= 3

    def test_search_all_returns_results(self, retriever):
        """Test that search_all returns results without type filtering."""
        results = retriever.search_all("pawn structure", k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_similarity_scores_are_valid(self, retriever):
        """Test that similarity scores are between 0 and 1."""
        results = retriever.search_text("checkmate", k=5)
        for result in results:
            assert 0 <= result["similarity"] <= 1

    def test_search_tables(self, retriever):
        """Test search_tables method (may return empty for chess PDF)."""
        results = retriever.search_tables("table data")
        assert isinstance(results, list)

    def test_search_images(self, retriever):
        """Test search_images method (may return empty for chess PDF)."""
        results = retriever.search_images("diagram")
        assert isinstance(results, list)


class TestMockRetriever:
    """Tests for MockRetriever class."""

    def test_mock_retriever_exists(self):
        """Test that MockRetriever can be imported."""
        from hack.retriever import MockRetriever
        mock = MockRetriever()
        assert mock is not None

    def test_mock_search_text_returns_mock_data(self):
        """Test that mock retriever returns mock data."""
        from hack.retriever import MockRetriever
        mock = MockRetriever()
        results = mock.search_text("test query")
        assert len(results) > 0
        assert results[0]["unit_id"] == "mock_1"
