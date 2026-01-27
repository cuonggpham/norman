"""
Unit tests for RAG Pipeline refactoring.

Tests shared BasePipeline methods and pipeline inheritance.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Test imports
from app.pipelines.base import BasePipeline, QueryTranslator
from app.pipelines.rag import RAGPipeline
from app.pipelines.graph_rag import GraphRAGPipeline


class TestBasePipelineInheritance:
    """Test that pipelines correctly inherit from BasePipeline."""
    
    def test_rag_pipeline_inherits_from_base(self):
        """RAGPipeline should inherit from BasePipeline."""
        assert issubclass(RAGPipeline, BasePipeline)
    
    def test_graph_rag_pipeline_inherits_from_base(self):
        """GraphRAGPipeline should inherit from BasePipeline."""
        assert issubclass(GraphRAGPipeline, BasePipeline)


class TestTranslateQuery:
    """Test _translate_query method."""
    
    def create_mock_pipeline(self, translator=None):
        """Create a minimal pipeline for testing."""
        return RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
            translator=translator,
        )
    
    def test_translate_query_without_translator(self):
        """Should return original query when no translator."""
        pipeline = self.create_mock_pipeline(translator=None)
        result = pipeline._translate_query("test query")
        assert result == "test query"
    
    def test_translate_query_with_translator(self):
        """Should return translated query when translator available."""
        translator = Mock()
        translator.translate.return_value = "翻訳されたクエリ"
        
        pipeline = self.create_mock_pipeline(translator=translator)
        result = pipeline._translate_query("test query")
        
        assert result == "翻訳されたクエリ"
        translator.translate.assert_called_once_with("test query")
    
    def test_translate_query_handles_exception(self):
        """Should return original query when translation fails."""
        translator = Mock()
        translator.translate.side_effect = Exception("Translation error")
        
        pipeline = self.create_mock_pipeline(translator=translator)
        result = pipeline._translate_query("test query")
        
        assert result == "test query"


class TestBuildContext:
    """Test _build_context method."""
    
    def create_mock_pipeline(self):
        return RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
        )
    
    def test_build_context_with_law_title(self):
        """Should format context with citation number and law title."""
        pipeline = self.create_mock_pipeline()
        results = [
            {
                "payload": {
                    "law_title": "労働基準法",
                    "article_title": "第32条",
                    "text": "使用者は、労働者に..."
                }
            }
        ]
        
        context = pipeline._build_context(results)
        
        assert len(context) == 1
        assert context[0].startswith("[1]【労働基準法 第32条】")
        assert "使用者は、労働者に..." in context[0]
    
    def test_build_context_without_law_title(self):
        """Should format context with just citation number when no title."""
        pipeline = self.create_mock_pipeline()
        results = [
            {"payload": {"text": "Some text content"}}
        ]
        
        context = pipeline._build_context(results)
        
        assert len(context) == 1
        assert context[0] == "[1] Some text content"
    
    def test_build_context_multiple_results(self):
        """Should number multiple results correctly."""
        pipeline = self.create_mock_pipeline()
        results = [
            {"payload": {"text": "First"}},
            {"payload": {"text": "Second"}},
            {"payload": {"text": "Third"}},
        ]
        
        context = pipeline._build_context(results)
        
        assert len(context) == 3
        assert context[0].startswith("[1]")
        assert context[1].startswith("[2]")
        assert context[2].startswith("[3]")


class TestToSourceDocument:
    """Test _to_source_document method."""
    
    def create_mock_pipeline(self):
        return RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
        )
    
    def test_to_source_document_full_fields(self):
        """Should extract all fields from result."""
        pipeline = self.create_mock_pipeline()
        result = {
            "score": 0.95,
            "payload": {
                "law_title": "労働基準法",
                "article_title": "第32条",
                "text": "A" * 600,  # Long text to test truncation
                "law_id": "law_123",
                "chapter_title": "第4章",
                "article_caption": "労働時間",
                "paragraph_num": "1",
                "highlight_path": {"law": "労働基準法"},
            }
        }
        
        source = pipeline._to_source_document(result)
        
        assert source.law_title == "労働基準法"
        assert source.article == "第32条"
        assert source.score == 0.95
        assert len(source.text) == 500  # Truncated
        assert source.law_id == "law_123"


class TestCanHybrid:
    """Test _can_hybrid method."""
    
    def test_can_hybrid_true(self):
        """Should return truthy when all hybrid components present."""
        pipeline = RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
            sparse_embedding=Mock(),
            hybrid_store=Mock(),
            use_hybrid_search=True,
        )
        
        # _can_hybrid returns truthy value when all components present
        assert pipeline._can_hybrid()
    
    def test_can_hybrid_false_no_sparse(self):
        """Should return falsy when sparse embedding missing."""
        pipeline = RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
            sparse_embedding=None,
            hybrid_store=Mock(),
            use_hybrid_search=True,
        )
        
        # Should be falsy
        assert not pipeline._can_hybrid()
    
    def test_can_hybrid_false_disabled(self):
        """Should return falsy when hybrid search disabled."""
        pipeline = RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
            sparse_embedding=Mock(),
            hybrid_store=Mock(),
            use_hybrid_search=False,
        )
        
        assert not pipeline._can_hybrid()


class TestFilterAndSortResults:
    """Test _filter_and_sort_results method."""
    
    def create_mock_pipeline(self, threshold=0.25):
        pipeline = RAGPipeline(
            embedding=Mock(),
            vector_store=Mock(),
            llm=Mock(),
        )
        pipeline.min_score_threshold = threshold
        return pipeline
    
    def test_filter_and_sort_by_score(self):
        """Should sort by score descending and filter by threshold."""
        pipeline = self.create_mock_pipeline(threshold=0.3)
        all_results = {
            "a": {"id": "a", "score": 0.9},
            "b": {"id": "b", "score": 0.2},  # Below threshold
            "c": {"id": "c", "score": 0.5},
        }
        
        results = pipeline._filter_and_sort_results(all_results)
        
        assert len(results) == 2
        assert results[0]["id"] == "a"  # Highest score first
        assert results[1]["id"] == "c"
    
    def test_filter_fallback_to_top3(self):
        """Should return top 3 results when all below threshold."""
        pipeline = self.create_mock_pipeline(threshold=0.9)
        all_results = {
            "a": {"id": "a", "score": 0.3},
            "b": {"id": "b", "score": 0.2},
            "c": {"id": "c", "score": 0.1},
            "d": {"id": "d", "score": 0.05},
        }
        
        results = pipeline._filter_and_sort_results(all_results)
        
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
