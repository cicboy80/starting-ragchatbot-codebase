"""Tests for RAGSystem content-query handling"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockConfig:
    """Mock configuration for tests"""
    ANTHROPIC_API_KEY = "test-api-key"
    ANTHROPIC_MODEL = "test-model"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "./test_chroma_db"


class TestRAGSystemQuery:
    """Test suite for RAGSystem.query() method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        with patch('rag_system.VectorStore') as mock_class:
            mock_store = Mock()
            mock_store.search = Mock()
            mock_store.get_lesson_link = Mock(return_value=None)
            mock_store.get_course_outline = Mock(return_value=None)
            mock_class.return_value = mock_store
            yield mock_store

    @pytest.fixture
    def mock_ai_generator(self):
        """Create a mock AIGenerator"""
        with patch('rag_system.AIGenerator') as mock_class:
            mock_generator = Mock()
            mock_generator.generate_response = Mock(return_value="Test response")
            mock_class.return_value = mock_generator
            yield mock_generator

    @pytest.fixture
    def mock_document_processor(self):
        """Create a mock DocumentProcessor"""
        with patch('rag_system.DocumentProcessor') as mock_class:
            mock_processor = Mock()
            mock_class.return_value = mock_processor
            yield mock_processor

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager"""
        with patch('rag_system.SessionManager') as mock_class:
            mock_session = Mock()
            mock_session.get_conversation_history = Mock(return_value=None)
            mock_session.create_session = Mock(return_value="test-session-id")
            mock_class.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def rag_system(self, mock_vector_store, mock_ai_generator, mock_document_processor, mock_session_manager):
        """Create RAGSystem with mocked dependencies"""
        from rag_system import RAGSystem
        return RAGSystem(MockConfig())

    def test_query_calls_ai_generator(self, rag_system, mock_ai_generator):
        """Test that query calls AI generator with correct parameters"""
        rag_system.query("What is Python?")

        mock_ai_generator.generate_response.assert_called_once()
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs

        assert "query" in call_kwargs
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs

    def test_query_passes_tools_to_ai(self, rag_system, mock_ai_generator):
        """Test that query passes tool definitions to AI generator"""
        rag_system.query("Search for Python content")

        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        tools = call_kwargs["tools"]

        # Should have at least search_course_content tool
        assert len(tools) >= 1
        tool_names = [t["name"] for t in tools]
        assert "search_course_content" in tool_names

    def test_query_passes_tool_manager(self, rag_system, mock_ai_generator):
        """Test that query passes tool manager for execution"""
        rag_system.query("Find information about MCP")

        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["tool_manager"] is not None

    def test_query_returns_response_and_sources(self, rag_system, mock_ai_generator):
        """Test that query returns both response and sources"""
        response, sources = rag_system.query("Test question")

        assert response == "Test response"
        assert isinstance(sources, list)

    def test_query_includes_session_history(self, rag_system, mock_ai_generator, mock_session_manager):
        """Test that query includes conversation history for sessions"""
        mock_session_manager.get_conversation_history.return_value = "Previous conversation"

        rag_system.query("Follow-up question", session_id="test-session")

        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "Previous conversation"

    def test_query_updates_session_history(self, rag_system, mock_session_manager):
        """Test that query updates session history after response"""
        rag_system.query("New question", session_id="test-session")

        mock_session_manager.add_exchange.assert_called_once()

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """Test that sources are reset after being retrieved"""
        # First query
        response1, sources1 = rag_system.query("Question 1")

        # Sources should be reset for next query
        response2, sources2 = rag_system.query("Question 2")

        # This tests the reset_sources behavior
        # Both should return without accumulated sources from previous queries


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem with real tool execution flow"""

    @pytest.fixture
    def mock_vector_store(self):
        with patch('rag_system.VectorStore') as mock_class:
            mock_store = Mock()
            mock_store.search = Mock()
            mock_store.get_lesson_link = Mock(return_value=None)
            mock_store.get_course_outline = Mock(return_value=None)
            mock_class.return_value = mock_store
            yield mock_store

    @pytest.fixture
    def mock_ai_generator(self):
        with patch('rag_system.AIGenerator') as mock_class:
            mock_generator = Mock()
            mock_class.return_value = mock_generator
            yield mock_generator

    @pytest.fixture
    def mock_document_processor(self):
        with patch('rag_system.DocumentProcessor') as mock_class:
            yield Mock()

    @pytest.fixture
    def mock_session_manager(self):
        with patch('rag_system.SessionManager') as mock_class:
            mock_session = Mock()
            mock_session.get_conversation_history = Mock(return_value=None)
            mock_class.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def rag_system(self, mock_vector_store, mock_ai_generator, mock_document_processor, mock_session_manager):
        from rag_system import RAGSystem
        return RAGSystem(MockConfig())

    def test_tool_manager_registered_with_search_tool(self, rag_system):
        """Test that search tool is properly registered"""
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]

        assert "search_course_content" in tool_names

    def test_tool_manager_registered_with_outline_tool(self, rag_system):
        """Test that outline tool is properly registered"""
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]

        assert "get_course_outline" in tool_names

    def test_tool_execution_uses_vector_store(self, rag_system, mock_vector_store):
        """Test that tool execution actually calls vector store"""
        from vector_store import SearchResults

        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )

        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        mock_vector_store.search.assert_called_once()
        assert "Test content" in result or "Test Course" in result


class TestRAGSystemErrorHandling:
    """Test RAGSystem error handling for content queries"""

    @pytest.fixture
    def mock_vector_store(self):
        with patch('rag_system.VectorStore') as mock_class:
            mock_store = Mock()
            mock_store.search = Mock()
            mock_store.get_lesson_link = Mock(return_value=None)
            mock_store.get_course_outline = Mock(return_value=None)
            mock_class.return_value = mock_store
            yield mock_store

    @pytest.fixture
    def mock_ai_generator(self):
        with patch('rag_system.AIGenerator') as mock_class:
            mock_generator = Mock()
            mock_class.return_value = mock_generator
            yield mock_generator

    @pytest.fixture
    def mock_document_processor(self):
        with patch('rag_system.DocumentProcessor') as mock_class:
            yield Mock()

    @pytest.fixture
    def mock_session_manager(self):
        with patch('rag_system.SessionManager') as mock_class:
            mock_session = Mock()
            mock_session.get_conversation_history = Mock(return_value=None)
            mock_class.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def rag_system(self, mock_vector_store, mock_ai_generator, mock_document_processor, mock_session_manager):
        from rag_system import RAGSystem
        return RAGSystem(MockConfig())

    def test_query_handles_ai_exception(self, rag_system, mock_ai_generator):
        """Test that query properly propagates AI generator exceptions"""
        mock_ai_generator.generate_response.side_effect = Exception("API Error")

        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test question")

        assert "API Error" in str(exc_info.value)

    def test_query_handles_empty_query(self, rag_system, mock_ai_generator):
        """Test handling of empty query string"""
        mock_ai_generator.generate_response.return_value = "I need a question to help you."

        response, sources = rag_system.query("")

        # Should still call AI generator
        mock_ai_generator.generate_response.assert_called_once()
