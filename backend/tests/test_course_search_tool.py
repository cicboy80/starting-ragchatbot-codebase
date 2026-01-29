"""Tests for CourseSearchTool.execute() method"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        store = Mock()
        store.search = Mock()
        store.get_lesson_link = Mock(return_value="https://example.com/lesson")
        return store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool with mocked VectorStore"""
        return CourseSearchTool(mock_vector_store)

    def test_execute_returns_formatted_results(self, search_tool, mock_vector_store):
        """Test that execute returns properly formatted results"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=["This is lesson content about Python."],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="Python basics")

        assert result is not None
        assert "Python Basics" in result
        assert "Lesson 1" in result
        assert "This is lesson content about Python." in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test that execute passes course_name filter to vector store"""
        mock_results = SearchResults(
            documents=["MCP content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="protocols", course_name="MCP")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="protocols",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test that execute passes lesson_number filter to vector store"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="advanced topics", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="advanced topics",
            course_name=None,
            lesson_number=3
        )

    def test_execute_handles_error_results(self, search_tool, mock_vector_store):
        """Test that execute properly handles error results from vector store"""
        mock_results = SearchResults.empty("No course found matching 'NonExistent'")
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test", course_name="NonExistent")

        assert "No course found matching 'NonExistent'" in result

    def test_execute_handles_empty_results(self, search_tool, mock_vector_store):
        """Test that execute handles empty results gracefully"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_tracks_sources(self, search_tool, mock_vector_store):
        """Test that execute populates last_sources for UI"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        search_tool.execute(query="test query")

        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["title"] == "Course A - Lesson 1"
        assert search_tool.last_sources[1]["title"] == "Course B - Lesson 2"

    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test execute with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "Intro Course", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        search_tool.execute(
            query="specific topic",
            course_name="Intro",
            lesson_number=5
        )

        mock_vector_store.search.assert_called_once_with(
            query="specific topic",
            course_name="Intro",
            lesson_number=5
        )


class TestToolManager:
    """Test suite for ToolManager"""

    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.search = Mock()
        store.get_lesson_link = Mock(return_value=None)
        return store

    @pytest.fixture
    def tool_manager(self, mock_vector_store):
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        return manager

    def test_get_tool_definitions(self, tool_manager):
        """Test that tool definitions are returned correctly"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) >= 1
        tool_def = definitions[0]
        assert tool_def["name"] == "search_course_content"
        assert "input_schema" in tool_def

    def test_execute_tool_by_name(self, tool_manager, mock_vector_store):
        """Test executing a tool by name"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = tool_manager.execute_tool("search_course_content", query="test")

        assert result is not None
        mock_vector_store.search.assert_called_once()

    def test_execute_unknown_tool(self, tool_manager):
        """Test that executing unknown tool returns error message"""
        result = tool_manager.execute_tool("unknown_tool", query="test")

        assert "not found" in result.lower()
