"""Integration tests using real components to identify failures"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


class TestVectorStoreReal:
    """Test VectorStore with real ChromaDB"""

    @pytest.fixture
    def real_vector_store(self):
        """Create real VectorStore pointing to existing DB"""
        # Use the actual chroma_db path
        chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "chroma_db"
        )
        return VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    def test_vector_store_has_courses(self, real_vector_store):
        """Test that vector store has course data"""
        course_count = real_vector_store.get_course_count()
        print(f"\nCourse count in vector store: {course_count}")
        assert course_count > 0, "Vector store has no courses loaded!"

    def test_vector_store_lists_courses(self, real_vector_store):
        """Test getting list of course titles"""
        titles = real_vector_store.get_existing_course_titles()
        print(f"\nCourses in vector store: {titles}")
        assert len(titles) > 0, "No course titles found!"

    def test_vector_store_search_returns_results(self, real_vector_store):
        """Test that basic search returns results"""
        results = real_vector_store.search(query="introduction")
        print(f"\nSearch results count: {len(results.documents)}")
        print(f"Search error: {results.error}")

        if results.error:
            pytest.fail(f"Search returned error: {results.error}")

        assert not results.is_empty(), "Search returned no results for 'introduction'"

    def test_vector_store_search_with_course_filter(self, real_vector_store):
        """Test search with course name filter"""
        # First get a course title
        titles = real_vector_store.get_existing_course_titles()
        if not titles:
            pytest.skip("No courses in vector store")

        course_name = titles[0]
        print(f"\nSearching in course: {course_name}")

        results = real_vector_store.search(
            query="lesson",
            course_name=course_name
        )
        print(f"Results: {len(results.documents)}, Error: {results.error}")

        # Should not error
        assert results.error is None, f"Search error: {results.error}"


class TestCourseSearchToolReal:
    """Test CourseSearchTool with real VectorStore"""

    @pytest.fixture
    def real_vector_store(self):
        chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "chroma_db"
        )
        return VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    @pytest.fixture
    def real_search_tool(self, real_vector_store):
        return CourseSearchTool(real_vector_store)

    def test_search_tool_execute_returns_content(self, real_search_tool):
        """Test that search tool execute returns actual content"""
        result = real_search_tool.execute(query="what is")
        print(f"\nSearch tool result: {result[:500] if len(result) > 500 else result}")

        # Should not be an error message
        assert "No relevant content found" not in result or "error" not in result.lower()

    def test_search_tool_tracks_sources_with_real_data(self, real_search_tool):
        """Test that sources are tracked with real data"""
        real_search_tool.execute(query="introduction lesson")

        print(f"\nSources tracked: {real_search_tool.last_sources}")
        # Should have sources if search found results


class TestToolManagerReal:
    """Test ToolManager with real tools"""

    @pytest.fixture
    def real_vector_store(self):
        chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "chroma_db"
        )
        return VectorStore(
            chroma_path=chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    @pytest.fixture
    def real_tool_manager(self, real_vector_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(real_vector_store))
        manager.register_tool(CourseOutlineTool(real_vector_store))
        return manager

    def test_tool_manager_execute_search(self, real_tool_manager):
        """Test executing search tool through manager"""
        result = real_tool_manager.execute_tool(
            "search_course_content",
            query="introduction to"
        )
        print(f"\nTool manager search result: {result[:500] if len(result) > 500 else result}")

    def test_tool_manager_get_sources(self, real_tool_manager):
        """Test getting sources after tool execution"""
        real_tool_manager.execute_tool(
            "search_course_content",
            query="lesson content"
        )
        sources = real_tool_manager.get_last_sources()
        print(f"\nSources from tool manager: {sources}")


class TestRAGSystemComponentsReal:
    """Test RAG system components with real dependencies"""

    def test_config_has_api_key(self):
        """Test that config has valid API key"""
        from config import config

        print(f"\nAPI key present: {bool(config.ANTHROPIC_API_KEY)}")
        print(f"API key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}")

        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set!"
        assert len(config.ANTHROPIC_API_KEY) > 10, "ANTHROPIC_API_KEY looks invalid"

    def test_chroma_db_exists(self):
        """Test that ChromaDB directory exists and has data"""
        from config import config

        chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "chroma_db"
        )

        print(f"\nChroma path: {chroma_path}")
        print(f"Path exists: {os.path.exists(chroma_path)}")

        if os.path.exists(chroma_path):
            files = os.listdir(chroma_path)
            print(f"Files in chroma_db: {files}")

        assert os.path.exists(chroma_path), "ChromaDB directory does not exist"


class TestAIGeneratorReal:
    """Test AIGenerator with real API (requires API key)"""

    @pytest.fixture
    def real_ai_generator(self):
        """Create real AIGenerator - skips if no API key"""
        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No ANTHROPIC_API_KEY configured")

        from ai_generator import AIGenerator
        return AIGenerator(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ANTHROPIC_MODEL
        )

    def test_ai_generator_simple_response(self, real_ai_generator):
        """Test AI generator can get a simple response"""
        try:
            response = real_ai_generator.generate_response(
                query="Say hello in one word."
            )
            print(f"\nAI response: {response}")
            assert response is not None
            assert len(response) > 0
        except Exception as e:
            pytest.fail(f"AI generator failed: {type(e).__name__}: {e}")

    def test_ai_generator_with_tools_no_execution(self, real_ai_generator):
        """Test AI generator with tools but no tool_manager (should not crash)"""
        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }]

        try:
            response = real_ai_generator.generate_response(
                query="What is 2+2? Answer briefly.",
                tools=tools
            )
            print(f"\nAI response with tools: {response}")
        except Exception as e:
            pytest.fail(f"AI generator with tools failed: {type(e).__name__}: {e}")


class TestFullRAGSystemReal:
    """Test the complete RAG system end-to-end"""

    @pytest.fixture
    def real_rag_system(self):
        """Create real RAGSystem - skips if dependencies missing"""
        from config import config
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No ANTHROPIC_API_KEY configured")

        from rag_system import RAGSystem
        return RAGSystem(config)

    def test_rag_system_query_simple(self, real_rag_system):
        """Test RAG system with a simple query"""
        try:
            response, sources = real_rag_system.query("What courses are available?")
            print(f"\nRAG response: {response}")
            print(f"Sources: {sources}")

            assert response is not None
            assert "query failed" not in response.lower(), f"Query failed! Response: {response}"
        except Exception as e:
            pytest.fail(f"RAG system query failed: {type(e).__name__}: {e}")

    def test_rag_system_query_course_content(self, real_rag_system):
        """Test RAG system with course content query"""
        try:
            response, sources = real_rag_system.query("What is covered in the introduction?")
            print(f"\nRAG content response: {response}")
            print(f"Sources: {sources}")

            assert response is not None
            assert "query failed" not in response.lower(), f"Query failed! Response: {response}"
        except Exception as e:
            pytest.fail(f"RAG system content query failed: {type(e).__name__}: {e}")
