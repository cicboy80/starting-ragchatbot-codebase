"""
Shared pytest fixtures for RAG system tests.

This module provides test fixtures that avoid static file mounting issues
by creating a test-specific FastAPI app with mocked dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """Provide a mock configuration object"""
    return MockConfig()


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    manager = Mock()
    manager.create_session.return_value = "test-session-123"
    manager.get_conversation_history.return_value = []
    manager.add_exchange.return_value = None
    return manager


@pytest.fixture
def mock_rag_system(mock_session_manager):
    """
    Create a mock RAG system with configurable responses.

    Returns a mock that can be customized per test.
    """
    rag = Mock()
    rag.session_manager = mock_session_manager
    rag.query.return_value = (
        "This is a test response about course materials.",
        ["Source 1: Test Course - Lesson 1"]
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Data Science 101", "Machine Learning"]
    }
    return rag


# Pydantic models (duplicated from app.py to avoid import issues)
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system: Mock) -> FastAPI:
    """
    Create a test FastAPI app without static file mounting.

    This avoids the issue where the main app mounts ../frontend
    which doesn't exist in the test environment.
    """
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system"""
    return create_test_app(mock_rag_system)


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Python used for?",
        "session_id": None
    }


@pytest.fixture
def sample_query_with_session():
    """Sample query request with existing session"""
    return {
        "query": "Tell me more about that topic",
        "session_id": "existing-session-456"
    }
