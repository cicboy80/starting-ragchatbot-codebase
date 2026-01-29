"""
API endpoint tests for the RAG system FastAPI application.

Tests cover:
- POST /api/query - Query processing endpoint
- GET /api/courses - Course statistics endpoint
- GET / - Health check / root endpoint
"""

import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_success_without_session(self, client, mock_rag_system):
        """Test successful query without providing a session ID"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_success_with_session(self, client, mock_rag_system):
        """Test successful query with an existing session ID"""
        response = client.post(
            "/api/query",
            json={
                "query": "Tell me more about neural networks",
                "session_id": "existing-session-456"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about neural networks",
            "existing-session-456"
        )

    def test_query_returns_sources(self, client, mock_rag_system):
        """Test that query returns sources from RAG system"""
        mock_rag_system.query.return_value = (
            "Python is a programming language.",
            ["Python Basics - Lesson 1", "Python Basics - Lesson 2"]
        )

        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert "Python Basics - Lesson 1" in data["sources"]

    def test_query_missing_query_field(self, client):
        """Test that missing query field returns 422 validation error"""
        response = client.post(
            "/api/query",
            json={"session_id": "some-session"}
        )

        assert response.status_code == 422

    def test_query_empty_string(self, client, mock_rag_system):
        """Test query with empty string is accepted (business logic should handle)"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty query is technically valid per schema, RAG system handles it
        assert response.status_code == 200

    def test_query_rag_system_error(self, client, mock_rag_system):
        """Test that RAG system errors return 500"""
        mock_rag_system.query.side_effect = Exception("Vector store unavailable")

        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]

    def test_query_invalid_json(self, client):
        """Test that invalid JSON returns 422"""
        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_empty_catalog(self, client, mock_rag_system):
        """Test response when no courses are loaded"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error(self, client, mock_rag_system):
        """Test that analytics errors return 500"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint (health check)"""

    def test_root_returns_ok(self, client):
        """Test that root endpoint returns OK status"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_message(self, client):
        """Test that root endpoint includes descriptive message"""
        response = client.get("/")

        data = response.json()
        assert "message" in data
        assert "RAG" in data["message"]


class TestRequestValidation:
    """Tests for request validation and edge cases"""

    def test_query_with_extra_fields_ignored(self, client, mock_rag_system):
        """Test that extra fields in request are ignored"""
        response = client.post(
            "/api/query",
            json={
                "query": "Test query",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )

        assert response.status_code == 200

    def test_query_with_long_input(self, client, mock_rag_system):
        """Test query with very long input string"""
        long_query = "What is Python? " * 1000

        response = client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_with_special_characters(self, client, mock_rag_system):
        """Test query with special characters and unicode"""
        response = client.post(
            "/api/query",
            json={"query": "What about émojis 🐍 and spëcial chars <>&?"}
        )

        assert response.status_code == 200


class TestResponseFormat:
    """Tests for response format compliance"""

    def test_query_response_structure(self, client, mock_rag_system):
        """Test that query response matches expected schema"""
        response = client.post(
            "/api/query",
            json={"query": "Test"}
        )

        data = response.json()

        # Verify all required fields present
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_structure(self, client, mock_rag_system):
        """Test that courses response matches expected schema"""
        response = client.get("/api/courses")

        data = response.json()

        # Verify all required fields present
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
