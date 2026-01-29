"""Tests for AIGenerator tool calling behavior"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class MockContentBlock:
    """Mock for Anthropic content blocks"""
    def __init__(self, block_type, text=None, tool_name=None, tool_input=None, tool_id=None):
        self.type = block_type
        self.text = text
        self.name = tool_name
        self.input = tool_input or {}
        self.id = tool_id


class MockResponse:
    """Mock for Anthropic API response"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator's tool calling behavior"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator with mocked client"""
        return AIGenerator(api_key="test-key", model="test-model")

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock ToolManager"""
        manager = Mock(spec=ToolManager)
        manager.execute_tool = Mock(return_value="Tool result: Found content about Python")
        return manager

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [{
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

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test response generation without tools"""
        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Hello, how can I help?")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("Hello")

        assert result == "Hello, how can I help?"

    def test_generate_response_passes_tools_to_api(self, ai_generator, mock_anthropic_client, sample_tools):
        """Test that tools are passed to the API when provided"""
        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Response without tools")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("question", tools=sample_tools)

        # Verify API was called with tools
        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == sample_tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_handles_tool_use(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that tool use is handled correctly"""
        # First response: Claude requests tool use
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "Python basics"},
                    tool_id="tool_123"
                )
            ],
            stop_reason="tool_use"
        )

        # Second response: Claude provides final answer
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Based on the search, Python is...")]
        )

        mock_anthropic_client.messages.create.side_effect = [tool_use_response, final_response]

        result = ai_generator.generate_response(
            "Tell me about Python",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )

        # Verify final response is returned
        assert result == "Based on the search, Python is..."

    def test_generate_response_includes_conversation_history(
        self, ai_generator, mock_anthropic_client
    ):
        """Test that conversation history is included in system prompt"""
        mock_response = MockResponse(
            content=[MockContentBlock("text", text="Response")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"
        ai_generator.generate_response("New question", conversation_history=history)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    def test_tool_execution_sends_results_back(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that tool results are sent back to Claude correctly"""
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_456"
                )
            ],
            stop_reason="tool_use"
        )

        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer")]
        )

        mock_anthropic_client.messages.create.side_effect = [tool_use_response, final_response]

        ai_generator.generate_response(
            "question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Get the second API call (after tool execution)
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result format
        tool_result_content = messages[2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_456"

    def test_no_tool_manager_skips_tool_execution(
        self, ai_generator, mock_anthropic_client, sample_tools
    ):
        """Test that without tool_manager, tool use is not executed"""
        # Response that would trigger tool use
        tool_use_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_789"
                )
            ],
            stop_reason="tool_use"
        )
        mock_anthropic_client.messages.create.return_value = tool_use_response

        # Call without tool_manager - should not raise, but behavior undefined
        # This tests current implementation
        result = ai_generator.generate_response("question", tools=sample_tools)

        # API should only be called once (no follow-up for tool results)
        assert mock_anthropic_client.messages.create.call_count == 1


class TestAIGeneratorSystemPrompt:
    """Test suite for AIGenerator system prompt"""

    @pytest.fixture
    def mock_anthropic_client(self):
        with patch('ai_generator.anthropic.Anthropic') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        return AIGenerator(api_key="test-key", model="test-model")

    def test_system_prompt_contains_tool_guidelines(self, ai_generator):
        """Test that system prompt contains tool usage guidelines"""
        prompt = ai_generator.SYSTEM_PROMPT

        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        assert "Tool Usage Guidelines" in prompt or "tool" in prompt.lower()

    def test_system_prompt_defines_response_protocol(self, ai_generator):
        """Test that system prompt defines response behavior"""
        prompt = ai_generator.SYSTEM_PROMPT

        assert "Response Protocol" in prompt or "response" in prompt.lower()

    def test_system_prompt_allows_multiple_tool_calls(self, ai_generator):
        """Test that system prompt mentions multi-step tool calling"""
        prompt = ai_generator.SYSTEM_PROMPT

        assert "2 sequential tool calls" in prompt or "multi-step" in prompt.lower()


class TestAIGeneratorMultiRoundToolCalling:
    """Test suite for sequential tool calling behavior"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator with mocked client"""
        return AIGenerator(api_key="test-key", model="test-model")

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock ToolManager"""
        manager = Mock(spec=ToolManager)
        manager.execute_tool = Mock(return_value="Tool result: Found content")
        return manager

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }, {
            "name": "get_course_outline",
            "description": "Get course outline",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {"type": "string"}
                },
                "required": ["course_name"]
            }
        }]

    def test_two_sequential_tool_calls(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that two sequential tool calls are handled correctly"""
        # First response: Claude requests get_course_outline
        first_tool_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "Python"},
                    tool_id="tool_1"
                )
            ],
            stop_reason="tool_use"
        )

        # Second response: Claude requests search_course_content
        second_tool_response = MockResponse(
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "lesson 3 content"},
                    tool_id="tool_2"
                )
            ],
            stop_reason="tool_use"
        )

        # Third response: Final answer
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Lesson 3 covers advanced topics.")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            first_tool_response, second_tool_response, final_response
        ]

        result = ai_generator.generate_response(
            "What does lesson 3 of Python cover?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0][0] == "get_course_outline"
        assert calls[1][0][0] == "search_course_content"

        # Verify API was called 3 times
        assert mock_anthropic_client.messages.create.call_count == 3

        # Verify final response is returned
        assert result == "Lesson 3 covers advanced topics."

    def test_loop_terminates_at_max_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that loop terminates after MAX_TOOL_ROUNDS even if Claude wants more"""
        # Create responses that keep requesting tools
        tool_response_1 = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="search_course_content",
                               tool_input={"query": "test1"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )
        tool_response_2 = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="search_course_content",
                               tool_input={"query": "test2"}, tool_id="tool_2")
            ],
            stop_reason="tool_use"
        )
        # After max rounds, even with tool_choice="none", Claude gives text response
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer after max rounds")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            tool_response_1, tool_response_2, final_response
        ]

        result = ai_generator.generate_response(
            "Complex question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify exactly MAX_TOOL_ROUNDS tool executions
        assert mock_tool_manager.execute_tool.call_count == ai_generator.MAX_TOOL_ROUNDS

        # Verify the last API call used tool_choice="none"
        last_call = mock_anthropic_client.messages.create.call_args_list[-1]
        assert last_call.kwargs["tool_choice"] == {"type": "none"}

        assert result == "Final answer after max rounds"

    def test_loop_terminates_when_claude_stops_calling_tools(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that loop terminates early if Claude provides answer without tool"""
        # First response: Tool use
        tool_response = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="get_course_outline",
                               tool_input={"course_name": "Python"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )

        # Second response: Claude has enough info, gives text answer
        final_response = MockResponse(
            content=[MockContentBlock("text", text="The course has 5 lessons.")],
            stop_reason="end_turn"
        )

        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        result = ai_generator.generate_response(
            "How many lessons in Python?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Only one tool execution
        assert mock_tool_manager.execute_tool.call_count == 1

        # Only 2 API calls (not 3)
        assert mock_anthropic_client.messages.create.call_count == 2

        assert result == "The course has 5 lessons."

    def test_tool_choice_auto_for_intermediate_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that tool_choice is 'auto' for rounds before the last allowed"""
        tool_response = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="get_course_outline",
                               tool_input={"course_name": "Test"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Answer")]
        )

        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        ai_generator.generate_response(
            "Question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Second call should have tool_choice="auto" (round 1 < MAX_TOOL_ROUNDS)
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        assert second_call.kwargs["tool_choice"] == {"type": "auto"}

    def test_message_accumulation_across_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that messages are correctly accumulated across tool rounds"""
        tool_response_1 = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="get_course_outline",
                               tool_input={"course_name": "Python"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )
        tool_response_2 = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="search_course_content",
                               tool_input={"query": "test"}, tool_id="tool_2")
            ],
            stop_reason="tool_use"
        )
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            tool_response_1, tool_response_2, final_response
        ]

        ai_generator.generate_response(
            "Question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Check message structure in final API call
        final_call = mock_anthropic_client.messages.create.call_args_list[2]
        messages = final_call.kwargs["messages"]

        # Should have: user, assistant (tool_use 1), user (tool_result 1),
        #              assistant (tool_use 2), user (tool_result 2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_empty_response_returns_fallback(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that empty response returns fallback message"""
        tool_response = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="search_course_content",
                               tool_input={"query": "test"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )
        # Empty final response
        empty_response = MockResponse(content=[])

        mock_anthropic_client.messages.create.side_effect = [tool_response, empty_response]

        result = ai_generator.generate_response(
            "Question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert "unable to generate a response" in result.lower()

    def test_tool_execution_error_returns_error_message(
        self, ai_generator, mock_anthropic_client, mock_tool_manager, sample_tools
    ):
        """Test that tool execution errors are handled gracefully"""
        tool_response = MockResponse(
            content=[
                MockContentBlock("tool_use", tool_name="search_course_content",
                               tool_input={"query": "test"}, tool_id="tool_1")
            ],
            stop_reason="tool_use"
        )

        mock_anthropic_client.messages.create.return_value = tool_response

        # Make tool execution raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        result = ai_generator.generate_response(
            "Question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert "error" in result.lower()
