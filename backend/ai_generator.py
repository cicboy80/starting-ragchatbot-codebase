import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool call rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
   - Use for questions about specific topics, concepts, or detailed educational content

2. **get_course_outline**: Get course structure including title, link, and lesson list
   - Use for questions about course structure, syllabus, what lessons are covered, or course overview
   - Returns: course title, course link, and complete lesson list (lesson number and title for each)

Tool Usage Guidelines:
- You may make up to 2 sequential tool calls per query when needed
- Use multi-step tool calling for:
  - Getting course structure before searching content
  - Refining searches based on initial results
  - Comparing information across courses
- After each tool result, decide if you have enough information or need another tool call
- Use **get_course_outline** for: "What lessons are in X?", "Show me the outline of X", "What does X course cover?", "Course structure of X"
- Use **search_course_content** for: detailed content questions, specific topics, concepts explained in lessons
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        messages = [{"role": "user", "content": query}]
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution loop if needed
        round_count = 0
        while response.stop_reason == "tool_use" and tool_manager and round_count < self.MAX_TOOL_ROUNDS:
            # Execute tools from this round
            tool_results = self._execute_tool_round(response, tool_manager)

            # Handle execution errors
            if tool_results is None:
                return "I encountered an error while processing your request. Please try again."

            # Add assistant's tool use response and tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            round_count += 1

            # Determine tool_choice for next round
            tool_choice = {"type": "none"} if round_count >= self.MAX_TOOL_ROUNDS else {"type": "auto"}

            # Make next API call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            if tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = tool_choice

            response = self.client.messages.create(**next_params)

        # Handle empty response gracefully
        if not response.content:
            return "I was unable to generate a response. Please try rephrasing your question."

        # Return final response
        return response.content[0].text
    
    def _execute_tool_round(self, response, tool_manager) -> Optional[List[Dict[str, Any]]]:
        """
        Execute all tool calls from a single response.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dicts for the API, or None if execution failed
        """
        tool_results = []

        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
        except Exception as e:
            # Return error as tool result so Claude can see and handle it
            if tool_results:
                # If we already have some results, add error for the failed one
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": f"Error executing tool: {str(e)}"
                })
            else:
                return None

        return tool_results if tool_results else None