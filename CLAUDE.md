# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Users interact via a web UI, queries go through FastAPI to Claude with tool-calling capability, and Claude can search a ChromaDB vector database for relevant course content.

## Commands

**Always use `uv` for running commands and managing dependencies - never use `pip` directly.**

### Run the application
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```
Or use `./run.sh` from the project root.

Access at http://localhost:8000 (API docs at /docs).

### Install dependencies
```bash
uv sync
```

## Architecture

### Query Flow
```
Frontend (script.js) → POST /api/query → app.py → RAGSystem.query()
    → AIGenerator calls Claude API with tools
    → Claude decides: answer directly OR use search_course_content tool
    → If tool used: ToolManager → VectorStore.search() → ChromaDB
    → Claude synthesizes final answer from search results
    → Response returns with answer + sources
```

### Key Components

**RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components. Entry point for queries.

**AIGenerator** (`backend/ai_generator.py`): Wraps Anthropic Claude API. Handles tool execution loop - if Claude returns `stop_reason: "tool_use"`, executes the tool and sends results back for final synthesis.

**VectorStore** (`backend/vector_store.py`): ChromaDB wrapper with two collections:
- `course_catalog`: Course metadata for semantic name resolution
- `course_content`: Text chunks for content search

**CourseSearchTool** (`backend/search_tools.py`): The tool Claude can invoke. Parameters: `query` (required), `course_name` (optional, supports partial matching), `lesson_number` (optional).

**DocumentProcessor** (`backend/document_processor.py`): Parses course documents, extracts metadata, chunks text with configurable size/overlap.

### Document Format
Course files in `docs/` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
...
```

### Configuration
All settings in `backend/config.py`: model names, chunk size (800), chunk overlap (100), max results (5), max history (2).

Requires `ANTHROPIC_API_KEY` in `.env` file (copy from `.env.example`).
