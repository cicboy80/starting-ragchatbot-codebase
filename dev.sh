#!/bin/bash

# Development helper script for code quality tools
# Usage: ./dev.sh [command]
#   format  - Format code with black
#   lint    - Run ruff linter
#   check   - Run all checks without modifying files
#   fix     - Auto-fix all issues (format + lint fixes)

set -e

# Ensure dev dependencies are installed
ensure_deps() {
    if ! uv run black --version > /dev/null 2>&1; then
        echo "Installing dev dependencies..."
        uv sync --extra dev
    fi
}

format() {
    echo "Formatting code with black..."
    uv run black backend/ main.py
}

lint() {
    echo "Running ruff linter..."
    uv run ruff check backend/ main.py
}

lint_fix() {
    echo "Running ruff with auto-fix..."
    uv run ruff check --fix backend/ main.py
}

check() {
    echo "Running format check..."
    uv run black --check backend/ main.py
    echo ""
    echo "Running lint check..."
    uv run ruff check backend/ main.py
    echo ""
    echo "All checks passed!"
}

fix() {
    format
    lint_fix
    echo ""
    echo "All fixes applied!"
}

# Main
ensure_deps

case "${1:-check}" in
    format)
        format
        ;;
    lint)
        lint
        ;;
    check)
        check
        ;;
    fix)
        fix
        ;;
    *)
        echo "Usage: $0 {format|lint|check|fix}"
        echo ""
        echo "Commands:"
        echo "  format  - Format code with black"
        echo "  lint    - Run ruff linter"
        echo "  check   - Run all checks without modifying files (default)"
        echo "  fix     - Auto-fix all issues (format + lint fixes)"
        exit 1
        ;;
esac
