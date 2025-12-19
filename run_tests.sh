#!/bin/bash
# Test runner script for TinyLLM
# This ensures tests run with the correct virtual environment

set -e

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/$VENV_DIR"
    echo "Please create a virtual environment first:"
    echo "  python -m venv .venv"
    echo "  .venv/bin/pip install -e \".[dev]\""
    exit 1
fi

# Run pytest with virtual environment Python
echo "Running tests with virtual environment..."
cd "$SCRIPT_DIR"
"$VENV_DIR/bin/python" -m pytest "$@"
