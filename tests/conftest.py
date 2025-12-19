"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_task_content():
    """Sample task content for testing."""
    return "Write a Python function to check if a number is prime"


@pytest.fixture
def sample_math_expression():
    """Sample math expression for calculator testing."""
    return "2 + 2 * 3"
