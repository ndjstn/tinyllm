"""Tests for profiling script."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestProfileScript:
    """Test profile.py script."""

    def test_script_exists(self):
        """Test that profile script exists and is executable."""
        script_path = Path("scripts/profile.py")
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111  # Check executable bit

    def test_help_message(self):
        """Test that script shows help message."""
        result = subprocess.run(
            [sys.executable, "scripts/profile.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Profile TinyLLM" in result.stdout
        assert "cpu" in result.stdout
        assert "memory" in result.stdout
        assert "graph" in result.stdout

    def test_cpu_help(self):
        """Test CPU profiling mode help."""
        result = subprocess.run(
            [sys.executable, "scripts/profile.py", "cpu", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "flamegraph" in result.stdout.lower()

    def test_memory_help(self):
        """Test memory profiling mode help."""
        result = subprocess.run(
            [sys.executable, "scripts/profile.py", "memory", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "memory" in result.stdout.lower()

    def test_graph_help(self):
        """Test graph profiling mode help."""
        result = subprocess.run(
            [sys.executable, "scripts/profile.py", "graph", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "graph" in result.stdout.lower()

    def test_no_mode(self):
        """Test that no mode shows help."""
        result = subprocess.run(
            [sys.executable, "scripts/profile.py"],
            capture_output=True,
            text=True,
        )
        # Should show help (exit code 1 because no mode specified)
        assert result.returncode == 1
        assert "Profiling mode" in result.stdout or "usage:" in result.stdout.lower()
