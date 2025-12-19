"""Sandbox utilities for safe code execution.

Provides isolated execution environment for untrusted code.
"""

import asyncio
import resource
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    timeout_seconds: int = 10
    max_memory_mb: int = 256
    max_output_size: int = 10000  # Characters
    allow_imports: list[str] | None = None  # None means use default whitelist


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool
    stdout: str
    stderr: str
    return_value: Optional[str]
    error: Optional[str]
    execution_time_ms: int


# Safe modules that can be imported
DEFAULT_ALLOWED_IMPORTS = [
    "math",
    "random",
    "datetime",
    "json",
    "re",
    "itertools",
    "functools",
    "collections",
    "string",
    "decimal",
    "fractions",
    "statistics",
    "typing",
]

# Dangerous modules that should never be imported
BLOCKED_IMPORTS = [
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "pickle",
    "marshal",
    "importlib",
    "__builtins__",
    "builtins",
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "breakpoint",
]


def create_restricted_globals() -> dict:
    """Create a restricted global namespace for code execution."""
    import math
    import random
    import datetime
    import json
    import re

    safe_builtins = {
        # Safe built-in functions
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "ZeroDivisionError": ZeroDivisionError,
        "StopIteration": StopIteration,
        # Constants
        "True": True,
        "False": False,
        "None": None,
    }

    return {
        "__builtins__": safe_builtins,
        "math": math,
        "random": random,
        "datetime": datetime,
        "json": json,
        "re": re,
    }


def validate_code(code: str, allowed_imports: list[str] | None = None) -> tuple[bool, str]:
    """Validate code for dangerous patterns.

    Args:
        code: Code to validate.
        allowed_imports: List of allowed module imports.

    Returns:
        Tuple of (is_valid, error_message).
    """
    import ast

    allowed = allowed_imports or DEFAULT_ALLOWED_IMPORTS

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Check for dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in BLOCKED_IMPORTS:
                    return False, f"Import of '{module_name}' is not allowed"
                if module_name not in allowed:
                    return False, f"Import of '{module_name}' is not in allowed list"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in BLOCKED_IMPORTS:
                    return False, f"Import from '{module_name}' is not allowed"
                if module_name not in allowed:
                    return False, f"Import from '{module_name}' is not in allowed list"

        # Check for dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval", "compile", "open", "input", "__import__"]:
                    return False, f"Function '{node.func.id}' is not allowed"

        # Check for attribute access to dangerous modules
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in BLOCKED_IMPORTS:
                    return False, f"Access to '{node.value.id}' module is not allowed"

    return True, ""


async def run_in_sandbox(
    code: str,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Run Python code in a sandboxed subprocess.

    Args:
        code: Python code to execute.
        config: Sandbox configuration.

    Returns:
        Execution result.
    """
    import time

    config = config or SandboxConfig()

    # Validate code first
    is_valid, error = validate_code(code, config.allow_imports)
    if not is_valid:
        return SandboxResult(
            success=False,
            stdout="",
            stderr="",
            return_value=None,
            error=f"Code validation failed: {error}",
            execution_time_ms=0,
        )

    # Create wrapper script that captures output
    wrapper_code = f'''
import sys
import io
import resource

# Set memory limit
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, ({config.max_memory_mb} * 1024 * 1024, hard))
except:
    pass

# Capture stdout
_stdout = io.StringIO()
sys.stdout = _stdout

try:
    # User code
{_indent_code(code, 4)}
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
finally:
    sys.stdout = sys.__stdout__
    print(_stdout.getvalue()[:10000])
'''

    start_time = time.perf_counter()

    try:
        # Run in subprocess with timeout
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            wrapper_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=config.max_output_size,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                return_value=None,
                error=f"Execution timed out after {config.timeout_seconds} seconds",
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        stdout_str = stdout.decode("utf-8", errors="replace")[: config.max_output_size]
        stderr_str = stderr.decode("utf-8", errors="replace")[: config.max_output_size]

        success = process.returncode == 0 and not stderr_str

        return SandboxResult(
            success=success,
            stdout=stdout_str.strip(),
            stderr=stderr_str.strip(),
            return_value=stdout_str.strip() if success else None,
            error=stderr_str.strip() if stderr_str else None,
            execution_time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return SandboxResult(
            success=False,
            stdout="",
            stderr="",
            return_value=None,
            error=f"Sandbox error: {str(e)}",
            execution_time_ms=elapsed_ms,
        )


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by specified number of spaces."""
    indent = " " * spaces
    lines = code.split("\n")
    return "\n".join(indent + line for line in lines)
