"""Tests for code executor tool."""

import pytest

from tinyllm.tools.code_executor import (
    CodeExecutorConfig,
    CodeExecutorInput,
    CodeExecutorOutput,
    CodeExecutorTool,
)
from tinyllm.tools.sandbox import (
    SandboxConfig,
    validate_code,
    create_restricted_globals,
    DEFAULT_ALLOWED_IMPORTS,
    BLOCKED_IMPORTS,
)


class TestCodeValidation:
    """Tests for code validation."""

    def test_valid_simple_code(self):
        """Should accept simple valid code."""
        code = "x = 1 + 2\nprint(x)"
        is_valid, error = validate_code(code)
        assert is_valid is True
        assert error == ""

    def test_valid_math_import(self):
        """Should accept allowed imports."""
        code = "import math\nprint(math.sqrt(16))"
        is_valid, error = validate_code(code)
        assert is_valid is True

    def test_valid_from_import(self):
        """Should accept from imports of allowed modules."""
        code = "from datetime import datetime\nprint(datetime.now())"
        is_valid, error = validate_code(code)
        assert is_valid is True

    def test_blocked_os_import(self):
        """Should reject os import."""
        code = "import os\nos.system('ls')"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "os" in error

    def test_blocked_subprocess_import(self):
        """Should reject subprocess import."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "subprocess" in error

    def test_blocked_sys_import(self):
        """Should reject sys import."""
        code = "import sys\nprint(sys.path)"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "sys" in error

    def test_blocked_eval(self):
        """Should reject eval calls."""
        code = "eval('print(1)')"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "eval" in error

    def test_blocked_exec(self):
        """Should reject exec calls."""
        code = "exec('x = 1')"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "exec" in error

    def test_blocked_open(self):
        """Should reject open calls."""
        code = "open('/etc/passwd').read()"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "open" in error

    def test_syntax_error(self):
        """Should reject syntax errors."""
        code = "def f(\nprint('broken'"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "Syntax error" in error

    def test_unlisted_import(self):
        """Should reject imports not in allowed list."""
        code = "import numpy"
        is_valid, error = validate_code(code)
        assert is_valid is False
        assert "numpy" in error


class TestRestrictedGlobals:
    """Tests for restricted globals."""

    def test_has_safe_builtins(self):
        """Should include safe built-in functions."""
        globals_dict = create_restricted_globals()
        builtins = globals_dict["__builtins__"]

        # Should have safe functions
        assert "print" in builtins
        assert "len" in builtins
        assert "range" in builtins
        assert "int" in builtins
        assert "str" in builtins

    def test_has_math_module(self):
        """Should include math module."""
        globals_dict = create_restricted_globals()
        assert "math" in globals_dict

    def test_has_datetime_module(self):
        """Should include datetime module."""
        globals_dict = create_restricted_globals()
        assert "datetime" in globals_dict


class TestCodeExecutorTool:
    """Tests for CodeExecutorTool."""

    def test_tool_metadata(self):
        """Should have correct metadata."""
        tool = CodeExecutorTool()
        assert tool.metadata.id == "code_executor"
        assert tool.metadata.category == "execution"
        assert tool.metadata.sandbox_required is True

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Should execute simple code."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="print('hello')")
        )
        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_math_execution(self):
        """Should execute math code."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="import math\nprint(math.sqrt(16))")
        )
        assert result.success is True
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_multiple_prints(self):
        """Should capture multiple print outputs."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="print('a')\nprint('b')\nprint('c')")
        )
        assert result.success is True
        assert "a" in result.output
        assert "b" in result.output
        assert "c" in result.output

    @pytest.mark.asyncio
    async def test_computation(self):
        """Should handle computation."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="result = sum(range(10))\nprint(result)")
        )
        assert result.success is True
        assert "45" in result.output

    @pytest.mark.asyncio
    async def test_blocked_import_rejected(self):
        """Should reject blocked imports."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="import os\nprint(os.getcwd())")
        )
        assert result.success is False
        assert result.error is not None
        assert "os" in result.error

    @pytest.mark.asyncio
    async def test_blocked_eval_rejected(self):
        """Should reject eval."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="eval('1+1')")
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Should enforce timeout."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(
                code="import time\nwhile True: time.sleep(0.1)",
                timeout_seconds=1,
            )
        )
        assert result.success is False
        assert "timeout" in result.error.lower() or "time" in result.error.lower()

    @pytest.mark.asyncio
    async def test_runtime_error_handling(self):
        """Should handle runtime errors gracefully."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="x = 1/0")
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self):
        """Should track execution time."""
        tool = CodeExecutorTool()
        result = await tool.execute(
            CodeExecutorInput(code="print('fast')")
        )
        assert result.execution_time_ms >= 0

    def test_validate_only(self):
        """Should validate without executing."""
        tool = CodeExecutorTool()

        # Valid code
        is_valid, error = tool.validate_only("print('hello')")
        assert is_valid is True

        # Invalid code
        is_valid, error = tool.validate_only("import os")
        assert is_valid is False


class TestCodeExecutorConfig:
    """Tests for CodeExecutorConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = CodeExecutorConfig()
        assert config.max_timeout_seconds == 30
        assert config.max_memory_mb == 256
        assert config.max_output_size == 10000

    def test_custom_config(self):
        """Should accept custom config."""
        config = CodeExecutorConfig(
            max_timeout_seconds=60,
            max_memory_mb=512,
        )
        tool = CodeExecutorTool(config)
        assert tool.executor_config.max_timeout_seconds == 60
        assert tool.executor_config.max_memory_mb == 512


class TestAllowedImports:
    """Tests for import allowlists."""

    def test_default_allowed_imports(self):
        """Should have expected default imports."""
        assert "math" in DEFAULT_ALLOWED_IMPORTS
        assert "random" in DEFAULT_ALLOWED_IMPORTS
        assert "datetime" in DEFAULT_ALLOWED_IMPORTS
        assert "json" in DEFAULT_ALLOWED_IMPORTS
        assert "re" in DEFAULT_ALLOWED_IMPORTS

    def test_blocked_imports(self):
        """Should block dangerous imports."""
        assert "os" in BLOCKED_IMPORTS
        assert "sys" in BLOCKED_IMPORTS
        assert "subprocess" in BLOCKED_IMPORTS
        assert "socket" in BLOCKED_IMPORTS
