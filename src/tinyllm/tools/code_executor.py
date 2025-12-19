"""Code executor tool for running Python code safely."""

from typing import Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.sandbox import SandboxConfig, run_in_sandbox, validate_code


class CodeExecutorInput(BaseModel):
    """Input for code executor tool."""

    code: str = Field(
        description="Python code to execute",
        max_length=10000,
    )
    timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum execution time in seconds",
    )


class CodeExecutorOutput(BaseModel):
    """Output from code executor tool."""

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: int = 0


class CodeExecutorConfig(ToolConfig):
    """Configuration for code executor."""

    max_timeout_seconds: int = Field(default=30, ge=1, le=60)
    max_memory_mb: int = Field(default=256, ge=64, le=1024)
    max_output_size: int = Field(default=10000, ge=1000, le=100000)
    allowed_imports: list[str] | None = None


class CodeExecutorTool(BaseTool[CodeExecutorInput, CodeExecutorOutput]):
    """Executes Python code in a sandboxed environment.

    Safety features:
    - Subprocess isolation
    - Timeout enforcement
    - Memory limits
    - Restricted imports (no os, sys, subprocess, etc.)
    - Output size limits
    """

    metadata = ToolMetadata(
        id="code_executor",
        name="Code Executor",
        description="Executes Python code safely in a sandboxed environment. "
        "Supports standard library modules like math, random, datetime, json, and re. "
        "Does NOT support file I/O, network access, or system operations.",
        category="execution",
        sandbox_required=True,
    )
    input_type = CodeExecutorInput
    output_type = CodeExecutorOutput

    def __init__(self, config: CodeExecutorConfig | None = None):
        """Initialize with optional configuration."""
        self.executor_config = config or CodeExecutorConfig()
        super().__init__(self.executor_config)

    async def execute(self, input: CodeExecutorInput) -> CodeExecutorOutput:
        """Execute Python code in sandbox."""
        # Validate code before execution
        is_valid, error = validate_code(
            input.code,
            self.executor_config.allowed_imports,
        )

        if not is_valid:
            return CodeExecutorOutput(
                success=False,
                error=f"Code validation failed: {error}",
            )

        # Enforce timeout limits
        timeout = min(input.timeout_seconds, self.executor_config.max_timeout_seconds)

        # Create sandbox config
        sandbox_config = SandboxConfig(
            timeout_seconds=timeout,
            max_memory_mb=self.executor_config.max_memory_mb,
            max_output_size=self.executor_config.max_output_size,
            allow_imports=self.executor_config.allowed_imports,
        )

        # Run in sandbox
        result = await run_in_sandbox(input.code, sandbox_config)

        return CodeExecutorOutput(
            success=result.success,
            output=result.stdout if result.success else None,
            error=result.error or result.stderr if not result.success else None,
            execution_time_ms=result.execution_time_ms,
        )

    def validate_only(self, code: str) -> tuple[bool, str]:
        """Validate code without executing.

        Args:
            code: Code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        return validate_code(code, self.executor_config.allowed_imports)
