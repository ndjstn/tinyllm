"""SSH and shell tools for TinyLLM.

Provides tools for:
- SSH connection management
- Remote command execution
- File transfer (upload/download)
- Local shell command execution
"""

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ShellType(str, Enum):
    """Shell type enum."""

    BASH = "bash"
    SH = "sh"
    ZSH = "zsh"
    POWERSHELL = "powershell"
    CMD = "cmd"


@dataclass
class SSHConfig:
    """SSH connection configuration."""

    host: str
    port: int = 22
    username: str = "root"
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    timeout: int = 30
    known_hosts_policy: str = "warn"  # "strict", "warn", "ignore"
    compress: bool = False


@dataclass
class ShellConfig:
    """Local shell configuration."""

    shell: ShellType = ShellType.BASH
    timeout: int = 60
    working_directory: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    allowed_commands: Optional[List[str]] = None  # None = all allowed
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=/dev/zero",
        ":(){:|:&};:",  # Fork bomb
    ])


@dataclass
class CommandResult:
    """Result from command execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SSHResult:
    """Result from SSH operation."""

    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


class SSHClient:
    """SSH client for remote operations.

    This is a subprocess-based implementation that uses the ssh command.
    For production use, consider using paramiko or asyncssh.
    """

    def __init__(self, config: SSHConfig):
        """Initialize client.

        Args:
            config: SSH configuration.
        """
        self.config = config
        self._connected = False

    def _build_ssh_command(self, remote_command: Optional[str] = None) -> List[str]:
        """Build SSH command with options."""
        cmd = ["ssh"]

        # Port
        if self.config.port != 22:
            cmd.extend(["-p", str(self.config.port)])

        # Timeout
        cmd.extend(["-o", f"ConnectTimeout={self.config.timeout}"])

        # Known hosts policy
        if self.config.known_hosts_policy == "ignore":
            cmd.extend(["-o", "StrictHostKeyChecking=no"])
            cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])
        elif self.config.known_hosts_policy == "warn":
            cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])

        # Compression
        if self.config.compress:
            cmd.append("-C")

        # Private key
        if self.config.private_key_path:
            cmd.extend(["-i", self.config.private_key_path])

        # No password prompt (batch mode)
        cmd.extend(["-o", "BatchMode=yes"])

        # User@host
        cmd.append(f"{self.config.username}@{self.config.host}")

        # Remote command
        if remote_command:
            cmd.append(remote_command)

        return cmd

    def _build_scp_command(
        self,
        source: str,
        destination: str,
        upload: bool = True,
    ) -> List[str]:
        """Build SCP command."""
        cmd = ["scp"]

        # Port
        if self.config.port != 22:
            cmd.extend(["-P", str(self.config.port)])

        # Timeout (via SSH options)
        cmd.extend(["-o", f"ConnectTimeout={self.config.timeout}"])

        # Known hosts policy
        if self.config.known_hosts_policy == "ignore":
            cmd.extend(["-o", "StrictHostKeyChecking=no"])
            cmd.extend(["-o", "UserKnownHostsFile=/dev/null"])

        # Compression
        if self.config.compress:
            cmd.append("-C")

        # Private key
        if self.config.private_key_path:
            cmd.extend(["-i", self.config.private_key_path])

        # Batch mode
        cmd.extend(["-o", "BatchMode=yes"])

        # Source and destination
        remote_path = f"{self.config.username}@{self.config.host}:"
        if upload:
            cmd.extend([source, f"{remote_path}{destination}"])
        else:
            cmd.extend([f"{remote_path}{source}", destination])

        return cmd

    async def test_connection(self) -> SSHResult:
        """Test SSH connection."""
        try:
            result = await self.execute_command("echo 'connection_test'")
            if result.success and "connection_test" in result.stdout:
                self._connected = True
                return SSHResult(success=True, data={"connected": True})
            return SSHResult(success=False, error=result.stderr or "Connection test failed")
        except Exception as e:
            return SSHResult(success=False, error=str(e))

    async def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Execute command on remote host.

        Args:
            command: Command to execute.
            timeout: Override timeout.

        Returns:
            Command result.
        """
        import time

        start_time = time.time()
        ssh_cmd = self._build_ssh_command(command)
        effective_timeout = timeout or self.config.timeout

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            duration_ms = (time.time() - start_time) * 1000

            return CommandResult(
                success=process.returncode == 0,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
    ) -> SSHResult:
        """Upload file to remote host.

        Args:
            local_path: Local file path.
            remote_path: Remote destination path.

        Returns:
            SSH result.
        """
        if not os.path.exists(local_path):
            return SSHResult(success=False, error=f"Local file not found: {local_path}")

        scp_cmd = self._build_scp_command(local_path, remote_path, upload=True)

        try:
            process = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout * 2,  # Allow more time for file transfer
            )

            if process.returncode == 0:
                return SSHResult(
                    success=True,
                    data={"uploaded": local_path, "destination": remote_path},
                )
            return SSHResult(
                success=False,
                error=stderr.decode("utf-8", errors="replace"),
            )

        except asyncio.TimeoutError:
            return SSHResult(success=False, error="File upload timed out")
        except Exception as e:
            return SSHResult(success=False, error=str(e))

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
    ) -> SSHResult:
        """Download file from remote host.

        Args:
            remote_path: Remote file path.
            local_path: Local destination path.

        Returns:
            SSH result.
        """
        scp_cmd = self._build_scp_command(remote_path, local_path, upload=False)

        try:
            process = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout * 2,
            )

            if process.returncode == 0:
                return SSHResult(
                    success=True,
                    data={"downloaded": remote_path, "destination": local_path},
                )
            return SSHResult(
                success=False,
                error=stderr.decode("utf-8", errors="replace"),
            )

        except asyncio.TimeoutError:
            return SSHResult(success=False, error="File download timed out")
        except Exception as e:
            return SSHResult(success=False, error=str(e))


class ShellExecutor:
    """Local shell command executor."""

    def __init__(self, config: ShellConfig):
        """Initialize executor.

        Args:
            config: Shell configuration.
        """
        self.config = config

    def _is_command_allowed(self, command: str) -> Tuple[bool, str]:
        """Check if command is allowed.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check blocked commands
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return False, f"Command contains blocked pattern: {blocked}"

        # Check allowed commands (if whitelist is set)
        if self.config.allowed_commands is not None:
            # Extract the base command
            parts = shlex.split(command)
            if parts:
                base_cmd = parts[0]
                if base_cmd not in self.config.allowed_commands:
                    return False, f"Command '{base_cmd}' not in allowed list"

        return True, ""

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_directory: Optional[str] = None,
    ) -> CommandResult:
        """Execute shell command.

        Args:
            command: Command to execute.
            timeout: Override timeout.
            working_directory: Override working directory.

        Returns:
            Command result.
        """
        import time

        # Check if command is allowed
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Command blocked: {reason}",
            )

        start_time = time.time()
        effective_timeout = timeout or self.config.timeout
        cwd = working_directory or self.config.working_directory

        # Build environment
        env = os.environ.copy()
        env.update(self.config.environment)

        # Determine shell executable
        shell_map = {
            ShellType.BASH: ["/bin/bash", "-c"],
            ShellType.SH: ["/bin/sh", "-c"],
            ShellType.ZSH: ["/bin/zsh", "-c"],
            ShellType.POWERSHELL: ["powershell", "-Command"],
            ShellType.CMD: ["cmd", "/c"],
        }

        shell_cmd = shell_map.get(self.config.shell, ["/bin/sh", "-c"])

        try:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            duration_ms = (time.time() - start_time) * 1000

            return CommandResult(
                success=process.returncode == 0,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )


class SSHManager:
    """Manager for SSH clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, SSHClient] = {}

    def add_client(self, name: str, client: SSHClient) -> None:
        """Add an SSH client."""
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[SSHClient]:
        """Get an SSH client."""
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove an SSH client."""
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Input/Output Models

class CreateSSHClientInput(BaseModel):
    """Input for creating an SSH client."""

    name: str = Field(..., description="Name for the client")
    host: str = Field(..., description="Remote host")
    port: int = Field(default=22, description="SSH port")
    username: str = Field(default="root", description="SSH username")
    password: Optional[str] = Field(default=None, description="SSH password")
    private_key_path: Optional[str] = Field(default=None, description="Path to private key")
    timeout: int = Field(default=30, description="Connection timeout")


class CreateSSHClientOutput(BaseModel):
    """Output from creating an SSH client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message")


class SSHExecuteInput(BaseModel):
    """Input for executing SSH command."""

    client: str = Field(default="default", description="SSH client name")
    command: str = Field(..., description="Command to execute")
    timeout: Optional[int] = Field(default=None, description="Command timeout")


class SSHCommandOutput(BaseModel):
    """Output from SSH command execution."""

    success: bool = Field(description="Whether command succeeded")
    exit_code: int = Field(default=0, description="Exit code")
    stdout: Optional[str] = Field(default=None, description="Standard output")
    stderr: Optional[str] = Field(default=None, description="Standard error")
    duration_ms: float = Field(default=0, description="Execution duration")
    error: Optional[str] = Field(default=None, description="Error message")


class SSHUploadInput(BaseModel):
    """Input for uploading file via SSH."""

    client: str = Field(default="default", description="SSH client name")
    local_path: str = Field(..., description="Local file path")
    remote_path: str = Field(..., description="Remote destination path")


class SSHDownloadInput(BaseModel):
    """Input for downloading file via SSH."""

    client: str = Field(default="default", description="SSH client name")
    remote_path: str = Field(..., description="Remote file path")
    local_path: str = Field(..., description="Local destination path")


class SSHTransferOutput(BaseModel):
    """Output from file transfer."""

    success: bool = Field(description="Whether transfer succeeded")
    source: Optional[str] = Field(default=None, description="Source path")
    destination: Optional[str] = Field(default=None, description="Destination path")
    error: Optional[str] = Field(default=None, description="Error message")


class ShellExecuteInput(BaseModel):
    """Input for local shell command."""

    command: str = Field(..., description="Command to execute")
    timeout: int = Field(default=60, description="Command timeout")
    working_directory: Optional[str] = Field(default=None, description="Working directory")
    shell: ShellType = Field(default=ShellType.BASH, description="Shell type")


class ShellCommandOutput(BaseModel):
    """Output from shell command execution."""

    success: bool = Field(description="Whether command succeeded")
    exit_code: int = Field(default=0, description="Exit code")
    stdout: Optional[str] = Field(default=None, description="Standard output")
    stderr: Optional[str] = Field(default=None, description="Standard error")
    duration_ms: float = Field(default=0, description="Execution duration")
    error: Optional[str] = Field(default=None, description="Error message")


class SSHSimpleOutput(BaseModel):
    """Simple output for SSH operations."""

    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message")


# Tools

class CreateSSHClientTool(BaseTool[CreateSSHClientInput, CreateSSHClientOutput]):
    """Tool for creating an SSH client."""

    metadata = ToolMetadata(
        id="create_ssh_client",
        name="Create SSH Client",
        description="Create an SSH client for remote connections",
        category="utility",
    )
    input_type = CreateSSHClientInput
    output_type = CreateSSHClientOutput

    def __init__(self, manager: SSHManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateSSHClientInput) -> CreateSSHClientOutput:
        """Create an SSH client."""
        config = SSHConfig(
            host=input.host,
            port=input.port,
            username=input.username,
            password=input.password,
            private_key_path=input.private_key_path,
            timeout=input.timeout,
        )
        client = SSHClient(config)
        self.manager.add_client(input.name, client)

        return CreateSSHClientOutput(
            success=True,
            name=input.name,
        )


class SSHConnectionTestInput(BaseModel):
    """Input for testing SSH connection."""

    client: str = Field(default="default", description="SSH client name")


class SSHConnectionTestTool(BaseTool[SSHConnectionTestInput, SSHSimpleOutput]):
    """Tool for testing SSH connection."""

    metadata = ToolMetadata(
        id="test_ssh_connection",
        name="Test SSH Connection",
        description="Test an SSH connection",
        category="utility",
    )
    input_type = SSHConnectionTestInput
    output_type = SSHSimpleOutput

    def __init__(self, manager: SSHManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SSHConnectionTestInput) -> SSHSimpleOutput:
        """Test SSH connection."""
        client = self.manager.get_client(input.client)

        if not client:
            return SSHSimpleOutput(success=False, error=f"Client '{input.client}' not found")

        result = await client.test_connection()
        return SSHSimpleOutput(success=result.success, error=result.error)


class SSHExecuteTool(BaseTool[SSHExecuteInput, SSHCommandOutput]):
    """Tool for executing SSH commands."""

    metadata = ToolMetadata(
        id="ssh_execute",
        name="Execute SSH Command",
        description="Execute a command on a remote host via SSH",
        category="execution",
    )
    input_type = SSHExecuteInput
    output_type = SSHCommandOutput

    def __init__(self, manager: SSHManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SSHExecuteInput) -> SSHCommandOutput:
        """Execute SSH command."""
        client = self.manager.get_client(input.client)

        if not client:
            return SSHCommandOutput(
                success=False,
                exit_code=-1,
                error=f"Client '{input.client}' not found",
            )

        result = await client.execute_command(input.command, timeout=input.timeout)

        return SSHCommandOutput(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
        )


class SSHUploadTool(BaseTool[SSHUploadInput, SSHTransferOutput]):
    """Tool for uploading files via SSH."""

    metadata = ToolMetadata(
        id="ssh_upload",
        name="Upload File via SSH",
        description="Upload a file to a remote host via SCP",
        category="execution",
    )
    input_type = SSHUploadInput
    output_type = SSHTransferOutput

    def __init__(self, manager: SSHManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SSHUploadInput) -> SSHTransferOutput:
        """Upload file via SSH."""
        client = self.manager.get_client(input.client)

        if not client:
            return SSHTransferOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = await client.upload_file(input.local_path, input.remote_path)

        if result.success:
            return SSHTransferOutput(
                success=True,
                source=input.local_path,
                destination=input.remote_path,
            )
        return SSHTransferOutput(success=False, error=result.error)


class SSHDownloadTool(BaseTool[SSHDownloadInput, SSHTransferOutput]):
    """Tool for downloading files via SSH."""

    metadata = ToolMetadata(
        id="ssh_download",
        name="Download File via SSH",
        description="Download a file from a remote host via SCP",
        category="execution",
    )
    input_type = SSHDownloadInput
    output_type = SSHTransferOutput

    def __init__(self, manager: SSHManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SSHDownloadInput) -> SSHTransferOutput:
        """Download file via SSH."""
        client = self.manager.get_client(input.client)

        if not client:
            return SSHTransferOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = await client.download_file(input.remote_path, input.local_path)

        if result.success:
            return SSHTransferOutput(
                success=True,
                source=input.remote_path,
                destination=input.local_path,
            )
        return SSHTransferOutput(success=False, error=result.error)


class ShellExecuteTool(BaseTool[ShellExecuteInput, ShellCommandOutput]):
    """Tool for executing local shell commands."""

    metadata = ToolMetadata(
        id="shell_execute",
        name="Execute Shell Command",
        description="Execute a command in the local shell",
        category="execution",
    )
    input_type = ShellExecuteInput
    output_type = ShellCommandOutput

    def __init__(self, config: Optional[ShellConfig] = None):
        """Initialize tool."""
        self.config = config or ShellConfig()
        self.executor = ShellExecutor(self.config)

    async def execute(self, input: ShellExecuteInput) -> ShellCommandOutput:
        """Execute shell command."""
        # Create executor with input shell type if different
        if input.shell != self.config.shell:
            config = ShellConfig(
                shell=input.shell,
                timeout=input.timeout,
                working_directory=input.working_directory,
                environment=self.config.environment,
                allowed_commands=self.config.allowed_commands,
                blocked_commands=self.config.blocked_commands,
            )
            executor = ShellExecutor(config)
        else:
            executor = self.executor

        result = await executor.execute(
            command=input.command,
            timeout=input.timeout,
            working_directory=input.working_directory,
        )

        return ShellCommandOutput(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
        )


# Helper functions

def create_ssh_config(
    host: str,
    port: int = 22,
    username: str = "root",
    password: Optional[str] = None,
    private_key_path: Optional[str] = None,
    timeout: int = 30,
) -> SSHConfig:
    """Create an SSH configuration."""
    return SSHConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        private_key_path=private_key_path,
        timeout=timeout,
    )


def create_ssh_client(config: SSHConfig) -> SSHClient:
    """Create an SSH client."""
    return SSHClient(config)


def create_ssh_manager() -> SSHManager:
    """Create an SSH manager."""
    return SSHManager()


def create_shell_config(
    shell: ShellType = ShellType.BASH,
    timeout: int = 60,
    working_directory: Optional[str] = None,
    allowed_commands: Optional[List[str]] = None,
) -> ShellConfig:
    """Create a shell configuration."""
    return ShellConfig(
        shell=shell,
        timeout=timeout,
        working_directory=working_directory,
        allowed_commands=allowed_commands,
    )


def create_shell_executor(config: Optional[ShellConfig] = None) -> ShellExecutor:
    """Create a shell executor."""
    return ShellExecutor(config or ShellConfig())


def create_ssh_tools(manager: SSHManager, shell_config: Optional[ShellConfig] = None) -> Dict[str, BaseTool]:
    """Create all SSH and shell tools."""
    return {
        "create_ssh_client": CreateSSHClientTool(manager),
        "test_ssh_connection": SSHConnectionTestTool(manager),
        "ssh_execute": SSHExecuteTool(manager),
        "ssh_upload": SSHUploadTool(manager),
        "ssh_download": SSHDownloadTool(manager),
        "shell_execute": ShellExecuteTool(shell_config),
    }
