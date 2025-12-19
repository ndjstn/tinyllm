"""Tests for SSH and shell tools."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.tools.ssh import (
    # Enums
    ShellType,
    # Config and dataclasses
    SSHConfig,
    ShellConfig,
    SSHClient,
    ShellExecutor,
    SSHManager,
    CommandResult,
    SSHResult,
    # Input models
    CreateSSHClientInput,
    SSHExecuteInput,
    SSHUploadInput,
    SSHDownloadInput,
    ShellExecuteInput,
    # Output models
    CreateSSHClientOutput,
    SSHCommandOutput,
    SSHTransferOutput,
    ShellCommandOutput,
    SSHSimpleOutput,
    # Tools
    CreateSSHClientTool,
    SSHConnectionTestTool,
    SSHExecuteTool,
    SSHUploadTool,
    SSHDownloadTool,
    ShellExecuteTool,
    # Helper functions
    create_ssh_config,
    create_ssh_client,
    create_ssh_manager,
    create_shell_config,
    create_shell_executor,
    create_ssh_tools,
)


# =============================================================================
# ShellType Tests
# =============================================================================


class TestShellType:
    """Tests for ShellType enum."""

    def test_shell_type_values(self):
        """Test shell type enum values."""
        assert ShellType.BASH == "bash"
        assert ShellType.SH == "sh"
        assert ShellType.ZSH == "zsh"
        assert ShellType.POWERSHELL == "powershell"
        assert ShellType.CMD == "cmd"

    def test_shell_type_count(self):
        """Test all shell types are defined."""
        assert len(ShellType) == 5


# =============================================================================
# SSHConfig Tests
# =============================================================================


class TestSSHConfig:
    """Tests for SSHConfig."""

    def test_config_required_field(self):
        """Test config with required host field."""
        config = SSHConfig(host="example.com")
        assert config.host == "example.com"
        assert config.port == 22
        assert config.username == "root"
        assert config.password is None
        assert config.timeout == 30

    def test_config_all_fields(self):
        """Test config with all fields."""
        config = SSHConfig(
            host="example.com",
            port=2222,
            username="admin",
            password="secret",
            private_key_path="/path/to/key",
            private_key_passphrase="keypass",
            timeout=60,
            known_hosts_policy="strict",
            compress=True,
        )
        assert config.host == "example.com"
        assert config.port == 2222
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.private_key_path == "/path/to/key"
        assert config.compress is True


# =============================================================================
# ShellConfig Tests
# =============================================================================


class TestShellConfig:
    """Tests for ShellConfig."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = ShellConfig()
        assert config.shell == ShellType.BASH
        assert config.timeout == 60
        assert config.working_directory is None
        assert config.allowed_commands is None
        assert len(config.blocked_commands) > 0

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = ShellConfig(
            shell=ShellType.ZSH,
            timeout=120,
            working_directory="/tmp",
            allowed_commands=["ls", "cat", "echo"],
        )
        assert config.shell == ShellType.ZSH
        assert config.timeout == 120
        assert config.working_directory == "/tmp"
        assert "ls" in config.allowed_commands


# =============================================================================
# CommandResult Tests
# =============================================================================


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = CommandResult(
            success=True,
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            duration_ms=50.5,
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"

    def test_error_result(self):
        """Test error result."""
        result = CommandResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Command not found",
        )
        assert result.success is False
        assert result.exit_code == 1
        assert result.stderr == "Command not found"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = CommandResult(
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=100.0,
        )
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["exit_code"] == 0
        assert result_dict["stdout"] == "output"


# =============================================================================
# SSHResult Tests
# =============================================================================


class TestSSHResult:
    """Tests for SSHResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = SSHResult(success=True, data={"connected": True})
        assert result.success is True
        assert result.data == {"connected": True}

    def test_error_result(self):
        """Test error result."""
        result = SSHResult(success=False, error="Connection refused")
        assert result.success is False
        assert result.error == "Connection refused"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = SSHResult(success=True, data={"key": "value"})
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["data"] == {"key": "value"}


# =============================================================================
# SSHClient Tests
# =============================================================================


class TestSSHClient:
    """Tests for SSHClient."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SSHConfig(
            host="example.com",
            port=22,
            username="testuser",
            private_key_path="/path/to/key",
        )

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return SSHClient(config)

    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client._connected is False

    def test_build_ssh_command_basic(self, client):
        """Test building basic SSH command."""
        cmd = client._build_ssh_command("ls -la")
        assert "ssh" in cmd
        assert "-i" in cmd
        assert "/path/to/key" in cmd
        assert "testuser@example.com" in cmd
        assert "ls -la" in cmd

    def test_build_ssh_command_custom_port(self):
        """Test building SSH command with custom port."""
        config = SSHConfig(host="example.com", port=2222)
        client = SSHClient(config)
        cmd = client._build_ssh_command("echo test")
        assert "-p" in cmd
        assert "2222" in cmd

    def test_build_ssh_command_ignore_host_key(self):
        """Test building SSH command with ignored host key."""
        config = SSHConfig(host="example.com", known_hosts_policy="ignore")
        client = SSHClient(config)
        cmd = client._build_ssh_command("echo test")
        assert "StrictHostKeyChecking=no" in " ".join(cmd)

    def test_build_scp_command_upload(self, client):
        """Test building SCP upload command."""
        cmd = client._build_scp_command("/local/file", "/remote/file", upload=True)
        assert "scp" in cmd
        assert "/local/file" in cmd
        assert "testuser@example.com:/remote/file" in cmd

    def test_build_scp_command_download(self, client):
        """Test building SCP download command."""
        cmd = client._build_scp_command("/remote/file", "/local/file", upload=False)
        assert "scp" in cmd
        assert "testuser@example.com:/remote/file" in cmd
        assert "/local/file" in cmd


class TestSSHClientOperations:
    """Tests for SSHClient operations with mocked subprocess."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SSHConfig(host="example.com", username="testuser")

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return SSHClient(config)

    @pytest.mark.asyncio
    async def test_execute_command_success(self, client):
        """Test successful command execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Hello\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.execute_command("echo Hello")

            assert result.success is True
            assert result.exit_code == 0
            assert "Hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_command_failure(self, client):
        """Test failed command execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error\n"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.execute_command("invalid_command")

            assert result.success is False
            assert result.exit_code == 1
            assert "Error" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, client):
        """Test command timeout."""
        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        async def slow_communicate():
            await asyncio.sleep(10)
            return (b"", b"")

        mock_process.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.execute_command("sleep 100", timeout=0.1)

            assert result.success is False
            assert "timed out" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_test_connection_success(self, client):
        """Test successful connection test."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"connection_test\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.test_connection()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, client):
        """Test upload with non-existent file."""
        result = await client.upload_file("/nonexistent/file", "/remote/path")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_upload_file_success(self, client, tmp_path):
        """Test successful file upload."""
        # Create temp file
        local_file = tmp_path / "test.txt"
        local_file.write_text("test content")

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.upload_file(str(local_file), "/remote/test.txt")

            assert result.success is True
            assert result.data["destination"] == "/remote/test.txt"

    @pytest.mark.asyncio
    async def test_download_file_success(self, client):
        """Test successful file download."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await client.download_file("/remote/file", "/local/file")

            assert result.success is True
            assert result.data["downloaded"] == "/remote/file"


# =============================================================================
# ShellExecutor Tests
# =============================================================================


class TestShellExecutor:
    """Tests for ShellExecutor."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ShellConfig(shell=ShellType.BASH, timeout=30)

    @pytest.fixture
    def executor(self, config):
        """Create test executor."""
        return ShellExecutor(config)

    def test_executor_initialization(self, executor, config):
        """Test executor initialization."""
        assert executor.config == config

    def test_is_command_allowed_blocked(self):
        """Test blocked command detection."""
        config = ShellConfig()
        executor = ShellExecutor(config)

        allowed, reason = executor._is_command_allowed("rm -rf /")
        assert allowed is False
        assert "blocked" in reason.lower()

    def test_is_command_allowed_whitelist(self):
        """Test whitelist command checking."""
        config = ShellConfig(allowed_commands=["ls", "cat"])
        executor = ShellExecutor(config)

        allowed, _ = executor._is_command_allowed("ls -la")
        assert allowed is True

        allowed, reason = executor._is_command_allowed("rm file.txt")
        assert allowed is False
        assert "not in allowed" in reason.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Test successful command execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"output\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await executor.execute("echo output")

            assert result.success is True
            assert "output" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self, executor):
        """Test executing blocked command."""
        result = await executor.execute("rm -rf /")

        assert result.success is False
        assert "blocked" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_with_working_directory(self):
        """Test execution with working directory."""
        config = ShellConfig()
        executor = ShellExecutor(config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"/tmp\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await executor.execute("pwd", working_directory="/tmp")

            # Verify cwd was passed
            call_kwargs = mock_exec.call_args[1]
            assert call_kwargs.get("cwd") == "/tmp"


# =============================================================================
# SSHManager Tests
# =============================================================================


class TestSSHManager:
    """Tests for SSHManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = SSHManager()
        assert manager._clients == {}

    def test_add_client(self):
        """Test adding a client."""
        manager = SSHManager()
        config = SSHConfig(host="example.com")
        client = SSHClient(config)

        manager.add_client("server1", client)

        assert "server1" in manager._clients
        assert manager.get_client("server1") == client

    def test_get_nonexistent_client(self):
        """Test getting a nonexistent client."""
        manager = SSHManager()
        assert manager.get_client("nonexistent") is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = SSHManager()
        config = SSHConfig(host="example.com")
        client = SSHClient(config)
        manager.add_client("test", client)

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None

    def test_remove_nonexistent_client(self):
        """Test removing a nonexistent client."""
        manager = SSHManager()
        result = manager.remove_client("nonexistent")
        assert result is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = SSHManager()
        config = SSHConfig(host="example.com")

        manager.add_client("server1", SSHClient(config))
        manager.add_client("server2", SSHClient(config))

        clients = manager.list_clients()

        assert "server1" in clients
        assert "server2" in clients
        assert len(clients) == 2


# =============================================================================
# Input/Output Models Tests
# =============================================================================


class TestInputModels:
    """Tests for input models."""

    def test_create_ssh_client_input(self):
        """Test CreateSSHClientInput."""
        input_model = CreateSSHClientInput(
            name="server1",
            host="example.com",
            port=2222,
            username="admin",
        )
        assert input_model.name == "server1"
        assert input_model.host == "example.com"
        assert input_model.port == 2222

    def test_ssh_execute_input(self):
        """Test SSHExecuteInput."""
        input_model = SSHExecuteInput(
            client="server1",
            command="ls -la",
            timeout=60,
        )
        assert input_model.client == "server1"
        assert input_model.command == "ls -la"
        assert input_model.timeout == 60

    def test_ssh_upload_input(self):
        """Test SSHUploadInput."""
        input_model = SSHUploadInput(
            local_path="/local/file",
            remote_path="/remote/file",
        )
        assert input_model.local_path == "/local/file"
        assert input_model.remote_path == "/remote/file"

    def test_shell_execute_input(self):
        """Test ShellExecuteInput."""
        input_model = ShellExecuteInput(
            command="echo hello",
            timeout=30,
            shell=ShellType.ZSH,
        )
        assert input_model.command == "echo hello"
        assert input_model.shell == ShellType.ZSH


class TestOutputModels:
    """Tests for output models."""

    def test_ssh_command_output(self):
        """Test SSHCommandOutput."""
        output = SSHCommandOutput(
            success=True,
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=50.0,
        )
        assert output.success is True
        assert output.stdout == "output"

    def test_ssh_transfer_output(self):
        """Test SSHTransferOutput."""
        output = SSHTransferOutput(
            success=True,
            source="/local/file",
            destination="/remote/file",
        )
        assert output.success is True
        assert output.source == "/local/file"

    def test_shell_command_output(self):
        """Test ShellCommandOutput."""
        output = ShellCommandOutput(
            success=True,
            exit_code=0,
            stdout="hello",
        )
        assert output.success is True
        assert output.stdout == "hello"


# =============================================================================
# Tool Tests
# =============================================================================


class TestCreateSSHClientTool:
    """Tests for CreateSSHClientTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager."""
        return SSHManager()

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return CreateSSHClientTool(manager)

    @pytest.mark.asyncio
    async def test_create_client(self, tool, manager):
        """Test creating a client."""
        input_model = CreateSSHClientInput(
            name="server1",
            host="example.com",
            username="admin",
        )

        result = await tool.execute(input_model)

        assert result.success is True
        assert result.name == "server1"
        assert manager.get_client("server1") is not None

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "create_ssh_client"
        assert tool.metadata.category == "utility"


class TestSSHExecuteTool:
    """Tests for SSHExecuteTool."""

    @pytest.fixture
    def manager(self):
        """Create test manager with mock client."""
        manager = SSHManager()
        config = SSHConfig(host="example.com")
        manager.add_client("default", SSHClient(config))
        return manager

    @pytest.fixture
    def tool(self, manager):
        """Create test tool."""
        return SSHExecuteTool(manager)

    @pytest.mark.asyncio
    async def test_execute_client_not_found(self, tool):
        """Test execution with nonexistent client."""
        input_model = SSHExecuteInput(client="nonexistent", command="ls")
        result = await tool.execute(input_model)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"output\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            input_model = SSHExecuteInput(client="default", command="echo output")
            result = await tool.execute(input_model)

            assert result.success is True
            assert "output" in result.stdout

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "ssh_execute"
        assert tool.metadata.category == "execution"


class TestShellExecuteTool:
    """Tests for ShellExecuteTool."""

    @pytest.fixture
    def tool(self):
        """Create test tool."""
        return ShellExecuteTool()

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful execution."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"hello\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            input_model = ShellExecuteInput(command="echo hello")
            result = await tool.execute(input_model)

            assert result.success is True
            assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self, tool):
        """Test blocked command execution."""
        input_model = ShellExecuteInput(command="rm -rf /")
        result = await tool.execute(input_model)

        assert result.success is False
        assert "blocked" in result.stderr.lower()

    def test_tool_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "shell_execute"
        assert tool.metadata.category == "execution"


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_ssh_config(self):
        """Test creating SSH config."""
        config = create_ssh_config(
            host="example.com",
            port=2222,
            username="admin",
            private_key_path="/path/to/key",
        )

        assert config.host == "example.com"
        assert config.port == 2222
        assert config.username == "admin"
        assert config.private_key_path == "/path/to/key"

    def test_create_ssh_client(self):
        """Test creating SSH client."""
        config = SSHConfig(host="example.com")
        client = create_ssh_client(config)

        assert isinstance(client, SSHClient)
        assert client.config == config

    def test_create_ssh_manager(self):
        """Test creating SSH manager."""
        manager = create_ssh_manager()

        assert isinstance(manager, SSHManager)
        assert manager._clients == {}

    def test_create_shell_config(self):
        """Test creating shell config."""
        config = create_shell_config(
            shell=ShellType.ZSH,
            timeout=120,
            working_directory="/tmp",
        )

        assert config.shell == ShellType.ZSH
        assert config.timeout == 120
        assert config.working_directory == "/tmp"

    def test_create_shell_executor(self):
        """Test creating shell executor."""
        executor = create_shell_executor()

        assert isinstance(executor, ShellExecutor)

    def test_create_ssh_tools(self):
        """Test creating SSH tools."""
        manager = SSHManager()
        tools = create_ssh_tools(manager)

        assert isinstance(tools, dict)
        assert "create_ssh_client" in tools
        assert "test_ssh_connection" in tools
        assert "ssh_execute" in tools
        assert "ssh_upload" in tools
        assert "ssh_download" in tools
        assert "shell_execute" in tools
        assert len(tools) == 6

    def test_all_tools_have_correct_manager(self):
        """Test all SSH tools reference the same manager."""
        manager = SSHManager()
        tools = create_ssh_tools(manager)

        for tool_name in ["create_ssh_client", "test_ssh_connection", "ssh_execute", "ssh_upload", "ssh_download"]:
            assert tools[tool_name].manager is manager
