"""Tests for filesystem tools."""

import os
import tempfile
from pathlib import Path

import pytest

from tinyllm.tools.filesystem import (
    CopyFileInput,
    CopyFileTool,
    CreateDirectoryInput,
    CreateDirectoryTool,
    DeleteFileInput,
    DeleteFileTool,
    FileInfoInput,
    FileInfoTool,
    FileSystemConfig,
    FileType,
    ListDirectoryInput,
    ListDirectoryTool,
    MoveFileInput,
    MoveFileTool,
    ReadFileInput,
    ReadFileTool,
    SearchFilesInput,
    SearchFilesTool,
    WriteFileInput,
    WriteFileTool,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    """Create config with temp directory allowed."""
    return FileSystemConfig(
        allowed_directories=[str(temp_dir)],
        require_confirmation_for_delete=False,  # Disable for tests
    )


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.mark.asyncio
    async def test_read_simple_file(self, temp_dir, config):
        """Should read a simple text file."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")

        tool = ReadFileTool(config)
        result = await tool.execute(ReadFileInput(path=str(test_file)))

        assert result.success is True
        assert result.content == "Hello, World!"
        assert result.encoding == "utf-8"
        assert result.size == 13
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_read_with_encoding(self, temp_dir, config):
        """Should read file with specified encoding."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test", encoding="utf-16")

        tool = ReadFileTool(config)
        result = await tool.execute(
            ReadFileInput(path=str(test_file), encoding="utf-16")
        )

        assert result.success is True
        assert result.content == "Test"

    @pytest.mark.asyncio
    async def test_read_with_max_bytes(self, temp_dir, config):
        """Should limit bytes read."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("0123456789" * 10, encoding="utf-8")

        tool = ReadFileTool(config)
        result = await tool.execute(
            ReadFileInput(path=str(test_file), max_bytes=10)
        )

        assert result.success is True
        assert len(result.content) == 10
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, temp_dir, config):
        """Should fail for nonexistent file."""
        tool = ReadFileTool(config)
        result = await tool.execute(
            ReadFileInput(path=str(temp_dir / "nonexistent.txt"))
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, temp_dir, config):
        """Should fail when trying to read a directory."""
        tool = ReadFileTool(config)
        result = await tool.execute(ReadFileInput(path=str(temp_dir)))

        assert result.success is False
        assert "not a file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_outside_allowed_dirs(self, temp_dir, config):
        """Should deny access outside allowed directories."""
        tool = ReadFileTool(config)
        result = await tool.execute(ReadFileInput(path="/etc/passwd"))

        assert result.success is False
        assert "access denied" in result.error.lower()


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    @pytest.mark.asyncio
    async def test_write_simple_file(self, temp_dir, config):
        """Should write a simple text file."""
        test_file = temp_dir / "output.txt"

        tool = WriteFileTool(config)
        result = await tool.execute(
            WriteFileInput(path=str(test_file), content="Hello, File!")
        )

        assert result.success is True
        assert result.path == str(test_file)
        assert result.bytes_written == 12
        assert test_file.read_text() == "Hello, File!"

    @pytest.mark.asyncio
    async def test_write_with_backup(self, temp_dir, config):
        """Should create backup of existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Original")

        tool = WriteFileTool(config)
        result = await tool.execute(
            WriteFileInput(
                path=str(test_file),
                content="Updated",
                create_backup=True,
            )
        )

        assert result.success is True
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()
        assert Path(result.backup_path).read_text() == "Original"
        assert test_file.read_text() == "Updated"

    @pytest.mark.asyncio
    async def test_write_append_mode(self, temp_dir, config):
        """Should append to existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Line 1\n")

        tool = WriteFileTool(config)
        result = await tool.execute(
            WriteFileInput(
                path=str(test_file),
                content="Line 2\n",
                append=True,
            )
        )

        assert result.success is True
        assert test_file.read_text() == "Line 1\nLine 2\n"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, temp_dir, config):
        """Should create parent directories."""
        test_file = temp_dir / "subdir" / "nested" / "test.txt"

        tool = WriteFileTool(config)
        result = await tool.execute(
            WriteFileInput(path=str(test_file), content="Test")
        )

        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == "Test"

    @pytest.mark.asyncio
    async def test_write_outside_allowed_dirs(self, temp_dir, config):
        """Should deny access outside allowed directories."""
        tool = WriteFileTool(config)
        result = await tool.execute(
            WriteFileInput(path="/tmp/test.txt", content="Test")
        )

        assert result.success is False
        assert "access denied" in result.error.lower()


class TestListDirectoryTool:
    """Tests for ListDirectoryTool."""

    @pytest.mark.asyncio
    async def test_list_directory(self, temp_dir, config):
        """Should list directory contents."""
        # Create test files
        (temp_dir / "file1.txt").write_text("test")
        (temp_dir / "file2.txt").write_text("test")
        (temp_dir / "subdir").mkdir()

        tool = ListDirectoryTool(config)
        result = await tool.execute(ListDirectoryInput(path=str(temp_dir)))

        assert result.success is True
        assert result.directory is not None
        assert result.directory.file_count == 2
        assert result.directory.directory_count == 1
        assert "file1.txt" in result.directory.files
        assert "file2.txt" in result.directory.files
        assert "subdir" in result.directory.directories

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, temp_dir, config):
        """Should filter by pattern."""
        (temp_dir / "test1.txt").write_text("test")
        (temp_dir / "test2.txt").write_text("test")
        (temp_dir / "other.py").write_text("test")

        tool = ListDirectoryTool(config)
        result = await tool.execute(
            ListDirectoryInput(path=str(temp_dir), pattern="*.txt")
        )

        assert result.success is True
        assert result.directory.file_count == 2
        assert len(result.entries) == 2

    @pytest.mark.asyncio
    async def test_list_recursive(self, temp_dir, config):
        """Should list recursively."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "file1.txt").write_text("test")
        (subdir / "file2.txt").write_text("test")

        tool = ListDirectoryTool(config)
        result = await tool.execute(
            ListDirectoryInput(path=str(temp_dir), recursive=True)
        )

        assert result.success is True
        # Should include files from subdirectory
        file_paths = [e.path for e in result.entries if e.type == FileType.FILE]
        assert any("file2.txt" in p for p in file_paths)

    @pytest.mark.asyncio
    async def test_list_hidden_files(self, temp_dir, config):
        """Should include/exclude hidden files."""
        (temp_dir / ".hidden").write_text("test")
        (temp_dir / "visible.txt").write_text("test")

        tool = ListDirectoryTool(config)

        # Without hidden
        result = await tool.execute(
            ListDirectoryInput(path=str(temp_dir), include_hidden=False)
        )
        assert result.directory.file_count == 1

        # With hidden
        result = await tool.execute(
            ListDirectoryInput(path=str(temp_dir), include_hidden=True)
        )
        assert result.directory.file_count == 2

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, temp_dir, config):
        """Should fail for nonexistent directory."""
        tool = ListDirectoryTool(config)
        result = await tool.execute(
            ListDirectoryInput(path=str(temp_dir / "nonexistent"))
        )

        assert result.success is False
        assert "not found" in result.error.lower()


class TestSearchFilesTool:
    """Tests for SearchFilesTool."""

    @pytest.mark.asyncio
    async def test_search_simple_pattern(self, temp_dir, config):
        """Should search for files matching pattern."""
        (temp_dir / "test1.txt").write_text("test")
        (temp_dir / "test2.txt").write_text("test")
        (temp_dir / "other.py").write_text("test")

        tool = SearchFilesTool(config)
        result = await tool.execute(
            SearchFilesInput(directory=str(temp_dir), pattern="*.txt")
        )

        assert result.success is True
        assert result.total_matches == 2

    @pytest.mark.asyncio
    async def test_search_recursive(self, temp_dir, config):
        """Should search recursively."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (temp_dir / "test1.txt").write_text("test")
        (subdir / "test2.txt").write_text("test")

        tool = SearchFilesTool(config)
        result = await tool.execute(
            SearchFilesInput(
                directory=str(temp_dir),
                pattern="*.txt",
                recursive=True,
            )
        )

        assert result.success is True
        assert result.total_matches == 2

    @pytest.mark.asyncio
    async def test_search_max_results(self, temp_dir, config):
        """Should limit results."""
        for i in range(10):
            (temp_dir / f"file{i}.txt").write_text("test")

        tool = SearchFilesTool(config)
        result = await tool.execute(
            SearchFilesInput(
                directory=str(temp_dir),
                pattern="*.txt",
                max_results=5,
            )
        )

        assert result.success is True
        assert len(result.matches) == 5
        assert result.truncated is True


class TestCreateDirectoryTool:
    """Tests for CreateDirectoryTool."""

    @pytest.mark.asyncio
    async def test_create_directory(self, temp_dir, config):
        """Should create directory."""
        new_dir = temp_dir / "newdir"

        tool = CreateDirectoryTool(config)
        result = await tool.execute(CreateDirectoryInput(path=str(new_dir)))

        assert result.success is True
        assert result.created is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.asyncio
    async def test_create_with_parents(self, temp_dir, config):
        """Should create parent directories."""
        new_dir = temp_dir / "parent" / "child" / "grandchild"

        tool = CreateDirectoryTool(config)
        result = await tool.execute(
            CreateDirectoryInput(path=str(new_dir), parents=True)
        )

        assert result.success is True
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_create_existing_directory(self, temp_dir, config):
        """Should handle existing directory."""
        new_dir = temp_dir / "existing"
        new_dir.mkdir()

        tool = CreateDirectoryTool(config)
        result = await tool.execute(
            CreateDirectoryInput(path=str(new_dir), exist_ok=True)
        )

        assert result.success is True
        assert result.created is False


class TestDeleteFileTool:
    """Tests for DeleteFileTool."""

    @pytest.mark.asyncio
    async def test_delete_file(self, temp_dir, config):
        """Should delete file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        tool = DeleteFileTool(config)
        result = await tool.execute(
            DeleteFileInput(path=str(test_file), confirm=True)
        )

        assert result.success is True
        assert result.deleted is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_directory(self, temp_dir, config):
        """Should delete directory recursively."""
        test_dir = temp_dir / "testdir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        tool = DeleteFileTool(config)
        result = await tool.execute(
            DeleteFileInput(path=str(test_dir), confirm=True, recursive=True)
        )

        assert result.success is True
        assert not test_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, temp_dir):
        """Should require confirmation when configured."""
        config = FileSystemConfig(
            allowed_directories=[str(temp_dir)],
            require_confirmation_for_delete=True,
        )
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        tool = DeleteFileTool(config)
        result = await tool.execute(DeleteFileInput(path=str(test_file)))

        assert result.success is False
        assert "confirmation" in result.error.lower()
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, temp_dir, config):
        """Should fail for nonexistent file."""
        tool = DeleteFileTool(config)
        result = await tool.execute(
            DeleteFileInput(path=str(temp_dir / "nonexistent.txt"), confirm=True)
        )

        assert result.success is False
        assert "not found" in result.error.lower()


class TestMoveFileTool:
    """Tests for MoveFileTool."""

    @pytest.mark.asyncio
    async def test_move_file(self, temp_dir, config):
        """Should move file."""
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        source.write_text("test")

        tool = MoveFileTool(config)
        result = await tool.execute(
            MoveFileInput(source=str(source), destination=str(dest))
        )

        assert result.success is True
        assert not source.exists()
        assert dest.exists()
        assert dest.read_text() == "test"

    @pytest.mark.asyncio
    async def test_rename_file(self, temp_dir, config):
        """Should rename file (same as move)."""
        source = temp_dir / "old_name.txt"
        dest = temp_dir / "new_name.txt"
        source.write_text("test")

        tool = MoveFileTool(config)
        result = await tool.execute(
            MoveFileInput(source=str(source), destination=str(dest))
        )

        assert result.success is True
        assert not source.exists()
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_move_with_overwrite(self, temp_dir, config):
        """Should overwrite destination."""
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        source.write_text("new")
        dest.write_text("old")

        tool = MoveFileTool(config)
        result = await tool.execute(
            MoveFileInput(source=str(source), destination=str(dest), overwrite=True)
        )

        assert result.success is True
        assert dest.read_text() == "new"

    @pytest.mark.asyncio
    async def test_move_without_overwrite(self, temp_dir, config):
        """Should fail when destination exists."""
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        source.write_text("new")
        dest.write_text("old")

        tool = MoveFileTool(config)
        result = await tool.execute(
            MoveFileInput(source=str(source), destination=str(dest), overwrite=False)
        )

        assert result.success is False
        assert "exists" in result.error.lower()


class TestCopyFileTool:
    """Tests for CopyFileTool."""

    @pytest.mark.asyncio
    async def test_copy_file(self, temp_dir, config):
        """Should copy file."""
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        source.write_text("test content")

        tool = CopyFileTool(config)
        result = await tool.execute(
            CopyFileInput(source=str(source), destination=str(dest))
        )

        assert result.success is True
        assert source.exists()
        assert dest.exists()
        assert dest.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_copy_directory(self, temp_dir, config):
        """Should copy directory recursively."""
        source = temp_dir / "source_dir"
        dest = temp_dir / "dest_dir"
        source.mkdir()
        (source / "file.txt").write_text("test")

        tool = CopyFileTool(config)
        result = await tool.execute(
            CopyFileInput(
                source=str(source),
                destination=str(dest),
                recursive=True,
            )
        )

        assert result.success is True
        assert source.exists()
        assert dest.exists()
        assert (dest / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_copy_directory_without_recursive(self, temp_dir, config):
        """Should fail to copy directory without recursive flag."""
        source = temp_dir / "source_dir"
        dest = temp_dir / "dest_dir"
        source.mkdir()

        tool = CopyFileTool(config)
        result = await tool.execute(
            CopyFileInput(source=str(source), destination=str(dest))
        )

        assert result.success is False
        assert "recursive" in result.error.lower()


class TestFileInfoTool:
    """Tests for FileInfoTool."""

    @pytest.mark.asyncio
    async def test_get_file_info(self, temp_dir, config):
        """Should get file information."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        tool = FileInfoTool(config)
        result = await tool.execute(FileInfoInput(path=str(test_file)))

        assert result.success is True
        assert result.info is not None
        assert result.info.name == "test.txt"
        assert result.info.size == 12
        assert result.info.type == FileType.FILE
        assert result.info.is_readable is True
        assert result.info.permissions is not None

    @pytest.mark.asyncio
    async def test_get_directory_info(self, temp_dir, config):
        """Should get directory information."""
        test_dir = temp_dir / "testdir"
        test_dir.mkdir()

        tool = FileInfoTool(config)
        result = await tool.execute(FileInfoInput(path=str(test_dir)))

        assert result.success is True
        assert result.info is not None
        assert result.info.type == FileType.DIRECTORY

    @pytest.mark.asyncio
    async def test_get_symlink_info(self, temp_dir, config):
        """Should get symlink information."""
        test_file = temp_dir / "test.txt"
        test_link = temp_dir / "link.txt"
        test_file.write_text("test")
        test_link.symlink_to(test_file)

        tool = FileInfoTool(config)
        result = await tool.execute(FileInfoInput(path=str(test_link)))

        assert result.success is True
        assert result.info is not None
        assert result.info.type == FileType.SYMLINK
        assert result.info.symlink_target is not None

    @pytest.mark.asyncio
    async def test_file_info_nonexistent(self, temp_dir, config):
        """Should fail for nonexistent file."""
        tool = FileInfoTool(config)
        result = await tool.execute(
            FileInfoInput(path=str(temp_dir / "nonexistent.txt"))
        )

        assert result.success is False
        assert "not found" in result.error.lower()


class TestFileSystemConfig:
    """Tests for FileSystemConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = FileSystemConfig()
        assert config.allowed_directories is None
        assert config.max_file_size == 100 * 1024 * 1024
        assert config.follow_symlinks is False
        assert config.require_confirmation_for_delete is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = FileSystemConfig(
            allowed_directories=["/tmp"],
            max_file_size=1024,
            follow_symlinks=True,
            require_confirmation_for_delete=False,
        )
        assert config.allowed_directories == ["/tmp"]
        assert config.max_file_size == 1024
        assert config.follow_symlinks is True
        assert config.require_confirmation_for_delete is False


class TestSecurityFeatures:
    """Tests for security features."""

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, temp_dir):
        """Should prevent path traversal attacks."""
        config = FileSystemConfig(allowed_directories=[str(temp_dir / "allowed")])
        (temp_dir / "allowed").mkdir()

        tool = ReadFileTool(config)
        # Try to access parent directory
        result = await tool.execute(
            ReadFileInput(path=str(temp_dir / "allowed" / ".." / "secret.txt"))
        )

        assert result.success is False
        assert "access denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_file_size_limit(self, temp_dir, config):
        """Should enforce file size limits."""
        # Create config with small size limit
        small_config = FileSystemConfig(
            allowed_directories=[str(temp_dir)],
            max_file_size=10,
        )

        test_file = temp_dir / "large.txt"
        test_file.write_text("x" * 100)

        tool = ReadFileTool(small_config)
        result = await tool.execute(ReadFileInput(path=str(test_file)))

        assert result.success is False
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_max_search_depth(self, temp_dir, config):
        """Should respect max search depth."""
        # Create nested directories beyond limit
        deep_dir = temp_dir
        for i in range(15):
            deep_dir = deep_dir / f"level{i}"
            deep_dir.mkdir()

        (deep_dir / "deep_file.txt").write_text("test")

        # Config with shallow depth
        shallow_config = FileSystemConfig(
            allowed_directories=[str(temp_dir)],
            max_search_depth=5,
        )

        tool = SearchFilesTool(shallow_config)
        result = await tool.execute(
            SearchFilesInput(
                directory=str(temp_dir),
                pattern="*.txt",
                recursive=True,
            )
        )

        # Should not find the deeply nested file
        assert result.success is True
        assert result.total_matches == 0


class TestToolMetadata:
    """Tests for tool metadata."""

    def test_read_file_metadata(self):
        """Should have correct metadata."""
        tool = ReadFileTool()
        assert tool.metadata.id == "read_file"
        assert tool.metadata.category == "utility"
        assert tool.metadata.sandbox_required is False

    def test_write_file_metadata(self):
        """Should have correct metadata."""
        tool = WriteFileTool()
        assert tool.metadata.id == "write_file"
        assert tool.metadata.category == "utility"

    def test_all_tools_have_metadata(self):
        """All tools should have required metadata."""
        tools = [
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            SearchFilesTool(),
            CreateDirectoryTool(),
            DeleteFileTool(),
            MoveFileTool(),
            CopyFileTool(),
            FileInfoTool(),
        ]

        for tool in tools:
            assert tool.metadata.id is not None
            assert tool.metadata.name is not None
            assert tool.metadata.description is not None
            assert tool.metadata.category in [
                "computation",
                "execution",
                "search",
                "memory",
                "utility",
            ]
