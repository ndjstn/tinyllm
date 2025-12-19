"""Filesystem tools for safe file operations.

This module provides a comprehensive set of filesystem tools with security guards:
- Path traversal prevention
- Allowed directories configuration
- File size limits
- Dangerous operation confirmation
- Symbolic link handling
- Glob pattern matching
- File type detection
"""

import asyncio
import os
import pathlib
import shutil
import stat
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import chardet
from pydantic import BaseModel, Field, field_validator

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


class FileType(str, Enum):
    """File type enumeration."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    OTHER = "other"


class FileInfo(BaseModel):
    """Information about a file."""

    path: str = Field(description="Absolute path to the file")
    name: str = Field(description="File name")
    size: int = Field(description="File size in bytes")
    type: FileType = Field(description="File type")
    is_readable: bool = Field(description="Whether file is readable")
    is_writable: bool = Field(description="Whether file is writable")
    is_executable: bool = Field(description="Whether file is executable")
    created_time: float = Field(description="Creation timestamp")
    modified_time: float = Field(description="Last modification timestamp")
    accessed_time: float = Field(description="Last access timestamp")
    permissions: str = Field(description="File permissions in octal format")
    owner: Optional[str] = Field(default=None, description="File owner")
    mime_type: Optional[str] = Field(default=None, description="MIME type of file")
    symlink_target: Optional[str] = Field(default=None, description="Target if symlink")


class DirectoryInfo(BaseModel):
    """Information about a directory."""

    path: str = Field(description="Absolute path to directory")
    name: str = Field(description="Directory name")
    file_count: int = Field(description="Number of files in directory")
    directory_count: int = Field(description="Number of subdirectories")
    total_size: int = Field(description="Total size of all files in bytes")
    files: list[str] = Field(default_factory=list, description="List of file names")
    directories: list[str] = Field(default_factory=list, description="List of subdirectory names")


class FileSystemConfig(ToolConfig):
    """Configuration for filesystem tools."""

    allowed_directories: Optional[list[str]] = Field(
        default=None,
        description="List of allowed directories. If None, all directories are allowed.",
    )
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100 MB
        ge=1024,
        le=1024 * 1024 * 1024,
        description="Maximum file size in bytes",
    )
    follow_symlinks: bool = Field(
        default=False,
        description="Whether to follow symbolic links",
    )
    require_confirmation_for_delete: bool = Field(
        default=True,
        description="Whether to require confirmation for delete operations",
    )
    allow_hidden_files: bool = Field(
        default=True,
        description="Whether to allow operations on hidden files",
    )
    max_search_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum depth for recursive operations",
    )
    encoding_detection_bytes: int = Field(
        default=10000,
        ge=1024,
        le=100000,
        description="Number of bytes to read for encoding detection",
    )


class FileSystemResult(BaseModel):
    """Base result for filesystem operations."""

    success: bool
    error: Optional[str] = None


# Read File Tool


class ReadFileInput(BaseModel):
    """Input for read file tool."""

    path: str = Field(description="Path to file to read")
    encoding: Optional[str] = Field(
        default=None,
        description="File encoding (auto-detect if None)",
    )
    max_bytes: Optional[int] = Field(
        default=None,
        ge=1,
        le=100 * 1024 * 1024,
        description="Maximum bytes to read (None for entire file)",
    )


class ReadFileOutput(FileSystemResult):
    """Output from read file tool."""

    content: Optional[str] = None
    encoding: Optional[str] = None
    size: int = 0
    truncated: bool = False


class ReadFileTool(BaseTool[ReadFileInput, ReadFileOutput]):
    """Read file contents with encoding detection."""

    metadata = ToolMetadata(
        id="read_file",
        name="Read File",
        description="Read text file contents with automatic encoding detection. "
        "Supports limiting bytes read for large files.",
        category="utility",
        sandbox_required=False,
    )
    input_type = ReadFileInput
    output_type = ReadFileOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: ReadFileInput) -> ReadFileOutput:
        """Read file contents."""
        try:
            path = Path(input.path).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return ReadFileOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            if not path.exists():
                return ReadFileOutput(success=False, error=f"File not found: {path}")

            if not path.is_file():
                return ReadFileOutput(success=False, error=f"Not a file: {path}")

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.fs_config.max_file_size:
                return ReadFileOutput(
                    success=False,
                    error=f"File too large: {file_size} bytes (max: {self.fs_config.max_file_size})",
                )

            # Detect encoding if not specified
            encoding = input.encoding
            if encoding is None:
                encoding = await self._detect_encoding(path)

            # Read file
            max_bytes = input.max_bytes or file_size
            truncated = max_bytes < file_size

            content = await asyncio.to_thread(
                self._read_file_sync, path, encoding, max_bytes
            )

            return ReadFileOutput(
                success=True,
                content=content,
                encoding=encoding,
                size=file_size,
                truncated=truncated,
            )

        except UnicodeDecodeError as e:
            return ReadFileOutput(
                success=False,
                error=f"Encoding error: {e}. Try specifying encoding explicitly.",
            )
        except Exception as e:
            return ReadFileOutput(success=False, error=f"Read error: {e}")

    def _read_file_sync(self, path: Path, encoding: str, max_bytes: int) -> str:
        """Read file synchronously (for thread execution)."""
        with open(path, "r", encoding=encoding) as f:
            if max_bytes:
                return f.read(max_bytes)
            return f.read()

    async def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
        try:
            # Read sample bytes
            sample_size = min(
                self.fs_config.encoding_detection_bytes,
                path.stat().st_size,
            )
            with open(path, "rb") as f:
                sample = f.read(sample_size)

            result = chardet.detect(sample)
            return result["encoding"] or "utf-8"
        except Exception:
            return "utf-8"

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Write File Tool


class WriteFileInput(BaseModel):
    """Input for write file tool."""

    path: str = Field(description="Path to file to write")
    content: str = Field(description="Content to write to file")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_backup: bool = Field(
        default=False,
        description="Create backup if file exists",
    )
    append: bool = Field(
        default=False,
        description="Append to file instead of overwriting",
    )


class WriteFileOutput(FileSystemResult):
    """Output from write file tool."""

    path: Optional[str] = None
    bytes_written: int = 0
    backup_path: Optional[str] = None


class WriteFileTool(BaseTool[WriteFileInput, WriteFileOutput]):
    """Write content to file with backup option."""

    metadata = ToolMetadata(
        id="write_file",
        name="Write File",
        description="Write content to a file. Supports creating backups and appending.",
        category="utility",
        sandbox_required=False,
    )
    input_type = WriteFileInput
    output_type = WriteFileOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: WriteFileInput) -> WriteFileOutput:
        """Write content to file."""
        try:
            path = Path(input.path).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return WriteFileOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            # Check content size
            content_size = len(input.content.encode(input.encoding))
            if content_size > self.fs_config.max_file_size:
                return WriteFileOutput(
                    success=False,
                    error=f"Content too large: {content_size} bytes (max: {self.fs_config.max_file_size})",
                )

            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested and file exists
            backup_path = None
            if input.create_backup and path.exists():
                backup_path = path.with_suffix(path.suffix + ".bak")
                await asyncio.to_thread(shutil.copy2, path, backup_path)

            # Write file
            mode = "a" if input.append else "w"
            bytes_written = await asyncio.to_thread(
                self._write_file_sync, path, input.content, input.encoding, mode
            )

            return WriteFileOutput(
                success=True,
                path=str(path),
                bytes_written=bytes_written,
                backup_path=str(backup_path) if backup_path else None,
            )

        except Exception as e:
            return WriteFileOutput(success=False, error=f"Write error: {e}")

    def _write_file_sync(
        self, path: Path, content: str, encoding: str, mode: str
    ) -> int:
        """Write file synchronously (for thread execution)."""
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
            return len(content.encode(encoding))

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# List Directory Tool


class ListDirectoryInput(BaseModel):
    """Input for list directory tool."""

    path: str = Field(description="Path to directory to list")
    recursive: bool = Field(default=False, description="List recursively")
    include_hidden: bool = Field(default=False, description="Include hidden files")
    pattern: Optional[str] = Field(default=None, description="Glob pattern to filter files")


class ListDirectoryOutput(FileSystemResult):
    """Output from list directory tool."""

    directory: Optional[DirectoryInfo] = None
    entries: list[FileInfo] = Field(default_factory=list)


class ListDirectoryTool(BaseTool[ListDirectoryInput, ListDirectoryOutput]):
    """List directory contents."""

    metadata = ToolMetadata(
        id="list_directory",
        name="List Directory",
        description="List directory contents with optional recursion and filtering.",
        category="utility",
        sandbox_required=False,
    )
    input_type = ListDirectoryInput
    output_type = ListDirectoryOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: ListDirectoryInput) -> ListDirectoryOutput:
        """List directory contents."""
        try:
            path = Path(input.path).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return ListDirectoryOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            if not path.exists():
                return ListDirectoryOutput(success=False, error=f"Directory not found: {path}")

            if not path.is_dir():
                return ListDirectoryOutput(success=False, error=f"Not a directory: {path}")

            # List entries
            entries: list[FileInfo] = []
            files: list[str] = []
            directories: list[str] = []
            total_size = 0

            if input.recursive:
                pattern = input.pattern or "**/*"
                paths = path.glob(pattern)
            else:
                pattern = input.pattern or "*"
                paths = path.glob(pattern)

            for entry_path in paths:
                # Skip hidden files if not included
                if not input.include_hidden and entry_path.name.startswith("."):
                    continue

                file_info = await self._get_file_info(entry_path)
                entries.append(file_info)

                if file_info.type == FileType.FILE:
                    files.append(file_info.name)
                    total_size += file_info.size
                elif file_info.type == FileType.DIRECTORY:
                    directories.append(file_info.name)

            directory_info = DirectoryInfo(
                path=str(path),
                name=path.name,
                file_count=len(files),
                directory_count=len(directories),
                total_size=total_size,
                files=sorted(files),
                directories=sorted(directories),
            )

            return ListDirectoryOutput(
                success=True,
                directory=directory_info,
                entries=entries,
            )

        except Exception as e:
            return ListDirectoryOutput(success=False, error=f"List error: {e}")

    async def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat_info = path.stat()

        # Determine file type
        if path.is_symlink():
            file_type = FileType.SYMLINK
            symlink_target = str(path.readlink())
        elif path.is_file():
            file_type = FileType.FILE
            symlink_target = None
        elif path.is_dir():
            file_type = FileType.DIRECTORY
            symlink_target = None
        else:
            file_type = FileType.OTHER
            symlink_target = None

        # Get permissions
        permissions = oct(stat_info.st_mode)[-3:]

        # Try to get owner (may not work on Windows)
        owner = None
        try:
            import pwd

            owner = pwd.getpwuid(stat_info.st_uid).pw_name
        except (ImportError, KeyError):
            pass

        # Detect MIME type for files
        mime_type = None
        if file_type == FileType.FILE:
            mime_type = self._detect_mime_type(path)

        return FileInfo(
            path=str(path),
            name=path.name,
            size=stat_info.st_size,
            type=file_type,
            is_readable=os.access(path, os.R_OK),
            is_writable=os.access(path, os.W_OK),
            is_executable=os.access(path, os.X_OK),
            created_time=stat_info.st_ctime,
            modified_time=stat_info.st_mtime,
            accessed_time=stat_info.st_atime,
            permissions=permissions,
            owner=owner,
            mime_type=mime_type,
            symlink_target=symlink_target,
        )

    def _detect_mime_type(self, path: Path) -> str:
        """Detect MIME type from file extension."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Search Files Tool


class SearchFilesInput(BaseModel):
    """Input for search files tool."""

    directory: str = Field(description="Directory to search in")
    pattern: str = Field(description="Glob pattern to match")
    recursive: bool = Field(default=True, description="Search recursively")
    include_hidden: bool = Field(default=False, description="Include hidden files")
    max_results: int = Field(default=100, ge=1, le=10000, description="Maximum results")


class SearchFilesOutput(FileSystemResult):
    """Output from search files tool."""

    matches: list[FileInfo] = Field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False


class SearchFilesTool(BaseTool[SearchFilesInput, SearchFilesOutput]):
    """Search for files matching patterns."""

    metadata = ToolMetadata(
        id="search_files",
        name="Search Files",
        description="Search for files using glob patterns. Supports recursive search.",
        category="utility",
        sandbox_required=False,
    )
    input_type = SearchFilesInput
    output_type = SearchFilesOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: SearchFilesInput) -> SearchFilesOutput:
        """Search for files."""
        try:
            path = Path(input.directory).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return SearchFilesOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            if not path.exists():
                return SearchFilesOutput(success=False, error=f"Directory not found: {path}")

            if not path.is_dir():
                return SearchFilesOutput(success=False, error=f"Not a directory: {path}")

            # Search for matches
            pattern = f"**/{input.pattern}" if input.recursive else input.pattern
            matches: list[FileInfo] = []

            for match_path in path.glob(pattern):
                # Skip hidden files if not included
                if not input.include_hidden and match_path.name.startswith("."):
                    continue

                # Skip if max depth exceeded
                try:
                    depth = len(match_path.relative_to(path).parts)
                    if depth > self.fs_config.max_search_depth:
                        continue
                except ValueError:
                    continue

                file_info = await self._get_file_info(match_path)
                matches.append(file_info)

                # Check max results
                if len(matches) >= input.max_results:
                    break

            total_matches = len(matches)
            truncated = len(list(path.glob(pattern))) > len(matches)

            return SearchFilesOutput(
                success=True,
                matches=matches,
                total_matches=total_matches,
                truncated=truncated,
            )

        except Exception as e:
            return SearchFilesOutput(success=False, error=f"Search error: {e}")

    async def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat_info = path.stat()

        # Determine file type
        if path.is_symlink():
            file_type = FileType.SYMLINK
            symlink_target = str(path.readlink())
        elif path.is_file():
            file_type = FileType.FILE
            symlink_target = None
        elif path.is_dir():
            file_type = FileType.DIRECTORY
            symlink_target = None
        else:
            file_type = FileType.OTHER
            symlink_target = None

        permissions = oct(stat_info.st_mode)[-3:]

        return FileInfo(
            path=str(path),
            name=path.name,
            size=stat_info.st_size,
            type=file_type,
            is_readable=os.access(path, os.R_OK),
            is_writable=os.access(path, os.W_OK),
            is_executable=os.access(path, os.X_OK),
            created_time=stat_info.st_ctime,
            modified_time=stat_info.st_mtime,
            accessed_time=stat_info.st_atime,
            permissions=permissions,
            owner=None,
            mime_type=None,
            symlink_target=symlink_target,
        )

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Create Directory Tool


class CreateDirectoryInput(BaseModel):
    """Input for create directory tool."""

    path: str = Field(description="Path to directory to create")
    parents: bool = Field(
        default=True,
        description="Create parent directories if needed",
    )
    exist_ok: bool = Field(
        default=True,
        description="Don't fail if directory already exists",
    )


class CreateDirectoryOutput(FileSystemResult):
    """Output from create directory tool."""

    path: Optional[str] = None
    created: bool = False


class CreateDirectoryTool(BaseTool[CreateDirectoryInput, CreateDirectoryOutput]):
    """Create directories."""

    metadata = ToolMetadata(
        id="create_directory",
        name="Create Directory",
        description="Create a directory with optional parent directory creation.",
        category="utility",
        sandbox_required=False,
    )
    input_type = CreateDirectoryInput
    output_type = CreateDirectoryOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: CreateDirectoryInput) -> CreateDirectoryOutput:
        """Create directory."""
        try:
            path = Path(input.path).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return CreateDirectoryOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            # Check if already exists
            if path.exists():
                if not input.exist_ok:
                    return CreateDirectoryOutput(
                        success=False,
                        error=f"Directory already exists: {path}",
                    )
                return CreateDirectoryOutput(
                    success=True,
                    path=str(path),
                    created=False,
                )

            # Create directory
            await asyncio.to_thread(path.mkdir, parents=input.parents, exist_ok=input.exist_ok)

            return CreateDirectoryOutput(
                success=True,
                path=str(path),
                created=True,
            )

        except Exception as e:
            return CreateDirectoryOutput(success=False, error=f"Create error: {e}")

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Delete File Tool


class DeleteFileInput(BaseModel):
    """Input for delete file tool."""

    path: str = Field(description="Path to file to delete")
    confirm: bool = Field(
        default=False,
        description="Confirmation flag for delete operation",
    )
    recursive: bool = Field(
        default=False,
        description="Delete directories recursively",
    )


class DeleteFileOutput(FileSystemResult):
    """Output from delete file tool."""

    path: Optional[str] = None
    deleted: bool = False


class DeleteFileTool(BaseTool[DeleteFileInput, DeleteFileOutput]):
    """Delete files with confirmation."""

    metadata = ToolMetadata(
        id="delete_file",
        name="Delete File",
        description="Delete a file or directory. Requires confirmation for safety.",
        category="utility",
        sandbox_required=False,
    )
    input_type = DeleteFileInput
    output_type = DeleteFileOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: DeleteFileInput) -> DeleteFileOutput:
        """Delete file."""
        try:
            path = Path(input.path).resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return DeleteFileOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            if not path.exists():
                return DeleteFileOutput(success=False, error=f"Path not found: {path}")

            # Require confirmation
            if self.fs_config.require_confirmation_for_delete and not input.confirm:
                return DeleteFileOutput(
                    success=False,
                    error="Delete operation requires confirmation (set confirm=True)",
                )

            # Delete
            if path.is_dir():
                if not input.recursive:
                    return DeleteFileOutput(
                        success=False,
                        error="Path is a directory, use recursive=True to delete",
                    )
                await asyncio.to_thread(shutil.rmtree, path)
            else:
                await asyncio.to_thread(path.unlink)

            return DeleteFileOutput(
                success=True,
                path=str(path),
                deleted=True,
            )

        except Exception as e:
            return DeleteFileOutput(success=False, error=f"Delete error: {e}")

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Move File Tool


class MoveFileInput(BaseModel):
    """Input for move file tool."""

    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")
    overwrite: bool = Field(default=False, description="Overwrite if destination exists")


class MoveFileOutput(FileSystemResult):
    """Output from move file tool."""

    source: Optional[str] = None
    destination: Optional[str] = None


class MoveFileTool(BaseTool[MoveFileInput, MoveFileOutput]):
    """Move or rename files."""

    metadata = ToolMetadata(
        id="move_file",
        name="Move File",
        description="Move or rename a file or directory.",
        category="utility",
        sandbox_required=False,
    )
    input_type = MoveFileInput
    output_type = MoveFileOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: MoveFileInput) -> MoveFileOutput:
        """Move file."""
        try:
            source = Path(input.source).resolve()
            destination = Path(input.destination).resolve()

            # Security checks
            if not self._is_path_allowed(source):
                return MoveFileOutput(
                    success=False,
                    error=f"Access denied: {source} is not in allowed directories",
                )

            if not self._is_path_allowed(destination):
                return MoveFileOutput(
                    success=False,
                    error=f"Access denied: {destination} is not in allowed directories",
                )

            if not source.exists():
                return MoveFileOutput(success=False, error=f"Source not found: {source}")

            if destination.exists() and not input.overwrite:
                return MoveFileOutput(
                    success=False,
                    error=f"Destination exists: {destination}. Use overwrite=True to replace.",
                )

            # Move
            await asyncio.to_thread(shutil.move, str(source), str(destination))

            return MoveFileOutput(
                success=True,
                source=str(source),
                destination=str(destination),
            )

        except Exception as e:
            return MoveFileOutput(success=False, error=f"Move error: {e}")

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# Copy File Tool


class CopyFileInput(BaseModel):
    """Input for copy file tool."""

    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")
    overwrite: bool = Field(default=False, description="Overwrite if destination exists")
    recursive: bool = Field(default=False, description="Copy directories recursively")


class CopyFileOutput(FileSystemResult):
    """Output from copy file tool."""

    source: Optional[str] = None
    destination: Optional[str] = None


class CopyFileTool(BaseTool[CopyFileInput, CopyFileOutput]):
    """Copy files."""

    metadata = ToolMetadata(
        id="copy_file",
        name="Copy File",
        description="Copy a file or directory.",
        category="utility",
        sandbox_required=False,
    )
    input_type = CopyFileInput
    output_type = CopyFileOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: CopyFileInput) -> CopyFileOutput:
        """Copy file."""
        try:
            source = Path(input.source).resolve()
            destination = Path(input.destination).resolve()

            # Security checks
            if not self._is_path_allowed(source):
                return CopyFileOutput(
                    success=False,
                    error=f"Access denied: {source} is not in allowed directories",
                )

            if not self._is_path_allowed(destination):
                return CopyFileOutput(
                    success=False,
                    error=f"Access denied: {destination} is not in allowed directories",
                )

            if not source.exists():
                return CopyFileOutput(success=False, error=f"Source not found: {source}")

            if destination.exists() and not input.overwrite:
                return CopyFileOutput(
                    success=False,
                    error=f"Destination exists: {destination}. Use overwrite=True to replace.",
                )

            # Copy
            if source.is_dir():
                if not input.recursive:
                    return CopyFileOutput(
                        success=False,
                        error="Source is a directory, use recursive=True to copy",
                    )
                await asyncio.to_thread(
                    shutil.copytree,
                    source,
                    destination,
                    dirs_exist_ok=input.overwrite,
                )
            else:
                await asyncio.to_thread(shutil.copy2, source, destination)

            return CopyFileOutput(
                success=True,
                source=str(source),
                destination=str(destination),
            )

        except Exception as e:
            return CopyFileOutput(success=False, error=f"Copy error: {e}")

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False


# File Info Tool


class FileInfoInput(BaseModel):
    """Input for file info tool."""

    path: str = Field(description="Path to file")


class FileInfoOutput(FileSystemResult):
    """Output from file info tool."""

    info: Optional[FileInfo] = None


class FileInfoTool(BaseTool[FileInfoInput, FileInfoOutput]):
    """Get file metadata."""

    metadata = ToolMetadata(
        id="file_info",
        name="File Info",
        description="Get detailed metadata about a file (size, permissions, dates, etc.).",
        category="utility",
        sandbox_required=False,
    )
    input_type = FileInfoInput
    output_type = FileInfoOutput

    def __init__(self, config: FileSystemConfig | None = None):
        """Initialize with configuration."""
        self.fs_config = config or FileSystemConfig()
        super().__init__(self.fs_config)

    async def execute(self, input: FileInfoInput) -> FileInfoOutput:
        """Get file info."""
        try:
            # Use unresolved path to detect symlinks
            path_unresolved = Path(input.path)
            path = path_unresolved.resolve()

            # Security checks
            if not self._is_path_allowed(path):
                return FileInfoOutput(
                    success=False,
                    error=f"Access denied: {path} is not in allowed directories",
                )

            if not path_unresolved.exists():
                return FileInfoOutput(success=False, error=f"Path not found: {path}")

            # Get file info (pass unresolved path to detect symlinks)
            file_info = await self._get_file_info(path_unresolved)

            return FileInfoOutput(
                success=True,
                info=file_info,
            )

        except Exception as e:
            return FileInfoOutput(success=False, error=f"Info error: {e}")

    async def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat_info = path.stat()

        # Determine file type
        if path.is_symlink():
            file_type = FileType.SYMLINK
            symlink_target = str(path.readlink())
        elif path.is_file():
            file_type = FileType.FILE
            symlink_target = None
        elif path.is_dir():
            file_type = FileType.DIRECTORY
            symlink_target = None
        else:
            file_type = FileType.OTHER
            symlink_target = None

        permissions = oct(stat_info.st_mode)[-3:]

        # Try to get owner (may not work on Windows)
        owner = None
        try:
            import pwd

            owner = pwd.getpwuid(stat_info.st_uid).pw_name
        except (ImportError, KeyError):
            pass

        # Detect MIME type for files
        mime_type = None
        if file_type == FileType.FILE:
            mime_type = self._detect_mime_type(path)

        return FileInfo(
            path=str(path),
            name=path.name,
            size=stat_info.st_size,
            type=file_type,
            is_readable=os.access(path, os.R_OK),
            is_writable=os.access(path, os.W_OK),
            is_executable=os.access(path, os.X_OK),
            created_time=stat_info.st_ctime,
            modified_time=stat_info.st_mtime,
            accessed_time=stat_info.st_atime,
            permissions=permissions,
            owner=owner,
            mime_type=mime_type,
            symlink_target=symlink_target,
        )

    def _detect_mime_type(self, path: Path) -> str:
        """Detect MIME type from file extension."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is in allowed directories."""
        if self.fs_config.allowed_directories is None:
            return True

        path_resolved = path.resolve()
        for allowed_dir in self.fs_config.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_resolved.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False
