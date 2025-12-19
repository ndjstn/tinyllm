"""Tests for GitHub tools."""

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from tinyllm.tools.github import (
    IssueState,
    PullRequestState,
    SortDirection,
    GitHubConfig,
    Issue,
    PullRequest,
    Repository,
    GitHubResult,
    GitHubClient,
    GitHubManager,
    CreateGitHubClientInput,
    CreateGitHubClientOutput,
    ListIssuesInput,
    IssueOutput,
    GetIssueInput,
    CreateIssueInput,
    UpdateIssueInput,
    AddCommentInput,
    CommentOutput,
    ListPullRequestsInput,
    PullRequestOutput,
    GetPullRequestInput,
    CreatePullRequestInput,
    GetRepositoryInput,
    RepositoryOutput,
    ListRepositoriesInput,
    CreateGitHubClientTool,
    ListIssuesTool,
    GetIssueTool,
    CreateIssueTool,
    UpdateIssueTool,
    AddCommentTool,
    ListPullRequestsTool,
    GetPullRequestTool,
    CreatePullRequestTool,
    GetRepositoryTool,
    ListRepositoriesTool,
    create_github_config,
    create_github_client,
    create_github_manager,
    create_github_tools,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_issue_state_values(self):
        """Test IssueState enum values."""
        assert IssueState.OPEN.value == "open"
        assert IssueState.CLOSED.value == "closed"
        assert IssueState.ALL.value == "all"

    def test_pull_request_state_values(self):
        """Test PullRequestState enum values."""
        assert PullRequestState.OPEN.value == "open"
        assert PullRequestState.CLOSED.value == "closed"
        assert PullRequestState.ALL.value == "all"

    def test_sort_direction_values(self):
        """Test SortDirection enum values."""
        assert SortDirection.ASC.value == "asc"
        assert SortDirection.DESC.value == "desc"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestGitHubConfig:
    """Tests for GitHubConfig dataclass."""

    def test_config_with_defaults(self):
        """Test config with default values."""
        config = GitHubConfig(token="test-token")

        assert config.token == "test-token"
        assert config.base_url == "https://api.github.com"
        assert config.timeout == 30

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = GitHubConfig(
            token="my-token",
            base_url="https://github.mycompany.com/api/v3",
            timeout=60,
        )

        assert config.base_url == "https://github.mycompany.com/api/v3"
        assert config.timeout == 60


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_from_api_response(self):
        """Test creating Issue from API response."""
        api_data = {
            "number": 42,
            "title": "Fix bug",
            "body": "This bug needs fixing",
            "state": "open",
            "html_url": "https://github.com/owner/repo/issues/42",
            "user": {"login": "testuser"},
            "labels": [{"name": "bug"}, {"name": "priority:high"}],
            "assignees": [{"login": "dev1"}, {"login": "dev2"}],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "closed_at": None,
            "comments": 5,
        }

        issue = Issue.from_api_response(api_data)

        assert issue.number == 42
        assert issue.title == "Fix bug"
        assert issue.body == "This bug needs fixing"
        assert issue.state == "open"
        assert issue.user == "testuser"
        assert issue.labels == ["bug", "priority:high"]
        assert issue.assignees == ["dev1", "dev2"]
        assert issue.comments == 5

    def test_issue_to_dict(self):
        """Test converting Issue to dict."""
        issue = Issue(
            number=1,
            title="Test",
            body="Body",
            state="open",
        )

        d = issue.to_dict()

        assert d["number"] == 1
        assert d["title"] == "Test"
        assert d["body"] == "Body"
        assert d["state"] == "open"


class TestPullRequest:
    """Tests for PullRequest dataclass."""

    def test_pull_request_from_api_response(self):
        """Test creating PullRequest from API response."""
        api_data = {
            "number": 123,
            "title": "Add feature",
            "body": "New feature description",
            "state": "open",
            "html_url": "https://github.com/owner/repo/pull/123",
            "user": {"login": "contributor"},
            "head": {"ref": "feature-branch"},
            "base": {"ref": "main"},
            "draft": False,
            "merged": False,
            "mergeable": True,
            "labels": [{"name": "enhancement"}],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "closed_at": None,
            "merged_at": None,
        }

        pr = PullRequest.from_api_response(api_data)

        assert pr.number == 123
        assert pr.title == "Add feature"
        assert pr.user == "contributor"
        assert pr.head == "feature-branch"
        assert pr.base == "main"
        assert pr.draft is False
        assert pr.merged is False
        assert pr.mergeable is True
        assert pr.labels == ["enhancement"]

    def test_pull_request_to_dict(self):
        """Test converting PullRequest to dict."""
        pr = PullRequest(
            number=1,
            title="Test PR",
            head="feature",
            base="main",
        )

        d = pr.to_dict()

        assert d["number"] == 1
        assert d["title"] == "Test PR"
        assert d["head"] == "feature"
        assert d["base"] == "main"


class TestRepository:
    """Tests for Repository dataclass."""

    def test_repository_from_api_response(self):
        """Test creating Repository from API response."""
        api_data = {
            "name": "my-repo",
            "full_name": "owner/my-repo",
            "description": "A test repository",
            "html_url": "https://github.com/owner/my-repo",
            "private": False,
            "fork": False,
            "default_branch": "main",
            "language": "Python",
            "stargazers_count": 100,
            "forks_count": 25,
            "open_issues_count": 10,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "pushed_at": "2024-01-03T00:00:00Z",
        }

        repo = Repository.from_api_response(api_data)

        assert repo.name == "my-repo"
        assert repo.full_name == "owner/my-repo"
        assert repo.description == "A test repository"
        assert repo.language == "Python"
        assert repo.stargazers_count == 100
        assert repo.forks_count == 25

    def test_repository_to_dict(self):
        """Test converting Repository to dict."""
        repo = Repository(
            name="test",
            full_name="owner/test",
        )

        d = repo.to_dict()

        assert d["name"] == "test"
        assert d["full_name"] == "owner/test"


class TestGitHubResult:
    """Tests for GitHubResult dataclass."""

    def test_result_success(self):
        """Test successful result."""
        result = GitHubResult(
            success=True,
            data={"id": 123},
            status_code=200,
        )

        assert result.success is True
        assert result.data == {"id": 123}
        assert result.error is None

    def test_result_failure(self):
        """Test failed result."""
        result = GitHubResult(
            success=False,
            error="Not found",
            status_code=404,
        )

        assert result.success is False
        assert result.error == "Not found"

    def test_result_to_dict(self):
        """Test converting to dict."""
        result = GitHubResult(success=True, data={"test": 1})

        d = result.to_dict()

        assert d["success"] is True
        assert d["data"] == {"test": 1}


# ============================================================================
# GitHubClient Tests
# ============================================================================


class TestGitHubClient:
    """Tests for GitHubClient class."""

    def test_create_client(self):
        """Test creating a client."""
        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        assert client.config == config

    @patch("urllib.request.urlopen")
    def test_list_issues_success(self, mock_urlopen):
        """Test listing issues successfully."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "number": 1,
                "title": "Bug 1",
                "state": "open",
                "user": {"login": "user1"},
                "labels": [],
                "assignees": [],
            },
            {
                "number": 2,
                "title": "Bug 2",
                "state": "open",
                "user": {"login": "user2"},
                "labels": [{"name": "bug"}],
                "assignees": [],
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.list_issues("owner", "repo")

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["number"] == 1
        assert result.data[1]["labels"] == ["bug"]

    @patch("urllib.request.urlopen")
    def test_list_issues_filters_pull_requests(self, mock_urlopen):
        """Test that PRs are filtered from issues."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "number": 1,
                "title": "Issue",
                "state": "open",
                "user": {"login": "user1"},
                "labels": [],
                "assignees": [],
            },
            {
                "number": 2,
                "title": "PR",
                "state": "open",
                "user": {"login": "user2"},
                "labels": [],
                "assignees": [],
                "pull_request": {"url": "..."},
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.list_issues("owner", "repo")

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["title"] == "Issue"

    @patch("urllib.request.urlopen")
    def test_get_issue_success(self, mock_urlopen):
        """Test getting a single issue."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 42,
            "title": "Test issue",
            "state": "open",
            "body": "Issue description",
            "user": {"login": "testuser"},
            "labels": [],
            "assignees": [],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.get_issue("owner", "repo", 42)

        assert result.success is True
        assert result.data["number"] == 42
        assert result.data["title"] == "Test issue"

    @patch("urllib.request.urlopen")
    def test_create_issue_success(self, mock_urlopen):
        """Test creating an issue."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 100,
            "title": "New issue",
            "state": "open",
            "body": "Issue body",
            "user": {"login": "creator"},
            "labels": [{"name": "bug"}],
            "assignees": [],
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.create_issue(
            owner="owner",
            repo="repo",
            title="New issue",
            body="Issue body",
            labels=["bug"],
        )

        assert result.success is True
        assert result.data["number"] == 100
        assert result.data["labels"] == ["bug"]

    @patch("urllib.request.urlopen")
    def test_update_issue_success(self, mock_urlopen):
        """Test updating an issue."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 42,
            "title": "Updated title",
            "state": "closed",
            "user": {"login": "testuser"},
            "labels": [],
            "assignees": [],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.update_issue(
            owner="owner",
            repo="repo",
            issue_number=42,
            title="Updated title",
            state="closed",
        )

        assert result.success is True
        assert result.data["state"] == "closed"

    def test_update_issue_no_updates(self):
        """Test update with no changes."""
        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.update_issue(
            owner="owner",
            repo="repo",
            issue_number=42,
        )

        assert result.success is False
        assert "No updates" in result.error

    @patch("urllib.request.urlopen")
    def test_add_comment_success(self, mock_urlopen):
        """Test adding a comment."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": 12345,
            "body": "This is a comment",
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.add_comment(
            owner="owner",
            repo="repo",
            issue_number=42,
            body="This is a comment",
        )

        assert result.success is True
        assert result.data["id"] == 12345

    @patch("urllib.request.urlopen")
    def test_list_pull_requests_success(self, mock_urlopen):
        """Test listing pull requests."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "number": 10,
                "title": "Feature PR",
                "state": "open",
                "user": {"login": "contributor"},
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "draft": False,
                "labels": [],
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.list_pull_requests("owner", "repo")

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["head"] == "feature"

    @patch("urllib.request.urlopen")
    def test_get_pull_request_success(self, mock_urlopen):
        """Test getting a pull request."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 123,
            "title": "Test PR",
            "state": "open",
            "user": {"login": "user"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "draft": False,
            "merged": False,
            "mergeable": True,
            "labels": [],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.get_pull_request("owner", "repo", 123)

        assert result.success is True
        assert result.data["number"] == 123
        assert result.data["mergeable"] is True

    @patch("urllib.request.urlopen")
    def test_create_pull_request_success(self, mock_urlopen):
        """Test creating a pull request."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 200,
            "title": "New Feature",
            "state": "open",
            "user": {"login": "creator"},
            "head": {"ref": "new-feature"},
            "base": {"ref": "main"},
            "draft": False,
            "labels": [],
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.create_pull_request(
            owner="owner",
            repo="repo",
            title="New Feature",
            head="new-feature",
            base="main",
            body="PR description",
        )

        assert result.success is True
        assert result.data["number"] == 200

    @patch("urllib.request.urlopen")
    def test_get_repository_success(self, mock_urlopen):
        """Test getting repository info."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "name": "my-repo",
            "full_name": "owner/my-repo",
            "description": "A repository",
            "private": False,
            "fork": False,
            "default_branch": "main",
            "language": "Python",
            "stargazers_count": 50,
            "forks_count": 10,
            "open_issues_count": 5,
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.get_repository("owner", "my-repo")

        assert result.success is True
        assert result.data["name"] == "my-repo"
        assert result.data["language"] == "Python"

    @patch("urllib.request.urlopen")
    def test_list_repositories_user(self, mock_urlopen):
        """Test listing user repositories."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "name": "repo1",
                "full_name": "user/repo1",
                "private": False,
                "fork": False,
                "default_branch": "main",
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.list_repositories(user="testuser")

        assert result.success is True
        assert len(result.data) == 1

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test handling HTTP errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/repos/owner/repo",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"message": "Not Found"}')),
        )

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.get_repository("owner", "nonexistent")

        assert result.success is False
        assert "404" in result.error
        assert result.status_code == 404

    @patch("urllib.request.urlopen")
    def test_url_error_handling(self, mock_urlopen):
        """Test handling connection errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        config = GitHubConfig(token="test-token")
        client = GitHubClient(config)

        result = client.list_issues("owner", "repo")

        assert result.success is False
        assert "Connection" in result.error


# ============================================================================
# GitHubManager Tests
# ============================================================================


class TestGitHubManager:
    """Tests for GitHubManager class."""

    def test_create_manager(self):
        """Test creating manager."""
        manager = GitHubManager()

        assert manager.list_clients() == []

    def test_add_client(self):
        """Test adding a client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test")
        client = GitHubClient(config)

        manager.add_client("default", client)

        assert "default" in manager.list_clients()
        assert manager.get_client("default") == client

    def test_get_client_not_found(self):
        """Test getting non-existent client."""
        manager = GitHubManager()

        result = manager.get_client("nonexistent")

        assert result is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test")
        manager.add_client("test", GitHubClient(config))

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None

    def test_remove_client_not_found(self):
        """Test removing non-existent client."""
        manager = GitHubManager()

        result = manager.remove_client("nonexistent")

        assert result is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = GitHubManager()
        config = GitHubConfig(token="test")

        for name in ["client1", "client2", "client3"]:
            manager.add_client(name, GitHubClient(config))

        clients = manager.list_clients()

        assert len(clients) == 3
        assert "client1" in clients
        assert "client2" in clients
        assert "client3" in clients


# ============================================================================
# Tool Tests
# ============================================================================


class TestCreateGitHubClientTool:
    """Tests for CreateGitHubClientTool."""

    @pytest.mark.asyncio
    async def test_create_client(self):
        """Test creating a GitHub client."""
        manager = GitHubManager()
        tool = CreateGitHubClientTool(manager)

        input_data = CreateGitHubClientInput(
            name="my-client",
            token="test-token",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.name == "my-client"
        assert manager.get_client("my-client") is not None

    def test_tool_metadata(self):
        """Test tool metadata."""
        manager = GitHubManager()
        tool = CreateGitHubClientTool(manager)

        assert tool.metadata.id == "create_github_client"


class TestListIssuesTool:
    """Tests for ListIssuesTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with a mock client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_client_not_found(self):
        """Test with non-existent client."""
        manager = GitHubManager()
        tool = ListIssuesTool(manager)

        input_data = ListIssuesInput(
            client="nonexistent",
            owner="owner",
            repo="repo",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_invalid_state(self, manager_with_client):
        """Test with invalid state value."""
        tool = ListIssuesTool(manager_with_client)

        input_data = ListIssuesInput(
            owner="owner",
            repo="repo",
            state="invalid",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "Invalid state" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_issues_success(self, mock_urlopen, manager_with_client):
        """Test listing issues successfully."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "number": 1,
                "title": "Test",
                "state": "open",
                "user": {"login": "user"},
                "labels": [],
                "assignees": [],
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListIssuesTool(manager_with_client)

        input_data = ListIssuesInput(
            owner="owner",
            repo="repo",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.issues) == 1


class TestGetIssueTool:
    """Tests for GetIssueTool."""

    @pytest.mark.asyncio
    async def test_client_not_found(self):
        """Test with non-existent client."""
        manager = GitHubManager()
        tool = GetIssueTool(manager)

        input_data = GetIssueInput(
            client="nonexistent",
            owner="owner",
            repo="repo",
            issue_number=1,
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error


class TestCreateIssueTool:
    """Tests for CreateIssueTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_create_issue_success(self, mock_urlopen, manager_with_client):
        """Test creating an issue."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 50,
            "title": "New Issue",
            "state": "open",
            "body": "Description",
            "user": {"login": "creator"},
            "labels": [],
            "assignees": [],
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = CreateIssueTool(manager_with_client)

        input_data = CreateIssueInput(
            owner="owner",
            repo="repo",
            title="New Issue",
            body="Description",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.issue["number"] == 50


class TestUpdateIssueTool:
    """Tests for UpdateIssueTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_update_issue_success(self, mock_urlopen, manager_with_client):
        """Test updating an issue."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 1,
            "title": "Updated Title",
            "state": "closed",
            "user": {"login": "user"},
            "labels": [],
            "assignees": [],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = UpdateIssueTool(manager_with_client)

        input_data = UpdateIssueInput(
            owner="owner",
            repo="repo",
            issue_number=1,
            state="closed",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.issue["state"] == "closed"


class TestAddCommentTool:
    """Tests for AddCommentTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_add_comment_success(self, mock_urlopen, manager_with_client):
        """Test adding a comment."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": 12345,
            "body": "Test comment",
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = AddCommentTool(manager_with_client)

        input_data = AddCommentInput(
            owner="owner",
            repo="repo",
            issue_number=1,
            body="Test comment",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.comment_id == 12345


class TestListPullRequestsTool:
    """Tests for ListPullRequestsTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_invalid_state(self, manager_with_client):
        """Test with invalid state."""
        tool = ListPullRequestsTool(manager_with_client)

        input_data = ListPullRequestsInput(
            owner="owner",
            repo="repo",
            state="invalid",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "Invalid state" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_prs_success(self, mock_urlopen, manager_with_client):
        """Test listing PRs."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "number": 10,
                "title": "PR",
                "state": "open",
                "user": {"login": "user"},
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "draft": False,
                "labels": [],
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListPullRequestsTool(manager_with_client)

        input_data = ListPullRequestsInput(
            owner="owner",
            repo="repo",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.pull_requests) == 1


class TestGetPullRequestTool:
    """Tests for GetPullRequestTool."""

    @pytest.mark.asyncio
    async def test_client_not_found(self):
        """Test with non-existent client."""
        manager = GitHubManager()
        tool = GetPullRequestTool(manager)

        input_data = GetPullRequestInput(
            client="nonexistent",
            owner="owner",
            repo="repo",
            pr_number=1,
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error


class TestCreatePullRequestTool:
    """Tests for CreatePullRequestTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_create_pr_success(self, mock_urlopen, manager_with_client):
        """Test creating a PR."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "number": 100,
            "title": "New PR",
            "state": "open",
            "user": {"login": "creator"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "draft": False,
            "labels": [],
        }).encode()
        mock_response.getcode.return_value = 201
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = CreatePullRequestTool(manager_with_client)

        input_data = CreatePullRequestInput(
            owner="owner",
            repo="repo",
            title="New PR",
            head="feature",
            base="main",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.pull_request["number"] == 100


class TestGetRepositoryTool:
    """Tests for GetRepositoryTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_get_repo_success(self, mock_urlopen, manager_with_client):
        """Test getting a repository."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "name": "my-repo",
            "full_name": "owner/my-repo",
            "description": "Test repo",
            "private": False,
            "fork": False,
            "default_branch": "main",
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = GetRepositoryTool(manager_with_client)

        input_data = GetRepositoryInput(
            owner="owner",
            repo="my-repo",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.repository["name"] == "my-repo"


class TestListRepositoriesTool:
    """Tests for ListRepositoriesTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GitHubManager()
        config = GitHubConfig(token="test-token")
        manager.add_client("default", GitHubClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_repos_success(self, mock_urlopen, manager_with_client):
        """Test listing repositories."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([
            {
                "name": "repo1",
                "full_name": "user/repo1",
                "private": False,
                "fork": False,
                "default_branch": "main",
            },
        ]).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListRepositoriesTool(manager_with_client)

        input_data = ListRepositoriesInput(
            user="testuser",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.repositories) == 1


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_github_config(self):
        """Test create_github_config function."""
        config = create_github_config(
            token="my-token",
            base_url="https://github.mycompany.com/api/v3",
            timeout=60,
        )

        assert config.token == "my-token"
        assert config.base_url == "https://github.mycompany.com/api/v3"
        assert config.timeout == 60

    def test_create_github_client(self):
        """Test create_github_client function."""
        config = GitHubConfig(token="test")
        client = create_github_client(config)

        assert isinstance(client, GitHubClient)
        assert client.config == config

    def test_create_github_manager(self):
        """Test create_github_manager function."""
        manager = create_github_manager()

        assert isinstance(manager, GitHubManager)
        assert manager.list_clients() == []

    def test_create_github_tools(self):
        """Test create_github_tools function."""
        manager = GitHubManager()
        tools = create_github_tools(manager)

        assert "create_github_client" in tools
        assert "list_github_issues" in tools
        assert "get_github_issue" in tools
        assert "create_github_issue" in tools
        assert "update_github_issue" in tools
        assert "add_github_comment" in tools
        assert "list_github_pull_requests" in tools
        assert "get_github_pull_request" in tools
        assert "create_github_pull_request" in tools
        assert "get_github_repository" in tools
        assert "list_github_repositories" in tools

        assert isinstance(tools["create_github_client"], CreateGitHubClientTool)
        assert isinstance(tools["list_github_issues"], ListIssuesTool)
