"""GitHub tools for TinyLLM.

This module provides tools for interacting with GitHub API
including issues, pull requests, and repositories.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class IssueState(str, Enum):
    """GitHub issue states."""

    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class PullRequestState(str, Enum):
    """GitHub pull request states."""

    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class SortDirection(str, Enum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class GitHubConfig:
    """GitHub API configuration."""

    token: str
    base_url: str = "https://api.github.com"
    timeout: int = 30


@dataclass
class Issue:
    """GitHub issue representation."""

    number: int
    title: str
    body: Optional[str] = None
    state: str = "open"
    html_url: Optional[str] = None
    user: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    comments: int = 0

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Issue":
        """Create Issue from GitHub API response."""
        return cls(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            html_url=data.get("html_url"),
            user=data.get("user", {}).get("login"),
            labels=[label["name"] for label in data.get("labels", [])],
            assignees=[a["login"] for a in data.get("assignees", [])],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            closed_at=data.get("closed_at"),
            comments=data.get("comments", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "html_url": self.html_url,
            "user": self.user,
            "labels": self.labels,
            "assignees": self.assignees,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "comments": self.comments,
        }


@dataclass
class PullRequest:
    """GitHub pull request representation."""

    number: int
    title: str
    body: Optional[str] = None
    state: str = "open"
    html_url: Optional[str] = None
    user: Optional[str] = None
    head: Optional[str] = None
    base: Optional[str] = None
    draft: bool = False
    merged: bool = False
    mergeable: Optional[bool] = None
    labels: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None
    merged_at: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "PullRequest":
        """Create PullRequest from GitHub API response."""
        return cls(
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=data["state"],
            html_url=data.get("html_url"),
            user=data.get("user", {}).get("login"),
            head=data.get("head", {}).get("ref"),
            base=data.get("base", {}).get("ref"),
            draft=data.get("draft", False),
            merged=data.get("merged", False),
            mergeable=data.get("mergeable"),
            labels=[label["name"] for label in data.get("labels", [])],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            closed_at=data.get("closed_at"),
            merged_at=data.get("merged_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "html_url": self.html_url,
            "user": self.user,
            "head": self.head,
            "base": self.base,
            "draft": self.draft,
            "merged": self.merged,
            "mergeable": self.mergeable,
            "labels": self.labels,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "merged_at": self.merged_at,
        }


@dataclass
class Repository:
    """GitHub repository representation."""

    name: str
    full_name: str
    description: Optional[str] = None
    html_url: Optional[str] = None
    private: bool = False
    fork: bool = False
    default_branch: str = "main"
    language: Optional[str] = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Repository":
        """Create Repository from GitHub API response."""
        return cls(
            name=data["name"],
            full_name=data["full_name"],
            description=data.get("description"),
            html_url=data.get("html_url"),
            private=data.get("private", False),
            fork=data.get("fork", False),
            default_branch=data.get("default_branch", "main"),
            language=data.get("language"),
            stargazers_count=data.get("stargazers_count", 0),
            forks_count=data.get("forks_count", 0),
            open_issues_count=data.get("open_issues_count", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            pushed_at=data.get("pushed_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "html_url": self.html_url,
            "private": self.private,
            "fork": self.fork,
            "default_branch": self.default_branch,
            "language": self.language,
            "stargazers_count": self.stargazers_count,
            "forks_count": self.forks_count,
            "open_issues_count": self.open_issues_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pushed_at": self.pushed_at,
        }


@dataclass
class GitHubResult:
    """Result from GitHub API operation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
        }


class GitHubClient:
    """Client for GitHub API."""

    def __init__(self, config: GitHubConfig):
        """Initialize client.

        Args:
            config: GitHub configuration.
        """
        self.config = config

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> GitHubResult:
        """Make HTTP request to GitHub API.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request body data.
            params: Query parameters.

        Returns:
            GitHub result.
        """
        url = f"{self.config.base_url}{endpoint}"

        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query_string:
                url = f"{url}?{query_string}"

        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "TinyLLM-GitHub-Tools",
        }

        request_data = json.dumps(data).encode("utf-8") if data else None

        try:
            req = urllib.request.Request(
                url,
                data=request_data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body) if response_body else None

                return GitHubResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                error_message = error_data.get("message", error_body)
            except Exception:
                error_message = error_body or str(e)

            return GitHubResult(
                success=False,
                error=f"HTTP {e.code}: {error_message}",
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return GitHubResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except Exception as e:
            return GitHubResult(
                success=False,
                error=str(e),
            )

    # Issue operations

    def list_issues(
        self,
        owner: str,
        repo: str,
        state: IssueState = IssueState.OPEN,
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> GitHubResult:
        """List issues for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: Issue state filter.
            labels: Filter by labels.
            assignee: Filter by assignee.
            per_page: Results per page.
            page: Page number.

        Returns:
            GitHub result with list of issues.
        """
        params = {
            "state": state.value,
            "per_page": str(per_page),
            "page": str(page),
        }

        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee

        result = self._make_request("GET", f"/repos/{owner}/{repo}/issues", params=params)

        if result.success and result.data:
            # Filter out pull requests (they come in issues endpoint)
            issues = [
                Issue.from_api_response(item).to_dict()
                for item in result.data
                if "pull_request" not in item
            ]
            result.data = issues

        return result

    def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubResult:
        """Get a specific issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            issue_number: Issue number.

        Returns:
            GitHub result with issue data.
        """
        result = self._make_request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")

        if result.success and result.data:
            result.data = Issue.from_api_response(result.data).to_dict()

        return result

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> GitHubResult:
        """Create a new issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            title: Issue title.
            body: Issue body.
            labels: Issue labels.
            assignees: Issue assignees.

        Returns:
            GitHub result with created issue.
        """
        data: Dict[str, Any] = {"title": title}

        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees

        result = self._make_request("POST", f"/repos/{owner}/{repo}/issues", data=data)

        if result.success and result.data:
            result.data = Issue.from_api_response(result.data).to_dict()

        return result

    def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> GitHubResult:
        """Update an existing issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            issue_number: Issue number.
            title: New title.
            body: New body.
            state: New state (open/closed).
            labels: New labels.
            assignees: New assignees.

        Returns:
            GitHub result with updated issue.
        """
        data: Dict[str, Any] = {}

        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            data["state"] = state
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees

        if not data:
            return GitHubResult(success=False, error="No updates provided")

        result = self._make_request(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/{issue_number}",
            data=data,
        )

        if result.success and result.data:
            result.data = Issue.from_api_response(result.data).to_dict()

        return result

    def add_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
    ) -> GitHubResult:
        """Add a comment to an issue.

        Args:
            owner: Repository owner.
            repo: Repository name.
            issue_number: Issue number.
            body: Comment body.

        Returns:
            GitHub result with created comment.
        """
        return self._make_request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            data={"body": body},
        )

    # Pull request operations

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: PullRequestState = PullRequestState.OPEN,
        head: Optional[str] = None,
        base: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> GitHubResult:
        """List pull requests for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: PR state filter.
            head: Filter by head branch.
            base: Filter by base branch.
            per_page: Results per page.
            page: Page number.

        Returns:
            GitHub result with list of PRs.
        """
        params = {
            "state": state.value,
            "per_page": str(per_page),
            "page": str(page),
        }

        if head:
            params["head"] = head
        if base:
            params["base"] = base

        result = self._make_request("GET", f"/repos/{owner}/{repo}/pulls", params=params)

        if result.success and result.data:
            result.data = [PullRequest.from_api_response(item).to_dict() for item in result.data]

        return result

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubResult:
        """Get a specific pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            GitHub result with PR data.
        """
        result = self._make_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")

        if result.success and result.data:
            result.data = PullRequest.from_api_response(result.data).to_dict()

        return result

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False,
    ) -> GitHubResult:
        """Create a new pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            title: PR title.
            head: Head branch.
            base: Base branch.
            body: PR body.
            draft: Create as draft.

        Returns:
            GitHub result with created PR.
        """
        data = {
            "title": title,
            "head": head,
            "base": base,
            "draft": draft,
        }

        if body:
            data["body"] = body

        result = self._make_request("POST", f"/repos/{owner}/{repo}/pulls", data=data)

        if result.success and result.data:
            result.data = PullRequest.from_api_response(result.data).to_dict()

        return result

    # Repository operations

    def get_repository(self, owner: str, repo: str) -> GitHubResult:
        """Get repository information.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            GitHub result with repository data.
        """
        result = self._make_request("GET", f"/repos/{owner}/{repo}")

        if result.success and result.data:
            result.data = Repository.from_api_response(result.data).to_dict()

        return result

    def list_repositories(
        self,
        org: Optional[str] = None,
        user: Optional[str] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> GitHubResult:
        """List repositories for org or user.

        Args:
            org: Organization name.
            user: Username.
            per_page: Results per page.
            page: Page number.

        Returns:
            GitHub result with list of repos.
        """
        params = {
            "per_page": str(per_page),
            "page": str(page),
        }

        if org:
            endpoint = f"/orgs/{org}/repos"
        elif user:
            endpoint = f"/users/{user}/repos"
        else:
            endpoint = "/user/repos"

        result = self._make_request("GET", endpoint, params=params)

        if result.success and result.data:
            result.data = [Repository.from_api_response(item).to_dict() for item in result.data]

        return result

    def search_code(
        self,
        query: str,
        per_page: int = 30,
        page: int = 1,
    ) -> GitHubResult:
        """Search code on GitHub.

        Args:
            query: Search query.
            per_page: Results per page.
            page: Page number.

        Returns:
            GitHub result with search results.
        """
        params = {
            "q": query,
            "per_page": str(per_page),
            "page": str(page),
        }

        return self._make_request("GET", "/search/code", params=params)


class GitHubManager:
    """Manager for GitHub clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, GitHubClient] = {}

    def add_client(self, name: str, client: GitHubClient) -> None:
        """Add a GitHub client.

        Args:
            name: Client name.
            client: GitHub client.
        """
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[GitHubClient]:
        """Get a GitHub client.

        Args:
            name: Client name.

        Returns:
            GitHub client or None.
        """
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a GitHub client.

        Args:
            name: Client name.

        Returns:
            True if removed.
        """
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Pydantic models for tool inputs/outputs


class CreateGitHubClientInput(BaseModel):
    """Input for creating a GitHub client."""

    name: str = Field(..., description="Name for the client")
    token: str = Field(..., description="GitHub personal access token")
    base_url: str = Field(
        default="https://api.github.com",
        description="GitHub API base URL",
    )


class CreateGitHubClientOutput(BaseModel):
    """Output from creating a GitHub client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListIssuesInput(BaseModel):
    """Input for listing issues."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    state: str = Field(default="open", description="Issue state (open/closed/all)")
    labels: Optional[List[str]] = Field(default=None, description="Filter by labels")
    assignee: Optional[str] = Field(default=None, description="Filter by assignee")
    per_page: int = Field(default=30, description="Results per page")
    page: int = Field(default=1, description="Page number")


class IssueOutput(BaseModel):
    """Output containing issue data."""

    success: bool = Field(description="Whether operation succeeded")
    issue: Optional[Dict[str, Any]] = Field(default=None, description="Issue data")
    issues: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of issues")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetIssueInput(BaseModel):
    """Input for getting a specific issue."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    issue_number: int = Field(..., description="Issue number")


class CreateIssueInput(BaseModel):
    """Input for creating an issue."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    title: str = Field(..., description="Issue title")
    body: Optional[str] = Field(default=None, description="Issue body")
    labels: Optional[List[str]] = Field(default=None, description="Issue labels")
    assignees: Optional[List[str]] = Field(default=None, description="Issue assignees")


class UpdateIssueInput(BaseModel):
    """Input for updating an issue."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    issue_number: int = Field(..., description="Issue number")
    title: Optional[str] = Field(default=None, description="New title")
    body: Optional[str] = Field(default=None, description="New body")
    state: Optional[str] = Field(default=None, description="New state (open/closed)")
    labels: Optional[List[str]] = Field(default=None, description="New labels")
    assignees: Optional[List[str]] = Field(default=None, description="New assignees")


class AddCommentInput(BaseModel):
    """Input for adding a comment."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    issue_number: int = Field(..., description="Issue or PR number")
    body: str = Field(..., description="Comment body")


class CommentOutput(BaseModel):
    """Output from adding a comment."""

    success: bool = Field(description="Whether operation succeeded")
    comment_id: Optional[int] = Field(default=None, description="Created comment ID")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListPullRequestsInput(BaseModel):
    """Input for listing pull requests."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    state: str = Field(default="open", description="PR state (open/closed/all)")
    head: Optional[str] = Field(default=None, description="Filter by head branch")
    base: Optional[str] = Field(default=None, description="Filter by base branch")
    per_page: int = Field(default=30, description="Results per page")
    page: int = Field(default=1, description="Page number")


class PullRequestOutput(BaseModel):
    """Output containing pull request data."""

    success: bool = Field(description="Whether operation succeeded")
    pull_request: Optional[Dict[str, Any]] = Field(default=None, description="PR data")
    pull_requests: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of PRs")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetPullRequestInput(BaseModel):
    """Input for getting a specific pull request."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    pr_number: int = Field(..., description="PR number")


class CreatePullRequestInput(BaseModel):
    """Input for creating a pull request."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    title: str = Field(..., description="PR title")
    head: str = Field(..., description="Head branch")
    base: str = Field(..., description="Base branch")
    body: Optional[str] = Field(default=None, description="PR body")
    draft: bool = Field(default=False, description="Create as draft")


class GetRepositoryInput(BaseModel):
    """Input for getting repository info."""

    client: str = Field(default="default", description="GitHub client name")
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class RepositoryOutput(BaseModel):
    """Output containing repository data."""

    success: bool = Field(description="Whether operation succeeded")
    repository: Optional[Dict[str, Any]] = Field(default=None, description="Repository data")
    repositories: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of repos")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListRepositoriesInput(BaseModel):
    """Input for listing repositories."""

    client: str = Field(default="default", description="GitHub client name")
    org: Optional[str] = Field(default=None, description="Organization name")
    user: Optional[str] = Field(default=None, description="Username")
    per_page: int = Field(default=30, description="Results per page")
    page: int = Field(default=1, description="Page number")


# Tool implementations


class CreateGitHubClientTool(BaseTool[CreateGitHubClientInput, CreateGitHubClientOutput]):
    """Tool for creating a GitHub client."""

    metadata = ToolMetadata(
        id="create_github_client",
        name="Create GitHub Client",
        description="Create a GitHub API client with authentication",
        category="utility",
    )
    input_type = CreateGitHubClientInput
    output_type = CreateGitHubClientOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateGitHubClientInput) -> CreateGitHubClientOutput:
        """Create a GitHub client."""
        config = GitHubConfig(
            token=input.token,
            base_url=input.base_url,
        )
        client = GitHubClient(config)
        self.manager.add_client(input.name, client)

        return CreateGitHubClientOutput(
            success=True,
            name=input.name,
        )


class ListIssuesTool(BaseTool[ListIssuesInput, IssueOutput]):
    """Tool for listing GitHub issues."""

    metadata = ToolMetadata(
        id="list_github_issues",
        name="List GitHub Issues",
        description="List issues in a GitHub repository",
        category="utility",
    )
    input_type = ListIssuesInput
    output_type = IssueOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListIssuesInput) -> IssueOutput:
        """List issues."""
        client = self.manager.get_client(input.client)

        if not client:
            return IssueOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        try:
            state = IssueState(input.state)
        except ValueError:
            return IssueOutput(
                success=False,
                error=f"Invalid state: {input.state}",
            )

        result = client.list_issues(
            owner=input.owner,
            repo=input.repo,
            state=state,
            labels=input.labels,
            assignee=input.assignee,
            per_page=input.per_page,
            page=input.page,
        )

        if result.success:
            return IssueOutput(success=True, issues=result.data)
        return IssueOutput(success=False, error=result.error)


class GetIssueTool(BaseTool[GetIssueInput, IssueOutput]):
    """Tool for getting a specific GitHub issue."""

    metadata = ToolMetadata(
        id="get_github_issue",
        name="Get GitHub Issue",
        description="Get a specific issue from a GitHub repository",
        category="utility",
    )
    input_type = GetIssueInput
    output_type = IssueOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetIssueInput) -> IssueOutput:
        """Get issue."""
        client = self.manager.get_client(input.client)

        if not client:
            return IssueOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_issue(
            owner=input.owner,
            repo=input.repo,
            issue_number=input.issue_number,
        )

        if result.success:
            return IssueOutput(success=True, issue=result.data)
        return IssueOutput(success=False, error=result.error)


class CreateIssueTool(BaseTool[CreateIssueInput, IssueOutput]):
    """Tool for creating a GitHub issue."""

    metadata = ToolMetadata(
        id="create_github_issue",
        name="Create GitHub Issue",
        description="Create a new issue in a GitHub repository",
        category="utility",
    )
    input_type = CreateIssueInput
    output_type = IssueOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateIssueInput) -> IssueOutput:
        """Create issue."""
        client = self.manager.get_client(input.client)

        if not client:
            return IssueOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.create_issue(
            owner=input.owner,
            repo=input.repo,
            title=input.title,
            body=input.body,
            labels=input.labels,
            assignees=input.assignees,
        )

        if result.success:
            return IssueOutput(success=True, issue=result.data)
        return IssueOutput(success=False, error=result.error)


class UpdateIssueTool(BaseTool[UpdateIssueInput, IssueOutput]):
    """Tool for updating a GitHub issue."""

    metadata = ToolMetadata(
        id="update_github_issue",
        name="Update GitHub Issue",
        description="Update an existing GitHub issue",
        category="utility",
    )
    input_type = UpdateIssueInput
    output_type = IssueOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: UpdateIssueInput) -> IssueOutput:
        """Update issue."""
        client = self.manager.get_client(input.client)

        if not client:
            return IssueOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.update_issue(
            owner=input.owner,
            repo=input.repo,
            issue_number=input.issue_number,
            title=input.title,
            body=input.body,
            state=input.state,
            labels=input.labels,
            assignees=input.assignees,
        )

        if result.success:
            return IssueOutput(success=True, issue=result.data)
        return IssueOutput(success=False, error=result.error)


class AddCommentTool(BaseTool[AddCommentInput, CommentOutput]):
    """Tool for adding a comment to an issue or PR."""

    metadata = ToolMetadata(
        id="add_github_comment",
        name="Add GitHub Comment",
        description="Add a comment to a GitHub issue or pull request",
        category="utility",
    )
    input_type = AddCommentInput
    output_type = CommentOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: AddCommentInput) -> CommentOutput:
        """Add comment."""
        client = self.manager.get_client(input.client)

        if not client:
            return CommentOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.add_comment(
            owner=input.owner,
            repo=input.repo,
            issue_number=input.issue_number,
            body=input.body,
        )

        if result.success:
            comment_id = result.data.get("id") if result.data else None
            return CommentOutput(success=True, comment_id=comment_id)
        return CommentOutput(success=False, error=result.error)


class ListPullRequestsTool(BaseTool[ListPullRequestsInput, PullRequestOutput]):
    """Tool for listing GitHub pull requests."""

    metadata = ToolMetadata(
        id="list_github_pull_requests",
        name="List GitHub Pull Requests",
        description="List pull requests in a GitHub repository",
        category="utility",
    )
    input_type = ListPullRequestsInput
    output_type = PullRequestOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListPullRequestsInput) -> PullRequestOutput:
        """List pull requests."""
        client = self.manager.get_client(input.client)

        if not client:
            return PullRequestOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        try:
            state = PullRequestState(input.state)
        except ValueError:
            return PullRequestOutput(
                success=False,
                error=f"Invalid state: {input.state}",
            )

        result = client.list_pull_requests(
            owner=input.owner,
            repo=input.repo,
            state=state,
            head=input.head,
            base=input.base,
            per_page=input.per_page,
            page=input.page,
        )

        if result.success:
            return PullRequestOutput(success=True, pull_requests=result.data)
        return PullRequestOutput(success=False, error=result.error)


class GetPullRequestTool(BaseTool[GetPullRequestInput, PullRequestOutput]):
    """Tool for getting a specific pull request."""

    metadata = ToolMetadata(
        id="get_github_pull_request",
        name="Get GitHub Pull Request",
        description="Get a specific pull request from a GitHub repository",
        category="utility",
    )
    input_type = GetPullRequestInput
    output_type = PullRequestOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetPullRequestInput) -> PullRequestOutput:
        """Get pull request."""
        client = self.manager.get_client(input.client)

        if not client:
            return PullRequestOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_pull_request(
            owner=input.owner,
            repo=input.repo,
            pr_number=input.pr_number,
        )

        if result.success:
            return PullRequestOutput(success=True, pull_request=result.data)
        return PullRequestOutput(success=False, error=result.error)


class CreatePullRequestTool(BaseTool[CreatePullRequestInput, PullRequestOutput]):
    """Tool for creating a pull request."""

    metadata = ToolMetadata(
        id="create_github_pull_request",
        name="Create GitHub Pull Request",
        description="Create a new pull request in a GitHub repository",
        category="utility",
    )
    input_type = CreatePullRequestInput
    output_type = PullRequestOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreatePullRequestInput) -> PullRequestOutput:
        """Create pull request."""
        client = self.manager.get_client(input.client)

        if not client:
            return PullRequestOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.create_pull_request(
            owner=input.owner,
            repo=input.repo,
            title=input.title,
            head=input.head,
            base=input.base,
            body=input.body,
            draft=input.draft,
        )

        if result.success:
            return PullRequestOutput(success=True, pull_request=result.data)
        return PullRequestOutput(success=False, error=result.error)


class GetRepositoryTool(BaseTool[GetRepositoryInput, RepositoryOutput]):
    """Tool for getting repository information."""

    metadata = ToolMetadata(
        id="get_github_repository",
        name="Get GitHub Repository",
        description="Get information about a GitHub repository",
        category="utility",
    )
    input_type = GetRepositoryInput
    output_type = RepositoryOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetRepositoryInput) -> RepositoryOutput:
        """Get repository."""
        client = self.manager.get_client(input.client)

        if not client:
            return RepositoryOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_repository(
            owner=input.owner,
            repo=input.repo,
        )

        if result.success:
            return RepositoryOutput(success=True, repository=result.data)
        return RepositoryOutput(success=False, error=result.error)


class ListRepositoriesTool(BaseTool[ListRepositoriesInput, RepositoryOutput]):
    """Tool for listing repositories."""

    metadata = ToolMetadata(
        id="list_github_repositories",
        name="List GitHub Repositories",
        description="List repositories for a user or organization",
        category="utility",
    )
    input_type = ListRepositoriesInput
    output_type = RepositoryOutput

    def __init__(self, manager: GitHubManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListRepositoriesInput) -> RepositoryOutput:
        """List repositories."""
        client = self.manager.get_client(input.client)

        if not client:
            return RepositoryOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.list_repositories(
            org=input.org,
            user=input.user,
            per_page=input.per_page,
            page=input.page,
        )

        if result.success:
            return RepositoryOutput(success=True, repositories=result.data)
        return RepositoryOutput(success=False, error=result.error)


# Convenience functions


def create_github_config(
    token: str,
    base_url: str = "https://api.github.com",
    timeout: int = 30,
) -> GitHubConfig:
    """Create a GitHub configuration.

    Args:
        token: GitHub personal access token.
        base_url: GitHub API base URL.
        timeout: Request timeout.

    Returns:
        GitHub configuration.
    """
    return GitHubConfig(token=token, base_url=base_url, timeout=timeout)


def create_github_client(config: GitHubConfig) -> GitHubClient:
    """Create a GitHub client.

    Args:
        config: GitHub configuration.

    Returns:
        GitHub client.
    """
    return GitHubClient(config)


def create_github_manager() -> GitHubManager:
    """Create a GitHub manager.

    Returns:
        GitHub manager.
    """
    return GitHubManager()


def create_github_tools(manager: GitHubManager) -> Dict[str, BaseTool]:
    """Create all GitHub tools.

    Args:
        manager: GitHub manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "create_github_client": CreateGitHubClientTool(manager),
        "list_github_issues": ListIssuesTool(manager),
        "get_github_issue": GetIssueTool(manager),
        "create_github_issue": CreateIssueTool(manager),
        "update_github_issue": UpdateIssueTool(manager),
        "add_github_comment": AddCommentTool(manager),
        "list_github_pull_requests": ListPullRequestsTool(manager),
        "get_github_pull_request": GetPullRequestTool(manager),
        "create_github_pull_request": CreatePullRequestTool(manager),
        "get_github_repository": GetRepositoryTool(manager),
        "list_github_repositories": ListRepositoriesTool(manager),
    }
