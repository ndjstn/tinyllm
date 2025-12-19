"""API calling tools for TinyLLM.

This module provides tools for making HTTP API requests with support for
various authentication methods, request types, and error handling.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class HttpMethod(str, Enum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Authentication types."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    BASIC = "basic"
    CUSTOM_HEADER = "custom_header"


class ContentType(str, Enum):
    """Common content types."""

    JSON = "application/json"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    XML = "application/xml"


@dataclass
class AuthConfig:
    """Authentication configuration."""

    auth_type: AuthType = AuthType.NONE
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retry_on_status: List[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


@dataclass
class ApiResponse:
    """Response from an API call."""

    status_code: int
    headers: Dict[str, str]
    body: Any = None
    text: str = ""
    elapsed_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "body": self.body,
            "text": self.text[:1000] if self.text else "",
            "elapsed_ms": self.elapsed_ms,
            "success": self.success,
            "error": self.error,
            "retries": self.retries,
        }

    @property
    def is_success(self) -> bool:
        """Check if response is successful (2xx status)."""
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        """Check if response is a client error (4xx status)."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if response is a server error (5xx status)."""
        return 500 <= self.status_code < 600


class HttpClient:
    """HTTP client for making API requests."""

    def __init__(
        self,
        base_url: str = "",
        auth: Optional[AuthConfig] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize HTTP client.

        Args:
            base_url: Base URL for all requests.
            auth: Authentication configuration.
            default_headers: Default headers for all requests.
            timeout: Request timeout in seconds.
            retry_config: Retry configuration.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth or AuthConfig()
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}

        if self.auth.auth_type == AuthType.BEARER:
            if self.auth.token:
                headers["Authorization"] = f"Bearer {self.auth.token}"

        elif self.auth.auth_type == AuthType.API_KEY:
            if self.auth.api_key:
                headers[self.auth.api_key_header] = self.auth.api_key

        elif self.auth.auth_type == AuthType.BASIC:
            if self.auth.username and self.auth.password:
                import base64

                credentials = f"{self.auth.username}:{self.auth.password}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif self.auth.auth_type == AuthType.CUSTOM_HEADER:
            headers.update(self.auth.custom_headers)

        return headers

    def _prepare_headers(
        self,
        headers: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare request headers."""
        result = dict(self.default_headers)
        result.update(self._get_auth_headers())

        if content_type:
            result["Content-Type"] = content_type

        if headers:
            result.update(headers)

        return result

    def _prepare_body(
        self,
        data: Any,
        content_type: str,
    ) -> Union[str, bytes, None]:
        """Prepare request body."""
        if data is None:
            return None

        if content_type == ContentType.JSON or "json" in content_type:
            return json.dumps(data)

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # URL-encoded form data
            from urllib.parse import urlencode

            return urlencode(data)

        return str(data)

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Check if request should be retried."""
        if attempt >= self.retry_config.max_retries:
            return False
        return status_code in self.retry_config.retry_on_status

    def _get_retry_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        delay = self.retry_config.initial_delay * (
            self.retry_config.exponential_base ** attempt
        )
        return min(delay, self.retry_config.max_delay)

    def request(
        self,
        method: HttpMethod,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Any = None,
        content_type: str = ContentType.JSON,
    ) -> ApiResponse:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            path: URL path or full URL.
            headers: Additional headers.
            params: Query parameters.
            data: Request body data.
            json_data: JSON request body (convenience for JSON content).
            content_type: Content type for the request.

        Returns:
            API response.
        """
        try:
            import urllib.request
            import urllib.error
            import urllib.parse

            url = self._build_url(path)

            # Add query parameters
            if params:
                query_string = urllib.parse.urlencode(params)
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{query_string}"

            # Prepare headers
            prepared_headers = self._prepare_headers(headers, content_type)

            # Prepare body
            body = None
            if json_data is not None:
                body = json.dumps(json_data).encode("utf-8")
                prepared_headers["Content-Type"] = ContentType.JSON
            elif data is not None:
                prepared_body = self._prepare_body(data, content_type)
                if prepared_body:
                    body = prepared_body.encode("utf-8") if isinstance(
                        prepared_body, str
                    ) else prepared_body

            attempt = 0
            last_error = None

            while True:
                start_time = time.time()

                try:
                    req = urllib.request.Request(
                        url,
                        data=body,
                        headers=prepared_headers,
                        method=method.value,
                    )

                    with urllib.request.urlopen(
                        req, timeout=self.timeout
                    ) as response:
                        elapsed_ms = (time.time() - start_time) * 1000
                        response_body = response.read().decode("utf-8")

                        # Parse response headers
                        response_headers = dict(response.headers)

                        # Try to parse JSON response
                        parsed_body: Any = response_body
                        if "application/json" in response.headers.get(
                            "Content-Type", ""
                        ):
                            try:
                                parsed_body = json.loads(response_body)
                            except json.JSONDecodeError:
                                pass

                        return ApiResponse(
                            status_code=response.status,
                            headers=response_headers,
                            body=parsed_body,
                            text=response_body,
                            elapsed_ms=elapsed_ms,
                            success=True,
                            retries=attempt,
                        )

                except urllib.error.HTTPError as e:
                    elapsed_ms = (time.time() - start_time) * 1000
                    response_body = ""
                    try:
                        response_body = e.read().decode("utf-8")
                    except Exception:
                        pass

                    if self._should_retry(e.code, attempt):
                        delay = self._get_retry_delay(attempt)
                        time.sleep(delay)
                        attempt += 1
                        last_error = str(e)
                        continue

                    # Parse error response body
                    parsed_body: Any = response_body
                    try:
                        parsed_body = json.loads(response_body)
                    except json.JSONDecodeError:
                        pass

                    return ApiResponse(
                        status_code=e.code,
                        headers=dict(e.headers) if e.headers else {},
                        body=parsed_body,
                        text=response_body,
                        elapsed_ms=elapsed_ms,
                        success=False,
                        error=str(e.reason),
                        retries=attempt,
                    )

                except urllib.error.URLError as e:
                    elapsed_ms = (time.time() - start_time) * 1000

                    if attempt < self.retry_config.max_retries:
                        delay = self._get_retry_delay(attempt)
                        time.sleep(delay)
                        attempt += 1
                        last_error = str(e)
                        continue

                    return ApiResponse(
                        status_code=0,
                        headers={},
                        elapsed_ms=elapsed_ms,
                        success=False,
                        error=str(e.reason),
                        retries=attempt,
                    )

        except Exception as e:
            return ApiResponse(
                status_code=0,
                headers={},
                success=False,
                error=str(e),
            )

    def get(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ApiResponse:
        """Make a GET request."""
        return self.request(HttpMethod.GET, path, headers=headers, params=params)

    def post(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Any = None,
    ) -> ApiResponse:
        """Make a POST request."""
        return self.request(
            HttpMethod.POST,
            path,
            headers=headers,
            params=params,
            data=data,
            json_data=json_data,
        )

    def put(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Any = None,
    ) -> ApiResponse:
        """Make a PUT request."""
        return self.request(
            HttpMethod.PUT,
            path,
            headers=headers,
            params=params,
            data=data,
            json_data=json_data,
        )

    def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ApiResponse:
        """Make a DELETE request."""
        return self.request(HttpMethod.DELETE, path, headers=headers, params=params)

    def patch(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
        json_data: Any = None,
    ) -> ApiResponse:
        """Make a PATCH request."""
        return self.request(
            HttpMethod.PATCH,
            path,
            headers=headers,
            params=params,
            data=data,
            json_data=json_data,
        )


class ApiClientManager:
    """Manager for API clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, HttpClient] = {}

    def add_client(self, name: str, client: HttpClient) -> None:
        """Add an API client.

        Args:
            name: Client name.
            client: HTTP client.
        """
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[HttpClient]:
        """Get an API client.

        Args:
            name: Client name.

        Returns:
            HTTP client or None.
        """
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove an API client.

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


class ApiCallInput(BaseModel):
    """Input for API call tool."""

    client: str = Field(
        default="default",
        description="Name of the API client to use",
    )
    method: str = Field(
        default="GET",
        description="HTTP method (GET, POST, PUT, DELETE, PATCH)",
    )
    path: str = Field(
        ...,
        description="URL path or full URL to call",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional request headers",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Query parameters",
    )
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Request body (JSON)",
    )


class ApiCallOutput(BaseModel):
    """Output from API call tool."""

    success: bool = Field(description="Whether the request succeeded")
    status_code: int = Field(description="HTTP status code")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Response headers",
    )
    body: Optional[Any] = Field(
        default=None,
        description="Response body (parsed if JSON)",
    )
    text: str = Field(
        default="",
        description="Raw response text",
    )
    elapsed_ms: float = Field(
        default=0.0,
        description="Request time in milliseconds",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if request failed",
    )


class CreateClientInput(BaseModel):
    """Input for creating an API client."""

    name: str = Field(
        ...,
        description="Name for the client",
    )
    base_url: str = Field(
        ...,
        description="Base URL for the API",
    )
    auth_type: str = Field(
        default="none",
        description="Authentication type (none, bearer, api_key, basic)",
    )
    token: Optional[str] = Field(
        default=None,
        description="Bearer token (for auth_type=bearer)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (for auth_type=api_key)",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key",
    )
    username: Optional[str] = Field(
        default=None,
        description="Username (for auth_type=basic)",
    )
    password: Optional[str] = Field(
        default=None,
        description="Password (for auth_type=basic)",
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default headers for all requests",
    )


class CreateClientOutput(BaseModel):
    """Output from creating an API client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(
        default=None,
        description="Error message if creation failed",
    )


class ListClientsOutput(BaseModel):
    """Output from listing API clients."""

    clients: List[str] = Field(
        default_factory=list,
        description="List of client names",
    )


# Tool implementations


class ApiCallTool(BaseTool[ApiCallInput, ApiCallOutput]):
    """Tool for making API calls."""

    metadata = ToolMetadata(
        id="api_call",
        name="API Call",
        description="Make HTTP API requests",
        category="utility",
    )
    input_type = ApiCallInput
    output_type = ApiCallOutput

    def __init__(self, manager: ApiClientManager):
        """Initialize tool.

        Args:
            manager: API client manager.
        """
        self.manager = manager

    async def execute(self, input: ApiCallInput) -> ApiCallOutput:
        """Execute an API call."""
        client = self.manager.get_client(input.client)

        if not client:
            return ApiCallOutput(
                success=False,
                status_code=0,
                error=f"Client '{input.client}' not found",
            )

        try:
            method = HttpMethod(input.method.upper())
        except ValueError:
            return ApiCallOutput(
                success=False,
                status_code=0,
                error=f"Invalid HTTP method: {input.method}",
            )

        response = client.request(
            method=method,
            path=input.path,
            headers=input.headers,
            params=input.params,
            json_data=input.body,
        )

        return ApiCallOutput(
            success=response.success and response.is_success,
            status_code=response.status_code,
            headers=response.headers,
            body=response.body,
            text=response.text[:5000] if response.text else "",
            elapsed_ms=response.elapsed_ms,
            error=response.error,
        )


class CreateApiClientTool(BaseTool[CreateClientInput, CreateClientOutput]):
    """Tool for creating API clients."""

    metadata = ToolMetadata(
        id="create_api_client",
        name="Create API Client",
        description="Create a new API client with authentication",
        category="utility",
    )
    input_type = CreateClientInput
    output_type = CreateClientOutput

    def __init__(self, manager: ApiClientManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateClientInput) -> CreateClientOutput:
        """Create an API client."""
        try:
            # Build auth config
            auth_type = AuthType(input.auth_type.lower())
            auth = AuthConfig(
                auth_type=auth_type,
                token=input.token,
                api_key=input.api_key,
                api_key_header=input.api_key_header,
                username=input.username,
                password=input.password,
            )

            # Create client
            client = HttpClient(
                base_url=input.base_url,
                auth=auth,
                default_headers=input.default_headers or {},
                timeout=input.timeout,
            )

            self.manager.add_client(input.name, client)

            return CreateClientOutput(
                success=True,
                name=input.name,
            )

        except ValueError as e:
            return CreateClientOutput(
                success=False,
                name=input.name,
                error=f"Invalid auth type: {input.auth_type}",
            )
        except Exception as e:
            return CreateClientOutput(
                success=False,
                name=input.name,
                error=str(e),
            )


class ListApiClientsTool(BaseTool[BaseModel, ListClientsOutput]):
    """Tool for listing API clients."""

    metadata = ToolMetadata(
        id="list_api_clients",
        name="List API Clients",
        description="List all configured API clients",
        category="utility",
    )
    input_type = BaseModel
    output_type = ListClientsOutput

    def __init__(self, manager: ApiClientManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: BaseModel) -> ListClientsOutput:
        """List API clients."""
        return ListClientsOutput(clients=self.manager.list_clients())


# Convenience functions


def create_http_client(
    base_url: str = "",
    auth_type: AuthType = AuthType.NONE,
    token: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_header: str = "X-API-Key",
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: float = 30.0,
    default_headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
) -> HttpClient:
    """Create an HTTP client.

    Args:
        base_url: Base URL for requests.
        auth_type: Authentication type.
        token: Bearer token.
        api_key: API key.
        api_key_header: API key header name.
        username: Basic auth username.
        password: Basic auth password.
        timeout: Request timeout.
        default_headers: Default headers.
        max_retries: Maximum retry attempts.

    Returns:
        HTTP client.
    """
    auth = AuthConfig(
        auth_type=auth_type,
        token=token,
        api_key=api_key,
        api_key_header=api_key_header,
        username=username,
        password=password,
    )

    retry_config = RetryConfig(max_retries=max_retries)

    return HttpClient(
        base_url=base_url,
        auth=auth,
        default_headers=default_headers or {},
        timeout=timeout,
        retry_config=retry_config,
    )


def create_api_client_manager() -> ApiClientManager:
    """Create an API client manager.

    Returns:
        API client manager.
    """
    return ApiClientManager()


def create_api_tools(manager: ApiClientManager) -> Dict[str, BaseTool]:
    """Create all API tools.

    Args:
        manager: API client manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "api_call": ApiCallTool(manager),
        "create_api_client": CreateApiClientTool(manager),
        "list_api_clients": ListApiClientsTool(manager),
    }
