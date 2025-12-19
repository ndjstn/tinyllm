"""Tests for API calling tools."""

import pytest
from unittest.mock import MagicMock, patch
import json
import urllib.error
import urllib.request

from tinyllm.tools.api_caller import (
    ApiCallInput,
    ApiCallOutput,
    ApiCallTool,
    ApiClientManager,
    ApiResponse,
    AuthConfig,
    AuthType,
    ContentType,
    CreateApiClientTool,
    CreateClientInput,
    CreateClientOutput,
    HttpClient,
    HttpMethod,
    ListApiClientsTool,
    ListClientsOutput,
    RetryConfig,
    create_api_client_manager,
    create_api_tools,
    create_http_client,
)


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AuthConfig()

        assert config.auth_type == AuthType.NONE
        assert config.token is None

    def test_bearer_config(self):
        """Test bearer token configuration."""
        config = AuthConfig(
            auth_type=AuthType.BEARER,
            token="my-token",
        )

        assert config.auth_type == AuthType.BEARER
        assert config.token == "my-token"

    def test_api_key_config(self):
        """Test API key configuration."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            api_key="my-key",
            api_key_header="X-Custom-Key",
        )

        assert config.api_key == "my-key"
        assert config.api_key_header == "X-Custom-Key"

    def test_basic_config(self):
        """Test basic auth configuration."""
        config = AuthConfig(
            auth_type=AuthType.BASIC,
            username="user",
            password="pass",
        )

        assert config.username == "user"
        assert config.password == "pass"


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert 429 in config.retry_on_status
        assert 503 in config.retry_on_status

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            retry_on_status=[500, 502],
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert len(config.retry_on_status) == 2


class TestApiResponse:
    """Tests for ApiResponse."""

    def test_success_response(self):
        """Test successful response."""
        response = ApiResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"data": "test"},
            text='{"data": "test"}',
            elapsed_ms=100.0,
            success=True,
        )

        assert response.is_success is True
        assert response.is_client_error is False
        assert response.is_server_error is False

    def test_client_error_response(self):
        """Test client error response."""
        response = ApiResponse(
            status_code=404,
            headers={},
            success=False,
            error="Not Found",
        )

        assert response.is_success is False
        assert response.is_client_error is True
        assert response.is_server_error is False

    def test_server_error_response(self):
        """Test server error response."""
        response = ApiResponse(
            status_code=500,
            headers={},
            success=False,
            error="Internal Server Error",
        )

        assert response.is_success is False
        assert response.is_client_error is False
        assert response.is_server_error is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        response = ApiResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"data": "test"},
            elapsed_ms=50.0,
        )

        d = response.to_dict()

        assert d["status_code"] == 200
        assert d["elapsed_ms"] == 50.0


class TestHttpClient:
    """Tests for HttpClient."""

    def test_creation(self):
        """Test client creation."""
        client = HttpClient(
            base_url="https://api.example.com",
            timeout=60.0,
        )

        assert client.base_url == "https://api.example.com"
        assert client.timeout == 60.0

    def test_build_url(self):
        """Test URL building."""
        client = HttpClient(base_url="https://api.example.com")

        # Relative path
        url1 = client._build_url("/users")
        assert url1 == "https://api.example.com/users"

        # Absolute URL
        url2 = client._build_url("https://other.com/endpoint")
        assert url2 == "https://other.com/endpoint"

    def test_bearer_auth_headers(self):
        """Test bearer auth headers."""
        client = HttpClient(
            auth=AuthConfig(
                auth_type=AuthType.BEARER,
                token="my-token",
            )
        )

        headers = client._get_auth_headers()

        assert headers["Authorization"] == "Bearer my-token"

    def test_api_key_auth_headers(self):
        """Test API key auth headers."""
        client = HttpClient(
            auth=AuthConfig(
                auth_type=AuthType.API_KEY,
                api_key="my-key",
                api_key_header="X-API-Key",
            )
        )

        headers = client._get_auth_headers()

        assert headers["X-API-Key"] == "my-key"

    def test_basic_auth_headers(self):
        """Test basic auth headers."""
        client = HttpClient(
            auth=AuthConfig(
                auth_type=AuthType.BASIC,
                username="user",
                password="pass",
            )
        )

        headers = client._get_auth_headers()

        assert headers["Authorization"].startswith("Basic ")

    def test_custom_header_auth(self):
        """Test custom header auth."""
        client = HttpClient(
            auth=AuthConfig(
                auth_type=AuthType.CUSTOM_HEADER,
                custom_headers={"X-Custom": "value"},
            )
        )

        headers = client._get_auth_headers()

        assert headers["X-Custom"] == "value"

    def test_prepare_headers(self):
        """Test header preparation."""
        client = HttpClient(
            default_headers={"User-Agent": "TestClient"},
        )

        headers = client._prepare_headers(
            headers={"X-Custom": "value"},
            content_type=ContentType.JSON,
        )

        assert headers["User-Agent"] == "TestClient"
        assert headers["X-Custom"] == "value"
        assert headers["Content-Type"] == ContentType.JSON

    def test_prepare_body_json(self):
        """Test JSON body preparation."""
        client = HttpClient()

        body = client._prepare_body(
            {"key": "value"},
            ContentType.JSON,
        )

        assert body == '{"key": "value"}'

    def test_prepare_body_string(self):
        """Test string body preparation."""
        client = HttpClient()

        body = client._prepare_body(
            "raw text",
            ContentType.TEXT,
        )

        assert body == "raw text"

    def test_should_retry(self):
        """Test retry decision."""
        client = HttpClient()

        assert client._should_retry(429, 0) is True
        assert client._should_retry(200, 0) is False
        assert client._should_retry(429, 5) is False

    def test_get_retry_delay(self):
        """Test retry delay calculation."""
        client = HttpClient(
            retry_config=RetryConfig(
                initial_delay=1.0,
                exponential_base=2.0,
            )
        )

        delay0 = client._get_retry_delay(0)
        delay1 = client._get_retry_delay(1)
        delay2 = client._get_retry_delay(2)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    @patch("urllib.request.urlopen")
    def test_get_request(self, mock_urlopen):
        """Test GET request."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.read.return_value = b'{"data": "test"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        response = client.get("/users")

        assert response.success is True
        assert response.status_code == 200

    @patch("urllib.request.urlopen")
    def test_post_request(self, mock_urlopen):
        """Test POST request."""
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.read.return_value = b'{"id": 1}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        response = client.post("/users", json_data={"name": "test"})

        assert response.success is True
        assert response.status_code == 201

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test HTTP error handling."""
        mock_error = urllib.error.HTTPError(
            url="https://api.example.com/users",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        mock_error.read = MagicMock(return_value=b'{"error": "not found"}')
        mock_error.headers = {}

        mock_urlopen.side_effect = mock_error

        client = HttpClient(
            base_url="https://api.example.com",
            retry_config=RetryConfig(max_retries=0),
        )
        response = client.get("/users")

        assert response.success is False
        assert response.status_code == 404

    @patch("urllib.request.urlopen")
    def test_url_error_handling(self, mock_urlopen):
        """Test URL error handling."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = HttpClient(
            base_url="https://api.example.com",
            retry_config=RetryConfig(max_retries=0),
        )
        response = client.get("/users")

        assert response.success is False
        assert response.status_code == 0

    @patch("urllib.request.urlopen")
    def test_query_params(self, mock_urlopen):
        """Test query parameters."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        client.get("/users", params={"page": 1, "limit": 10})

        # Check the URL used in the request
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "page=1" in request.full_url
        assert "limit=10" in request.full_url


class TestApiClientManager:
    """Tests for ApiClientManager."""

    def test_add_client(self):
        """Test adding a client."""
        manager = ApiClientManager()
        client = HttpClient()

        manager.add_client("default", client)

        assert manager.get_client("default") == client

    def test_get_client_not_found(self):
        """Test getting non-existent client."""
        manager = ApiClientManager()

        client = manager.get_client("nonexistent")

        assert client is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = ApiClientManager()
        client = HttpClient()

        manager.add_client("default", client)
        removed = manager.remove_client("default")

        assert removed is True
        assert manager.get_client("default") is None

    def test_list_clients(self):
        """Test listing clients."""
        manager = ApiClientManager()

        manager.add_client("api1", HttpClient())
        manager.add_client("api2", HttpClient())

        clients = manager.list_clients()

        assert "api1" in clients
        assert "api2" in clients


class TestApiCallTool:
    """Tests for ApiCallTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up API client manager."""
        manager = ApiClientManager()
        client = HttpClient(base_url="https://api.example.com")
        manager.add_client("default", client)
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_successful_call(self, mock_urlopen, setup_manager):
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.read.return_value = b'{"data": "test"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = mock_response

        tool = ApiCallTool(setup_manager)

        result = await tool.execute(
            ApiCallInput(
                client="default",
                method="GET",
                path="/users",
            )
        )

        assert result.success is True
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_client_not_found(self, setup_manager):
        """Test with non-existent client."""
        tool = ApiCallTool(setup_manager)

        result = await tool.execute(
            ApiCallInput(
                client="nonexistent",
                method="GET",
                path="/users",
            )
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_invalid_method(self, setup_manager):
        """Test with invalid HTTP method."""
        tool = ApiCallTool(setup_manager)

        result = await tool.execute(
            ApiCallInput(
                client="default",
                method="INVALID",
                path="/users",
            )
        )

        assert result.success is False
        assert "Invalid HTTP method" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_post_with_body(self, mock_urlopen, setup_manager):
        """Test POST with request body."""
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.read.return_value = b'{"id": 1}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.return_value = mock_response

        tool = ApiCallTool(setup_manager)

        result = await tool.execute(
            ApiCallInput(
                client="default",
                method="POST",
                path="/users",
                body={"name": "test"},
            )
        )

        assert result.success is True
        assert result.status_code == 201


class TestCreateApiClientTool:
    """Tests for CreateApiClientTool."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        return ApiClientManager()

    @pytest.mark.asyncio
    async def test_create_basic_client(self, manager):
        """Test creating a basic client."""
        tool = CreateApiClientTool(manager)

        result = await tool.execute(
            CreateClientInput(
                name="test",
                base_url="https://api.example.com",
            )
        )

        assert result.success is True
        assert result.name == "test"
        assert manager.get_client("test") is not None

    @pytest.mark.asyncio
    async def test_create_bearer_client(self, manager):
        """Test creating a client with bearer auth."""
        tool = CreateApiClientTool(manager)

        result = await tool.execute(
            CreateClientInput(
                name="auth_client",
                base_url="https://api.example.com",
                auth_type="bearer",
                token="my-token",
            )
        )

        assert result.success is True

        client = manager.get_client("auth_client")
        assert client.auth.auth_type == AuthType.BEARER
        assert client.auth.token == "my-token"

    @pytest.mark.asyncio
    async def test_create_api_key_client(self, manager):
        """Test creating a client with API key auth."""
        tool = CreateApiClientTool(manager)

        result = await tool.execute(
            CreateClientInput(
                name="api_client",
                base_url="https://api.example.com",
                auth_type="api_key",
                api_key="my-key",
                api_key_header="X-Custom-Key",
            )
        )

        assert result.success is True

        client = manager.get_client("api_client")
        assert client.auth.api_key == "my-key"
        assert client.auth.api_key_header == "X-Custom-Key"

    @pytest.mark.asyncio
    async def test_invalid_auth_type(self, manager):
        """Test with invalid auth type."""
        tool = CreateApiClientTool(manager)

        result = await tool.execute(
            CreateClientInput(
                name="bad_client",
                base_url="https://api.example.com",
                auth_type="invalid",
            )
        )

        assert result.success is False
        assert "Invalid auth type" in result.error


class TestListApiClientsTool:
    """Tests for ListApiClientsTool."""

    @pytest.mark.asyncio
    async def test_list_clients(self):
        """Test listing clients."""
        manager = ApiClientManager()
        manager.add_client("api1", HttpClient())
        manager.add_client("api2", HttpClient())

        tool = ListApiClientsTool(manager)

        result = await tool.execute(ListClientsOutput())

        assert "api1" in result.clients
        assert "api2" in result.clients

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Test listing with no clients."""
        manager = ApiClientManager()
        tool = ListApiClientsTool(manager)

        result = await tool.execute(ListClientsOutput())

        assert len(result.clients) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_http_client(self):
        """Test creating HTTP client."""
        client = create_http_client(
            base_url="https://api.example.com",
            auth_type=AuthType.BEARER,
            token="my-token",
            timeout=60.0,
        )

        assert client.base_url == "https://api.example.com"
        assert client.timeout == 60.0
        assert client.auth.auth_type == AuthType.BEARER

    def test_create_api_client_manager(self):
        """Test creating API client manager."""
        manager = create_api_client_manager()

        assert isinstance(manager, ApiClientManager)

    def test_create_api_tools(self):
        """Test creating all API tools."""
        manager = create_api_client_manager()
        tools = create_api_tools(manager)

        assert "api_call" in tools
        assert "create_api_client" in tools
        assert "list_api_clients" in tools


class TestHttpMethods:
    """Tests for different HTTP methods."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.read.return_value = b'{"success": true}'
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    @patch("urllib.request.urlopen")
    def test_put_request(self, mock_urlopen, mock_response):
        """Test PUT request."""
        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        response = client.put("/users/1", json_data={"name": "updated"})

        assert response.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.method == "PUT"

    @patch("urllib.request.urlopen")
    def test_delete_request(self, mock_urlopen, mock_response):
        """Test DELETE request."""
        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        response = client.delete("/users/1")

        assert response.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.method == "DELETE"

    @patch("urllib.request.urlopen")
    def test_patch_request(self, mock_urlopen, mock_response):
        """Test PATCH request."""
        mock_urlopen.return_value = mock_response

        client = HttpClient(base_url="https://api.example.com")
        response = client.patch("/users/1", json_data={"name": "patched"})

        assert response.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.method == "PATCH"
