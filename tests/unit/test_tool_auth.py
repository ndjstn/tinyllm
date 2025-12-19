"""Tests for tool authentication."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.auth import (
    ApiKeyAuthenticator,
    AuthContext,
    AuthCredentials,
    AuthenticatedToolWrapper,
    AuthenticationError,
    AuthMethod,
    AuthorizationError,
    Authorizer,
    HmacAuthenticator,
    NoAuthenticator,
    Permission,
    TokenAuthenticator,
    ToolAuthManager,
)


class AuthInput(BaseModel):
    """Input for auth test tool."""

    data: str = ""


class AuthOutput(BaseModel):
    """Output for auth test tool."""

    success: bool = True
    error: str | None = None


class AuthTool(BaseTool[AuthInput, AuthOutput]):
    """Tool for auth tests."""

    metadata = ToolMetadata(
        id="auth_tool",
        name="Auth Tool",
        description="A tool for auth tests",
        category="utility",
    )
    input_type = AuthInput
    output_type = AuthOutput

    async def execute(self, input: AuthInput) -> AuthOutput:
        return AuthOutput()


class TestAuthCredentials:
    """Tests for AuthCredentials."""

    def test_credentials_api_key(self):
        """Test API key credentials."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="test-key",
        )

        assert creds.method == AuthMethod.API_KEY
        assert creds.api_key == "test-key"

    def test_credentials_token(self):
        """Test token credentials."""
        creds = AuthCredentials(
            method=AuthMethod.TOKEN,
            token="test-token",
        )

        assert creds.method == AuthMethod.TOKEN
        assert creds.token == "test-token"


class TestAuthContext:
    """Tests for AuthContext."""

    def test_context_defaults(self):
        """Test default context."""
        ctx = AuthContext()

        assert ctx.user_id is None
        assert len(ctx.permissions) == 0
        assert not ctx.authenticated

    def test_context_with_data(self):
        """Test context with data."""
        ctx = AuthContext(
            user_id="user123",
            permissions={Permission.EXECUTE, Permission.READ},
            authenticated=True,
        )

        assert ctx.user_id == "user123"
        assert Permission.EXECUTE in ctx.permissions
        assert ctx.authenticated


class TestNoAuthenticator:
    """Tests for NoAuthenticator."""

    @pytest.mark.asyncio
    async def test_always_succeeds(self):
        """Test that no auth always succeeds."""
        auth = NoAuthenticator()
        creds = AuthCredentials(method=AuthMethod.NONE)

        ctx = await auth.authenticate(creds)

        assert ctx.authenticated
        assert Permission.EXECUTE in ctx.permissions


class TestApiKeyAuthenticator:
    """Tests for ApiKeyAuthenticator."""

    @pytest.fixture
    def auth(self):
        """Create authenticator with registered key."""
        auth = ApiKeyAuthenticator()
        auth.register_key(
            "test-api-key",
            user_id="user123",
            permissions={Permission.EXECUTE, Permission.READ},
            roles={"developer"},
        )
        return auth

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self, auth):
        """Test authentication with valid key."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="test-api-key",
        )

        ctx = await auth.authenticate(creds)

        assert ctx.authenticated
        assert ctx.user_id == "user123"
        assert Permission.EXECUTE in ctx.permissions
        assert "developer" in ctx.roles

    @pytest.mark.asyncio
    async def test_authenticate_invalid_key(self, auth):
        """Test authentication with invalid key."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="wrong-key",
        )

        with pytest.raises(AuthenticationError):
            await auth.authenticate(creds)

    @pytest.mark.asyncio
    async def test_authenticate_no_key(self, auth):
        """Test authentication without key."""
        creds = AuthCredentials(method=AuthMethod.API_KEY)

        with pytest.raises(AuthenticationError):
            await auth.authenticate(creds)

    def test_revoke_key(self, auth):
        """Test revoking a key."""
        assert auth.revoke_key("test-api-key")
        assert not auth.revoke_key("nonexistent")


class TestTokenAuthenticator:
    """Tests for TokenAuthenticator."""

    @pytest.fixture
    def auth(self):
        """Create authenticator."""
        return TokenAuthenticator(token_ttl=3600)

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, auth):
        """Test authentication with valid token."""
        token = auth.generate_token(
            "user123",
            permissions={Permission.EXECUTE},
        )
        creds = AuthCredentials(method=AuthMethod.TOKEN, token=token)

        ctx = await auth.authenticate(creds)

        assert ctx.authenticated
        assert ctx.user_id == "user123"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, auth):
        """Test authentication with invalid token."""
        creds = AuthCredentials(method=AuthMethod.TOKEN, token="invalid")

        with pytest.raises(AuthenticationError):
            await auth.authenticate(creds)

    @pytest.mark.asyncio
    async def test_authenticate_expired_token(self):
        """Test authentication with expired token."""
        auth = TokenAuthenticator(token_ttl=0)  # Immediate expiration
        token = auth.generate_token("user123")
        creds = AuthCredentials(method=AuthMethod.TOKEN, token=token)

        with pytest.raises(AuthenticationError) as exc_info:
            await auth.authenticate(creds)

        assert "expired" in str(exc_info.value).lower()

    def test_revoke_token(self, auth):
        """Test revoking a token."""
        token = auth.generate_token("user123")

        assert auth.revoke_token(token)
        assert not auth.revoke_token("nonexistent")


class TestHmacAuthenticator:
    """Tests for HmacAuthenticator."""

    @pytest.fixture
    def auth(self):
        """Create authenticator with registered secret."""
        auth = HmacAuthenticator()
        auth.register_secret(
            "key1",
            "secret123",
            user_id="user123",
            permissions={Permission.EXECUTE},
        )
        return auth

    def test_verify_valid_signature(self, auth):
        """Test verifying valid signature."""
        import hashlib
        import hmac

        message = "test message"
        signature = hmac.new(
            b"secret123", message.encode(), hashlib.sha256
        ).hexdigest()

        assert auth.verify_signature("key1", message, signature)

    def test_verify_invalid_signature(self, auth):
        """Test verifying invalid signature."""
        assert not auth.verify_signature("key1", "message", "invalid")

    @pytest.mark.asyncio
    async def test_authenticate_valid(self, auth):
        """Test authentication with valid HMAC."""
        import hashlib
        import hmac

        message = "test message"
        signature = hmac.new(
            b"secret123", message.encode(), hashlib.sha256
        ).hexdigest()

        creds = AuthCredentials(
            method=AuthMethod.HMAC,
            api_key="key1",
            secret=signature,
            metadata={"message": message},
        )

        ctx = await auth.authenticate(creds)
        assert ctx.authenticated


class TestAuthorizer:
    """Tests for Authorizer."""

    @pytest.fixture
    def authorizer(self):
        """Create authorizer."""
        auth = Authorizer()
        auth.set_tool_requirements(
            "admin_tool",
            {Permission.ADMIN},
        )
        auth.set_role_permissions(
            "admin",
            {Permission.ADMIN},
        )
        return auth

    def test_authorize_basic_permission(self, authorizer):
        """Test basic permission check."""
        ctx = AuthContext(
            authenticated=True,
            permissions={Permission.EXECUTE},
        )

        assert authorizer.authorize(ctx, "any_tool", Permission.EXECUTE)
        assert not authorizer.authorize(ctx, "any_tool", Permission.ADMIN)

    def test_authorize_with_role(self, authorizer):
        """Test authorization with role permissions."""
        ctx = AuthContext(
            authenticated=True,
            permissions=set(),
            roles={"admin"},
        )

        assert authorizer.authorize(ctx, "admin_tool", Permission.ADMIN)

    def test_authorize_not_authenticated(self, authorizer):
        """Test authorization when not authenticated."""
        ctx = AuthContext(authenticated=False)

        assert not authorizer.authorize(ctx, "any_tool", Permission.EXECUTE)

    def test_authorize_admin_grants_all(self, authorizer):
        """Test that admin grants all permissions."""
        ctx = AuthContext(
            authenticated=True,
            permissions={Permission.ADMIN},
        )

        assert authorizer.authorize(ctx, "admin_tool", Permission.ADMIN)
        assert authorizer.authorize(ctx, "any_tool", Permission.EXECUTE)

    def test_require_raises(self, authorizer):
        """Test require raises on failure."""
        ctx = AuthContext(
            authenticated=True,
            permissions={Permission.EXECUTE},
        )

        with pytest.raises(AuthorizationError):
            authorizer.require(ctx, "admin_tool", Permission.ADMIN)


class TestToolAuthManager:
    """Tests for ToolAuthManager."""

    @pytest.fixture
    def manager(self):
        """Create auth manager."""
        auth = ApiKeyAuthenticator()
        auth.register_key(
            "test-key",
            "user123",
            permissions={Permission.EXECUTE},
        )
        return ToolAuthManager(authenticator=auth)

    @pytest.mark.asyncio
    async def test_authenticate(self, manager):
        """Test authentication."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="test-key",
        )

        ctx = await manager.authenticate(creds)
        assert ctx.authenticated

    def test_authorize(self, manager):
        """Test authorization."""
        ctx = AuthContext(
            authenticated=True,
            permissions={Permission.EXECUTE},
        )

        assert manager.authorize(ctx, "tool", Permission.EXECUTE)

    @pytest.mark.asyncio
    async def test_authenticate_and_authorize(self, manager):
        """Test combined auth."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="test-key",
        )

        ctx = await manager.authenticate_and_authorize(
            creds, "tool", Permission.EXECUTE
        )
        assert ctx.authenticated


class TestAuthenticatedToolWrapper:
    """Tests for AuthenticatedToolWrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with auth."""
        auth = ApiKeyAuthenticator()
        auth.register_key("test-key", "user123", {Permission.EXECUTE})
        manager = ToolAuthManager(authenticator=auth)
        tool = AuthTool()
        return AuthenticatedToolWrapper(tool, auth_manager=manager)

    @pytest.mark.asyncio
    async def test_execute_with_auth(self, wrapper):
        """Test execution with valid auth."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="test-key",
        )

        result = await wrapper.execute(AuthInput(), credentials=creds)
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_no_credentials(self, wrapper):
        """Test execution without credentials."""
        with pytest.raises(AuthenticationError):
            await wrapper.execute(AuthInput())

    @pytest.mark.asyncio
    async def test_execute_invalid_credentials(self, wrapper):
        """Test execution with invalid credentials."""
        creds = AuthCredentials(
            method=AuthMethod.API_KEY,
            api_key="wrong-key",
        )

        with pytest.raises(AuthenticationError):
            await wrapper.execute(AuthInput(), credentials=creds)

    @pytest.mark.asyncio
    async def test_execute_no_auth_required(self):
        """Test execution when auth not required."""
        tool = AuthTool()
        wrapper = AuthenticatedToolWrapper(tool, require_auth=False)

        result = await wrapper.execute(AuthInput())
        assert result.success
