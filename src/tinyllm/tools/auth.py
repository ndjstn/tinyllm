"""Tool authentication for TinyLLM.

This module provides authentication and authorization
for tool access and execution.
"""

import hashlib
import hmac
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class AuthMethod(str, Enum):
    """Authentication methods."""

    NONE = "none"
    API_KEY = "api_key"
    TOKEN = "token"
    HMAC = "hmac"
    CUSTOM = "custom"


class Permission(str, Enum):
    """Tool permissions."""

    EXECUTE = "execute"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class AuthCredentials:
    """Authentication credentials."""

    method: AuthMethod
    api_key: Optional[str] = None
    token: Optional[str] = None
    secret: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthContext:
    """Authentication context for a request."""

    user_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    authenticated: bool = False
    credentials: Optional[AuthCredentials] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthenticationError(Exception):
    """Authentication failed."""

    pass


class AuthorizationError(Exception):
    """Authorization failed."""

    pass


class Authenticator(ABC):
    """Abstract base class for authenticators."""

    @abstractmethod
    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Authenticate with credentials.

        Args:
            credentials: Authentication credentials.

        Returns:
            Authentication context.

        Raises:
            AuthenticationError: If authentication fails.
        """
        pass


class NoAuthenticator(Authenticator):
    """No authentication required."""

    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Always succeeds with basic permissions."""
        return AuthContext(
            authenticated=True,
            permissions={Permission.EXECUTE, Permission.READ},
        )


class ApiKeyAuthenticator(Authenticator):
    """API key based authentication."""

    def __init__(self):
        """Initialize authenticator."""
        self._keys: Dict[str, Dict[str, Any]] = {}

    def register_key(
        self,
        api_key: str,
        user_id: str,
        permissions: Optional[Set[Permission]] = None,
        roles: Optional[Set[str]] = None,
    ) -> None:
        """Register an API key.

        Args:
            api_key: The API key.
            user_id: Associated user ID.
            permissions: Granted permissions.
            roles: Assigned roles.
        """
        key_hash = self._hash_key(api_key)
        self._keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions or {Permission.EXECUTE},
            "roles": roles or set(),
        }

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: The API key to revoke.

        Returns:
            True if key was revoked.
        """
        key_hash = self._hash_key(api_key)
        if key_hash in self._keys:
            del self._keys[key_hash]
            return True
        return False

    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Authenticate with API key."""
        if not credentials.api_key:
            raise AuthenticationError("API key required")

        key_hash = self._hash_key(credentials.api_key)
        if key_hash not in self._keys:
            raise AuthenticationError("Invalid API key")

        key_data = self._keys[key_hash]
        return AuthContext(
            user_id=key_data["user_id"],
            permissions=key_data["permissions"],
            roles=key_data["roles"],
            authenticated=True,
            credentials=credentials,
        )


class TokenAuthenticator(Authenticator):
    """Token based authentication with expiration."""

    def __init__(self, token_ttl: int = 3600):
        """Initialize authenticator.

        Args:
            token_ttl: Token time-to-live in seconds.
        """
        self.token_ttl = token_ttl
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def generate_token(
        self,
        user_id: str,
        permissions: Optional[Set[Permission]] = None,
        roles: Optional[Set[str]] = None,
    ) -> str:
        """Generate a new token.

        Args:
            user_id: User identifier.
            permissions: Granted permissions.
            roles: Assigned roles.

        Returns:
            Generated token string.
        """
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            "user_id": user_id,
            "permissions": permissions or {Permission.EXECUTE},
            "roles": roles or set(),
            "expires_at": time.time() + self.token_ttl,
        }
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token.

        Args:
            token: Token to revoke.

        Returns:
            True if token was revoked.
        """
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired tokens.

        Returns:
            Number of tokens removed.
        """
        now = time.time()
        expired = [t for t, d in self._tokens.items() if d["expires_at"] < now]
        for token in expired:
            del self._tokens[token]
        return len(expired)

    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Authenticate with token."""
        if not credentials.token:
            raise AuthenticationError("Token required")

        if credentials.token not in self._tokens:
            raise AuthenticationError("Invalid token")

        token_data = self._tokens[credentials.token]

        if token_data["expires_at"] < time.time():
            del self._tokens[credentials.token]
            raise AuthenticationError("Token expired")

        return AuthContext(
            user_id=token_data["user_id"],
            permissions=token_data["permissions"],
            roles=token_data["roles"],
            authenticated=True,
            credentials=credentials,
        )


class HmacAuthenticator(Authenticator):
    """HMAC signature based authentication."""

    def __init__(self):
        """Initialize authenticator."""
        self._secrets: Dict[str, Dict[str, Any]] = {}

    def register_secret(
        self,
        key_id: str,
        secret: str,
        user_id: str,
        permissions: Optional[Set[Permission]] = None,
    ) -> None:
        """Register a signing secret.

        Args:
            key_id: Key identifier.
            secret: Signing secret.
            user_id: Associated user ID.
            permissions: Granted permissions.
        """
        self._secrets[key_id] = {
            "secret": secret,
            "user_id": user_id,
            "permissions": permissions or {Permission.EXECUTE},
        }

    def verify_signature(
        self,
        key_id: str,
        message: str,
        signature: str,
    ) -> bool:
        """Verify an HMAC signature.

        Args:
            key_id: Key identifier.
            message: Message that was signed.
            signature: Signature to verify.

        Returns:
            True if signature is valid.
        """
        if key_id not in self._secrets:
            return False

        secret = self._secrets[key_id]["secret"]
        expected = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Authenticate with HMAC."""
        if not credentials.api_key or not credentials.secret:
            raise AuthenticationError("Key ID and signature required")

        key_id = credentials.api_key
        signature = credentials.secret
        message = credentials.metadata.get("message", "")

        if not self.verify_signature(key_id, message, signature):
            raise AuthenticationError("Invalid signature")

        secret_data = self._secrets[key_id]
        return AuthContext(
            user_id=secret_data["user_id"],
            permissions=secret_data["permissions"],
            authenticated=True,
            credentials=credentials,
        )


class Authorizer:
    """Authorizes access to tools based on permissions."""

    def __init__(self):
        """Initialize authorizer."""
        self._tool_permissions: Dict[str, Set[Permission]] = {}
        self._role_permissions: Dict[str, Set[Permission]] = {}

    def set_tool_requirements(
        self,
        tool_id: str,
        required_permissions: Set[Permission],
    ) -> None:
        """Set required permissions for a tool.

        Args:
            tool_id: Tool identifier.
            required_permissions: Required permissions.
        """
        self._tool_permissions[tool_id] = required_permissions

    def set_role_permissions(
        self,
        role: str,
        permissions: Set[Permission],
    ) -> None:
        """Set permissions for a role.

        Args:
            role: Role name.
            permissions: Permissions granted by role.
        """
        self._role_permissions[role] = permissions

    def get_effective_permissions(self, context: AuthContext) -> Set[Permission]:
        """Get all effective permissions for a context.

        Args:
            context: Authentication context.

        Returns:
            Set of all permissions.
        """
        permissions = set(context.permissions)

        for role in context.roles:
            if role in self._role_permissions:
                permissions |= self._role_permissions[role]

        return permissions

    def authorize(
        self,
        context: AuthContext,
        tool_id: str,
        action: Permission = Permission.EXECUTE,
    ) -> bool:
        """Check if context is authorized for a tool action.

        Args:
            context: Authentication context.
            tool_id: Tool identifier.
            action: Action to perform.

        Returns:
            True if authorized.
        """
        if not context.authenticated:
            return False

        # Check if tool has specific requirements
        if tool_id in self._tool_permissions:
            required = self._tool_permissions[tool_id]
            effective = self.get_effective_permissions(context)

            # Admin permission grants all access
            if Permission.ADMIN in effective:
                return True

            # Check if all required permissions are present
            if not required.issubset(effective):
                return False

        # Check if action is allowed
        effective = self.get_effective_permissions(context)
        return action in effective or Permission.ADMIN in effective

    def require(
        self,
        context: AuthContext,
        tool_id: str,
        action: Permission = Permission.EXECUTE,
    ) -> None:
        """Require authorization, raising if not authorized.

        Args:
            context: Authentication context.
            tool_id: Tool identifier.
            action: Action to perform.

        Raises:
            AuthorizationError: If not authorized.
        """
        if not self.authorize(context, tool_id, action):
            raise AuthorizationError(
                f"Not authorized for {action.value} on {tool_id}"
            )


class ToolAuthManager:
    """Manages authentication and authorization for tools."""

    def __init__(
        self,
        authenticator: Optional[Authenticator] = None,
        authorizer: Optional[Authorizer] = None,
    ):
        """Initialize manager.

        Args:
            authenticator: Authenticator to use.
            authorizer: Authorizer to use.
        """
        self.authenticator = authenticator or NoAuthenticator()
        self.authorizer = authorizer or Authorizer()

    async def authenticate(self, credentials: AuthCredentials) -> AuthContext:
        """Authenticate with credentials.

        Args:
            credentials: Credentials to authenticate.

        Returns:
            Authentication context.
        """
        return await self.authenticator.authenticate(credentials)

    def authorize(
        self,
        context: AuthContext,
        tool_id: str,
        action: Permission = Permission.EXECUTE,
    ) -> bool:
        """Check authorization.

        Args:
            context: Authentication context.
            tool_id: Tool identifier.
            action: Action to check.

        Returns:
            True if authorized.
        """
        return self.authorizer.authorize(context, tool_id, action)

    async def authenticate_and_authorize(
        self,
        credentials: AuthCredentials,
        tool_id: str,
        action: Permission = Permission.EXECUTE,
    ) -> AuthContext:
        """Authenticate and check authorization in one step.

        Args:
            credentials: Credentials.
            tool_id: Tool identifier.
            action: Action to check.

        Returns:
            Authentication context if successful.

        Raises:
            AuthenticationError: If authentication fails.
            AuthorizationError: If authorization fails.
        """
        context = await self.authenticate(credentials)
        self.authorizer.require(context, tool_id, action)
        return context


class AuthenticatedToolWrapper:
    """Wrapper that adds authentication to tool execution."""

    def __init__(
        self,
        tool: Any,
        auth_manager: Optional[ToolAuthManager] = None,
        require_auth: bool = True,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            auth_manager: Authentication manager.
            require_auth: Whether auth is required.
        """
        self.tool = tool
        self.auth_manager = auth_manager or ToolAuthManager()
        self.require_auth = require_auth

    async def execute(
        self,
        input: Any,
        credentials: Optional[AuthCredentials] = None,
    ) -> Any:
        """Execute tool with authentication.

        Args:
            input: Tool input.
            credentials: Authentication credentials.

        Returns:
            Tool output.

        Raises:
            AuthenticationError: If authentication fails.
            AuthorizationError: If authorization fails.
        """
        tool_id = self.tool.metadata.id

        if self.require_auth:
            if not credentials:
                raise AuthenticationError("Credentials required")

            await self.auth_manager.authenticate_and_authorize(
                credentials, tool_id, Permission.EXECUTE
            )

        return await self.tool.execute(input)
