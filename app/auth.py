"""
Direct Authentication Module using MSAL

This module provides OAuth 2.0 authentication with Microsoft Entra ID (Azure AD)
without relying on Azure App Service Easy Auth.

Features:
- Authorization code flow with PKCE
- Token caching and refresh
- Session management
- Middleware for protecting endpoints
"""

import os
import logging
import secrets
import hashlib
import base64
from typing import Optional
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import msal
from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Session configuration
SESSION_COOKIE_NAME = "session_id"
SESSION_MAX_AGE = 60 * 60 * 8  # 8 hours


class AuthConfig(BaseModel):
    """Authentication configuration from environment variables"""
    client_id: str
    client_secret: Optional[str] = None
    tenant_id: str
    redirect_uri: str
    mcp_api_client_id: str
    authority: str = ""
    scopes: list[str] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.authority:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        if not self.scopes:
            # MSAL automatically adds openid, profile, offline_access
            # We only need to specify our custom API scope
            self.scopes = [
                f"api://{self.mcp_api_client_id}/Mcp.Tools.ReadWrite"
            ]


class UserSession(BaseModel):
    """User session data stored in session store"""
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    access_token: str
    refresh_token: Optional[str] = None
    token_expires_at: datetime
    id_token: Optional[str] = None
    
    def is_token_expired(self) -> bool:
        """Check if the access token is expired (with 5 min buffer)"""
        return datetime.now(timezone.utc) >= (self.token_expires_at - timedelta(minutes=5))


class AuthManager:
    """
    Manages authentication flow and session state.
    
    This class handles:
    - MSAL client initialization
    - Authorization URL generation
    - Token acquisition from auth code
    - Token refresh
    - Session management
    """
    
    def __init__(self, config: AuthConfig, secret_key: str):
        self.config = config
        self.secret_key = secret_key
        self.serializer = URLSafeTimedSerializer(secret_key)
        
        # In-memory session store (use Redis in production)
        self._sessions: dict[str, UserSession] = {}
        
        # PKCE state store (code_verifier by state)
        # Store for CSRF protection (state -> True mapping)
        self._state_store: dict[str, bool] = {}
        
        # Initialize MSAL confidential client
        # Note: We're a confidential client with client_secret, so PKCE is not required
        self._msal_app = msal.ConfidentialClientApplication(
            client_id=config.client_id,
            client_credential=config.client_secret,
            authority=config.authority,
            token_cache=msal.TokenCache()  # In-memory cache
        )
        
        logger.info(f"AuthManager initialized for tenant {config.tenant_id}")
    
    def get_auth_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Generate authorization URL for login.
        
        Returns:
            Tuple of (auth_url, state)
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Store state for CSRF protection
        self._state_store[state] = True
        
        auth_url = self._msal_app.get_authorization_request_url(
            scopes=self.config.scopes,
            state=state,
            redirect_uri=self.config.redirect_uri
        )
        
        logger.debug(f"Generated auth URL with state: {state}")
        return auth_url, state
    
    async def handle_callback(self, code: str, state: str) -> UserSession:
        """
        Handle OAuth callback and exchange code for tokens.
        
        Args:
            code: Authorization code from callback
            state: State parameter for CSRF protection
            
        Returns:
            UserSession with tokens
            
        Raises:
            HTTPException: If authentication fails
        """
        # Validate state for CSRF protection
        if not self._state_store.pop(state, None):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter - possible CSRF attack"
            )
        
        # Exchange code for tokens
        result = self._msal_app.acquire_token_by_authorization_code(
            code=code,
            scopes=self.config.scopes,
            redirect_uri=self.config.redirect_uri
        )
        
        if "error" in result:
            logger.error(f"Token acquisition failed: {result.get('error_description', result.get('error'))}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {result.get('error_description', 'Unknown error')}"
            )
        
        # Parse ID token claims
        id_token_claims = result.get("id_token_claims", {})
        
        # Calculate token expiration
        expires_in = result.get("expires_in", 3600)
        token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        session = UserSession(
            user_id=id_token_claims.get("oid", id_token_claims.get("sub", "unknown")),
            email=id_token_claims.get("preferred_username") or id_token_claims.get("email"),
            name=id_token_claims.get("name"),
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            token_expires_at=token_expires_at,
            id_token=result.get("id_token")
        )
        
        logger.info(f"User authenticated: {session.email} (id: {session.user_id})")
        return session
    
    async def refresh_token(self, session: UserSession) -> Optional[UserSession]:
        """
        Refresh an expired access token.
        
        Args:
            session: Current user session
            
        Returns:
            Updated session with new tokens, or None if refresh failed
        """
        if not session.refresh_token:
            logger.warning(f"No refresh token for user {session.user_id}")
            return None
        
        # Try to get cached token first
        accounts = self._msal_app.get_accounts()
        if accounts:
            result = self._msal_app.acquire_token_silent(
                scopes=self.config.scopes,
                account=accounts[0]
            )
            if result and "access_token" in result:
                expires_in = result.get("expires_in", 3600)
                session.access_token = result["access_token"]
                session.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                if "refresh_token" in result:
                    session.refresh_token = result["refresh_token"]
                logger.info(f"Token refreshed silently for user {session.user_id}")
                return session
        
        # Fall back to refresh token flow
        result = self._msal_app.acquire_token_by_refresh_token(
            refresh_token=session.refresh_token,
            scopes=self.config.scopes
        )
        
        if "error" in result:
            logger.warning(f"Token refresh failed for user {session.user_id}: {result.get('error')}")
            return None
        
        expires_in = result.get("expires_in", 3600)
        session.access_token = result["access_token"]
        session.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        if "refresh_token" in result:
            session.refresh_token = result["refresh_token"]
        
        logger.info(f"Token refreshed for user {session.user_id}")
        return session
    
    def create_session_id(self, session: UserSession) -> str:
        """Create a signed session ID and store session data"""
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = session
        
        # Return signed session cookie value
        return self.serializer.dumps(session_id)
    
    def get_session(self, signed_session_id: str) -> Optional[UserSession]:
        """Retrieve session from signed session ID"""
        try:
            session_id = self.serializer.loads(signed_session_id, max_age=SESSION_MAX_AGE)
            return self._sessions.get(session_id)
        except (BadSignature, SignatureExpired):
            return None
    
    def delete_session(self, signed_session_id: str) -> bool:
        """Delete a session"""
        try:
            session_id = self.serializer.loads(signed_session_id, max_age=SESSION_MAX_AGE)
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        except (BadSignature, SignatureExpired):
            pass
        return False
    
    async def get_valid_session(self, request: Request) -> Optional[UserSession]:
        """
        Get a valid session from request, refreshing token if needed.
        
        This is the main method for checking if a request is authenticated.
        """
        signed_session_id = request.cookies.get(SESSION_COOKIE_NAME)
        if not signed_session_id:
            return None
        
        session = self.get_session(signed_session_id)
        if not session:
            return None
        
        # Check if token needs refresh
        if session.is_token_expired():
            logger.debug(f"Token expired for user {session.user_id}, attempting refresh")
            refreshed = await self.refresh_token(session)
            if not refreshed:
                # Token refresh failed, session is invalid
                self.delete_session(signed_session_id)
                return None
            session = refreshed
        
        return session


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce authentication on protected routes.
    
    Excludes:
    - /health, /ping, /alive (health checks)
    - /login, /auth/callback, /logout (auth endpoints)
    - Static files
    """
    
    EXCLUDED_PATHS = {
        "/health",
        "/ping", 
        "/alive",
        "/login",
        "/auth/callback",
        "/logout",
        "/",  # Allow index page to load (will redirect if not authenticated)
        "/favicon.ico",
    }
    
    def __init__(self, app, auth_manager: AuthManager):
        super().__init__(app)
        self.auth_manager = auth_manager
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Skip auth for static files
        if request.url.path.startswith("/static"):
            return await call_next(request)
        
        # Check for valid session
        session = await self.auth_manager.get_valid_session(request)
        
        if not session:
            # For API endpoints, return 401
            if request.url.path.startswith("/api") or request.url.path == "/chat":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            # For page requests, redirect to login
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        
        # Add session to request state for use in endpoints
        request.state.user_session = session
        request.state.user_id = session.user_id
        request.state.access_token = session.access_token
        
        return await call_next(request)


def get_auth_config() -> AuthConfig:
    """Load auth configuration from environment variables"""
    
    # Determine redirect URI based on environment
    # Use APP_BASE_URL to avoid conflicts with other libraries that might use BASE_URL
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
    redirect_uri = f"{base_url}/auth/callback"
    
    return AuthConfig(
        client_id=os.getenv("AZURE_CLIENT_ID", "4835f9b9-7b7f-433a-acd1-0545bd15b7cb"),
        client_secret=os.getenv("MICROSOFT_PROVIDER_AUTHENTICATION_SECRET"),
        tenant_id=os.getenv("TENANT_ID", "9412b47a-813f-4d21-85a5-7772f28bf719"),
        redirect_uri=redirect_uri,
        mcp_api_client_id=os.getenv("MCP_API_CLIENT_ID", "f29dd31f-0747-46d7-8cbc-c846e315eb88"),
    )


def get_secret_key() -> str:
    """Get or generate secret key for session signing"""
    secret = os.getenv("SESSION_SECRET_KEY")
    if not secret:
        # Generate a random key (note: sessions won't survive restarts)
        logger.warning("SESSION_SECRET_KEY not set, generating random key. Sessions won't survive restarts!")
        secret = secrets.token_urlsafe(32)
    return secret
