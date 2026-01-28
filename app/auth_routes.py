"""
Authentication Routes for Direct Auth

This module provides FastAPI routes for MSAL-based authentication.
Import and include these routes when USE_DIRECT_AUTH is enabled.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import RedirectResponse, HTMLResponse

from auth import (
    AuthManager,
    AuthConfig,
    UserSession,
    get_auth_config,
    get_secret_key,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global auth manager instance (initialized in setup_auth_routes)
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the auth manager instance"""
    if _auth_manager is None:
        raise RuntimeError("Auth manager not initialized. Call setup_auth_routes first.")
    return _auth_manager


def setup_auth_routes(app) -> AuthManager:
    """
    Initialize auth manager and add routes to the app.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        AuthManager instance for use with middleware
    """
    global _auth_manager
    
    config = get_auth_config()
    secret_key = get_secret_key()
    
    _auth_manager = AuthManager(config, secret_key)
    
    # Include the router
    app.include_router(router)
    
    logger.info("Direct auth routes configured")
    return _auth_manager


@router.get("/login")
async def login(request: Request, next: str = "/"):
    """
    Initiate login flow.
    
    Redirects user to Microsoft login page.
    """
    auth = get_auth_manager()
    
    # Store the 'next' URL in state for redirect after login
    # Using simple approach - in production, use a more secure state store
    state = f"next={next}"
    
    auth_url, _ = auth.get_auth_url(state=state)
    
    logger.info(f"Redirecting to login, next={next}")
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/auth/callback")
async def auth_callback(request: Request, code: str = None, state: str = None, error: str = None, error_description: str = None):
    """
    Handle OAuth callback from Microsoft.
    
    Exchanges authorization code for tokens and creates session.
    """
    if error:
        logger.error(f"Auth callback error: {error} - {error_description}")
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family: sans-serif; padding: 40px; text-align: center;">
                    <h1>Authentication Failed</h1>
                    <p style="color: red;">{error_description or error}</p>
                    <a href="/login">Try Again</a>
                </body>
            </html>
            """,
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No authorization code received"
        )
    
    auth = get_auth_manager()
    
    try:
        # Exchange code for tokens
        session = await auth.handle_callback(code, state)
        
        # Create session cookie
        signed_session_id = auth.create_session_id(session)
        
        # Parse next URL from state
        next_url = "/"
        if state and state.startswith("next="):
            next_url = state[5:]  # Remove "next=" prefix
        
        # Redirect to the next URL with session cookie
        response = RedirectResponse(url=next_url, status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=signed_session_id,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=not os.getenv("DEBUG", "false").lower() == "true"  # Secure in production
        )
        
        logger.info(f"User {session.email} logged in successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth callback failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )


@router.get("/logout")
async def logout(request: Request):
    """
    Log out the user.
    
    Clears session and redirects to Microsoft logout.
    """
    auth = get_auth_manager()
    
    # Clear session
    signed_session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if signed_session_id:
        auth.delete_session(signed_session_id)
    
    # Create redirect response that clears the cookie
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    
    logger.info("User logged out")
    return response


@router.get("/auth/me")
async def get_current_user(request: Request):
    """
    Get current user information.
    
    Returns user details if authenticated, or 401 if not.
    """
    auth = get_auth_manager()
    session = await auth.get_valid_session(request)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return {
        "user_id": session.user_id,
        "email": session.email,
        "name": session.name,
        "token_expires_at": session.token_expires_at.isoformat(),
        "is_token_expired": session.is_token_expired()
    }


@router.get("/auth/token")
async def get_token(request: Request):
    """
    Get the current access token.
    
    This endpoint can be used by the frontend to get the token for API calls.
    Only accessible to authenticated users.
    """
    auth = get_auth_manager()
    session = await auth.get_valid_session(request)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return {
        "access_token": session.access_token,
        "expires_at": session.token_expires_at.isoformat(),
        "user_id": session.user_id
    }
