import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from visionagent.api.deps import AuthorizedDeleteAccount, CurrentUser
from visionagent.api.observability import get_api_logger, translated_http_error
from visionagent.api.security import access_security, create_token, get_current_user_id
from visionagent.database.postgres.repositories import UserRepository
from visionagent.exceptions.auth import AuthError
from visionagent.models import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
)
from visionagent.pipeline.account import (
    authenticate,
    change_password,
    delete_account,
    register_user,
)

router = APIRouter()
logger = get_api_logger(__name__)

@router.post("/login")
async def login(request: LoginRequest) -> dict[str, Any]:
    try:
        user_id, username = authenticate(request.username, request.password)
        # The credential is minted here, not in the workflow: its format and
        # lifetime are HTTP decisions. It carries the row's auth_version so a
        # later password change ends it.
        version = UserRepository().auth_version(str(user_id)) or 0
        token = str(create_token(user_id, username, auth_version=version))
        return {"access_token": token, "token_type": "bearer"}
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.public_message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        raise translated_http_error(
            e,
            operation="login",
            detail="Failed to log in",
        ) from e

@router.post("/register")
async def register(request: RegisterRequest) -> dict[str, Any]:
    try:
        register_user(request.username, request.password)
        return {"message": "User registered successfully"}
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.public_message,
        ) from e
    except Exception as e:
        raise translated_http_error(
            e,
            operation="register",
            detail="Failed to register user",
        ) from e

@router.post("/logout")
async def logout() -> dict[str, Any]:
    # Revocation is client-side because access tokens are stateless.
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_current_user(credentials: Any = Depends(access_security)) -> dict[str, Any]:
    try:
        # Through get_current_user_id, not the raw claims: it is what refuses
        # a deleted account's token and a token from before a password change.
        user_id = get_current_user_id(credentials)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        username = credentials["user_name"]
        return {"user_id": int(user_id), "username": username}
    except HTTPException:
        raise
    except Exception as exc:
        # `from None` on purpose: the cause is a malformed or expired token, and
        # echoing its parse error into a 401 tells an unauthenticated caller
        # more about the token format than they should learn.
        logger.debug(
            "token_rejected",
            exception_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        ) from None

@router.post("/me/password")
async def change_my_password(request: ChangePasswordRequest, user_id: CurrentUser) -> dict[str, Any]:
    """Change the caller's password.

    The current password is required even though the caller is already
    authenticated: a session left open on a shared machine should not be enough
    to lock the real owner out.
    """
    try:
        change_password(int(user_id), request.current_password, request.new_password)
        return {"message": "Password changed successfully"}
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.public_message,
        ) from e


@router.delete("/me")
async def delete_my_account(command: AuthorizedDeleteAccount) -> dict[str, Any]:
    """Delete the caller's account and everything they own.

    The dependency has already checked the password and the literal "DELETE"
    confirmation and minted the command; the workflow runs under it and never
    sees the credential. A second call finds nothing left and reports zeros.
    """
    try:
        # This workflow may wait on a cross-process lock and erase several
        # synchronous stores. Keep that bounded but potentially long work off
        # the application loop so other requests and supervisors keep moving.
        removed = await asyncio.to_thread(delete_account, command)
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account cleanup is busy; try again",
        ) from e
    return {"message": "Account deleted", "removed": removed}
