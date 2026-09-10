"""The HTTP trust boundary: who the caller is, and what they own.

This layer authenticates bearer tokens and checks session ownership. Account
state changes remain workflows in ``pipeline/account.py``.

A token is not the whole answer to the first question. It proves the caller
held the password at some point in the last two days; it does not prove the
account still exists or that the password has not changed since. Both are
checked against the users table on every request, through the `auth_version`
the token carries: a password change or a deletion invalidates every token
issued before it, without keeping a revocation list.
"""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

from fastapi import HTTPException, status
from fastapi_jwt import JwtAccessBearerCookie

from visionagent.config.settings import settings
from visionagent.database.postgres.repositories import SessionRepository, UserRepository

JWT_SECRET_KEY = settings.jwt_secret_key


class _AccessSecurity(JwtAccessBearerCookie):
    """The library's dependency, with every decode failure answered as 401.

    fastapi_jwt converts only its own BackendException. authlib raises
    BadSignatureError (and friends) straight through it, so a token signed
    with a previous secret -- or a forged one -- surfaced as a 500 with a
    traceback rather than as "log in again".
    """

    async def _get_payload(self, bearer: Any, cookie: Any) -> Any:
        try:
            return await super()._get_payload(bearer, cookie)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - any decode failure is the same answer
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from exc


# Header first, then cookie.
access_security = _AccessSecurity(
    secret_key=JWT_SECRET_KEY,
    auto_error=True,
    access_expires_delta=timedelta(days=2)
)

def create_token(user_id: int, user_name: str, auth_version: int = 0, salting: str = "") -> Any:
    subject = {
        "user_id": user_id,
        "user_name": user_name,
        # The users row's counter at issue time. require_auth compares it on
        # every request; a password change bumps the row and this token dies.
        "auth_version": int(auth_version),
        "salting": secrets.token_hex(16)
    }

    access_token = access_security.create_access_token(subject=subject)

    return access_token


def _claims(credentials: Any) -> dict[str, Any]:
    """The token's subject as a dict, whatever object the library handed us."""
    if credentials is None:
        return {}
    if hasattr(credentials, "subject"):
        subject = credentials.subject
        return dict(subject) if isinstance(subject, dict) else {}
    if isinstance(credentials, dict):
        return credentials
    return {}


def get_current_user_id(credentials: Any) -> str | None:
    """
    Extract user_id from JWT credentials
    Returns None if no valid credentials

    Beyond the signature: the user row must still exist and its auth_version
    must match the token's. A deleted account's token stops working the moment
    the row is gone, and a password change stops every older token -- neither
    of which a signature check alone can tell. The check lives here, on the
    function every route calls, rather than only on require_auth: a route
    that read the claims directly would otherwise keep serving a revoked
    token.
    """
    try:
        claims = _claims(credentials)
        user_id = claims.get("user_id")
        if user_id is None:
            return None
        current = UserRepository().auth_version(str(user_id))
        if current is None or claims.get("auth_version") != current:
            return None
        return str(user_id)
    except Exception:
        return None

def require_auth(credentials: Any) -> str:
    """
    Require valid authentication and return user_id
    Raises HTTPException if not authenticated
    """
    user_id = get_current_user_id(credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user_id


def verify_session_owner(session_id: str, user_id: str) -> None:
    """
    Ensure `session_id` exists and belongs to `user_id`.

    Authentication alone is not enough: session_id arrives from the client as a
    plain query/path parameter, and messages inherit ownership from their
    parent session rather than carrying a user_id. Without this check any
    authenticated user could pass another user's session_id and probe it.

    Raises 404 for both "no such session" and "not yours" so the endpoint does
    not confirm the existence of other users' sessions.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # The query lives in the repository, not inlined in a util: it is the same
    # query the session routes need, and two copies drift.
    if not SessionRepository().owns(session_id, user_id):
        raise HTTPException(status_code=404, detail="Session not found")
