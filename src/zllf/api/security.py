"""Password hashing, server-side sessions, login rate limiting and CSRF protection.

CSRF: every state-changing request (POST/PUT/PATCH/DELETE) under ``/api`` must carry the
header ``X-ZLLF-CSRF: 1``. Browsers cannot add custom headers cross-site without a CORS
preflight (which this server never grants), and the session cookie is SameSite=Strict.
If an ``Origin`` header is present it must match the request's ``Host``.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlmodel import Session, select

from zllf.api.errors import error_body
from zllf.db.models import AuthSession, User

SESSION_COOKIE = "zllf_session"
CSRF_HEADER = "x-zllf-csrf"
CSRF_VALUE = "1"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_hasher = PasswordHasher()
# Verified against when the user does not exist, so timing does not reveal usernames.
_DUMMY_HASH = _hasher.hash("zllf-dummy-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def check_user_password(user: User | None, password: str) -> bool:
    if user is None or user.disabled:
        verify_password(_DUMMY_HASH, password)
        return False
    return verify_password(user.password_hash, password)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, user_id: str, ttl_hours: float) -> str:
    """Create a session row and return the raw cookie token (only its hash is stored)."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    db.add(AuthSession(id=token_hash(token), user_id=user_id, created_at=now,
                       expires_at=now + ttl_hours * 3600, last_seen_at=now))
    # opportunistic cleanup of expired sessions
    for old in db.exec(select(AuthSession).where(AuthSession.expires_at < now)).all():
        db.delete(old)
    return token


def lookup_session(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    sess = db.get(AuthSession, token_hash(token))
    if sess is None or sess.expires_at < time.time():
        return None
    user = db.get(User, sess.user_id)
    if user is None or user.disabled:
        return None
    return user


def delete_session(db: Session, token: str | None) -> None:
    if token:
        sess = db.get(AuthSession, token_hash(token))
        if sess is not None:
            db.delete(sess)


class LoginRateLimiter:
    """Sliding-window limit on failed logins per client address and per username."""

    def __init__(self, max_failures: int, window_seconds: float) -> None:
        self.max_failures = max_failures
        self.window = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        q = self._failures[key]
        while q and q[0] < now - self.window:
            q.popleft()
        return q

    def blocked(self, *keys: str) -> bool:
        now = time.time()
        with self._lock:
            return any(len(self._prune(k, now)) >= self.max_failures for k in keys)

    def fail(self, *keys: str) -> None:
        now = time.time()
        with self._lock:
            for k in keys:
                self._prune(k, now).append(now)

    def reset(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._failures.pop(k, None)


class CSRFMiddleware:
    """Pure ASGI middleware (does not buffer streaming uploads or SSE responses)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            method = scope["method"].upper()
            if method not in SAFE_METHODS:
                headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                           for k, v in scope.get("headers", [])}
                reason = _csrf_violation(headers)
                if reason:
                    await _send_json(send, 403, error_body("E_CSRF", reason))
                    return
        await self.app(scope, receive, send)


def _csrf_violation(headers: dict[str, str]) -> str | None:
    if headers.get(CSRF_HEADER) != CSRF_VALUE:
        return "This request was blocked for your security. Reload the page and try again."
    origin = headers.get("origin")
    if origin and origin != "null":
        host = headers.get("host", "")
        if urlsplit(origin).netloc.lower() != host.lower():
            return "Cross-site requests are not allowed."
    return None


async def _send_json(send: Any, status: int, body: dict[str, Any]) -> None:
    import json

    payload = json.dumps(body).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})
