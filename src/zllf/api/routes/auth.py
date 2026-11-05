"""Login/logout/me and first-run setup."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from sqlalchemy import func
from sqlmodel import Session, select

from zllf.api.deps import DB, CurrentUser, SettingsDep
from zllf.api.errors import ApiError
from zllf.api.schemas import Credentials, SetupStatus, UserOut
from zllf.api.security import (
    SESSION_COOKIE,
    LoginRateLimiter,
    check_user_password,
    create_session,
    delete_session,
    hash_password,
)
from zllf.api.settings import Settings
from zllf.db.engine import audit
from zllf.db.models import User

router = APIRouter(prefix="/api", tags=["auth"])


def user_out(user: User) -> UserOut:
    return UserOut(id=user.id, username=user.username, role=user.role)  # type: ignore[arg-type]


def _user_count(db: Session) -> int:
    return int(db.exec(select(func.count()).select_from(User)).one())


def _set_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(settings.session_ttl_hours * 3600),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )


def validate_new_credentials(creds: Credentials, settings: Settings) -> str:
    username = creds.username.strip()
    if not username or any(c.isspace() for c in username):
        raise ApiError(422, "E_VALIDATION", "Enter a user name without spaces.")
    if len(creds.password) < settings.min_password_length:
        raise ApiError(
            422, "E_WEAK_PASSWORD",
            f"Choose a password with at least {settings.min_password_length} characters.",
        )
    return username


@router.get("/setup", response_model=SetupStatus)
def setup_status(db: DB) -> SetupStatus:
    return SetupStatus(needs_setup=_user_count(db) == 0)


@router.post("/setup", response_model=UserOut)
def setup(creds: Credentials, db: DB, settings: SettingsDep, response: Response) -> UserOut:
    if _user_count(db) > 0:
        raise ApiError(409, "E_SETUP_DONE", "Setup is already complete. Sign in instead.")
    username = validate_new_credentials(creds, settings)
    user = User(username=username, password_hash=hash_password(creds.password), role="admin")
    db.add(user)
    db.flush()
    audit(db, user.id, "setup.admin_created", "user", user.id)
    _set_cookie(response, create_session(db, user.id, settings.session_ttl_hours), settings)
    return user_out(user)


@router.post("/auth/login", response_model=UserOut)
def login(
    creds: Credentials, request: Request, response: Response, db: DB, settings: SettingsDep
) -> UserOut:
    limiter: LoginRateLimiter = request.app.state.login_limiter
    client = request.client.host if request.client else "unknown"
    keys = (f"ip:{client}", f"user:{creds.username.strip().lower()}")
    if limiter.blocked(*keys):
        raise ApiError(429, "E_RATE_LIMITED",
                       "Too many failed sign-in attempts. Wait a few minutes and try again.")
    user = db.exec(select(User).where(User.username == creds.username.strip())).first()
    if not check_user_password(user, creds.password):
        limiter.fail(*keys)
        audit(db, "anonymous", "auth.login_failed", "user", user.id if user else None)
        raise ApiError(401, "E_BAD_CREDENTIALS", "User name or password is wrong. Try again.")
    assert user is not None
    limiter.reset(*keys)
    audit(db, user.id, "auth.login", "user", user.id)
    _set_cookie(response, create_session(db, user.id, settings.session_ttl_hours), settings)
    return user_out(user)


@router.post("/auth/logout", status_code=204)
def logout(request: Request, db: DB) -> Response:
    token = request.cookies.get(SESSION_COOKIE)
    delete_session(db, token)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return response


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return user_out(user)
