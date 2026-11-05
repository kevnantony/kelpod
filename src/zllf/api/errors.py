"""API errors in the ``{"error": {"code", "message"}}`` shape (docs/API.md)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from zllf.core.logging import get_logger

log = get_logger("zllf.api")


class ApiError(Exception):
    """Raise from any endpoint to return a structured error."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def not_found(what: str) -> ApiError:
    return ApiError(404, "E_NOT_FOUND", f"This {what} was not found. It may have been deleted.")


def conflict_state(message: str) -> ApiError:
    return ApiError(409, "E_STATE", message)


_HTTP_CODES = {
    400: ("E_BAD_REQUEST", "The request is not valid."),
    401: ("E_AUTH", "Please sign in to continue."),
    403: ("E_FORBIDDEN", "You do not have permission to do this."),
    404: ("E_NOT_FOUND", "Not found."),
    405: ("E_METHOD", "This method is not allowed here."),
    413: ("E_TOO_LARGE", "The upload is too large."),
}


def _validation_message(exc: RequestValidationError) -> str:
    # Only field locations and messages: never echo the submitted input (may hold passwords).
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path"))
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "Some fields are missing or invalid: " + "; ".join(parts)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body("E_VALIDATION", _validation_message(exc)), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, default = _HTTP_CODES.get(exc.status_code, (f"E_HTTP_{exc.status_code}", ""))
        message = exc.detail if isinstance(exc.detail, str) and exc.detail else default
        if exc.status_code == 404 and message == "Not Found":
            message = default
        return JSONResponse(error_body(code, message), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _internal(req: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", path=req.url.path, error=type(exc).__name__)
        return JSONResponse(
            error_body("E_INTERNAL", "An internal error occurred. Your data is safe."),
            status_code=500,
        )
