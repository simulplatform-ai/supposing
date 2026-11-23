"""Errors from the Supposing API, with the API's stable ``code`` and a ``hint`` saying what to do next."""
from __future__ import annotations

from typing import Any

import httpx

# Used when the server sent no code of its own (an old server, a proxy, a crash).
_STATUS_CODES = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict",
    413: "too_large", 422: "invalid_request", 429: "rate_limited",
}
_STATUS_HINTS = {
    401: "Check your API key: run `supposing login`, or set SUPPOSING_API_KEY.",
    429: "Wait a moment and try again.",
}


class CloudError(Exception):
    """A request the API refused or could not serve.

    ``code`` is stable and safe to branch on; ``message`` is for people; ``hint`` says what to do next.
    """

    def __init__(self, message: str, *, code: str, status: int | None = None, hint: str | None = None,
                 details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.hint = hint
        self.details = details or []

    def __str__(self) -> str:
        return self.message + (" " + self.hint if self.hint else "")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint, "status": self.status,
                "details": self.details or None}

    @classmethod
    def from_response(cls, resp: httpx.Response) -> CloudError:
        status = resp.status_code
        try:
            body: Any = resp.json()
        except ValueError:
            body = None
        code = _STATUS_CODES.get(status, "server_error" if status >= 500 else "http_%d" % status)
        message = resp.reason_phrase or "HTTP %d" % status
        hint: str | None = _STATUS_HINTS.get(status)
        details: list[dict[str, Any]] | None = None
        if isinstance(body, dict):
            nested = body.get("error")
            err: dict[str, Any] = nested if isinstance(nested, dict) else body
            if isinstance(err.get("code"), str):
                code = err["code"]
            if isinstance(err.get("message"), str):
                message = err["message"]
            if "hint" in err:
                hint = err.get("hint")
            if isinstance(err.get("details"), list):
                details = err["details"]
            detail = err.get("detail")  # FastAPI's default shape
            if isinstance(detail, str):
                message = detail
            elif isinstance(detail, list):
                details = [{"field": ".".join(str(p) for p in d.get("loc", [])[1:]), "message": d.get("msg", "")}
                           for d in detail if isinstance(d, dict)]
                message = "; ".join("%s: %s" % (d["field"], d["message"]) for d in details) or message
        if status >= 500 and not hint:
            hint = "The server failed. Try again; if it keeps failing, report the request id or time."
        return cls(message, code=code, status=status, hint=hint, details=details)


def connection_error(base_url: str, exc: Exception) -> CloudError:
    return CloudError("Could not reach %s (%s)." % (base_url, type(exc).__name__), code="connection_error",
                      hint="Check the URL and your network. Set SUPPOSING_API_URL, or pass --api-url.")
