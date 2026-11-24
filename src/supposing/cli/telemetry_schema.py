"""The anonymous usage event the ``supposing`` command line sends, and nothing else: one fixed allow-list of fields,
each a closed enum, a bounded integer, a boolean or a random id. The CLI builds events with it, and the API
(POST /v1/telemetry) validates against it, so client and server share one definition. Pure data: importing it
sends nothing and reads nothing.
"""
from __future__ import annotations

import re
from typing import Any

ENDPOINT = "https://api.supposing.app/v1/telemetry"
DOCS_URL = "https://app.supposing.app/docs/telemetry/"
MAX_BYTES = 2048

COMMANDS = (
    "rank", "run", "models", "samples", "results", "comparisons", "login", "logout", "whoami",
    "datasets_push", "datasets_list", "datasets_get", "datasets_delete",
    "runs_list", "runs_get", "runs_watch", "runs_cancel",
    "billing", "billing_ledger", "billing_topup", "billing_portal", "pricing", "usage", "other",
)
OS = ("linux", "darwin", "windows", "other")
JUDGES = ("openrouter", "anthropic", "openai", "ollama", "claude_cli", "mock", "other")
# Known models as a short name; any other model string becomes "other" and is never sent.
MODELS = ("quick", "balanced", "thorough", "haiku", "sonnet", "opus", "mock", "other")
ITEMS = ("1-5", "6-20", "21-50", "51-200", "200+")
DURATIONS = ("<1s", "1-10s", "10-60s", "1-10m", "10m+")
OUTCOMES = ("ok", "usage", "auth", "not_found", "refused", "unavailable", "needs_confirmation", "needs_credit",
            "run_failed", "interrupted", "error")
SAMPLES = ("stories", "grants", "other")

_UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_VERSION = re.compile(r"[0-9A-Za-z.+-]{1,32}")
_PYTHON = re.compile(r"[0-9]{1,2}\.[0-9]{1,3}")

# field -> (allowed values or a check, may be null)
FIELDS: dict[str, tuple[Any, bool]] = {
    "install_id": (_UUID4, False),
    "command": (COMMANDS, False),
    "version": (_VERSION, False),
    "python": (_PYTHON, False),
    "os": (OS, False),
    "local": (bool, True),  # True: `rank` on this machine; False: the hosted API
    "judge": (JUDGES, True),
    "model": (MODELS, True),
    "items": (ITEMS, True),
    "rounds": (range(1, 201), True),
    "both_orders": (bool, True),
    "sample": (SAMPLES, True),
    "duration": (DURATIONS, False),
    "outcome": (OUTCOMES, False),
}


def problems(event: Any) -> list[str]:
    """Why ``event`` is not a valid telemetry event; empty when it is. Unknown and missing fields both count."""
    if not isinstance(event, dict):
        return ["not an object"]
    out = ["unknown field %s" % k for k in event if k not in FIELDS]
    out += ["missing field %s" % k for k in FIELDS if k not in event]
    for name, (allowed, nullable) in FIELDS.items():
        if name not in event:
            continue
        v = event[name]
        if v is None:
            if not nullable:
                out.append("%s is null" % name)
        elif allowed is bool:
            if not isinstance(v, bool):
                out.append("%s is not a boolean" % name)
        elif isinstance(allowed, range):
            if isinstance(v, bool) or not isinstance(v, int) or v not in allowed:
                out.append("%s is out of range" % name)
        elif isinstance(allowed, re.Pattern):
            if not isinstance(v, str) or not allowed.fullmatch(v):
                out.append("%s has an invalid value" % name)
        elif v not in allowed:
            out.append("%s is not one of the allowed values" % name)
    return out


def valid(event: Any) -> bool:
    return not problems(event)


def items_bucket(n: int | None) -> str | None:
    if n is None or n < 1:
        return None
    return "1-5" if n <= 5 else "6-20" if n <= 20 else "21-50" if n <= 50 else "51-200" if n <= 200 else "200+"


def duration_bucket(seconds: float) -> str:
    return ("<1s" if seconds < 1 else "1-10s" if seconds < 10 else "10-60s" if seconds < 60
            else "1-10m" if seconds < 600 else "10m+")
