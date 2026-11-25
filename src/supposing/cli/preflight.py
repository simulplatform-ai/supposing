"""Checks before `supposing rank` asks to spend or starts: is the judge reachable, does it accept the key, does the
model exist? Each is one free request with a short timeout, so a bad setup fails in seconds with a hint instead of
after a round of failed calls. A check that can't decide (offline, an endpoint without a model list) lets the run
go ahead; the run's own errors still apply.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from .output import USAGE, CliError

TIMEOUT = 3.0
OPENROUTER = "https://openrouter.ai/api/v1"
ANTHROPIC = "https://api.anthropic.com/v1"


def _unavailable(message: str, hint: str) -> CliError:
    return CliError(message, code="provider_unavailable", exit_code=USAGE, hint=hint)


def _get(url: str, headers: dict[str, str] | None = None) -> httpx.Response | None:
    try:
        return httpx.get(url, headers=headers or {}, timeout=TIMEOUT)
    except httpx.HTTPError:
        return None


def ollama_hint(base_url: str) -> str:
    if "11434" in base_url or "ollama" in base_url.lower():
        return "Is Ollama running? Start it with `ollama serve`, then check --base-url (now %s)." % base_url
    return "Check that the server is running and that --base-url is right (now %s)." % base_url


def check(provider: str, model: str, base_url: str | None = None) -> None:
    """Raise a CliError that says what to fix, or return when the judge looks usable."""
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            r = _get(OPENROUTER + "/key", {"Authorization": "Bearer " + key})
            if r is not None and r.status_code in (401, 403):
                raise _unavailable("OpenRouter refused the key in OPENROUTER_API_KEY.",
                                   "Check it, or create one, at https://openrouter.ai/settings/keys.")
        r = _get("%s/models/%s/endpoints" % (OPENROUTER, model))
        if r is not None and r.status_code == 404:
            raise _unavailable("OpenRouter has no model %s." % model,
                               "Use a level (--judge quick, balanced or thorough) or an id from "
                               "https://openrouter.ai/models, e.g. --judge anthropic/claude-haiku-5.5.")
    elif provider == "openai_compat" and base_url:
        try:
            r = httpx.get(base_url.rstrip("/") + "/models", timeout=TIMEOUT)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise _unavailable("Can't reach the judge at %s." % base_url, ollama_hint(base_url)) from exc
        except httpx.HTTPError:
            return
        ids = _model_ids(r)
        if ids and model not in ids and not any(i.split(":")[0] == model for i in ids):
            pull = " With Ollama: `ollama pull %s`." % model if ("11434" in base_url or "ollama" in base_url) else ""
            raise _unavailable("No model %s at %s." % (model, base_url),
                               "Models there: %s.%s" % (", ".join(sorted(ids)[:10]), pull))
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return
        r = _get("%s/models/%s" % (ANTHROPIC, model), {"x-api-key": key, "anthropic-version": "2023-06-01"})
        if r is not None and r.status_code in (401, 403):
            raise _unavailable("Anthropic refused the key in ANTHROPIC_API_KEY.",
                               "Check it in the Anthropic Console, under API keys.")
        if r is not None and r.status_code == 404:
            raise _unavailable("Anthropic has no model %s." % model,
                               "Pass an Anthropic model id with --model, e.g. --model claude-haiku-5-5.")


def _model_ids(r: httpx.Response) -> set[str]:
    if r.status_code != 200:
        return set()
    try:
        body: Any = r.json()
    except ValueError:
        return set()
    rows = body.get("data") if isinstance(body, dict) else None
    return {str(m.get("id")) for m in rows or [] if isinstance(m, dict) and m.get("id")}
