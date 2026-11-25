"""What every model provider implements: given a system and a user prompt, return text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    """A call failed. ``retryable`` is False when trying again cannot help (bad key, no credit, unknown model)."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def is_local(base_url: str) -> bool:
    """Whether a base URL points at this machine or a private network (where a refused connection means the
    server is not running, not a passing network fault)."""
    import ipaddress
    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or ""
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def unreachable(base_url: str, exc: Exception) -> str:
    """One actionable line for an endpoint that cannot be reached."""
    local = is_local(base_url)
    hint = (" Is the server running (for Ollama: `ollama serve`)? Check the base URL." if local else
            " Check the base URL and your network.")
    return "Could not connect to %s (%s).%s" % (base_url, type(exc).__name__, hint)


@dataclass
class Completion:
    """A model's reply, with usage when the provider reports it. ``cost_usd`` is None when unknown."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


class Provider(Protocol):
    """A judge model. Implement ``complete`` (and ``close``, for cleanup) to plug in any model. Both may be
    plain functions or ``async``; a plain ``complete`` runs in a worker thread.

    ``complete`` should retry transient failures itself and raise ``ProviderError`` when it gives up, with
    ``retryable=False`` if trying again cannot help.
    """

    name: str
    model: str

    async def complete(self, system: str, user: str) -> Completion: ...

    async def close(self) -> None: ...
