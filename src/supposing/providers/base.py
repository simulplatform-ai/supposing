"""What every model provider implements: given a system and a user prompt, return text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    """A call failed. ``retryable`` is False when trying again cannot help (bad key, no credit, unknown model)."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Completion:
    """A model's reply, with usage when the provider reports it. ``cost_usd`` is None when unknown."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


class Provider(Protocol):
    """A judge model. Implement ``complete`` (and ``close``, for cleanup) to plug in any model.

    ``complete`` should retry transient failures itself and raise ``ProviderError`` when it gives up, with
    ``retryable=False`` if trying again cannot help.
    """

    name: str
    model: str

    async def complete(self, system: str, user: str) -> Completion: ...

    async def close(self) -> None: ...
