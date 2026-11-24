"""Model providers, and ``make_provider`` to build one from a plain settings dict."""
from __future__ import annotations

import os
from typing import Any

from .anthropic import AnthropicProvider
from .base import Completion, Provider, ProviderError
from .claude_cli import ClaudeCLIProvider
from .mock import MockProvider, mock_quality
from .openai_compat import OpenAICompatProvider, openrouter

__all__ = [
    "AnthropicProvider", "ClaudeCLIProvider", "Completion", "MockProvider", "OpenAICompatProvider", "Provider",
    "ProviderError", "make_provider", "mock_quality", "openrouter",
]

KINDS = ("openrouter", "openai_compat", "anthropic", "claude_cli", "mock")


def make_provider(spec: dict[str, Any]) -> Provider:
    """Build a provider from ``{"provider": kind, "model": ..., ...}``. Keys default to the usual env vars."""
    kind = spec.get("provider", "openrouter")
    model = (spec.get("model") or "").strip()
    if kind == "openrouter":
        return openrouter(
            model,
            api_key=spec.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""),
            zero_data_retention=bool(spec.get("zero_data_retention")),
            temperature=spec.get("temperature"),
            headers=spec.get("headers"),
            **({"max_tokens": int(spec["max_tokens"])} if spec.get("max_tokens") else {}),
        )
    if kind == "openai_compat":
        return OpenAICompatProvider(
            model, api_key=spec.get("api_key"), base_url=spec["base_url"], temperature=spec.get("temperature")
        )
    if kind == "anthropic":
        return AnthropicProvider(model, api_key=spec.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", ""),
                                 temperature=spec.get("temperature"))
    if kind == "claude_cli":
        return ClaudeCLIProvider(model or "haiku")
    if kind == "mock":
        return MockProvider(seed=int(spec.get("seed") or 0), position_bias=float(spec.get("position_bias") or 0.0),
                            delay_s=float(spec.get("delay_s", 0.002)), fail_rate=float(spec.get("fail_rate") or 0.0))
    raise ProviderError("Unknown provider: %r. Use one of %s." % (kind, ", ".join(KINDS)), retryable=False)
