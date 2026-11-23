"""The Anthropic Messages API, called directly."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from .base import Completion, ProviderError

ANTHROPIC_URL = "https://api.anthropic.com"
FATAL_STATUS = (400, 401, 402, 403, 404)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, max_tokens: int = 2000, temperature: float | None = None,
                 max_attempts: int = 4, base_url: str = ANTHROPIC_URL):
        if not api_key:
            raise ProviderError("No Anthropic API key.", retryable=False)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=httpx.Timeout(300, connect=20),
        )

    async def complete(self, system: str, user: str) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt is identical across a run, so cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        last_error = "unknown error"
        for attempt in range(self.max_attempts):
            started = time.monotonic()
            try:
                response = await self._client.post("/v1/messages", json=body)
            except httpx.HTTPError as exc:
                last_error = "network error: %r" % exc
            else:
                if response.status_code == 200:
                    data = response.json()
                    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                    usage = data.get("usage") or {}
                    if text.strip():
                        prompt_tokens = sum(
                            usage.get(k) or 0
                            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                        )
                        return Completion(
                            text=text.strip(),
                            prompt_tokens=prompt_tokens or None,
                            completion_tokens=usage.get("output_tokens"),
                            latency_ms=int((time.monotonic() - started) * 1000),
                        )
                    last_error = "empty response from model"
                else:
                    last_error = "HTTP %s: %s" % (response.status_code, response.text[:300])
                    if response.status_code in FATAL_STATUS:
                        raise ProviderError(last_error, retryable=False)
            if attempt + 1 < self.max_attempts:
                await asyncio.sleep(1.5 * (2**attempt) + random.random())
        raise ProviderError(last_error)

    async def close(self) -> None:
        await self._client.aclose()
