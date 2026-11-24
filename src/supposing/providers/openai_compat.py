"""Any OpenAI-compatible chat completions API: OpenRouter, Ollama, vLLM, LM Studio, OpenAI itself."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from .base import Completion, ProviderError

OPENROUTER_URL = "https://openrouter.ai/api/v1"
FATAL_STATUS = (400, 401, 402, 403, 404)


def error_message(response: httpx.Response) -> str:
    """The API's own error message, without the rest of the body (OpenRouter adds the account's user_id)."""
    try:
        err = response.json().get("error")
    except ValueError:
        err = None
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])[:300]
    if isinstance(err, str) and err:
        return err[:300]
    return response.text[:300]


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = OPENROUTER_URL,
        temperature: float | None = None,
        max_tokens: int = 2000,
        extra_body: dict[str, Any] | None = None,
        max_attempts: int = 4,
        headers: dict[str, str] | None = None,
    ):
        if not model:
            raise ProviderError("Choose a model.", retryable=False)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_attempts = max_attempts
        all_headers = {"X-Title": "Supposing", **(headers or {})}
        if api_key:
            all_headers["Authorization"] = "Bearer " + api_key
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=all_headers, timeout=httpx.Timeout(300, connect=20)
        )

    async def complete(self, system: str, user: str) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        last_error = "unknown error"
        for attempt in range(self.max_attempts):
            started = time.monotonic()
            try:
                response = await self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last_error = "network error: %r" % exc
            else:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("error"):
                        last_error = str(data["error"].get("message", data["error"]))
                    else:
                        usage = data.get("usage") or {}
                        # An empty reply is returned, not retried here: the model was paid for it, and the
                        # caller decides whether to try again (judge_pair tries once more).
                        return Completion(
                            text=(data["choices"][0]["message"].get("content") or "").strip(),
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            cost_usd=usage.get("cost"),
                            latency_ms=int((time.monotonic() - started) * 1000),
                        )
                else:
                    last_error = "HTTP %s: %s" % (response.status_code, error_message(response))
                    if response.status_code in FATAL_STATUS:
                        raise ProviderError(last_error, retryable=False)
            if attempt + 1 < self.max_attempts:
                await asyncio.sleep(1.5 * (2**attempt) + random.random())
        raise ProviderError(last_error)

    async def close(self) -> None:
        await self._client.aclose()


def openrouter(
    model: str, api_key: str, zero_data_retention: bool = False, **kwargs: Any
) -> OpenAICompatProvider:
    """OpenRouter, with usage accounting on and, optionally, routing only to providers that keep no data."""
    extra: dict[str, Any] = {"usage": {"include": True}}
    if zero_data_retention:
        extra["provider"] = {"data_collection": "deny", "zdr": True}
    if not api_key:
        raise ProviderError("No OpenRouter key.", retryable=False)
    provider = OpenAICompatProvider(model, api_key=api_key, base_url=OPENROUTER_URL, extra_body=extra, **kwargs)
    provider.name = "openrouter"
    return provider
