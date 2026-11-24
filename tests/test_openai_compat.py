"""The OpenAI-compatible provider's error text: the API's message, never the rest of the body."""
from __future__ import annotations

import httpx
import pytest

from supposing.providers.base import ProviderError
from supposing.providers.openai_compat import OpenAICompatProvider


def provider(handler) -> OpenAICompatProvider:
    p = OpenAICompatProvider("acme/model", api_key="k", base_url="https://api.test/v1", max_attempts=1)
    p._client = httpx.AsyncClient(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    return p


async def test_error_keeps_the_message_and_drops_the_account_id():
    p = provider(lambda r: httpx.Response(400, json={"error": {"message": "x/y is not a valid model ID", "code": 400},
                                                     "user_id": "user_secret123"}))
    with pytest.raises(ProviderError) as exc:
        await p.complete("s", "u")
    assert str(exc.value) == "HTTP 400: x/y is not a valid model ID"
    assert not exc.value.retryable
    await p.close()


async def test_error_without_json_keeps_the_text():
    p = provider(lambda r: httpx.Response(503, text="upstream down"))
    with pytest.raises(ProviderError) as exc:
        await p.complete("s", "u")
    assert str(exc.value) == "HTTP 503: upstream down" and exc.value.retryable
    await p.close()


async def test_make_provider_passes_extra_headers_to_openrouter():
    from supposing import make_provider

    p = make_provider({"provider": "openrouter", "model": "m", "api_key": "k", "headers": {"x-session-id": "s1"}})
    assert p._client.headers["x-session-id"] == "s1"  # type: ignore[attr-defined]
    await p.close()


async def test_make_provider_passes_max_tokens_to_openrouter_and_sends_it():
    import json

    from supposing import make_provider

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"winner": "A"}'}}]})

    capped = make_provider({"provider": "openrouter", "model": "m", "api_key": "k", "max_tokens": 600})
    default = make_provider({"provider": "openrouter", "model": "m", "api_key": "k"})
    assert capped.max_tokens == 600 and default.max_tokens == 2000  # type: ignore[attr-defined]
    capped._client = httpx.AsyncClient(base_url="https://api.test/v1",  # type: ignore[attr-defined]
                                       transport=httpx.MockTransport(handler))
    await capped.complete("s", "u")
    assert sent[0]["max_tokens"] == 600
    await capped.close()
    await default.close()
