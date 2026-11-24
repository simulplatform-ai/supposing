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


async def test_an_empty_reply_is_returned_with_its_usage_not_retried():
    """An empty reply was paid for: the provider returns it, and judge_pair decides about trying again."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 600, "cost": 0.004}})

    p = provider(handler)
    p.max_attempts = 4
    c = await p.complete("s", "u")
    assert c.text == "" and c.cost_usd == 0.004 and c.completion_tokens == 600 and len(calls) == 1
    await p.close()


class Replies:
    """A provider that gives the replies it is handed, in order, each billed 0.004."""

    name = model = "replies"

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls = 0

    async def complete(self, system, user):
        from supposing import Completion

        self.calls += 1
        return Completion(self.texts.pop(0), prompt_tokens=900, completion_tokens=100, cost_usd=0.004)

    async def close(self):
        pass


@pytest.mark.parametrize(("texts", "winner", "attempts"), [
    (['{"reasoning": "r", "winner": "B"}'], "B", 1),
    (["", '{"reasoning": "r", "winner": "A"}'], "A", 2),
    (["no verdict here", "", '{"winner": "A"}'], None, 2),  # at most one more call inside judge_pair
])
async def test_judge_pair_tries_once_more_and_reports_every_attempt(texts, winner, attempts):
    from supposing import Comparison, apply_verdict, judge_pair

    p = Replies(*texts)
    v = await judge_pair(p, "Better?", "a", "b")
    assert (v.winner, v.attempts, p.calls) == (winner, attempts, attempts)
    assert v.completion.cost_usd == pytest.approx(0.004 * attempts)
    assert (v.completion.prompt_tokens, v.completion.completion_tokens) == (900 * attempts, 100 * attempts)
    comp = Comparison(id="c", round=1, pair_id=1, item_a="x", item_b="y")
    apply_verdict(comp, v)
    assert comp.cost_usd == pytest.approx(0.004 * attempts)
