from __future__ import annotations

import json
import stat

import httpx
import pytest
from fake_api import KEY, FakeAPI

from supposing.cloud import AsyncClient, Client, CloudError, config_path, resolve, run_body, write_config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPOSING_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("SUPPOSING_API_KEY", raising=False)
    monkeypatch.delenv("SUPPOSING_API_URL", raising=False)


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "a.md").write_text("Alpha proposal")
    (d / "b.txt").write_text("Beta proposal")
    (d / "sub" / "c.html").write_text("<p>Gamma</p>")
    (d / "empty.md").write_text("  ")
    (d / "image.png").write_bytes(b"\x89PNG")
    (d / ".hidden.md").write_text("ignored")
    return d


def make(api: FakeAPI, **kw) -> Client:
    return Client(KEY, "http://fake", transport=api.transport(), **kw)


def test_push_estimate_run_wait_results(docs):
    api = FakeAPI(polls_until_done=3)
    with make(api) as sp:
        ds = sp.datasets.push(docs)
        assert ds.name == "docs" and ds.n_items == 3
        assert [i.source_id for i in ds.items or []] == ["a.md", "b.txt", "c.html"]  # uploaded: file names
        assert {s["path"] for s in ds.skipped} == {"empty.md", "image.png"}

        est = sp.runs.estimate(ds.id, "Which is best?", judge="mock", rounds=2)
        assert est.max_calls == 12 and est.describe() == "3 items, 3 to 12 judge calls, free, about 12 seconds"

        run = sp.runs.create(ds.id, "Which is best?", judge="mock", rounds=2, max_cost_usd=1.0)
        assert run.status == "queued" and run.type == "ranking" and run.config["rubric"] == "Which is best?"
        seen = []
        run = sp.runs.wait(run.id, on_progress=lambda r: seen.append(r.status), poll_interval=0)
        assert seen == ["running", "running", "done"] and run.finished
        assert any(r.url.path.endswith("/events") for r in api.requests)

        res = sp.runs.results(run.id)
        assert res.final and res.summary == "%s ranks first." % res.leaderboard[0].title
        assert [e.source_id for e in res.leaderboard] == ["a.md", "b.txt", "c.html"]
        assert res.leaderboard[0].rank_hi == 1 and res.interval == "90%"
        assert sp.datasets.list().data[0].id == ds.id
        assert [d.id for d in sp.datasets.iter()] == [ds.id]


def test_run_body_shape():
    body = run_body("ds1", "Clearer?", judge="mock", rounds=3, goal="top_n", top_n=2, seed=None)
    assert body == {"type": "ranking", "dataset_id": "ds1", "judge": "mock",
                    "ranking": {"rubric": "Clearer?", "rounds": 3, "goal": "top_n", "top_n": 2}}
    with pytest.raises(TypeError):
        run_body("ds1", "x", round=3)


def test_errors_carry_code_and_hint():
    api = FakeAPI()
    with make(api) as sp, pytest.raises(CloudError) as e:
        sp.datasets.get("nope")
    assert e.value.code == "dataset_not_found" and e.value.status == 404
    assert e.value.hint == "List datasets to find its id."
    assert str(e.value) == "Dataset not found. List datasets to find its id."


def test_bad_key_and_no_key():
    api = FakeAPI()
    with Client("sp_wrong", "http://fake", transport=api.transport()) as sp, pytest.raises(CloudError) as e:
        sp.me()
    assert e.value.code == "unauthorized"
    with Client(api_url="http://fake", transport=api.transport()) as sp, pytest.raises(CloudError) as e:
        sp.me()
    assert e.value.code == "no_api_key" and "supposing login" in (e.value.hint or "")
    assert not api.requests[1:]  # no request without a key


def test_fastapi_default_error_shapes():
    r = httpx.Response(422, json={"detail": [{"loc": ["body", "ranking", "rubric"], "msg": "Field required"}]})
    e = CloudError.from_response(r)
    assert e.code == "invalid_request" and e.details == [{"field": "ranking.rubric", "message": "Field required"}]
    e = CloudError.from_response(httpx.Response(404, json={"detail": "Run not found."}))
    assert e.code == "not_found" and e.message == "Run not found."
    e = CloudError.from_response(httpx.Response(500, text="oops"))
    assert e.code == "server_error" and e.hint


def test_retries_idempotent_requests_only(monkeypatch, docs):
    monkeypatch.setattr("supposing.cloud.client.time.sleep", lambda s: None)
    api = FakeAPI()
    with make(api) as sp:
        api.fail_next = [503, 502]
        assert sp.me().via == "api_key"
        ds = sp.datasets.push(docs)
        api.fail_next = [503]
        with pytest.raises(CloudError) as e:
            sp.runs.create(ds.id, "x")
        assert e.value.code == "server_error"


def test_connection_error_says_where():
    def boom(req):
        raise httpx.ConnectError("refused")
    with Client(KEY, "http://nowhere:1", transport=httpx.MockTransport(boom), retries=0) as sp, \
            pytest.raises(CloudError) as e:
        sp.me()
    assert e.value.code == "connection_error" and "http://nowhere:1" in e.value.message


def test_wait_timeout():
    api = FakeAPI(polls_until_done=10**6)
    with make(api) as sp:
        ds = sp.datasets.create("d", [{"title": "a", "text": "a"}, {"title": "b", "text": "b"}])
        run = sp.runs.create(ds.id, "x")
        with pytest.raises(CloudError) as e:
            sp.runs.wait(run.id, poll_interval=0, timeout=0.01)
        assert e.value.code == "wait_timeout"


async def test_async_client(docs):
    api = FakeAPI()
    async with AsyncClient(KEY, "http://fake", transport=api.async_transport()) as sp:
        ds = await sp.datasets.push(docs)
        run = await sp.runs.create(ds.id, "Which is best?")
        run = await sp.runs.wait(run.id, poll_interval=0)
        assert run.status == "done"
        assert (await sp.runs.results(run.id)).leaderboard[0].source_id == "a.md"
        assert [d.id async for d in sp.datasets.iter()] == [ds.id]
        canceled = await sp.runs.cancel((await sp.runs.create(ds.id, "x")).id)
        assert canceled.status == "canceled"


def test_config_resolution(monkeypatch):
    assert resolve().source == "none" and resolve().api_url == "https://api.supposing.app"
    p = write_config("sp_file", "http://localhost:8000/")
    assert p == config_path() and stat.S_IMODE(p.stat().st_mode) == 0o600
    c = resolve()
    assert (c.api_key, c.api_url, c.source) == ("sp_file", "http://localhost:8000", "file")
    monkeypatch.setenv("SUPPOSING_API_KEY", "sp_env")
    assert resolve().source == "env" and resolve("sp_arg").api_key == "sp_arg"


def test_source_header_and_ephemeral():
    api = FakeAPI()
    with Client(KEY, "http://fake", transport=api.transport(), source="cli") as sp:
        ds = sp.datasets.create("d", [{"title": "a", "text": "a"}, {"title": "b", "text": "b"}], ephemeral=True)
        sp.runs.create(ds.id, "x", ephemeral=True)
    posts = [r for r in api.requests if r.method == "POST"]
    assert all(r.headers["supposing-source"] == "cli" for r in posts)
    assert all(json.loads(r.content)["ephemeral"] is True for r in posts)
    with make(api) as sp:  # the library itself sends no source: the API records "api"
        sp.me()
    assert "supposing-source" not in api.requests[-1].headers


def test_billing_and_pricing_client():
    api = FakeAPI()
    with make(api) as sp:
        b = sp.billing.get()
        assert (b.enabled, b.balance_usd, b.topup_url) == (True, 2.0, "https://app.test/billing")
        assert sp.billing.ledger().data[0].kind == "grant"
        assert sp.billing.checkout(25).url.startswith("https://checkout.test/")
        assert sp.billing.portal().url == "https://portal.test/p"
    with Client(api_url="http://fake", transport=api.transport()) as anon:  # no key: pricing is public
        p = anon.pricing()
        assert p.provisional and p.unit_chars == 24000 and p.tiers[0].name == "economy"
    with make(api) as sp:
        ds = sp.datasets.create("d", [{"title": "a", "text": "a"}, {"title": "b", "text": "b"}])
        est = sp.runs.estimate(ds.id, "x", judge="quick")
        assert est.billed and est.price_tier == "economy" and est.price_per_call_usd == 0.004
        assert est.describe() == "2 items, 2 to 8 judge calls, up to $0.03 of credit at $0.004 a call (economy tier), about 12 seconds"
        api.refuse_runs_for_credit = True
        with pytest.raises(CloudError) as e:
            sp.runs.create(ds.id, "x")
        assert e.value.code == "insufficient_credit" and e.value.status == 402 and "billing" in (e.value.hint or "")


def test_a_short_rate_limit_is_waited_out_once(monkeypatch):
    import httpx

    from supposing.cloud import Client, CloudError

    waits: list[float] = []
    monkeypatch.setattr("supposing.cloud.client.time.sleep", waits.append)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if req.url.path == "/v1/runs" and calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "4"}, json={
                "code": "rate_limited", "message": "Too many new runs. Try again in 4 seconds.", "hint": "Wait."})
        if req.url.path == "/v1/runs":
            return httpx.Response(429, headers={"Retry-After": "400"}, json={
                "code": "rate_limited", "message": "Too many new runs. Try again in 400 seconds.", "hint": "Wait."})
        return httpx.Response(404, json={"code": "not_found", "message": "x", "hint": None})

    with Client("sp_test", "http://fake", transport=httpx.MockTransport(handler)) as sp, \
            pytest.raises(CloudError) as exc:
        sp.runs.create("ds", "Better?", judge="mock")  # 4 s: waited once; then 400 s: reported
    assert waits == [4.0] and calls["n"] == 2
    assert exc.value.code == "rate_limited" and "400 seconds" in exc.value.message

