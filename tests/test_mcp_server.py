"""The stdio MCP server in local mode: the engine ranks in-process with the mock judge."""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from supposing.mcp_server import LocalBackend, build_server  # noqa: E402

MOCK = "mock"


def text(res) -> str:
    return res.content[0].text


def local() -> Client:
    return Client(build_server(LocalBackend()), raise_exceptions=True)


async def test_same_tools_as_the_hosted_server():
    async with local() as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"list_samples", "rank_documents", "get_run", "get_results", "compare_options"}
        from supposing import __version__
        assert client.server_info.version == __version__
        assert "folder" in tools["rank_documents"].input_schema["properties"]
        for name, read_only in (("list_samples", True), ("rank_documents", False), ("compare_options", False), ("get_run", True),
                                ("get_results", True)):
            a = tools[name].annotations
            assert tools[name].title and a is not None and a.destructive_hint is False, name
            assert (a.read_only_hint, a.idempotent_hint, a.open_world_hint) == (read_only, read_only, False), name


async def test_compare_options_locally():
    async with local() as client:
        options = ["Vague. [[q=0.1]]", "Clear and specific. [[q=0.9]]", "Okay. [[q=0.5]]", "Fine. [[q=0.3]]"]
        res = await client.call_tool("compare_options", {"rubric": "Which is clearest?", "options": options,
                                                         "judge": MOCK})
        assert not res.is_error, text(res)
        data = res.structured_content
        assert data["final"] and data["run"]["status"] == "done" and data["run"]["progress"]["done"] == 6
        assert data["ranking"]["leaderboard"][0]["title"] == "option 2"
        assert 'Top option: "option 2" (rank 1-' in text(res)
        res = await client.call_tool("compare_options", {"rubric": "Which is clearest?", "options": options,
                                                         "judge": MOCK, "both_orders": True})
        assert res.structured_content["run"]["progress"]["done"] == 12  # every pair in both orders


async def test_rank_a_sample():
    async with local() as client:
        listed = await client.call_tool("list_samples", {})
        ids = [s["id"] for s in listed.structured_content["data"]]
        assert "stories" in ids and text(listed).startswith("%d samples: " % len(ids))
        est = await client.call_tool("rank_documents", {"sample": "stories", "judge": MOCK, "estimate_only": True})
        assert not est.is_error, text(est)
        assert est.structured_content["estimate"]["n_items"] == listed.structured_content["data"][ids.index(
            "stories")]["n_items"]
        run = await client.call_tool("rank_documents", {"sample": "stories", "judge": MOCK, "rounds": 2})
        assert run.structured_content["run"]["ranking"]["rubric"] == listed.structured_content["data"][ids.index(
            "stories")]["rubric"]  # the sample's own rubric
        res = await client.call_tool("rank_documents", {"sample": "nope", "judge": MOCK})
        assert res.is_error and "sample_not_found" in text(res) and "list_samples" in text(res)
        res = await client.call_tool("rank_documents", {"items": [{"title": "a", "text": "a"}] * 2, "judge": MOCK})
        assert res.is_error and "rubric" in text(res)


async def test_rank_a_folder_then_poll(tmp_path):
    async with local() as client:
        for i in range(6):
            (tmp_path / ("doc%d.md" % i)).write_text("Doc %d [[q=%.2f]]" % (i, i / 6))
        (tmp_path / "notes.pdf").write_bytes(b"%PDF")
        est = await client.call_tool("rank_documents", {"rubric": "Which is better?", "folder": str(tmp_path),
                                                        "judge": MOCK, "rounds": 3, "estimate_only": True})
        assert text(est).startswith("Estimate: 6 items, 9 judge calls, free"), text(est)
        assert est.structured_content["skipped_files"][0]["path"] == "notes.pdf"
        started = await client.call_tool("rank_documents", {"rubric": "Which is better?", "judge": MOCK, "rounds": 3,
                                                            "dataset_id": est.structured_content["dataset_id"]})
        run_id = started.structured_content["run"]["id"]
        for _ in range(200):
            run = await client.call_tool("get_run", {"run_id": run_id})
            if run.structured_content["run"]["status"] == "done":
                break
        assert text(run).startswith("Run %s is done: 9 of 9 judge calls done" % run_id)
        res = await client.call_tool("get_results", {"run_id": run_id, "top": 2})
        assert res.structured_content["final"] and len(res.structured_content["ranking"]["leaderboard"]) == 2
        assert res.structured_content["ranking"]["leaderboard"][0]["title"] == "doc5.md"


async def test_errors_say_what_to_do_next(monkeypatch):
    async with local() as client:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        res = await client.call_tool("compare_options", {"rubric": "x", "options": ["a", "b"]})
        assert res.is_error and "provider_unavailable" in text(res) and "mock" in text(res)
        res = await client.call_tool("get_run", {"run_id": "00000000-0000-0000-0000-000000000000"})
        assert res.is_error and "run_not_found" in text(res)
        res = await client.call_tool("get_run", {"run_id": "../api-keys"})  # ids are UUIDs, never path text
        assert res.is_error and "uuid" in text(res).lower()
        res = await client.call_tool("rank_documents", {"rubric": "x", "folder": "/no/such/folder"})
        assert res.is_error and "not_a_folder" in text(res)
        res = await client.call_tool("compare_options", {"rubric": "x", "options": ["a", "b"], "judge": "fast"})
        assert res.is_error and "invalid_request" in text(res) and "balanced" in text(res)


async def test_runs_over_real_stdio(tmp_path):
    env = {"SUPPOSING_CONFIG": str(tmp_path / "none.json"), "PATH": "/usr/bin:/bin"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "supposing.mcp_server"], env=env)
    async with Client(params) as c:
        res = await c.call_tool("compare_options", {"rubric": "x", "options": ["a [[q=0.2]]", "b [[q=0.8]]"],
                                                    "judge": MOCK})
    assert not res.is_error, text(res)
    assert res.structured_content["ranking"]["leaderboard"][0]["title"] == "option 2"


async def test_context_and_examples_reach_the_local_judge(monkeypatch):
    from supposing import mcp_server
    from supposing.providers import make_provider as real_make

    seen: list[str] = []
    real = real_make({"provider": "mock"})

    class Spy:
        name, model = "mock", "mock"

        async def complete(self, system, user):
            seen.append(user)
            return await real.complete(system, user)

        async def close(self):
            await real.close()

    monkeypatch.setattr(mcp_server, "make_provider", lambda spec: Spy())
    async with local() as client:
        res = await client.call_tool("compare_options", {
            "rubric": "Which is clearer?", "options": ["a [[q=0.2]]", "b [[q=0.8]]"], "judge": MOCK,
            "context": "BRIEF-MARK: for nurses", "examples": [{"a": "EX-A", "b": "EX-B", "winner": "A"}]})
    assert not res.is_error, text(res)
    assert seen and "BRIEF-MARK" in seen[0] and "EX-A" in seen[0]


async def test_cloud_mode_refuses_context_for_now():
    from supposing.mcp_server import BackendError, CloudBackend, RunSpec

    with pytest.raises(BackendError) as exc:
        await CloudBackend(None).estimate(RunSpec("d", "r", context="x"))  # type: ignore[arg-type]
    assert exc.value.code == "unsupported" and "--local" in (exc.value.hint or "")

