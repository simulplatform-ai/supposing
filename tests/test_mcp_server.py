"""The stdio MCP server in local mode: the engine ranks in-process with the mock judge."""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from supposing.mcp_server import LocalBackend, build_server  # noqa: E402

MOCK = {"provider": "mock", "model": "mock"}


def text(res) -> str:
    return res.content[0].text


def local() -> Client:
    return Client(build_server(LocalBackend()), raise_exceptions=True)


async def test_same_four_tools_as_the_hosted_server():
    async with local() as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"rank_documents", "get_run", "get_results", "compare_options"}
        assert "folder" in tools["rank_documents"].input_schema["properties"]


async def test_compare_options_locally():
    async with local() as client:
        options = ["Vague. [[q=0.1]]", "Clear and specific. [[q=0.9]]", "Okay. [[q=0.5]]", "Fine. [[q=0.3]]"]
        res = await client.call_tool("compare_options", {"rubric": "Which is clearest?", "options": options,
                                                         "judge": MOCK})
        assert not res.is_error, text(res)
        data = res.structured_content
        assert data["final"] and data["run"]["status"] == "done" and data["run"]["progress"]["done"] == 12
        assert data["ranking"]["leaderboard"][0]["title"] == "option 2"
        assert 'Top option: "option 2" (rank 1-' in text(res)


async def test_rank_a_folder_then_poll(tmp_path):
    async with local() as client:
        for i in range(6):
            (tmp_path / ("doc%d.md" % i)).write_text("Doc %d [[q=%.2f]]" % (i, i / 6))
        (tmp_path / "notes.pdf").write_bytes(b"%PDF")
        est = await client.call_tool("rank_documents", {"rubric": "Which is better?", "folder": str(tmp_path),
                                                        "judge": MOCK, "rounds": 3, "estimate_only": True})
        assert text(est).startswith("Estimate: 6 items, 18 judge calls, free"), text(est)
        assert est.structured_content["skipped_files"][0]["path"] == "notes.pdf"
        started = await client.call_tool("rank_documents", {"rubric": "Which is better?", "judge": MOCK, "rounds": 3,
                                                            "dataset_id": est.structured_content["dataset_id"]})
        run_id = started.structured_content["run"]["id"]
        for _ in range(200):
            run = await client.call_tool("get_run", {"run_id": run_id})
            if run.structured_content["run"]["status"] == "done":
                break
        assert text(run).startswith("Run %s is done: 18 of 18 judge calls done" % run_id)
        res = await client.call_tool("get_results", {"run_id": run_id, "top": 2})
        assert res.structured_content["final"] and len(res.structured_content["ranking"]["leaderboard"]) == 2
        assert res.structured_content["ranking"]["leaderboard"][0]["title"] == "doc5.md"


async def test_errors_say_what_to_do_next(monkeypatch):
    async with local() as client:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        res = await client.call_tool("compare_options", {"rubric": "x", "options": ["a", "b"]})
        assert res.is_error and "provider_unavailable" in text(res) and "mock" in text(res)
        res = await client.call_tool("get_run", {"run_id": "nope"})
        assert res.is_error and "run_not_found" in text(res)
        res = await client.call_tool("rank_documents", {"rubric": "x", "folder": "/no/such/folder"})
        assert res.is_error and "not_a_folder" in text(res)


async def test_runs_over_real_stdio(tmp_path):
    env = {"SUPPOSING_CONFIG": str(tmp_path / "none.json"), "PATH": "/usr/bin:/bin"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "supposing.mcp_server"], env=env)
    async with Client(params) as c:
        res = await c.call_tool("compare_options", {"rubric": "x", "options": ["a [[q=0.2]]", "b [[q=0.8]]"],
                                                    "judge": MOCK})
    assert not res.is_error, text(res)
    assert res.structured_content["ranking"]["leaderboard"][0]["title"] == "option 2"
