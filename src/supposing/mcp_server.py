"""MCP server over stdio, for coding agents on this machine: ``supposing mcp``.

Needs the optional extra: ``uv tool install 'supposing[mcp]'``, or ``uvx 'supposing[mcp]' mcp``.

It offers the same tools as the hosted server at https://api.supposing.app/mcp, with the same arguments
and result shapes. Where the work happens depends on credentials:

- **cloud**, when an API key is found (``SUPPOSING_API_KEY`` or ``supposing login``): runs go to the Supposing
  API through ``supposing.cloud``, so they show up in the web app and survive restarts.
- **local**, otherwise: the engine ranks in this process with your own judge (an OpenRouter key in
  ``OPENROUTER_API_KEY``, the ``claude`` CLI login, or the free ``mock`` judge). Local runs live as long as the
  server process.

Claude Code: ``claude mcp add supposing -- supposing mcp``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Protocol
from uuid import UUID

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from . import present, samples
from . import ranking as rk
from .cloud import JUDGE_HELP, AsyncClient, CloudError, JudgeError, parse_judge, read_folder, resolve
from .providers import ProviderError, make_provider
from .report import summarize
from .runner import RunError, run_ranking
from .store import MemoryStore

MAX_OPTIONS = 8
ROUND_ROBIN = {"rounds": 1, "pairing": "round_robin"}
MAX_OPTION_CHARS = 20_000
POLL_SECONDS = 0.5
DONE = ("done", "failed", "canceled")
FREE = ("mock", "claude_cli")

INSTRUCTIONS = """\
Supposing ranks texts with an LLM judge: it runs a pairwise tournament and reports each item's rank with a 90% interval, so you know when a winner is real and when \
items are tied.

- Picking the best of 2 to 8 short options (drafts, prompts, names, answers): call compare_options. \
It waits for the answer, usually 10 to 60 seconds.
- No documents at hand: list_samples, then rank_documents(sample=<id>).
- Ranking more or longer documents, or a folder of files: call rank_documents (it returns at once with a run \
id and the number of judge calls), then get_run until status is done, then get_results. Pass \
estimate_only=true first to see the number of calls before anything is spent.

Write the rubric as a short paragraph saying what "better" means and what to ignore. Results lead with \
a one-sentence verdict; treat items whose rank intervals overlap as tied."""


class BackendError(Exception):
    """The REST error shape, ``{code, message, hint}``, from either backend."""

    def __init__(self, code: str, message: str, hint: str | None = None,
                 details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code, self.message, self.hint, self.details = code, message, hint, details

    def text(self) -> str:
        return present.error_text(self.code, self.message, self.hint, self.details)


@dataclass
class RunSpec:
    dataset_id: str
    rubric: str
    name: str | None = None
    judge: str | None = None  # quick, balanced, thorough, an OpenRouter model id, mock or claude_cli
    max_cost_usd: float | None = None
    options: dict[str, Any] = field(default_factory=dict)  # the ranking's own: rounds, goal, top_n, ...
    ephemeral: bool = False  # a one-off: kept out of the web app's lists


class Backend(Protocol):
    """The few API calls the tools make. Both return the API's JSON shapes."""
    mode: str

    async def create_dataset(self, name: str, items: list[dict[str, str]], ephemeral: bool = False
                             ) -> dict[str, Any]: ...
    async def push_folder(self, folder: str, name: str | None) -> dict[str, Any]: ...
    async def estimate(self, spec: RunSpec) -> dict[str, Any]: ...
    async def create_run(self, spec: RunSpec) -> dict[str, Any]: ...
    async def get_run(self, run_id: str) -> dict[str, Any]: ...
    async def cancel_run(self, run_id: str) -> dict[str, Any]: ...
    async def results(self, run_id: str) -> dict[str, Any]: ...
    async def open_sample(self, sample_id: str) -> dict[str, Any] | None: ...


# --- cloud: the Supposing API through supposing.cloud ----------------------------------------------------------
class CloudBackend:
    mode = "cloud"

    def __init__(self, client: AsyncClient):
        self.c = client

    async def _do[T](self, aw: Awaitable[T]) -> T:
        try:
            return await aw
        except CloudError as exc:
            raise BackendError(exc.code, exc.message, exc.hint, exc.details) from exc

    @staticmethod
    def _kw(spec: RunSpec) -> dict[str, Any]:
        return {"name": spec.name, "judge": spec.judge, "max_cost_usd": spec.max_cost_usd, "ephemeral": spec.ephemeral, **spec.options}

    async def create_dataset(self, name: str, items: list[dict[str, str]], ephemeral: bool = False
                             ) -> dict[str, Any]:
        return (await self._do(self.c.datasets.create(name, items, ephemeral))).raw

    async def push_folder(self, folder: str, name: str | None) -> dict[str, Any]:
        ds = await self._do(self.c.datasets.push(folder, name))
        return {**ds.raw, "skipped": ds.skipped}

    async def estimate(self, spec: RunSpec) -> dict[str, Any]:
        return (await self._do(self.c.runs.estimate(spec.dataset_id, spec.rubric, **self._kw(spec)))).raw

    async def create_run(self, spec: RunSpec) -> dict[str, Any]:
        return (await self._do(self.c.runs.create(spec.dataset_id, spec.rubric, **self._kw(spec)))).raw

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return (await self._do(self.c.runs.get(run_id))).raw

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        return (await self._do(self.c.runs.cancel(run_id))).raw

    async def results(self, run_id: str) -> dict[str, Any]:
        return (await self._do(self.c.runs.results(run_id))).raw

    async def open_sample(self, sample_id: str) -> dict[str, Any] | None:
        return (await self._do(self.c.open_sample(sample_id))).raw


# --- local: the engine, in this process ---------------------------------------------------------------------
def _iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, UTC).isoformat() if t else None


@dataclass
class LocalRun:
    id: str
    name: str
    dataset: dict[str, Any]
    spec: RunSpec
    cfg: rk.RankConfig
    judge: dict[str, Any]
    store: MemoryStore = field(default_factory=MemoryStore)
    status: str = "queued"
    note: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    task: asyncio.Task[None] | None = None

    @property
    def items(self) -> list[rk.Item]:
        return [rk.Item(id=i["id"], text=i["text"], title=i["title"]) for i in self.dataset["items"]]


class LocalBackend:
    """Datasets and runs held in memory; judging with the engine and your own provider."""
    mode = "local"

    def __init__(self, concurrency: int = 8):
        self.concurrency = concurrency
        self.datasets: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, LocalRun] = {}

    async def create_dataset(self, name: str, items: list[dict[str, str]], ephemeral: bool = False
                             ) -> dict[str, Any]:
        ds = {"id": str(uuid.uuid4()), "name": name, "created_at": _iso(time.time()), "n_items": len(items),
              "items": [{"id": str(uuid.uuid4()), "title": it["title"], "text": it["text"],
                         "n_words": len(it["text"].split()), "n_chars": len(it["text"])} for it in items]}
        self.datasets[ds["id"]] = ds
        return ds

    async def push_folder(self, folder: str, name: str | None) -> dict[str, Any]:
        try:
            found = read_folder(Path(folder).expanduser())
        except FileNotFoundError as exc:
            raise BackendError("not_a_folder", str(exc), "Pass the absolute path of a folder of .md, .txt or "
                               ".html files.") from exc
        ds = await self.create_dataset(name or Path(folder).name, [{"title": t, "text": x} for t, x in found.items])
        return {**ds, "skipped": [{"path": p, "reason": r} for p, r in found.skipped]}

    def _dataset(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self.datasets:
            raise BackendError("dataset_not_found", "Dataset not found.", "Local datasets last as long as this MCP "
                               "server; pass items or folder instead.")
        return self.datasets[dataset_id]

    def _run(self, run_id: str) -> LocalRun:
        if run_id not in self.runs:
            raise BackendError("run_not_found", "Run not found.", "Local runs last as long as this MCP server; "
                               "start it again with rank_documents.")
        return self.runs[run_id]

    @staticmethod
    def _config(spec: RunSpec) -> rk.RankConfig:
        try:
            return rk.RankConfig.from_dict({**spec.options, "max_cost_usd": spec.max_cost_usd})
        except rk.ConfigError as exc:
            raise BackendError("invalid_request", str(exc)) from exc

    async def estimate(self, spec: RunSpec) -> dict[str, Any]:
        ds, cfg = self._dataset(spec.dataset_id), self._config(spec)
        n = len(ds["items"])
        hi, lo = rk.planned_calls(n, cfg), rk.fewest_calls(n, cfg)
        chars = sum(min(i["n_chars"], cfg.max_chars or 10**9) for i in ds["items"]) / max(1, n)
        per_call = int(2 * chars / 4 + 450 + len(spec.rubric) / 4)
        free = self._parse(spec)["provider"] in FREE
        return {"type": "ranking", "n_items": n, "min_calls": lo, "max_calls": hi, "est_input_tokens": hi * per_call,
                "est_output_tokens": hi * 150, "est_cost_usd": 0.0 if free else None,
                "est_cost_usd_min": 0.0 if free else None, "est_seconds": None}

    @staticmethod
    def _parse(spec: RunSpec) -> dict[str, Any]:
        try:
            return parse_judge(spec.judge)
        except JudgeError as exc:
            raise BackendError("invalid_request", str(exc), "Leave judge out for quick, or pass balanced, "
                               "thorough, an OpenRouter model id, mock or claude_cli.") from exc

    def _judge(self, spec: RunSpec) -> dict[str, Any]:
        judge = self._parse(spec)
        provider = judge["provider"]
        if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
            raise BackendError("provider_unavailable", "No OpenRouter key: OPENROUTER_API_KEY is not set for the "
                               "MCP server.", "Set it in the server's env, log in with `supposing login` to use "
                               "the cloud, or pass judge claude_cli (your Claude login) or mock "
                               "(free, fake verdicts).")
        if provider == "claude_cli" and not shutil.which("claude"):
            raise BackendError("provider_unavailable", "The claude CLI is not on PATH.",
                               "Install Claude Code, or use another judge.")
        return judge

    async def create_run(self, spec: RunSpec) -> dict[str, Any]:
        ds, cfg = self._dataset(spec.dataset_id), self._config(spec)
        if len(ds["items"]) < 2:
            raise BackendError("too_few_items", "A ranking needs at least two items; this dataset has %d."
                               % len(ds["items"]))
        judge = self._judge(spec)
        run = LocalRun(id=str(uuid.uuid4()), name=spec.name or "%s · %s" % (ds["name"], judge["model"]),
                       dataset=ds, spec=spec, cfg=cfg, judge=judge)
        self.runs[run.id] = run
        run.task = asyncio.create_task(self._drive(run))
        return await self.get_run(run.id)

    async def _drive(self, run: LocalRun) -> None:
        provider = make_provider(run.judge)
        run.status, run.started_at = "running", time.time()
        try:
            result = await run_ranking(run.items, run.spec.rubric, provider, run.cfg, run.store, run.id,
                                       self.concurrency)
            run.status, run.note = "done", result.note
        except asyncio.CancelledError:
            run.status = "canceled"
        except (RunError, ProviderError) as exc:
            run.status, run.error = "failed", str(exc)
        finally:
            run.finished_at = time.time()
            await provider.close()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        comps = run.store.load(run.id)
        progress = {"total": len(comps), "done": sum(c.status == "done" for c in comps),
                    "failed": sum(c.status == "failed" for c in comps), "in_flight": 0,
                    "pending": sum(c.status == "pending" for c in comps),
                    "round": max((c.round for c in comps), default=0), "cost_usd": rk.spent(comps)}
        cfg = run.cfg.to_dict()
        cfg.pop("max_cost_usd", None)
        return {"id": run.id, "type": "ranking", "name": run.name, "status": run.status,
                "dataset_id": run.dataset["id"], "dataset_name": run.dataset["name"], "judge": run.judge,
                "max_cost_usd": run.cfg.max_cost_usd, "ranking": {"rubric": run.spec.rubric, **cfg},
                "planned_calls": rk.planned_calls(len(run.dataset["items"]), run.cfg), "note": run.note,
                "error": run.error, "progress": progress, "created_at": _iso(run.created_at),
                "started_at": _iso(run.started_at), "finished_at": _iso(run.finished_at)}

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run.task and not run.task.done():
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)
        return await self.get_run(run_id)

    async def open_sample(self, sample_id: str) -> dict[str, Any] | None:
        return None  # no stored rankings on this machine: the sample is ranked with your judge

    async def results(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        cfg = run.cfg
        res = rk.results(run.items, run.store.load(run.id), top_n=cfg.top_n if cfg.goal == "top_n" else None,
                         seed=cfg.seed)
        final = run.status in DONE
        return {"run_id": run.id, "type": "ranking", "final": final,
                "summary": summarize(res["leaderboard"], res["diagnostics"], final=final,
                                     top_n=cfg.top_n if cfg.goal == "top_n" else None),
                "ranking": res}


# --- results for agents: wording from supposing.present, shared with the CLI and the other MCP server ------------
def result(summary: str, data: dict[str, Any], text: str | None = None) -> CallToolResult:
    """``summary`` leads both the text the model reads and the structured content."""
    body = {"summary": summary, **data}
    shown = (text or summary) + "\n\n" + json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str)
    return CallToolResult(content=[TextContent(type="text", text=shown)], structured_content=body)


# --- tools ------------------------------------------------------------------------------------------------
class Item(BaseModel):
    title: str = Field(min_length=1, max_length=300, description="A name for the item, e.g. a file name.")
    text: str = Field(min_length=1, description="The full text to judge.")


Rubric = Annotated[str, Field(min_length=1, max_length=20000, description=(
    "What makes one text better than another, in a sentence or short paragraph. Say what to ignore too, "
    "e.g. 'Which is clearest to a new user? Ignore length.'"))]
JudgeArg = Annotated[str | None, Field(description="The judge: " + JUDGE_HELP)]
BothOrders = Annotated[bool, Field(description=(
    "Judge every pair twice, once in each order. Twice the cost; adds the order check."))]
RubricOrSample = Annotated[str | None, Field(min_length=1, max_length=20000, description=(
    "What makes one text better than another, in a sentence or short paragraph. Say what to ignore too, "
    "e.g. 'Which is clearest to a new user? Ignore length.' Required unless you pass sample."))]
RunId = Annotated[UUID, Field(description="The run_id returned by rank_documents or compare_options.")]


def _sample(sample_id: str) -> dict[str, Any]:
    """A built-in sample with its items as {title, text}; an unknown id says which exist."""
    try:
        s = samples.load(sample_id)
    except KeyError as exc:
        raise ToolError(present.error_text("sample_not_found", exc.args[0], "Call list_samples.")) from exc
    return {**s, "items": [{"title": i["title"], "text": i["text"]} for i in s["items"]]}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _spec(dataset_id: str, rubric: str, name: str | None, judge: str | None, options: dict[str, Any],
          max_cost_usd: float | None = None) -> RunSpec:
    return RunSpec(dataset_id, rubric, name, judge, max_cost_usd, {k: v for k, v in options.items() if v is not None})


async def wait_for_run(backend: Backend, run_id: str, wait_seconds: float,
                       report: Callable[..., Awaitable[None]]) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    last = -1
    while True:
        run = await backend.get_run(str(run_id))
        done = (run.get("progress") or {}).get("done", 0)
        if done > last:
            last = done
            await report(last, run["planned_calls"], "%d of %s judge calls done" % (last, run["planned_calls"]))
        if run["status"] in DONE or time.monotonic() >= deadline:
            return run
        await asyncio.sleep(POLL_SECONDS)


# Tool annotations, explicit on every tool (OpenAI requires destructiveHint; Claude's directory scan wants them all).
# Starting a ranking spends the workspace's credit or model key and creates a run, but changes or deletes nothing
# that exists: not read-only, not destructive. Calling it twice starts two runs and spends twice: not idempotent.
# Reading a run is read-only and idempotent. No tool reaches beyond Supposing and its judge model: not open-world.
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def hints(title: str, base: ToolAnnotations) -> ToolAnnotations:
    return base.model_copy(update={"title": title})


def build_server(backend: Backend) -> MCPServer:
    mcp = MCPServer("supposing", title="Supposing", instructions=INSTRUCTIONS + (
        "\n\nThis server runs in %s mode." % backend.mode), website_url="https://supposing.app", subscriptions=False)

    async def call[T](aw: Awaitable[T]) -> T:
        try:
            return await aw
        except BackendError as exc:
            raise ToolError(exc.text()) from exc

    @mcp.tool(title="List samples", annotations=hints("List samples", READ))
    async def list_samples() -> CallToolResult:
        """Built-in sample datasets, for trying a ranking when you have no documents of your own.

        Free and instant. Each has an id, a description, its rubric and how many items it holds; pass the id as
        rank_documents(sample=...).
        """
        rows = samples.list_samples()
        return result("%d samples: %s. Rank one with rank_documents(sample=<id>)."
                      % (len(rows), ", ".join(s["id"] for s in rows)), {"data": rows})

    @mcp.tool(title="Rank documents", annotations=hints("Rank documents", WRITE))
    async def rank_documents(
        rubric: RubricOrSample = None,
        items: Annotated[list[Item] | None, Field(
            description="The documents to rank, 2 to 2000, as {title, text}. Or pass folder or dataset_id.")] = None,
        folder: Annotated[str | None, Field(description=(
            "Absolute path of a folder on this machine: every .md, .txt and .html file in it is one item."))] = None,
        dataset_id: Annotated[UUID | None, Field(description="Rank an existing dataset instead.")] = None,
        sample: Annotated[str | None, Field(description=(
            "A built-in sample instead (ids from list_samples). In cloud mode, alone it opens the sample's stored "
            "ranking, free and instant; with a judge or a rubric it is a new run. Locally it is ranked with your "
            "judge and the sample's rubric unless you pass one."))] = None,
        judge: JudgeArg = None,
        rounds: Annotated[int | None, Field(ge=1, le=200, description=(
            "Opponents per document (default 8). More rounds, tighter intervals, more calls."))] = None,
        top_n: Annotated[int | None, Field(ge=1, description=(
            "Only settle the top N. Cheaper on big sets: after a warm-up, only contenders keep playing."))] = None,
        max_cost_usd: Annotated[float | None, Field(ge=0, description="Stop the run once it has spent this much.")
                                ] = None,
        estimate_only: Annotated[bool, Field(description=(
            "Return the number of judge calls without starting anything. Items and folders are still saved as a "
            "dataset, whose id you can pass next time."))] = False,
        both_orders: BothOrders = False,
        name: Annotated[str | None, Field(description="A name for the run.")] = None,
    ) -> CallToolResult:
        """Start ranking a set of documents against your rubric, in the background.

        Use for more than 8 options, for long documents, or for a folder of files; for 2 to 8 short options use
        compare_options, which waits for the answer. Returns at once with a run_id and the estimate. Each
        document meets `rounds` opponents, one judge call each, so a run makes about rounds x documents / 2
        judge calls (8 rounds over 20 documents: 80 calls; twice that with both_orders). Small runs finish in a
        minute or two. Follow with get_run, then get_results. Costs go to the judge's account.
        """
        if sum(x is not None for x in (items, folder, dataset_id, sample)) != 1:
            raise ToolError("Pass exactly one of items (a list of {title, text}), folder, dataset_id or sample.")
        skipped: list[Any] = []
        if sample is not None and judge is None and rubric is None and backend.mode == "cloud":
            s = _sample(sample)
            if estimate_only:
                return result("Free: the %s sample's stored ranking opens at once, with no judge calls. Call "
                              "rank_documents with sample=%s and estimate_only=false to open it." % (s["name"], sample),
                              {"sample": sample, "estimate": None})
            run = await call(backend.open_sample(sample))
            assert run is not None
            res = await call(backend.results(str(run["id"])))
            text, data = present.results_payload(res, run, 20)
            lead = "The %s sample's stored ranking (judged by %s, free): " % (
                s["name"], (run.get("judge") or {}).get("model") or "a real model")
            return result(res["summary"], data, lead + text)
        if sample is not None:
            s = _sample(sample)
            rubric = rubric or s["rubric"]
            ds = await call(backend.create_dataset("Sample: %s" % s["name"], s["items"]))
            dataset_id = UUID(str(ds["id"]))
        if rubric is None:
            raise ToolError("Pass a rubric: what makes one text better than another.")
        if items is not None:
            if len(items) < 2:
                raise ToolError("A ranking needs at least two items.")
            ds = await call(backend.create_dataset(name or "rank_documents %s" % _now(),
                                                   [d.model_dump() for d in items]))
            dataset_id = UUID(str(ds["id"]))
        elif folder is not None:
            ds = await call(backend.push_folder(folder, name))
            dataset_id, skipped = UUID(str(ds["id"])), ds.get("skipped") or []
        assert dataset_id is not None
        options: dict[str, Any] = {"rounds": rounds, "both_orders": both_orders}
        if top_n:
            options.update(goal="top_n", top_n=top_n)
        spec = _spec(str(dataset_id), rubric, name, judge, options, max_cost_usd)
        est = await call(backend.estimate(spec))
        extra = {"skipped_files": skipped[:20]} if skipped else {}
        if estimate_only:
            return result("Estimate: %s. Nothing was started; call rank_documents with dataset_id=%s and "
                          "estimate_only=false to run it." % (present.estimate_line(est), dataset_id),
                          {"dataset_id": dataset_id, "estimate": est, **extra})
        run = await call(backend.create_run(spec))
        return result("Started run %s: %s. Call get_run(run_id) to follow it and get_results(run_id) for the "
                      "ranking; results are readable while it runs." % (run["id"], present.estimate_line(est)),
                      {"run": run, "estimate": est, **extra})

    @mcp.tool(title="Get run", annotations=hints("Get run", READ))
    async def get_run(run_id: RunId) -> CallToolResult:
        """Status and progress of a run: queued, running, done, failed or canceled, with calls done and money spent.

        Free and instant. Poll it every 10 to 30 seconds while a run is going; when status is done, call
        get_results.
        """
        run = await call(backend.get_run(str(run_id)))
        return result(present.run_line(run), {"run": run})

    @mcp.tool(title="Get results", annotations=hints("Get results", READ))
    async def get_results(
        run_id: RunId,
        top: Annotated[int, Field(ge=1, le=2000, description="How many leaderboard rows to return.")] = 20,
    ) -> CallToolResult:
        """The ranking of a run: a one-sentence verdict, then the leaderboard with 90% rank intervals and judge
        diagnostics.

        Free and instant. Works while the run is still going (marked provisional). Items whose rank intervals
        overlap may be tied; diagnostics.reliability grades the judge (the order check only with both_orders).
        """
        run = await call(backend.get_run(str(run_id)))
        res = await call(backend.results(str(run_id)))
        text, data = present.results_payload(res, run, top)
        return result(res["summary"], data, text)

    @mcp.tool(title="Compare options", annotations=hints("Compare options", WRITE))
    async def compare_options(
        rubric: Rubric,
        options: Annotated[list[Annotated[str, Field(min_length=1, max_length=MAX_OPTION_CHARS)]], Field(
            min_length=2, max_length=MAX_OPTIONS, description="2 to 8 candidate texts, each up to 20,000 characters.")],
        ctx: Context,
        judge: JudgeArg = None,
        both_orders: BothOrders = False,
        wait_seconds: Annotated[int, Field(ge=0, le=300, description=(
            "How long to wait for the answer before handing back a run_id to poll. Keep it under your client's "
            "tool timeout."))] = 45,
    ) -> CallToolResult:
        """Pick the best of 2 to 8 short options (drafts, prompts, names, answers) and wait for the answer.

        Every option meets every other once: n x (n-1) / 2 judge calls (4 options: 6 calls, 8 options: 28;
        twice that with both_orders), a few cents with the default quick judge. It usually answers in 10 to 60 seconds. If it
        is not finished within wait_seconds you get a run_id and a provisional ranking: call get_results later.
        Options are named "option 1", "option 2", ... in the order given.
        """
        n = len(options)
        # A one-off: ephemeral keeps it out of the web app's lists. Every pair is judged in one round.
        ds = await call(backend.create_dataset("compare_options %s" % _now(), [
            {"title": "option %d" % (i + 1), "text": text} for i, text in enumerate(options)], ephemeral=True))
        spec = _spec(ds["id"], rubric, "compare_options %s" % _now(), judge, {**ROUND_ROBIN, "both_orders": both_orders})
        spec.ephemeral = True
        run = await call(backend.create_run(spec))
        texts = {str(item["id"]): text for item, text in zip(ds["items"], options, strict=True)}
        try:
            run = await call(wait_for_run(backend, run["id"], wait_seconds, ctx.report_progress))
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):  # the caller gave up: stop spending on its behalf
                await call(backend.cancel_run(run["id"]))
            raise
        res = await call(backend.results(run["id"]))
        text, data = present.results_payload(res, run, MAX_OPTIONS, texts, wait_seconds)
        if run["status"] == "failed":
            return result(present.run_line(run), data)
        board = data[data["type"]]["leaderboard"]
        if board:  # the summary may not name it when options are close; the agent asked which is best
            top = board[0]
            text += ' Top option: "%s" (rank %d-%d of %d).' % (top["title"], top["rank_lo"], top["rank_hi"], n)
        return result(res["summary"], data, text)

    return mcp


def make_backend(local: bool = False) -> Backend:
    creds = resolve()
    if creds.api_key and not local:
        return CloudBackend(AsyncClient(creds.api_key, creds.api_url, source="mcp"))
    return LocalBackend()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supposing mcp", description="MCP server over stdio for coding agents.")
    parser.add_argument("--local", action="store_true", help="rank on this machine even when an API key is set")
    args = parser.parse_args(argv)
    build_server(make_backend(args.local)).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
