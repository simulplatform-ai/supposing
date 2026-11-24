"""Running a ranking to completion in one process."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import judge, report
from .providers import Provider, ProviderError, make_provider
from .ranking import Comparison, Item, RankConfig, affordable, next_action, plan_round, planned_calls, results, spent
from .rubric import Rubric
from .store import MemoryStore, RunStore

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


class RunError(RuntimeError):
    """A run cannot continue: the provider refused a call for good (bad key, no credit, unknown model), every
    call in a round failed, or the items are invalid. With a ``SQLiteStore``, fix the cause and run again to
    continue where it stopped."""


@dataclass
class Ranking:
    """A finished run.

    ``leaderboard`` and ``diagnostics`` are as described in ``supposing.results``. ``comparisons`` holds every
    judge call with its verdict and reasoning. ``note`` says why the run stopped early, if it did. ``cost_usd``
    is the recorded spend (zero for providers that do not report a price). ``summary`` is the verdict in plain
    words (``supposing.summarize``). ``interval`` is the confidence level of every interval and ``n_bootstrap``
    the number of resamples behind them. ``groups`` is the leaderboard split for display (``supposing.display_groups``). ``calls_saved`` is how many calls fewer than the fixed plan
    (``planned_calls``) the run made, when a top-N run stopped early or the spend cap was reached.
    """

    leaderboard: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    comparisons: list[Comparison]
    note: str | None
    cost_usd: float
    summary: str
    interval: str = "90%"
    n_bootstrap: int = 0
    calls_saved: int = 0
    groups: list[dict[str, Any]] = field(default_factory=list)

    def top(self, n: int) -> list[dict[str, Any]]:
        """The best ``n`` leaderboard rows."""
        return self.leaderboard[:n]


Progress = Callable[[int, int], None]  # (judged so far, comparisons planned so far)


async def run_ranking(
    items: Sequence[Item],
    rubric: str | Rubric,
    provider: Provider,
    config: RankConfig | None = None,
    store: RunStore | None = None,
    run_id: str = "run",
    concurrency: int = 8,
    on_progress: Progress | None = None,
) -> Ranking:
    """Rank ``items`` against ``rubric`` with ``provider`` as the judge, and return the finished ``Ranking``.

    If ``store`` already holds comparisons for ``run_id``, the run continues from them: a decided comparison
    is never judged again. ``concurrency`` caps judge calls in flight. ``on_progress(judged, planned)`` is
    called after each verdict. Raises ``RunError`` when the run cannot continue.
    """
    cfg = config or RankConfig()
    store = store or MemoryStore()
    item_list = list(items)
    texts = {it.id: it.text for it in item_list}
    if len(texts) != len(item_list):
        raise RunError("Item ids must be unique.")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    budget = asyncio.Condition()  # guards in_flight, so concurrent calls cannot jointly pass the spend cap
    in_flight = 0
    note: str | None = None

    async def one(comp: Comparison) -> None:
        nonlocal in_flight
        async with semaphore:
            async with budget:
                while not affordable(comparisons, cfg, in_flight):
                    if not in_flight:
                        return  # the cap: next_action ends the run and says why
                    await budget.wait()
                in_flight += 1
            try:
                await judge_one(comp)
            finally:
                async with budget:
                    in_flight -= 1
                    budget.notify_all()

    async def judge_one(comp: Comparison) -> None:
        try:
            verdict = await judge.judge_pair(provider, rubric, texts[comp.item_a], texts[comp.item_b],
                                             cfg.max_chars)
        except ProviderError as exc:
            judge.apply_error(comp, exc)
            store.update(run_id, comp)
            if not exc.retryable:
                raise RunError(str(exc)) from exc
            return
        judge.apply_verdict(comp, verdict)
        store.update(run_id, comp)
        if on_progress:
            done = [c for c in comparisons if c.status == "done"]
            on_progress(len(done), len(comparisons))

    while True:
        comparisons = store.load(run_id)
        action = next_action(comparisons, cfg, len(item_list))
        if action.kind == "done":
            note = action.note
            break
        if action.kind == "plan":
            new, note = plan_round(item_list, comparisons, cfg, action.round)
            if not new:
                break
            store.add(run_id, new)
            continue
        by_id = {c.id: c for c in comparisons}
        batch = [by_id[c.id] for c in action.pending]
        await asyncio.gather(*(one(c) for c in batch))
        if batch and all(c.status == "failed" for c in batch):
            raise RunError("Every call in this round failed. Last error: %s" % batch[-1].error)

    comparisons = store.load(run_id)
    top_n = cfg.top_n if cfg.goal == "top_n" else None
    res = results(item_list, comparisons, top_n=top_n, seed=cfg.seed)
    summary = report.summarize(res["leaderboard"], res["diagnostics"], top_n=top_n)
    saved = max(0, planned_calls(len(item_list), cfg) - len(comparisons))
    return Ranking(res["leaderboard"], res["diagnostics"], comparisons, note, spent(comparisons), summary,
                   res["interval"], res["n_bootstrap"], saved, res["groups"])


def _as_items(docs: Mapping[str, str] | Sequence[str] | Sequence[Item]) -> list[Item]:
    if isinstance(docs, Mapping):
        return [Item(id=str(k), text=v, title=str(k)) for k, v in docs.items()]
    out: list[Item] = []
    for i, d in enumerate(docs):
        out.append(d if isinstance(d, Item) else Item(id=str(i), text=d, title=d.strip().split("\n")[0][:80]))
    return out


def rank(
    docs: Mapping[str, str] | Sequence[str] | Sequence[Item],
    rubric: str | Rubric,
    model: str = DEFAULT_MODEL,
    provider: str | Provider = "openrouter",
    concurrency: int = 8,
    store: RunStore | None = None,
    run_id: str = "run",
    **config: Any,
) -> Ranking:
    """Rank documents against a rubric and return the ``Ranking``. Blocks until done.

    ``docs`` is a mapping of id to text, a list of texts (ids are their positions), or a list of ``Item``.
    ``rubric`` says in plain words what the judge should prefer: a string, or a ``Rubric`` (for example
    ``Rubric.load("rubric.md")``). ``provider`` is a name for
    ``make_provider`` (``"openrouter"``, ``"anthropic"``, ``"openai_compat"``, ``"claude_cli"`` or the free
    ``"mock"``) or a ``Provider``. Other keyword arguments are ``RankConfig`` settings, such as
    ``rounds=8`` or ``goal="top_n", top_n=10``. Pass a ``SQLiteStore`` to make the run resumable.

    >>> ranking = rank({"a.md": "...", "b.md": "..."}, "Which is clearer?", provider="mock")

    Inside a running event loop (Jupyter, an async app), await ``run_ranking`` instead.
    """
    cfg = RankConfig.from_dict(config)
    prov = provider if not isinstance(provider, str) else make_provider({"provider": provider, "model": model})

    async def go() -> Ranking:
        try:
            return await run_ranking(_as_items(docs), rubric, prov, cfg, store, run_id, concurrency)
        finally:
            if isinstance(provider, str):
                await prov.close()

    return asyncio.run(go())
