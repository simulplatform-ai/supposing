"""Running a ranking to completion in one process."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import judge, report
from .judge import Example, JudgePrompt, Verdict
from .providers import Provider, ProviderError, make_provider
from .ranking import Comparison, Item, RankConfig, affordable, next_action, plan_round, planned_calls, results, spent
from .rubric import Rubric
from .store import MemoryStore, RunStore

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


class RunError(RuntimeError):
    """A run cannot continue: the provider refused a call for good (bad key, no credit, unknown model), every
    call in a round failed, the items are invalid, or the store holds verdicts made under a different rubric,
    prompt or judge. With a ``SQLiteStore``, fix the cause and run again to continue where it stopped.

    ``code`` is a stable name for the cause (``"state_mismatch"``, ``"run_failed"``) and ``hint`` says what to
    do, when there is something to say."""

    def __init__(self, message: str, code: str = "run_failed", hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass
class Ranking:
    """A finished run.

    ``leaderboard`` and ``diagnostics`` are as described in ``supposing.results``. ``fingerprint`` identifies
    what the verdicts mean (``supposing.judge.run_fingerprint``): two runs with the same fingerprint asked the
    same judge the same question. ``comparisons`` holds every
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
    fingerprint: str = ""

    def top(self, n: int) -> list[dict[str, Any]]:
        """The best ``n`` leaderboard rows."""
        return self.leaderboard[:n]


Progress = Callable[[int, int], None]  # (judged so far, comparisons planned so far)
# A judge you write: given the two items in the order the library chose (a is shown first) and the rubric,
# return "A" or "B", a Verdict, or a dict with "winner" (and optionally "reasoning" and more fields).
JudgeFunction = Callable[[Item, Item, "str | Rubric"], "str | Verdict | dict[str, Any] | Awaitable[Any]"]


def _is_provider(judge_: Any) -> bool:
    return hasattr(judge_, "complete")


async def _call_function(fn: JudgeFunction, a: Item, b: Item, rubric: str | Rubric) -> Verdict:
    if inspect.iscoroutinefunction(fn):
        result = await fn(a, b, rubric)
    else:
        result = await asyncio.to_thread(fn, a, b, rubric)
        if inspect.isawaitable(result):
            result = await result
    return judge.verdict_from(result)


async def run_ranking(
    items: Sequence[Item],
    rubric: str | Rubric,
    provider: Provider | JudgeFunction,
    config: RankConfig | None = None,
    store: RunStore | None = None,
    run_id: str = "run",
    concurrency: int = 8,
    on_progress: Progress | None = None,
    *,
    context: str | None = None,
    examples: Sequence[Example | Mapping[str, Any]] | None = None,
    prompt: JudgePrompt | str | None = None,
) -> Ranking:
    """Rank ``items`` against ``rubric`` and return the finished ``Ranking``.

    ``provider`` is the judge: a ``Provider`` (a model backend), or a function ``fn(a, b, rubric)`` that gets
    the two ``Item`` objects in the order the library chose (``a`` is shown first) and returns ``"A"``,
    ``"B"``, a ``Verdict`` or a dict with ``winner``. Plain and ``async`` functions both work.

    ``context`` (a brief shared by every comparison), ``examples`` (past judgements, ``Example`` or dicts) and
    each item's own ``context`` are shown to a model judge in their own blocks. ``prompt`` replaces the system
    prompt (a ``JudgePrompt``, or a template string for one). Whatever is customised, the library picks which
    item is shown first, parses the verdict, and runs the judge checks.

    If ``store`` already holds comparisons for ``run_id``, the run continues from them: a decided comparison
    is never judged again. They must have been made with the same rubric, prompt, judge, order mode and text
    cut (the run's fingerprint); otherwise ``RunError`` with ``code="state_mismatch"`` is raised before any
    call, and nothing is reused. ``concurrency`` caps judge calls in flight. ``on_progress(judged, planned)`` is
    called after each verdict. Raises ``RunError`` when the run cannot continue.
    """
    cfg = config or RankConfig()
    store = store or MemoryStore()
    item_list = list(items)
    texts = {it.id: it.text for it in item_list}
    if len(texts) != len(item_list):
        raise RunError("Item ids must be unique.")
    if isinstance(prompt, str):
        prompt = JudgePrompt(system=prompt)
    shown = judge.as_examples(examples)
    by_item = {it.id: it for it in item_list}
    fingerprint = judge.run_fingerprint(rubric, provider, cfg.both_orders, cfg.max_chars,
                                        system=prompt.render() if prompt else judge.SYSTEM_PROMPT,
                                        context=context, examples=shown)
    stale = [c for c in store.load(run_id) if c.fingerprint != fingerprint]
    if stale and all(c.fingerprint is None for c in stale):
        raise RunError(
            "This saved run was made by supposing 0.3.0 or earlier, before runs recorded what their verdicts mean, "
            "so it cannot be resumed (%d saved verdicts)." % len(stale), code="state_mismatch",
            hint="Use a new state file or run id to start fresh; the old file is unchanged.")
    if stale:
        raise RunError(
            "This run's saved verdicts were made with a different rubric, prompt, judge, order mode or text cut "
            "(%d of them), so they cannot be reused." % len(stale), code="state_mismatch",
            hint="Use a new state file or run id to start fresh, or go back to the settings the run was made with.")
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
        a, b = by_item[comp.item_a], by_item[comp.item_b]
        try:
            if _is_provider(provider):
                verdict = await judge.judge_pair(provider, rubric, a.text, b.text, cfg.max_chars,  # type: ignore[arg-type]
                                                 prompt=prompt, context=context, examples=shown,
                                                 context_a=a.context, context_b=b.context)
            else:
                verdict = await _call_function(provider, a, b, rubric)  # type: ignore[arg-type]
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
        action = next_action(comparisons, cfg, len(item_list), [it.id for it in item_list])
        if action.kind == "done":
            note = action.note
            break
        if action.kind == "plan":
            new, note = plan_round(item_list, comparisons, cfg, action.round)
            if not new:
                break
            for c in new:
                c.fingerprint = fingerprint
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
                   res["interval"], res["n_bootstrap"], saved, res["groups"], fingerprint)


def _as_items(docs: Mapping[str, str] | Sequence[str | Item | Mapping[str, Any]]) -> list[Item]:
    if isinstance(docs, Mapping):
        return [Item(id=str(k), text=v, title=str(k)) for k, v in docs.items()]
    out: list[Item] = []
    for i, d in enumerate(docs):
        if isinstance(d, Item):
            out.append(d)
        elif isinstance(d, str):
            out.append(Item(id=str(i), text=d, title=d.strip().split("\n")[0][:80]))
        elif isinstance(d, Mapping) and isinstance(d.get("text"), str):
            out.append(Item(id=str(d.get("id", i)), text=d["text"], title=d.get("title"), context=d.get("context")))
        else:
            raise RunError("Item %d is a %s; give a text, an Item, or a dict with \"text\" (and optionally \"id\", "
                           "\"title\" and \"context\")." % (i, type(d).__name__), code="invalid_items")
    empty = [it.id for it in out if not it.text.strip()]
    if empty:
        raise RunError("Item %s has no text." % empty[0], code="invalid_items")
    return out


def rank(
    docs: Mapping[str, str] | Sequence[str | Item | Mapping[str, Any]],
    rubric: str | Rubric,
    model: str | None = None,
    provider: str | Provider = "openrouter",
    concurrency: int = 8,
    store: RunStore | None = None,
    run_id: str = "run",
    *,
    judge: str | Provider | JudgeFunction | None = None,
    context: str | None = None,
    examples: Sequence[Example | Mapping[str, Any]] | None = None,
    prompt: JudgePrompt | str | None = None,
    **config: Any,
) -> Ranking:
    """Rank documents against a rubric and return the ``Ranking``. Blocks until done.

    ``docs`` is a mapping of id to text, or a list of texts (ids are their positions), ``Item`` objects or
    dicts with ``text`` (and optionally ``id``, ``title`` and ``context``). ``rubric`` says in plain words what
    the judge should prefer: a string, or a ``Rubric`` (for example ``Rubric.load("rubric.md")``).

    The judge is ``provider`` (a name for ``make_provider``: ``"openrouter"``, ``"anthropic"``,
    ``"openai_compat"``, ``"claude_cli"`` or the free ``"mock"``; or a ``Provider``) with ``model`` (default:
    ``DEFAULT_MODEL`` on OpenRouter, the provider's own default otherwise).
    ``judge=`` takes precedence: a level (``"quick"``, ``"balanced"``, ``"thorough"``), an OpenRouter model id
    such as ``"anthropic/claude-haiku-5.5"``, a provider name (``"mock"``, ``"claude_cli"``, ...; ``model=`` then
    picks its model), a ``Provider``, or your own function ``fn(a, b, rubric)``, plain or ``async``,
    that returns ``"A"``, ``"B"``, a ``Verdict`` or a dict with ``winner``.

    Customising what a model judge is told, one step at a time: ``context`` (a brief shared by every
    comparison), ``examples`` (past judgements: ``Example`` objects or dicts with ``a``, ``b``, ``winner`` and
    optionally ``why``), each ``Item``'s own ``context``, and ``prompt`` (a ``JudgePrompt`` that replaces the
    system prompt). Other keyword arguments are ``RankConfig`` settings, such as ``rounds=8`` or
    ``goal="top_n", top_n=10``. Pass a ``SQLiteStore`` to make the run resumable.

    >>> ranking = rank({"a.md": "...", "b.md": "..."}, "Which is clearer?", provider="mock")

    Inside a running event loop (Jupyter, an async app), await ``run_ranking`` instead.
    """
    cfg = RankConfig.from_dict(config)
    item_list = _as_items(docs)
    if len(item_list) < 2:
        raise RunError("A ranking needs at least two items; got %d." % len(item_list), code="too_few_items",
                       hint="Pass two or more documents.")
    chosen: Any = judge if judge is not None else provider
    owned = isinstance(chosen, str)
    if owned and judge is not None:  # the same judge strings as the CLI: levels, model ids, provider names
        from .cloud.judges import JudgeError, parse_judge

        try:
            spec = parse_judge(chosen, model, local=True)
        except JudgeError as exc:
            raise RunError(str(exc), code="invalid_judge") from exc
        chosen, model = spec["provider"], spec["model"]
    if owned and chosen == "openrouter" and not model:
        model = DEFAULT_MODEL
    prov: Any = make_provider({"provider": chosen, "model": model}) if owned else chosen

    async def go() -> Ranking:
        try:
            return await run_ranking(item_list, rubric, prov, cfg, store, run_id, concurrency,
                                     context=context, examples=examples, prompt=prompt)
        finally:
            if owned:
                await prov.close()

    return asyncio.run(go())
