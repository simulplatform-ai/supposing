"""Drive a run step by step, the way a job queue or a fleet of workers would.

    python examples/state_machine.py

``next_action`` looks only at the comparisons recorded so far and says what to do: judge the pending ones,
plan the next round, or stop. ``plan_round`` is deterministic, so any process holding the same rows makes
the same plan. Here one loop plays every role; in production each step could run on a different machine,
with the rows in a database.
"""
import asyncio

from supposing import (
    Item,
    MemoryStore,
    RankConfig,
    apply_error,
    apply_verdict,
    judge_pair,
    next_action,
    plan_round,
    results,
)
from supposing.providers import MockProvider, ProviderError

items = [Item(id="essay-%d" % i, text="Essay %d. [[q=%.2f]]" % (i, (i * 37 % 11) / 10)) for i in range(8)]
rubric = "Which essay argues its point more clearly?"
cfg = RankConfig(rounds=5)
store = MemoryStore()
judge = MockProvider()


async def main() -> None:
    while True:
        comparisons = store.load("run-1")
        action = next_action(comparisons, cfg, len(items))
        if action.kind == "done":
            break
        if action.kind == "plan":
            new, note = plan_round(items, comparisons, cfg, action.round)
            if not new:
                print(note)
                break
            store.add("run-1", new)
            print("planned round %d: %d comparisons" % (action.round, len(new)))
            continue
        texts = {it.id: it.text for it in items}
        for comp in action.pending:  # a worker would claim one of these at a time
            try:
                verdict = await judge_pair(judge, rubric, texts[comp.item_a], texts[comp.item_b])
                apply_verdict(comp, verdict)
            except ProviderError as exc:
                apply_error(comp, exc)
            store.update("run-1", comp)

    final = results(items, store.load("run-1"))
    for row in final["leaderboard"]:
        print(row["rank"], row["item_id"], row["rating"], "ranks %d-%d" % (row["rank_lo"], row["rank_hi"]))


asyncio.run(main())
