import os
import signal
import subprocess
import sys
import time

import pytest

from supposing import (
    Item,
    RankConfig,
    SQLiteStore,
    next_action,
    plan_round,
    planned_calls,
    rank,
    run_ranking,
)
from supposing.providers import MockProvider


def corpus(n: int = 12) -> list[Item]:
    # Quality rises with the index, so a good ranking puts the last item first.
    return [Item(id="doc%02d" % i, text="Document %d. [[q=%.3f]]" % (i, i / n), title="Doc %d" % i) for i in range(n)]


async def test_ranking_recovers_latent_order():
    items = corpus(16)
    ranking = await run_ranking(items, "Which is better?", MockProvider(sharpness=12), RankConfig(rounds=10))
    top = [e["item_id"] for e in ranking.top(3)]
    assert top[0] in {"doc15", "doc14"}
    assert set(top) <= {"doc15", "doc14", "doc13", "doc12"}
    assert len([c for c in ranking.comparisons if c.status == "done"]) == planned_calls(16, RankConfig(rounds=10))
    assert ranking.diagnostics["order_consistency"] is None  # one order: no order check
    assert "order" not in {c["key"] for c in ranking.diagnostics["reliability"]["checks"]}


def test_rank_sync_entry_point():
    ranking = rank({"a": "[[q=0.9]] good", "b": "[[q=0.1]] bad", "c": "[[q=0.5]] fine"}, "Better?",
                   provider="mock", rounds=4)
    assert len(ranking.leaderboard) == 3


def test_plan_round_is_deterministic():
    items = corpus(10)
    cfg = RankConfig(rounds=3, seed=7)
    a, _ = plan_round(items, [], cfg, 1)
    b, _ = plan_round(items, [], cfg, 1)
    assert [(c.id, c.item_a, c.item_b) for c in a] == [(c.id, c.item_a, c.item_b) for c in b]
    assert len(a) == 5  # 5 pairs, one order each
    both, _ = plan_round(items, [], RankConfig(rounds=3, seed=7, both_orders=True), 1)
    assert len(both) == 10


def test_next_action_stops_at_round_limit():
    cfg = RankConfig(rounds=1)
    items = corpus(4)
    comps, _ = plan_round(items, [], cfg, 1)
    assert next_action(comps, cfg, 4).kind == "judge"
    for c in comps:
        c.status, c.winner_item = "done", c.item_a
    assert next_action(comps, cfg, 4).kind == "done"


async def test_top_n_narrows():
    items = corpus(20)
    cfg = RankConfig(rounds=8, goal="top_n", top_n=3)
    ranking = await run_ranking(items, "Better?", MockProvider(sharpness=15), cfg)
    assert len(ranking.comparisons) < planned_calls(20, cfg)


async def test_flaky_provider_retries_and_finishes():
    items = corpus(8)
    ranking = await run_ranking(items, "Better?", MockProvider(fail_rate=0.3), RankConfig(rounds=3), concurrency=4)
    done = [c for c in ranking.comparisons if c.status == "done"]
    assert len(done) >= 0.8 * len(ranking.comparisons)


async def test_memory_and_resume_give_identical_results(tmp_path):
    items = corpus(10)
    cfg = RankConfig(rounds=5)
    full = await run_ranking(items, "Better?", MockProvider(), cfg)
    store = SQLiteStore(tmp_path / "s.db")
    # Run the first round only, then continue in a new "process" from the same file.
    await run_ranking(items, "Better?", MockProvider(), RankConfig(rounds=1), store, "r1")
    store.close()
    resumed = await run_ranking(items, "Better?", MockProvider(), cfg, SQLiteStore(tmp_path / "s.db"), "r1")
    assert [e["item_id"] for e in resumed.leaderboard] == [e["item_id"] for e in full.leaderboard]


@pytest.mark.slow
def test_kill_and_resume(tmp_path):
    """Kill the CLI mid-run with SIGKILL, restart it, and check nothing is lost or judged twice."""
    folder = tmp_path / "docs"
    folder.mkdir()
    for item in corpus(30):
        (folder / ("%s.md" % item.id)).write_text(item.text)
    state = tmp_path / "state.db"
    cmd = [sys.executable, "-m", "supposing.cli", "rank", str(folder), "--rubric", "Better?", "--judge",
           "mock", "--rounds", "10", "--concurrency", "4", "--state", str(state), "--json"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    # Uninterrupted reference run.
    ref = subprocess.run([*cmd[:-3], "--state", str(tmp_path / "ref.db"), "--json"], capture_output=True,
                         text=True, env=env, check=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    run_id = str(folder.resolve())
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(0.05)
        if state.exists():
            done = [c for c in SQLiteStore(state).load(run_id) if c.status == "done"]
            if len(done) >= 60:
                break
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    partial = [c for c in SQLiteStore(state).load(run_id) if c.status == "done"]
    total = planned_calls(30, RankConfig(rounds=10))
    assert 0 < len(partial) < total, "the kill should land mid-run"

    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    final = SQLiteStore(state).load(run_id)
    assert len(final) == total
    assert all(c.status == "done" for c in final)
    assert all(c.attempts == 1 for c in final), "a decided comparison was judged twice"
    assert out.stdout == ref.stdout, "a resumed run must give the same result as an uninterrupted one"
