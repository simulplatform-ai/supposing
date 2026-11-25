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


def test_a_saved_run_is_refused_under_a_different_rubric_or_judge(tmp_path):
    """sp-4lk: verdicts are never reused under a different question or judge; resuming the same run still works."""
    from supposing import RunError, SQLiteStore

    items = corpus(6)
    path = tmp_path / "run.db"
    first = rank(items, "Which is clearer?", provider="mock", rounds=2, store=SQLiteStore(path))
    again = rank(items, "Which is clearer?", provider="mock", rounds=3, store=SQLiteStore(path))
    assert again.fingerprint == first.fingerprint
    assert {c.id for c in first.comparisons} <= {c.id for c in again.comparisons}
    for change in (dict(rubric="Which is shortest?"), dict(provider=MockProvider(model="another-model")),
                   dict(both_orders=True), dict(max_chars=100)):
        kw = {"rubric": "Which is clearer?", "provider": "mock", "rounds": 2, **change}
        with pytest.raises(RunError) as err:
            rank(items, kw.pop("rubric"), store=SQLiteStore(path), **kw)
        assert err.value.code == "state_mismatch" and "start fresh" in (err.value.hint or "")
    other = rank(items, "Which is shortest?", provider="mock", rounds=2, store=SQLiteStore(path), run_id="new")
    assert other.fingerprint != first.fingerprint and len(other.comparisons) == len(first.comparisons)


def test_cli_refuses_a_state_file_made_with_another_rubric(tmp_path, capsys):
    import json

    from supposing.cli import main

    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(4):
        (docs / ("d%d.md" % i)).write_text("Doc %d [[q=%.2f]]" % (i, i / 4))
    base = ["rank", str(docs), "--judge", "mock", "--rounds", "1", "--state", str(tmp_path / "s.db"), "--json"]
    assert main(base + ["--rubric", "Which is clearer?"]) == 0
    capsys.readouterr()
    code = main(base + ["--rubric", "Which is shortest?"])
    err = json.loads(capsys.readouterr().out)["error"]
    assert code != 0 and err["code"] == "state_mismatch" and "start fresh" in err["hint"]


@pytest.mark.parametrize("n", [2, 3, 5, 8, 9, 13])
@pytest.mark.parametrize("rounds", [1, 3, 8])
def test_a_run_without_added_items_makes_exactly_its_planned_rounds(n, rounds):
    """Catch-up rounds are for added items only: odd counts and byes never trigger them."""
    from supposing.ranking import Comparison as C
    from supposing.ranking import behind

    items = corpus(n)
    cfg = RankConfig(rounds=rounds, seed=n)
    comps: list[C] = []
    while True:
        action = next_action(comps, cfg, n, [it.id for it in items])
        if action.kind == "done":
            break
        if action.kind == "plan":
            new, _ = plan_round(items, comps, cfg, action.round)
            if not new:
                break
            comps += new
            continue
        for c in action.pending:
            c.status, c.winner_item = "done", c.item_a
    assert max(c.round for c in comps) == rounds
    assert not behind([it.id for it in items], comps, cfg)


def test_an_added_item_is_judged_before_it_is_ranked(tmp_path):
    """sp-ll9: an item added to a finished run gets catch-up comparisons, then a range from its verdicts."""
    from supposing import SQLiteStore

    items = corpus(10)
    path = tmp_path / "run.db"
    first = rank(items, "Which is better?", provider=MockProvider(sharpness=12), rounds=4, store=SQLiteStore(path))
    late = Item(id="late", text="Late document. [[q=0.999]]", title="Late")
    again = rank(items + [late], "Which is better?", provider=MockProvider(sharpness=12), rounds=4,
                 store=SQLiteStore(path))
    games = [c for c in again.comparisons if "late" in (c.item_a, c.item_b) and c.status == "done"]
    assert len(games) >= 3 and len(again.comparisons) > len(first.comparisons)  # 11 items: 4 rounds, 3+ games each
    row = next(r for r in again.leaderboard if r["item_id"] == "late")
    assert not row["few_verdicts"] and row["rank_lo"] <= 3


def test_items_with_too_few_verdicts_get_the_widest_range():
    from supposing import Comparison, results

    items = corpus(6)
    comps = [Comparison(id="c%d" % i, round=1, pair_id=i, item_a=a, item_b=b, status="done", winner_item=a)
             for i, (a, b) in enumerate([("doc00", "doc01"), ("doc00", "doc02"), ("doc00", "doc03"),
                                         ("doc01", "doc02"), ("doc01", "doc03"), ("doc02", "doc03")])]
    res = results(items, comps)
    rows = {r["item_id"]: r for r in res["leaderboard"]}
    assert rows["doc04"]["few_verdicts"] and (rows["doc04"]["rank_lo"], rows["doc04"]["rank_hi"]) == (1, 6)
    assert not rows["doc00"]["few_verdicts"]
    from supposing import summarize

    assert "2 items have too few verdicts to place yet." in summarize(res["leaderboard"], res["diagnostics"])
    two = rank({"a": "[[q=0.9]] a", "b": "[[q=0.1]] b"}, "Better?", provider="mock", rounds=2)
    assert "clearly" not in two.summary


def test_max_calls_stops_a_run_and_says_so():
    from supposing import RankConfig, planned_calls, rank

    cfg = RankConfig(max_calls=5)
    assert planned_calls(20, cfg) == 5 and RankConfig.from_dict(cfg.to_dict()) == cfg
    docs = {"d%d" % i: "text %d [[q=0.%d]]" % (i, i) for i in range(10)}
    r = rank(docs, "Better?", provider="mock", max_calls=5, concurrency=4)
    assert len([c for c in r.comparisons if c.status in ("done", "failed")]) == 5
    assert r.note and "call cap: 5 of 5" in r.note



def test_a_state_file_from_before_fingerprints_is_refused_with_its_own_message(tmp_path):
    from supposing import Comparison, RunError, SQLiteStore

    store = SQLiteStore(tmp_path / "old.db")
    store.add("run", [Comparison(id="r1-p1-a", round=1, pair_id=1, item_a="doc00", item_b="doc01", status="done",
                                 winner_item="doc00")])  # no fingerprint, as 0.3.0 wrote them
    with pytest.raises(RunError) as err:
        rank(corpus(4), "Q?", provider="mock", rounds=1, store=store)
    assert err.value.code == "state_mismatch" and "0.3.0 or earlier" in str(err.value)


def test_results_can_cap_the_bootstrap_for_live_views():
    from supposing import results

    items = corpus(12)
    comps, _ = plan_round(items, [], RankConfig(rounds=1), 1)
    for c in comps:
        c.status, c.winner_item = "done", c.item_a
    assert results(items, comps, max_bootstrap=40)["n_bootstrap"] == 40
    assert results(items, comps)["n_bootstrap"] > 40
