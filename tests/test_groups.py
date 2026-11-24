"""Display groups: settled / close calls / out for top-N runs, clear tiers for full rankings."""
import re

import numpy as np
import pytest
from test_statistics import judge_run

from supposing import RankConfig, display_groups, results, summarize


def row(item, rank, lo, hi, p=None):
    out = {"item_id": item, "title": item, "rank": rank, "rank_lo": lo, "rank_hi": hi}
    if p is not None:
        out["p_top_n"] = p
    return out


def test_top_n_groups_follow_the_rule():
    board = [row("a", 1, 1, 1, 1.0), row("b", 2, 1, 3, 0.97), row("c", 3, 2, 5, 0.7), row("d", 4, 3, 6, 0.96),
             row("e", 5, 3, 7, 0.2), row("f", 6, 5, 8, 0.03), row("g", 7, 6, 8, 0.0), row("h", 8, 7, 8, 0.0)]
    groups = display_groups(board, top_n=3)
    assert [(g["key"], g["item_ids"]) for g in groups] == [
        ("settled", ["a", "b"]), ("close", ["c", "d", "e"]), ("out", ["f", "g", "h"])]
    assert [g["label"] for g in groups] == ["2 settled in the top 3", "1 place left · 3 contenders", "3 out"]
    assert [g["close"] for g in groups] == [False, True, False]


def test_settled_cut_has_no_close_group():
    board = [row("a", 1, 1, 2, 1.0), row("b", 2, 1, 2, 0.99), row("c", 3, 3, 5, 0.01), row("d", 4, 3, 5, 0.0)]
    assert [g["label"] for g in display_groups(board, top_n=2)] == ["2 settled in the top 2", "2 out"]


def test_full_ranking_groups_are_clear_tiers():
    board = [row("a", 1, 1, 1), row("b", 2, 2, 3), row("c", 3, 2, 3), row("d", 4, 4, 5), row("e", 5, 4, 5)]
    groups = display_groups(board)
    assert [(g["label"], g["item_ids"]) for g in groups] == [
        ("Rank 1", ["a"]), ("Ranks 2-3", ["b", "c"]), ("Ranks 4-5", ["d", "e"])]
    assert [g["tier"] for g in groups] == [1, 3, 4]  # colour steps spread from 1 to 4
    assert display_groups([row(x, i + 1, 1, 3) for i, x in enumerate("abc")])[0]["label"] == "Ranks 1-3"


def check_run(n: int, top_n: int, sd: float, seed: int) -> None:
    theta = np.random.default_rng(seed).normal(0, sd, n)
    items, comps = judge_run(theta, RankConfig(rounds=8, goal="top_n", top_n=top_n, seed=seed), seed=seed)
    res = results(items, comps, top_n=top_n, seed=seed)
    board, groups = res["leaderboard"], res["groups"]
    # The groups partition the leaderboard in rank order.
    assert [i for g in groups for i in g["item_ids"]] == [r["item_id"] for r in board]
    p = {r["item_id"]: r["p_top_n"] for r in board}
    by = {g["key"]: g["item_ids"] for g in groups}
    assert all(p[i] >= 0.95 for i in by.get("settled", []))
    if "close" in by:  # exactly the span from the first to the last open item
        assert 0.05 < p[by["close"][0]] < 0.95 and 0.05 < p[by["close"][-1]] < 0.95
    assert not any(0.05 < p[i] < 0.95 for i in by.get("settled", []) + by.get("out", []))
    # No group swallows nearly everything (the bug: close calls held 38 of 40). Below 20 items most of a
    # small set can honestly be in contention.
    if n >= 20:
        assert len(by.get("close", [])) <= 0.6 * n, (n, top_n, seed, len(by["close"]))
    # The verdict sentence says what the groups say.
    sentence = summarize(board, res["diagnostics"], top_n=top_n, caution=False)
    s, c = len(by.get("settled", [])), len(by.get("close", []))
    if c:
        assert re.search(r"\b%d items? (is|are) contesting|%d items are contesting" % (c, c), sentence), sentence
    if s and c:
        assert re.match(r"%d of the top %d (is|are) settled" % (s, top_n), sentence), sentence


@pytest.mark.parametrize(("n", "top_n"), [(30, 3)])
def test_groups_on_simulated_runs(fewer_resamples, n, top_n):
    check_run(n, top_n, 1.5, 0)


@pytest.mark.stats
@pytest.mark.parametrize(("n", "top_n", "sd"), [(8, 2, 1.0), (20, 3, 1.5), (40, 5, 1.0), (40, 5, 1.5),
                                                (100, 10, 1.5), (200, 20, 1.5), (200, 10, 1.0)])
def test_groups_on_many_simulated_runs(n, top_n, sd):
    for seed in range(5):
        check_run(n, top_n, sd, seed)


def test_live_labels_for_a_run_still_going():
    """While a run is live the groups read '1st so far', 'Close behind', 'Still sorting'; below the second
    tier, everything is one 'Still sorting' group. Keys, tiers and item order stay those of the finished groups."""
    board = [row("a", 1, 1, 1), row("b", 2, 2, 3), row("c", 3, 2, 3), row("d", 4, 4, 4), row("e", 5, 5, 6),
             row("f", 6, 5, 6)]
    final = display_groups(board)
    live = display_groups(board, live=True)
    assert [g["label"] for g in final] == ["Rank 1", "Ranks 2-3", "Rank 4", "Ranks 5-6"]
    assert [g["label"] for g in live] == ["1st so far", "Close behind", "Still sorting"]
    assert [g["item_ids"] for g in live] == [["a"], ["b", "c"], ["d", "e", "f"]]
    assert [g["key"] for g in live] == ["tier-1", "tier-2", "tier-3"]
    tied = display_groups([row("a", 1, 1, 2), row("b", 2, 1, 2), row("c", 3, 3, 3)], live=True)
    assert [g["label"] for g in tied] == ["Top 2 so far", "Close behind"]


def test_live_labels_for_a_top_n_run():
    board = [row("a", 1, 1, 1, 1.0), row("b", 2, 1, 3, 0.97), row("c", 3, 2, 5, 0.7), row("d", 4, 3, 6, 0.96),
             row("e", 5, 3, 7, 0.2), row("f", 6, 5, 8, 0.03)]
    live = display_groups(board, top_n=3, live=True)
    assert [(g["key"], g["label"]) for g in live] == [
        ("settled", "2 in the top 3 so far"), ("close", "Close behind"), ("out", "Still sorting")]
    assert live == [{**g, "label": lg["label"]} for g, lg in zip(display_groups(board, top_n=3), live, strict=True)]


def test_results_passes_live_through():
    items, comps = judge_run(np.linspace(0, 4, 10), RankConfig(rounds=4), seed=1)
    assert results(items, comps, live=True)["groups"][0]["label"].endswith("so far")
    assert not results(items, comps)["groups"][0]["label"].endswith("so far")
