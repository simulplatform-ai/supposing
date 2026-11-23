"""Planted-truth checks: simulate a judge whose preferences we know, and test that the engine recovers them.

The simulated judge follows the Bradley-Terry model the engine fits: item i beats item j with probability
1 / (1 + exp(theta_j - theta_i - bias)) when i is shown first. On the engine's scale a gap of one unit of
theta is 400 / ln(10), about 174 rating points.

Tests marked ``stats`` run many simulated tournaments each and take most of the time; ``make check`` skips
them and ``make check-stats`` (run by the release workflow) runs them.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from supposing import (
    Comparison,
    ConfigError,
    Item,
    MemoryStore,
    RankConfig,
    SQLiteStore,
    fewest_calls,
    next_action,
    plan_round,
    planned_calls,
    rating,
    results,
)
from supposing.ranking import WARMUP_ROUNDS, cut_status
from supposing.scheduler import make_round

POINTS_PER_UNIT = 400 / math.log(10)


def judge_run(theta, cfg: RankConfig, seed: int, bias: float = 0.0, store=None, run_id: str = "run",
              stop_after: int | None = None) -> tuple[list[Item], list[Comparison]]:
    """Drive the state machine to the end with a simulated judge. Every verdict is drawn from its own
    seeded generator, so a run cut short and resumed draws exactly the same verdicts."""
    items = [Item(id="i%02d" % i, text="item %d" % i) for i in range(len(theta))]
    index = {it.id: i for i, it in enumerate(items)}
    store = store or MemoryStore()
    judged = 0
    while True:
        comps = store.load(run_id)
        action = next_action(comps, cfg, len(items))
        if action.kind == "done":
            return items, comps
        if action.kind == "plan":
            new, _ = plan_round(items, comps, cfg, action.round)
            if not new:
                return items, comps
            store.add(run_id, new)
            continue
        for c in action.pending:
            if stop_after is not None and judged >= stop_after:
                return items, store.load(run_id)
            d = theta[index[c.item_a]] - theta[index[c.item_b]] + bias
            u = random.Random("%d|%s" % (seed, c.id)).random()
            c.status, c.attempts = "done", c.attempts + 1
            c.winner_item = c.item_a if u < 1 / (1 + math.exp(-d)) else c.item_b
            store.update(run_id, c)
            judged += 1


def true_ratings(theta) -> np.ndarray:
    r = POINTS_PER_UNIT * np.asarray(theta, dtype=float)
    return 1500 + r - r.mean()


def point_ratings(items, comps) -> np.ndarray:
    """Fitted ratings in item order, without the bootstrap."""
    log = rating.MatchLog([it.id for it in items], [c.as_rating_row() for c in comps if c.winner_item])
    return rating.to_elo(log.fit())


def board(items, comps, seed=0) -> dict[str, dict]:
    return {e["item_id"]: e for e in results(items, comps, seed=seed)["leaderboard"]}


@pytest.fixture
def fewer_resamples(monkeypatch):
    """100 bootstrap resamples instead of up to 300: coverage estimates stay honest, the tests stay quick."""
    monkeypatch.setattr(rating, "n_bootstrap_for", lambda n_items, n_matches: 100)


# Bradley-Terry recovers a planted order -------------------------------------------------------------


def test_fit_recovers_planted_strengths_from_a_large_sample():
    """With many verdicts per pair the fitted ratings match the planted ones, not just their order."""
    rng = np.random.default_rng(0)
    theta = np.linspace(-1.5, 1.5, 8)
    rows = []
    k = 0
    for i in range(8):
        for j in range(i + 1, 8):
            p = 1 / (1 + math.exp(theta[j] - theta[i]))
            for _ in range(400):
                k += 1
                winner = "i%d" % i if rng.random() < p else "i%d" % j
                rows.append({"id": str(k), "round": 1, "pair_id": k, "item_a": "i%d" % i, "item_b": "i%d" % j,
                             "winner_item": winner})
    log = rating.MatchLog(["i%d" % i for i in range(8)], rows)
    fitted = rating.to_elo(log.fit())
    assert np.max(np.abs(fitted - true_ratings(theta))) < 25  # points; the planted range is 520


@pytest.mark.parametrize("pairing", ["swiss", "random"])
def test_tournament_recovers_planted_order(pairing):
    theta = np.linspace(0, 4, 16)
    items, comps = judge_run(theta, RankConfig(rounds=10, pairing=pairing), seed=1)
    b = board(items, comps)
    fitted = [b[it.id]["rating"] for it in items]
    assert rating.spearman(fitted, theta) > 0.9
    assert {e for e, row in b.items() if row["rank"] <= 3} <= {"i15", "i14", "i13", "i12"}


# Intervals cover at about the stated rate ---------------------------------------------------------


def coverage(n: int, theta_of, rounds: int, reps: int, pairing: str = "swiss") -> tuple[float, float]:
    """Share of items whose 90% rating interval, and whose rank range, contains the planted value."""
    hits = rank_hits = total = 0
    for rep in range(reps):
        theta = theta_of(rep)
        truth = true_ratings(theta)
        true_rank = rating.ranks_of(truth)
        items, comps = judge_run(theta, RankConfig(rounds=rounds, seed=rep, pairing=pairing), seed=rep)
        b = board(items, comps, seed=rep)
        for i, it in enumerate(items):
            e = b[it.id]
            hits += e["rating_lo"] <= truth[i] <= e["rating_hi"]
            rank_hits += e["rank_lo"] <= true_rank[i] <= e["rank_hi"]
            total += 1
    return hits / total, rank_hits / total


@pytest.mark.stats
@pytest.mark.parametrize(("n", "spread", "rounds"), [(12, 4.0, 8), (20, 4.0, 3)])
def test_intervals_cover_planted_ratings_at_about_90_percent(fewer_resamples, n, spread, rounds):
    """Evenly spread items: the 90% intervals should contain the planted rating about 90% of the time."""
    rate, rank_rate = coverage(n, lambda rep: np.linspace(0, spread, n), rounds, reps=15)
    assert 0.80 <= rate <= 0.97, "rating coverage %.3f" % rate
    assert rank_rate >= 0.85, "rank coverage %.3f" % rank_rate


@pytest.mark.stats
def test_intervals_cover_when_items_are_close(fewer_resamples):
    """40 items of similar quality (planted ratings with a spread of 174 points). With strict neighbour pairing
    the fitted scale came out about 1.4 times too wide here and rating coverage fell to 0.69."""
    rate, rank_rate = coverage(40, lambda rep: np.random.default_rng(rep).normal(0, 1, 40), 8, reps=6)
    assert 0.78 <= rate <= 0.97, "rating coverage %.3f" % rate
    assert rank_rate >= 0.80, "rank coverage %.3f" % rank_rate


@pytest.mark.stats
def test_items_that_win_almost_everything_are_understated_not_overstated(fewer_resamples):
    """Known limit. With a wide spread the top and bottom items win or lose nearly every comparison, the prior
    pulls their ratings toward the middle, and their rating intervals often miss the planted value. The error
    is in the safe direction: gaps are understated, and rank ranges still cover."""
    n = 12
    theta = np.linspace(0, 10, n)
    truth = true_ratings(theta)
    rate, rank_rate = coverage(n, lambda rep: theta, 8, reps=6)
    assert rank_rate >= 0.9
    error = np.mean([point_ratings(*judge_run(theta, RankConfig(rounds=8, seed=rep), seed=rep)) - truth
                     for rep in range(6)], axis=0)
    assert error[-1] < -50 and error[0] > 50  # the best is rated too low, the worst too high


def test_equal_items_get_overlapping_ranks():
    """When nothing separates the items, no item's rank range should claim it is clearly first or last."""
    items, comps = judge_run(np.zeros(10), RankConfig(rounds=8), seed=3)
    for e in board(items, comps).values():
        assert e["rank_hi"] - e["rank_lo"] >= 3


# Both orders cancel a planted position bias -------------------------------------------------------


def test_both_orders_show_every_item_first_half_the_time():
    items, comps = judge_run(np.linspace(0, 3, 11), RankConfig(rounds=6), seed=4)
    for it in items:
        first = sum(c.item_a == it.id for c in comps)
        second = sum(c.item_b == it.id for c in comps)
        assert first == second


def test_both_orders_cancel_a_strong_position_bias():
    """A judge that favours the first text by 1.5 units (82% first-position wins between equals) still gives
    the planted order, and the diagnostics report the bias."""
    theta = np.linspace(0, 4, 16)
    seeds = range(6)
    rho_clean = np.mean([rating.spearman(point_ratings(*judge_run(theta, RankConfig(rounds=10, seed=s), seed=s)),
                                         theta) for s in seeds])
    biased_runs = [judge_run(theta, RankConfig(rounds=10, seed=s), seed=s, bias=1.5) for s in seeds]
    rho_biased = np.mean([rating.spearman(point_ratings(*run), theta) for run in biased_runs])
    assert rho_biased > 0.85
    assert rho_biased > rho_clean - 0.08
    diagnostics = results(*biased_runs[0])["diagnostics"]
    assert diagnostics["first_position_win_rate"] > 0.65
    assert diagnostics["order_consistency"] < 0.6


# Swiss pairing never repeats a pair before it must ------------------------------------------------


def fresh_round_exists(ids: list[str], played: set, games: dict[str, int]) -> bool:
    """Brute force: can the items be paired with no pair that has met already? With an odd count, one of the
    items that have played most sits out, so game counts stay level."""
    def pairable(rest: list[str]) -> bool:
        if len(rest) < 2:
            return True
        a = rest[0]
        return any(frozenset((a, b)) not in played and pairable([x for x in rest[1:] if x != b]) for b in rest[1:])
    if len(ids) % 2 == 0:
        return pairable(ids)
    most = max(games[i] for i in ids)
    return any(pairable([x for x in ids if x != bye]) for bye in ids if games[bye] == most)


@pytest.mark.parametrize("pairing", ["swiss", "random"])
@pytest.mark.parametrize("n", [4, 6, 7, 8, 9, 10, 12])
def test_no_rematch_while_a_fresh_round_exists(pairing, n):
    ids = ["i%02d" % i for i in range(n)]
    for seed in range(15):
        rng = random.Random(seed)
        played: set = set()
        games = dict.fromkeys(ids, 0)
        ratings = {i: rng.gauss(1500, 150) for i in ids}
        for _ in range(n + 2):
            pairs = make_round(ids, ratings, played, games, rng, pairing)
            assert len(pairs) == n // 2
            assert len({x for p in pairs for x in p}) == 2 * len(pairs)
            if fresh_round_exists(ids, played, games):
                assert all(frozenset(p) not in played for p in pairs), (seed, pairs)
            for a, b in pairs:
                played.add(frozenset((a, b)))
                games[a] += 1
                games[b] += 1
                ratings[a if rng.random() < 0.5 else b] += 30


@pytest.mark.parametrize("n", [5, 9, 11])
def test_byes_rotate(n):
    """With an odd count, game counts never drift more than one apart."""
    ids = ["i%02d" % i for i in range(n)]
    rng = random.Random(0)
    played: set = set()
    games = dict.fromkeys(ids, 0)
    for _ in range(2 * n):
        for a, b in make_round(ids, {i: rng.gauss(1500, 150) for i in ids}, played, games, rng):
            played.add(frozenset((a, b)))
            games[a] += 1
            games[b] += 1
        assert max(games.values()) - min(games.values()) <= 1


def test_tournament_has_no_avoidable_rematches():
    """End to end through plan_round: 12 items for 11 rounds is a full round robin when the pairing allows."""
    theta = np.linspace(0, 3, 12)
    for seed in range(5):
        items, comps = judge_run(theta, RankConfig(rounds=11, seed=seed), seed=seed)
        played: set = set()
        games = {it.id: 0 for it in items}
        for r in range(1, 12):
            ids = [it.id for it in items]
            pairs = {frozenset((c.item_a, c.item_b)) for c in comps if c.round == r}
            if fresh_round_exists(ids, played, games):
                assert not (pairs & played), "round %d replayed a pair" % r
            played |= pairs
            for p in pairs:
                for x in p:
                    games[x] += 1


# Resuming gives the same result as an uninterrupted run -------------------------------------------


@pytest.mark.parametrize("stop_after", [1, 41, 79])
def test_resume_from_store_matches_an_uninterrupted_run(tmp_path, fewer_resamples, stop_after):
    """Stop mid-round (or on a round boundary), reopen the file as a new process would, and finish."""
    theta = np.linspace(0, 3, 10)
    cfg = RankConfig(rounds=8, seed=5)
    ref_items, ref = judge_run(theta, cfg, seed=5)

    path = tmp_path / "run.db"
    store = SQLiteStore(path)
    _, partial = judge_run(theta, cfg, seed=5, store=store, stop_after=stop_after)
    store.close()
    assert 0 < sum(c.status == "done" for c in partial) < len(ref)

    items, resumed = judge_run(theta, cfg, seed=5, store=SQLiteStore(path))
    key = [(c.id, c.item_a, c.item_b, c.winner_item) for c in ref]
    assert sorted(key) == sorted((c.id, c.item_a, c.item_b, c.winner_item) for c in resumed)
    assert results(ref_items, ref) == results(items, resumed)


# Top-N runs spend their calls near the cut and stop when it is settled -------------------------------


@pytest.mark.stats
def test_top_n_stops_when_the_cut_is_settled(fewer_resamples):
    theta = np.r_[np.linspace(0, 3, 17), [6.0, 6.5, 7.0]]  # a clear gap below the top 3
    cfg = RankConfig(rounds=15, goal="top_n", top_n=3)
    items, comps = judge_run(theta, cfg, seed=0)
    assert max(c.round for c in comps) < cfg.rounds
    assert len(comps) < planned_calls(20, cfg)
    assert len(comps) >= fewest_calls(20, cfg)
    top = {e["item_id"] for e in results(items, comps, top_n=3)["leaderboard"][:3]}
    assert top == {"i19", "i18", "i17"}
    assert plan_round(items, comps, cfg, max(c.round for c in comps) + 1)[1].startswith("Stopped early")


@pytest.mark.stats
def test_top_n_judges_only_open_items_and_anchors(fewer_resamples):
    theta = np.random.default_rng(1).normal(0, 1.5, 30)
    cfg = RankConfig(rounds=8, goal="top_n", top_n=5)
    items, comps = judge_run(theta, cfg, seed=1)
    for r in range(WARMUP_ROUNDS + 1, max(c.round for c in comps) + 1):
        before = [c for c in comps if c.round < r]
        status, ratings = cut_status(items, before, cfg)
        ins = [i for i, s in status.items() if s == "in"]
        outs = [i for i, s in status.items() if s == "out"]
        allowed = {i for i, s in status.items() if s == "open"}
        allowed |= {min(ins, key=ratings.get)} if ins else set()
        allowed |= {max(outs, key=ratings.get)} if outs else set()
        played = {x for c in comps if c.round == r for x in (c.item_a, c.item_b)}
        assert played <= allowed
        assert len(played) < len(items)


@pytest.mark.stats
async def test_ranking_reports_calls_saved(fewer_resamples):
    from supposing import run_ranking
    from supposing.providers import MockProvider

    docs = [Item(id="d%02d" % i, text="Doc %d [[q=%.3f]]" % (i, i / 20)) for i in range(20)]
    cfg = RankConfig(rounds=8, goal="top_n", top_n=3)
    ranking = await run_ranking(docs, "Better?", MockProvider(sharpness=12), cfg)
    assert ranking.calls_saved == planned_calls(20, cfg) - len(ranking.comparisons) > 0


# Round robin: every pair in one round ------------------------------------------------------------


def test_round_robin_plans_every_pair_at_once():
    items = [Item(id="o%d" % i, text="option %d" % i) for i in range(6)]
    cfg = RankConfig(rounds=1, pairing="round_robin")
    comps, _ = plan_round(items, [], cfg, 1)
    assert len(comps) == planned_calls(6, cfg) == fewest_calls(6, cfg) == 30
    assert {frozenset((c.item_a, c.item_b)) for c in comps} == {
        frozenset((a.id, b.id)) for i, a in enumerate(items) for b in items[i + 1:]}
    for it in items:  # both orders: each option is shown first exactly as often as second
        assert sum(c.item_a == it.id for c in comps) == sum(c.item_b == it.id for c in comps) == 5


def test_round_robin_recovers_planted_order():
    theta = np.linspace(0, 3, 8)
    items, comps = judge_run(theta, RankConfig(rounds=2, pairing="round_robin"), seed=2)
    assert max(c.round for c in comps) == 2 and len(comps) == 2 * 56
    assert rating.spearman(point_ratings(items, comps), theta) > 0.85


def test_round_robin_cannot_narrow():
    with pytest.raises(ConfigError):
        RankConfig(pairing="round_robin", goal="top_n")
