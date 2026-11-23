"""Turning pairwise verdicts into a ranking.

Ratings are a Bradley-Terry fit over the whole match log, shown on the familiar Elo
scale (mean 1500, a 400-point gap means 10:1 odds). Unlike sequential Elo the result
does not depend on the order the matches were played in, which is what you want for a
fixed corpus. Uncertainty comes from bootstrapping over pairs.
"""
from __future__ import annotations

import numpy as np

PRIOR = 0.5  # each item starts with half a win and half a loss against an average opponent
CI = (5, 95)  # 90% intervals


def _fit(n: int, win_idx, lose_idx, weights, start=None, max_iter: int = 2000, tol: float = 1e-7):
    """Bradley-Terry strengths by the MM algorithm, in O(matches) per iteration.

    The prior is a virtual opponent of strength 1 that every item has beaten and lost
    to PRIOR times. It keeps unbeaten items finite and pins the scale.
    """
    wins = np.bincount(win_idx, weights=weights, minlength=n) + PRIOR
    p = np.ones(n) if start is None else start.copy()
    for _ in range(max_iter):
        share = weights / (p[win_idx] + p[lose_idx])
        denom = (
            np.bincount(win_idx, weights=share, minlength=n)
            + np.bincount(lose_idx, weights=share, minlength=n)
            + 2 * PRIOR / (p + 1.0)
        )
        new = wins / denom
        converged = np.max(np.abs(np.log(new) - np.log(p))) < tol
        p = new
        if converged:
            break
    return p


def to_elo(p):
    r = 400 * np.log10(p)
    return 1500 + r - r.mean()


def ranks_of(ratings):
    """1 = best."""
    order = np.argsort(-ratings, kind="stable")
    ranks = np.empty(len(ratings), dtype=int)
    ranks[order] = np.arange(1, len(ratings) + 1)
    return ranks


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


class MatchLog:
    """Decided comparisons as index arrays over a fixed item list."""

    def __init__(self, item_ids: list, comparisons: list):
        self.item_ids = list(item_ids)
        self.index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.comparisons = [c for c in comparisons if c.get("winner_item") in self.index]
        winners, losers, pairs = [], [], []
        for c in self.comparisons:
            loser = c["item_b"] if c["winner_item"] == c["item_a"] else c["item_a"]
            winners.append(self.index[c["winner_item"]])
            losers.append(self.index[loser])
            pairs.append(c["pair_id"])
        self.win_idx = np.array(winners, dtype=int)
        self.lose_idx = np.array(losers, dtype=int)
        # Bootstrap unit: both orderings of a pair move together.
        _, self.group = np.unique(np.array(pairs, dtype=int), return_inverse=True) if pairs else (None, np.array([], dtype=int))
        self.n_groups = int(self.group.max()) + 1 if len(self.group) else 0

    @property
    def n(self) -> int:
        return len(self.item_ids)

    def fit(self, weights=None, start=None):
        if weights is None:
            weights = np.ones(len(self.win_idx))
        return _fit(self.n, self.win_idx, self.lose_idx, weights, start=start)


def n_bootstrap_for(n_items: int, n_matches: int) -> int:
    """Keep the bootstrap around a second or two regardless of corpus size."""
    work = max(1, n_items + n_matches)
    return int(min(300, max(40, 3_000_000 // work)))


def compute_results(items: list, comparisons: list, top_n: int | None = None, seed: int = 0,
                    max_bootstrap: int | None = None) -> dict:
    """Leaderboard with intervals, plus diagnostics about how far the judge can be trusted.

    items: dicts with id, and optionally title and n_words.
    comparisons: decided comparisons as dicts with id, round, pair_id, item_a, item_b, winner_item.
    max_bootstrap: cap on resamples, for planning where speed matters more than the last digit.
    """
    log = MatchLog([it["id"] for it in items], comparisons)
    n, m = log.n, len(log.win_idx)
    rng = np.random.default_rng(seed)

    strengths = log.fit()
    ratings = to_elo(strengths)
    ranks = ranks_of(ratings)

    n_boot = n_bootstrap_for(n, m) if m else 0
    if max_bootstrap is not None:
        n_boot = min(n_boot, max_bootstrap)
    boot_ratings = np.empty((n_boot, n))
    boot_ranks = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        counts = rng.multinomial(log.n_groups, np.full(log.n_groups, 1.0 / log.n_groups))
        boot = to_elo(log.fit(weights=counts[log.group].astype(float), start=strengths))
        boot_ratings[b] = boot
        boot_ranks[b] = ranks_of(boot)

    wins = np.bincount(log.win_idx, minlength=n)
    losses = np.bincount(log.lose_idx, minlength=n)

    leaderboard = []
    for i, item in enumerate(items):
        entry = {
            "item_id": item["id"],
            "title": item.get("title"),
            "n_words": item.get("n_words"),
            "rating": round(float(ratings[i]), 1),
            "rank": int(ranks[i]),
            "wins": int(wins[i]),
            "losses": int(losses[i]),
        }
        if n_boot:
            entry["rating_lo"] = round(float(np.percentile(boot_ratings[:, i], CI[0])), 1)
            entry["rating_hi"] = round(float(np.percentile(boot_ratings[:, i], CI[1])), 1)
            entry["rank_lo"] = int(np.percentile(boot_ranks[:, i], CI[0]))
            entry["rank_hi"] = int(np.ceil(np.percentile(boot_ranks[:, i], CI[1])))
            # The point estimate always sits inside its own interval.
            entry["rating_lo"] = min(entry["rating_lo"], entry["rating"])
            entry["rating_hi"] = max(entry["rating_hi"], entry["rating"])
            entry["rank_lo"] = min(entry["rank_lo"], entry["rank"])
            entry["rank_hi"] = max(entry["rank_hi"], entry["rank"])
            if top_n:
                entry["p_top_n"] = round(float(np.mean(boot_ranks[:, i] <= top_n)), 3)
        else:
            entry.update(rating_lo=entry["rating"], rating_hi=entry["rating"], rank_lo=1, rank_hi=n)
            if top_n:
                entry["p_top_n"] = None
        leaderboard.append(entry)
    leaderboard.sort(key=lambda e: e["rank"])

    return {
        "leaderboard": leaderboard,
        "diagnostics": _diagnostics(log, items, ratings, rng),
        "n_bootstrap": n_boot,
        "interval": "90%",
    }


def _diagnostics(log: MatchLog, items: list, ratings, rng) -> dict:
    comps = log.comparisons
    m = len(comps)
    out = {
        "decided": m,
        "n_items": len(items),
        "first_position_win_rate": None,
        "order_consistency": None,
        "pairs_both_orders": 0,
        "first_shown_won_both": None,
        "second_shown_won_both": None,
        "ranking_agreement": None,
        "split_half_reliability": None,
        "length_correlation": None,
        "upsets": [],
    }
    if not m:
        return out

    out["first_position_win_rate"] = round(sum(c["winner_item"] == c["item_a"] for c in comps) / m, 3)

    by_pair: dict = {}
    for c in comps:
        by_pair.setdefault(c["pair_id"], []).append(c)
    both = [cs for cs in by_pair.values() if len({c["item_a"] for c in cs}) == 2]
    out["pairs_both_orders"] = len(both)
    if both:
        consistent = sum(len({c["winner_item"] for c in cs}) == 1 for cs in both)
        out["order_consistency"] = round(consistent / len(both), 3)
        # A flip is a pair whose winner changed with the order: the position, not the text, decided it.
        first_both = sum(all(c["winner_item"] == c["item_a"] for c in cs) for cs in both)
        second_both = sum(all(c["winner_item"] == c["item_b"] for c in cs) for cs in both)
        out["first_shown_won_both"] = round(first_both / len(both), 3)
        out["second_shown_won_both"] = round(second_both / len(both), 3)

    # How many verdicts the final ranking gets "right": a ceiling on how much of the
    # judge's behaviour a single quality scale can explain.
    rating_of = {item_id: float(ratings[i]) for item_id, i in log.index.items()}
    agree = 0
    upsets = []
    for c in comps:
        loser = c["item_b"] if c["winner_item"] == c["item_a"] else c["item_a"]
        gap = rating_of[c["winner_item"]] - rating_of[loser]
        if gap >= 0:
            agree += 1
        else:
            upsets.append({"comparison_id": c["id"], "pair_id": c["pair_id"], "winner_item": c["winner_item"],
                           "loser_item": loser, "rating_gap": round(-gap, 1)})
    out["ranking_agreement"] = round(agree / m, 3)
    out["upsets"] = sorted(upsets, key=lambda u: -u["rating_gap"])[:25]

    # Split the pairs into random halves, rank from each half alone, and correlate.
    if log.n_groups >= 8 and log.n >= 4:
        values = []
        for _ in range(20):
            mask = rng.permutation(log.n_groups) < log.n_groups // 2
            in_half = mask[log.group].astype(float)
            a = log.fit(weights=in_half)
            b = log.fit(weights=1.0 - in_half)
            rho = spearman(a, b)
            if rho is not None:
                values.append(rho)
        if values:
            out["split_half_reliability"] = round(float(np.mean(values)), 3)

    rho = spearman([it.get("n_words") or 0 for it in items], ratings)
    out["length_correlation"] = None if rho is None else round(rho, 3)
    return out


def timeline(items: list, comparisons: list) -> list:
    """Ratings after each round, for playback."""
    item_ids = [it["id"] for it in items]
    rounds = sorted({c["round"] for c in comparisons if c.get("winner_item")})
    frames = []
    start = None
    for r in rounds:
        log = MatchLog(item_ids, [c for c in comparisons if c["round"] <= r])
        strengths = log.fit(start=start)
        start = strengths
        ratings = to_elo(strengths)
        ranks = ranks_of(ratings)
        frames.append({
            "round": r,
            "comparisons": len(log.comparisons),
            "ratings": {item_id: round(float(ratings[i]), 1) for i, item_id in enumerate(item_ids)},
            "ranks": {item_id: int(ranks[i]) for i, item_id in enumerate(item_ids)},
        })
    return frames
