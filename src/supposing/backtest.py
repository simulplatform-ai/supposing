"""Check a judge against past decisions, and tune its rubric on a training split with a locked test split.

``outcomes`` map item ids to what really happened, as a number where higher is better: 1 and 0 for selected
and rejected, or a panel's scores or stages. Items without an outcome are ignored.

- ``backtest(leaderboard, outcomes)`` scores a finished ranking against the decisions. Pure.
- ``train_test_split(item_ids, outcomes)`` divides the items into train and test, the same way every time. Pure.
- ``tune(items, outcomes, rubrics, provider)`` judges pairs of past decisions with each candidate rubric on
  the train items, keeps a candidate only if it beats the first (the base) by more than chance, and then, once,
  measures the base and the choice on test items the judge has not seen.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .judge import judge_pair
from .providers.base import Provider, ProviderError
from .ranking import MAX_ATTEMPTS, Item
from .rubric import Rubric

N_BOOTSTRAP = 1000


@dataclass(frozen=True)
class Split:
    """Item ids for training and for the held-out test."""

    train: list[str]
    test: list[str]


def train_test_split(item_ids: Sequence[str], outcomes: Mapping[str, float], test_share: float = 0.4, seed: int = 0,
          groups: Mapping[str, str] | None = None) -> Split:
    """Divide the items with an outcome into train and test, stratified by outcome.

    ``groups`` (item id to a key) keeps items that belong together, such as resubmissions, on one side; a
    group counts at its best outcome. Deterministic in its arguments.
    """
    units: dict[str, list[str]] = {}
    for item_id in item_ids:
        if item_id in outcomes:
            units.setdefault((groups or {}).get(item_id, item_id), []).append(item_id)
    by_level: dict[float, list[str]] = {}
    for key, members in units.items():
        by_level.setdefault(max(outcomes[m] for m in members), []).append(key)
    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []
    for level in sorted(by_level):
        keys = sorted(by_level[level])
        rng.shuffle(keys)
        n_test = round(test_share * len(keys))
        for i, key in enumerate(keys):
            (test if i < n_test else train).extend(units[key])
    return Split(train=sorted(train), test=sorted(test))


def _interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.05, 0.95])
    return round(float(lo), 3), round(float(hi), 3)


def _auc(rank: np.ndarray, y: np.ndarray) -> float | None:
    """Chance that an item with a better outcome is ranked above one with a worse outcome (rank 1 is best)."""
    better = y[:, None] > y[None, :]
    if not better.any():
        return None
    above = (rank[:, None] < rank[None, :]) + 0.5 * (rank[:, None] == rank[None, :])
    return float(above[better].mean())


def _top_k(rank: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """k is the number of items at the best outcome; the share of them the ranking puts in its top k."""
    best = y == y.max()
    k = int(best.sum())
    picked = np.argsort(rank, kind="stable")[:k]
    return k, float(best[picked].mean())


def backtest(leaderboard: Sequence[Mapping[str, Any]], outcomes: Mapping[str, float], seed: int = 0
             ) -> dict[str, Any]:
    """How well a ranking agrees with past decisions.

    Returns ``n`` (items with an outcome), ``auc`` (the chance that an item with a better outcome is ranked
    above one with a worse outcome; 0.5 is chance), ``top_k`` (items at the best outcome, such as the number
    selected), ``top_k_share`` (the share of them in the ranking's top ``top_k``; chance is ``chance_top_k``),
    90% intervals ``auc_lo``/``auc_hi`` and ``top_k_lo``/``top_k_hi`` from a bootstrap over items, and
    ``line``, one plain sentence. The intervals are for the sample of decisions; they leave out the
    ranking's own uncertainty. Raises ``ValueError`` with fewer than two outcome levels among ranked items.
    """
    rows = [r for r in leaderboard if r["item_id"] in outcomes]
    rank = np.array([float(r["rank"]) for r in rows])
    y = np.array([float(outcomes[r["item_id"]]) for r in rows])
    if len(set(y.tolist())) < 2:
        raise ValueError("Back-testing needs ranked items with at least two different outcomes.")
    auc = _auc(rank, y)
    assert auc is not None  # two outcome levels, checked above
    k, share = _top_k(rank, y)
    rng = np.random.default_rng(seed)
    aucs, shares = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(rows), len(rows))
        if len(set(y[idx].tolist())) < 2:
            continue
        aucs.append(_auc(rank[idx], y[idx]))
        shares.append(_top_k(rank[idx], y[idx])[1])
    auc_lo, auc_hi = _interval(np.array(aucs))
    top_lo, top_hi = _interval(np.array(shares))
    chance = k / len(rows)
    line = ("Against %d past decisions, the ranking puts the better outcome above the worse %d%% of the time "
            "(90%% interval %d-%d%%; chance is 50%%). Of the %d best, %d are in its top %d (chance: %.1f)." % (
                len(rows), round(100 * auc), round(100 * auc_lo), round(100 * auc_hi), k, round(share * k), k,
                chance * k))
    return {"n": len(rows), "auc": round(auc, 3), "auc_lo": auc_lo, "auc_hi": auc_hi, "top_k": k,
            "top_k_share": round(share, 3), "top_k_lo": top_lo, "top_k_hi": top_hi,
            "chance_top_k": round(chance, 3), "line": line}


# Tuning ---------------------------------------------------------------------------------------------------


@dataclass
class TuneCall:
    """One judge call made while tuning: which split and candidate rubric, the pair, and the verdict."""

    split: str  # "train" or "test"
    candidate: int  # index into the rubrics
    item_a: str  # shown first
    item_b: str
    better: str  # the item with the better outcome
    winner_item: str | None = None  # None if the call failed
    reasoning: str = ""
    cost_usd: float = 0.0


@dataclass
class Tuning:
    """The result of ``tune``.

    ``chosen`` is the index of the rubric to use (0, the base, unless a candidate beat it by more than chance).
    ``train`` has one row per candidate: ``agreement`` (the share of train pairs where the judge picked the
    item with the better outcome), its 90% interval, and ``gain`` over the base with ``gain_lo``, a one-sided
    95% lower bound corrected for the number of candidates. ``test`` has ``base`` and ``chosen`` agreement on
    the held-out pairs, each with a 90% interval, and ``gain`` with its interval. ``split`` says which items
    trained and which tested. ``line`` is the verdict in one sentence.
    """

    chosen: int
    train: list[dict[str, Any]]
    test: dict[str, Any]
    split: Split
    calls: list[TuneCall] = field(default_factory=list)
    cost_usd: float = 0.0
    line: str = ""


def decided_pairs(item_ids: Sequence[str], outcomes: Mapping[str, float], n: int, seed: int
                  ) -> list[tuple[str, str]]:
    """Up to ``n`` pairs of items with different outcomes, each in a random order (first shown first)."""
    ids = sorted(item_ids)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if outcomes[a] != outcomes[b]]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return [(a, b) if rng.random() < 0.5 else (b, a) for a, b in pairs[:n]]


def planned_tune_calls(n_rubrics: int, train_pairs: int, test_pairs: int) -> int:
    """Most calls ``tune`` makes: every rubric on the train pairs, then two rubrics on the test pairs."""
    return n_rubrics * train_pairs + min(2, n_rubrics) * test_pairs


async def _judge_all(provider: Provider, rubric: str | Rubric, pairs: list[tuple[str, str]],
                     text: Mapping[str, str], outcomes: Mapping[str, float], split_name: str, candidate: int,
                     max_chars: int, concurrency: int) -> list[TuneCall]:
    gate = asyncio.Semaphore(concurrency)

    async def one(a: str, b: str) -> TuneCall:
        call = TuneCall(split_name, candidate, a, b, a if outcomes[a] > outcomes[b] else b)
        async with gate:
            for _ in range(MAX_ATTEMPTS):
                try:
                    verdict = await judge_pair(provider, rubric, text[a], text[b], max_chars)
                except ProviderError as exc:
                    if not exc.retryable:
                        raise
                    continue
                call.cost_usd += verdict.completion.cost_usd or 0.0
                if verdict.winner:
                    call.winner_item = a if verdict.winner == "A" else b
                    call.reasoning = verdict.reasoning
                    break
        return call

    return list(await asyncio.gather(*(one(a, b) for a, b in pairs)))


def _hits(calls: list[TuneCall]) -> np.ndarray:
    """1 where the judge picked the better outcome, 0 where not, NaN where the call failed."""
    return np.array([np.nan if c.winner_item is None else float(c.winner_item == c.better) for c in calls])


def _item_weights(pairs: list[tuple[str, str]], rng: np.random.Generator) -> np.ndarray:
    """Bootstrap weights for the pairs, ``(N_BOOTSTRAP, len(pairs))``, from resampling items rather than pairs.

    Pairs that share an item are not independent: an item the judge likes wins most of its pairs. Resampling
    pairs as if they were would make the intervals too narrow (in a null test it picked a candidate in 9 of
    30 runs instead of about 1).
    """
    ids = sorted({x for pair in pairs for x in pair})
    index = {x: i for i, x in enumerate(ids)}
    counts = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=N_BOOTSTRAP)
    a = np.array([index[x] for x, _ in pairs])
    b = np.array([index[y] for _, y in pairs])
    return counts[:, a] * counts[:, b]


def _boot_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(values)
    w = weights[:, ok]
    total = w.sum(axis=1)
    keep = total > 0
    return (w[keep] @ values[ok]) / total[keep]


def _agreement(hits: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    ok = hits[~np.isnan(hits)]
    if not len(ok):
        return {"agreement": None, "lo": None, "hi": None, "pairs": 0}
    lo, hi = _interval(_boot_mean(hits, weights))
    return {"agreement": round(float(ok.mean()), 3), "lo": lo, "hi": hi, "pairs": int(len(ok))}


def _gain(hits: np.ndarray, base: np.ndarray, weights: np.ndarray, alpha: float) -> dict[str, Any]:
    """Paired difference over the pairs both rubrics judged."""
    d = hits - base  # NaN where either call failed
    if np.isnan(d).all():
        return {"gain": None, "gain_lo": None, "gain_hi": None}
    boot = _boot_mean(d, weights)
    return {"gain": round(float(np.nanmean(d)), 3), "gain_lo": round(float(np.quantile(boot, alpha)), 3),
            "gain_hi": round(float(np.quantile(boot, 1 - alpha)), 3)}


async def tune(items: Sequence[Item], outcomes: Mapping[str, float], rubrics: Sequence[str | Rubric],
               provider: Provider, *, test_share: float = 0.4, train_pairs: int = 60, test_pairs: int = 100,
               seed: int = 0, groups: Mapping[str, str] | None = None, max_chars: int = 60000,
               concurrency: int = 8) -> Tuning:
    """Pick the rubric that best predicts past decisions, and measure it on decisions it was not chosen on.

    ``rubrics[0]`` is the base; the rest are candidates (written by a person, an agent or a proposer). Each
    rubric judges the same ``train_pairs`` pairs of train items with different outcomes, once each, with a
    random item first. A candidate replaces the base only if its one-sided 95% lower bound on the gain in
    agreement, corrected for the number of candidates, is above zero. Then, and only then, the base and the
    chosen rubric judge the same ``test_pairs`` pairs of test items. No test item reaches the judge before
    the choice is made, so the test result is honest once; re-running ``tune`` on the same test items with new
    candidates is choosing on the test set. At most ``planned_tune_calls(len(rubrics), train_pairs,
    test_pairs)`` calls.
    """
    if not rubrics:
        raise ValueError("Give at least one rubric; the first is the base.")
    text = {it.id: it.text for it in items}
    sp = train_test_split([it.id for it in items], outcomes, test_share=test_share, seed=seed, groups=groups)
    rng = np.random.default_rng(seed)
    alpha = 0.05 / max(1, len(rubrics) - 1)

    train = decided_pairs(sp.train, outcomes, train_pairs, seed)
    if not train:
        raise ValueError("The train items need at least two different outcomes.")
    calls: list[TuneCall] = []
    hits = []
    for i, rubric in enumerate(rubrics):
        judged = await _judge_all(provider, rubric, train, text, outcomes, "train", i, max_chars, concurrency)
        calls += judged
        hits.append(_hits(judged))
    rows = []
    weights = _item_weights(train, rng)
    for i, h in enumerate(hits):
        row = {"candidate": i, **_agreement(h, weights)}
        row.update(_gain(h, hits[0], weights, alpha) if i else {"gain": 0.0, "gain_lo": None, "gain_hi": None})
        rows.append(row)
    better = [r for r in rows[1:] if r["gain_lo"] is not None and r["gain_lo"] > 0]
    chosen = max(better, key=lambda r: r["gain"])["candidate"] if better else 0

    test = decided_pairs(sp.test, outcomes, test_pairs, seed + 1)
    test_result: dict[str, Any] = {"pairs": len(test)}
    if test:
        base_calls = await _judge_all(provider, rubrics[0], test, text, outcomes, "test", 0, max_chars,
                                      concurrency)
        calls += base_calls
        base_hits = _hits(base_calls)
        weights = _item_weights(test, rng)
        test_result["base"] = _agreement(base_hits, weights)
        if chosen:
            chosen_calls = await _judge_all(provider, rubrics[chosen], test, text, outcomes, "test", chosen,
                                            max_chars, concurrency)
            calls += chosen_calls
            test_result["chosen"] = _agreement(_hits(chosen_calls), weights)
            test_result.update(_gain(_hits(chosen_calls), base_hits, weights, 0.05))
        else:
            test_result["chosen"] = test_result["base"]
            test_result.update({"gain": 0.0, "gain_lo": None, "gain_hi": None})

    return Tuning(chosen=chosen, train=rows, test=test_result, split=sp, calls=calls,
                  cost_usd=round(sum(c.cost_usd for c in calls), 6), line=_line(chosen, test_result))


def _line(chosen: int, test: dict[str, Any]) -> str:
    if "base" not in test:
        return "No test pairs: the test items need at least two different outcomes."
    b = test["base"]
    if b["agreement"] is None:
        return "Every test call failed, so nothing was measured."
    base = "the base rubric picks the better outcome in %d%% of %d held-out pairs (90%% interval %d-%d%%)" % (
        round(100 * b["agreement"]), b["pairs"], round(100 * b["lo"]), round(100 * b["hi"]))
    if not chosen:
        return "No candidate beat the base by more than chance on the train pairs; %s." % base
    c = test["chosen"]
    return ("Candidate %d beat the base on the train pairs. Held out, it picks the better outcome in %d%% of "
            "pairs against %d%% for the base (gain %+d points, 90%% interval %+d to %+d)." % (
                chosen, round(100 * c["agreement"]), round(100 * b["agreement"]), round(100 * test["gain"]),
                round(100 * test["gain_lo"]), round(100 * test["gain_hi"])))
