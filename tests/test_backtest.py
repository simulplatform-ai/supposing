import hashlib
import math
import random
import re

import numpy as np
import pytest

from supposing import Completion, Item, backtest, planned_tune_calls, train_test_split, tune


class FeatureJudge:
    """A rubric-sensitive fake judge: each text carries ``[[f: evidence=0.8 polish=0.3]]``, and the judge
    scores a text by the features its rubric names. Records which texts it was shown, in order."""

    name = "features"
    model = "features"

    def __init__(self, sharpness: float = 6.0):
        self.sharpness = sharpness
        self.seen: list[tuple[str, str]] = []

    async def complete(self, system, user):
        criteria = re.search(r"<criteria>\n(.*?)\n</criteria>", user, re.S).group(1).lower()
        a, b = (re.search(r"<%s>\n(.*?)\n</%s>" % (t, t), user, re.S).group(1) for t in ("text_a", "text_b"))
        self.seen.append((a.split()[0], b.split()[0]))

        def score(text):
            feats = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([0-9.]+)", text))
            return sum(v for k, v in feats.items() if k in criteria)

        p = 1 / (1 + math.exp(-self.sharpness * (score(a) - score(b))))
        u = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return Completion('{"reasoning": "r", "winner": "%s"}' % ("A" if u < p else "B"), cost_usd=0.001)

    async def close(self):
        pass


def corpus(n=80, seed=0, shuffle=False):
    """The panel selected on evidence, with noise; polish is unrelated."""
    rng = np.random.default_rng(seed)
    evidence, polish = rng.random(n), rng.random(n)
    items = [Item(id="d%02d" % i, text="d%02d [[f: evidence=%.3f polish=%.3f]]" % (i, evidence[i], polish[i]))
             for i in range(n)]
    noisy = evidence + rng.normal(0, 0.15, n)
    cut = np.quantile(noisy, 0.7)
    outcomes = {it.id: int(noisy[i] >= cut) for i, it in enumerate(items)}
    if shuffle:
        values = list(outcomes.values())
        random.Random(seed).shuffle(values)
        outcomes = dict(zip(outcomes, values, strict=True))
    return items, outcomes


BASE = "Prefer the more polished text."
EVIDENCE = "Prefer the text with stronger evidence."


def test_split_is_stratified_deterministic_and_keeps_groups():
    ids = ["d%02d" % i for i in range(20)]
    outcomes = {i: int(k < 5) for k, i in enumerate(ids)}
    groups = {"d00": "g", "d01": "g"}
    a = train_test_split(ids, outcomes, test_share=0.4, seed=3, groups=groups)
    assert a == train_test_split(list(reversed(ids)), outcomes, test_share=0.4, seed=3, groups=groups)
    assert set(a.train) | set(a.test) == set(ids) and not set(a.train) & set(a.test)
    assert ({"d00", "d01"} <= set(a.train)) or ({"d00", "d01"} <= set(a.test))
    selected_units = {groups.get(i, i) for i in a.test if outcomes[i]}
    assert len(selected_units) == 2  # 4 selected units (one is a group of two): 40% of them


def test_backtest_against_hand_computed_values():
    board = [{"item_id": i, "rank": r} for r, i in enumerate(["a", "b", "c", "d", "e"], start=1)]
    report = backtest(board, {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0, "z": 1})
    # Selected a (rank 1) and c (rank 3) against rejected b, d, e: 5 of the 6 pairs are in the right order.
    assert report["n"] == 5 and report["auc"] == round(5 / 6, 3)
    assert report["top_k"] == 2 and report["top_k_share"] == 0.5 and report["chance_top_k"] == 0.4
    assert report["auc_lo"] <= report["auc"] <= report["auc_hi"]
    assert report["line"].startswith("Against 5 past decisions")
    with pytest.raises(ValueError):
        backtest(board, {"a": 1, "b": 1})


async def test_tuning_finds_the_rubric_the_panel_used_and_says_so_held_out():
    items, outcomes = corpus()
    tuning = await tune(items, outcomes, [BASE, EVIDENCE], FeatureJudge(), train_pairs=60, test_pairs=100)
    assert tuning.chosen == 1
    assert tuning.train[1]["gain_lo"] > 0
    assert tuning.test["chosen"]["agreement"] > 0.75 and tuning.test["base"]["agreement"] < 0.65
    assert tuning.test["gain_lo"] > 0
    assert tuning.line.startswith("Candidate 1 beat the base")
    assert len(tuning.calls) <= planned_tune_calls(2, 60, 100)
    assert tuning.cost_usd == pytest.approx(0.001 * len(tuning.calls))


async def test_no_test_item_is_judged_before_the_choice():
    items, outcomes = corpus()
    judge = FeatureJudge()
    tuning = await tune(items, outcomes, [BASE, EVIDENCE, "Prefer the shorter text."], judge, train_pairs=40)
    test_ids = set(tuning.split.test)
    n_train = sum(c.split == "train" for c in tuning.calls)
    assert n_train == 3 * 40
    assert not test_ids & {x for pair in judge.seen[:n_train] for x in pair}
    assert {x for pair in judge.seen[n_train:] for x in pair} <= test_ids


@pytest.mark.stats
async def test_shuffled_outcomes_give_no_reliable_improvement():
    """With outcomes unrelated to any feature, the acceptance rule should almost never pick a candidate."""
    chosen = 0
    for seed in range(30):
        items, outcomes = corpus(seed=seed, shuffle=True)
        tuning = await tune(items, outcomes, [BASE, EVIDENCE], FeatureJudge(), seed=seed, test_pairs=10)
        chosen += tuning.chosen != 0
    assert chosen <= 3, "a candidate was chosen in %d of 30 null runs" % chosen
