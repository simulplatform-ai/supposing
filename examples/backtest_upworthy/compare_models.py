"""Paired differences between the Haiku and Opus runs on the same 60 tests (pre-registered addendum A).

    uv run python examples/backtest_upworthy/compare_models.py     # from the recorded verdicts: no calls

Pairwise agreement on the significantly different pairs; paired bootstrap over tests (1,000 resamples, seed 1),
90% intervals and the share of resamples above zero.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backtest_ellipse"))
from common import ratings_from  # noqa: E402

from supposing import Item  # noqa: E402

spec = importlib.util.spec_from_file_location("upworthy", HERE / "backtest.py")
assert spec and spec.loader
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def scores(tests: list[dict], recorded: dict) -> dict[str, dict[str, float]]:
    engine: dict[str, float] = {}
    for t in tests:
        engine.update(ratings_from([Item(id=a["id"], text=a["headline"]) for a in t["arms"]],
                                   recorded["pairwise"][t["test"]], bt.SEED))
    listed: dict[str, float] = {}
    for t in tests:
        order = recorded["list_ranking"].get(t["test"], [])
        for i, a in enumerate(t["arms"]):
            listed[a["id"]] = float(len(t["arms"]) - order.index(i + 1)) if (i + 1) in order else 0.0
    return {"engine": engine, "alone": {k: float(v) for k, v in recorded["score_alone_1to10"].items()}, "list": listed}


def main() -> None:
    tests = sorted(random.Random(bt.SEED).sample(bt.eligible(), bt.N_TESTS), key=lambda t: t["test"])
    for t in tests:
        t["pairs"] = bt.significant_pairs(t["arms"])
    runs = {m: scores(tests, json.loads(bt.verdicts_path(m).read_text())) for m in ("haiku", "opus")}
    diffs = {
        "opus engine - haiku engine": (("opus", "engine"), ("haiku", "engine")),
        "opus alone - haiku alone": (("opus", "alone"), ("haiku", "alone")),
        "opus list - haiku list": (("opus", "list"), ("haiku", "list")),
        "opus engine - opus alone": (("opus", "engine"), ("opus", "alone")),
        "opus engine - opus list": (("opus", "engine"), ("opus", "list")),
        "opus list - opus alone": (("opus", "list"), ("opus", "alone")),
    }

    def pw(ts: list[dict], key: tuple[str, str]) -> float:
        return bt.measures(ts, runs[key[0]][key[1]])[2]

    rng = random.Random(1)
    samples = [[rng.choice(tests) for _ in tests] for _ in range(1000)]
    out = {"pairwise_agreement": {"%s %s" % k: round(pw(tests, k), 3)
                                  for k in [(m, a) for m in ("haiku", "opus") for a in ("engine", "alone", "list")]}}
    for name, (a, b) in diffs.items():
        vals = sorted(pw(s, a) - pw(s, b) for s in samples)
        out[name] = {"diff": round(pw(tests, a) - pw(tests, b), 3), "ci90": [round(vals[50], 3), round(vals[949], 3)],
                     "share_above_0": round(sum(v > 0 for v in vals) / len(vals), 3)}
    lo = out["opus engine - haiku engine"]["ci90"][0]
    out["reading"] = "model quality matters" if lo > 0 else "no detectable gain from the stronger model on this sample"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
