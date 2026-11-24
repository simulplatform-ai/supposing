"""Argument back-test: rank arguments by convincingness and compare with crowd judgements (UKPConvArg1).

    uv run python examples/backtest_ukpconvarg/backtest.py            # replays the recorded verdicts: no calls
    uv run python examples/backtest_ukpconvarg/backtest.py --fresh    # judges again with Claude Haiku (claude_cli)

Data: UKPConvArg1 (Habernal and Gurevych, ACL 2016), https://github.com/UKPLab/acl2016-convincing-arguments,
under CC BY 4.0 (arguments from createdebate.com, CC BY 3.0, and convinceme.net, public domain). Per debate side,
``UKPConvArg1Strict-CSV`` gives argument pairs with the crowd's gold label for the more convincing one, and
``UKPConvArg1-Ranking-CSV`` each argument's derived rank score. Files are downloaded from GitHub; no argument text
is stored here.

Design, pre-registered on 2026-11-24 before any judge call (knowledge note 2026-11-24-ukpconvarg-pilot):
- two debates drawn with seed 20261124 from the 16, both sides of each, 140 arguments; each side ranked alone;
- plain rubric per side: Debate "<title>", both arguments argue for "<side>", which is more convincing?;
- engine defaults (8 Swiss rounds, one order), Haiku via claude_cli; baselines word count and scoring each
  argument alone 1 to 10;
- metrics: agreement with the crowd's gold pairs, and Spearman with the derived rank score within each side
  (averaged, weighted by size), each with a 90% interval resampling arguments within sides;
- positive if pairwise beats both baselines on both metrics with its interval wholly above length's point estimate.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backtest_ellipse"))
from common import fetch, judge_pairs, provider_for, ratings_from, score_alone, spearman  # noqa: E402

from supposing import Item, RankConfig  # noqa: E402

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache" / "supposing" / "backtest_ukpconvarg"
REPO = "https://raw.githubusercontent.com/UKPLab/acl2016-convincing-arguments/master/data/"
SEED = 20261124
DEBATES = {  # file stem prefix -> title, sides (file stem suffix -> side as written)
    "human-growth-and-development-should-parents-use-spanking-as-an-option-to-discipline-": (
        "Human growth and development: should parents use spanking as an option to discipline?",
        {"no": "No", "yes": "Yes"}),
    "is-the-school-uniform-a-good-or-bad-idea-": ("Is the school uniform a good or bad idea?",
                                                  {"bad": "Bad", "good": "Good"}),
}
ALL_DEBATES = 16
VERDICTS = HERE / "verdicts-haiku-2026-11-24.json"
RUBRIC = 'Debate: "%s". Both arguments argue for the side "%s". Which argument is more convincing?'


def read(path: Path) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t"))
    return [r for r in rows if r and not r[0].startswith("#")]


def clean(text: str) -> str:
    return html.unescape(re.sub(r"<br\s*/?>", "\n", text)).strip()


def load() -> list[dict]:
    sides = []
    for stem, (title, names) in DEBATES.items():
        for suffix, side in names.items():
            name = "%s_%s.csv" % (stem, suffix)
            ranking = read(fetch(REPO + "UKPConvArg1-Ranking-CSV/" + name, CACHE / "rank" / name))
            pairs = read(fetch(REPO + "UKPConvArg1Strict-CSV/" + name, CACHE / "strict" / name))
            # The lower rank score is the crowd's gold winner in 87-94% of gold pairs in these files, so it is
            # negated: higher means more convincing.
            args = {r[0]: {"id": r[0], "rank": -float(r[1]), "text": clean(r[2])} for r in ranking}
            gold = []
            for r in pairs:
                a, b = r[0].split("_")
                if a in args and b in args:
                    gold.append((a, b) if r[1] == "a1" else (b, a))  # (more convincing, less)
            sides.append({"key": "%s_%s" % (stem, suffix), "rubric": RUBRIC % (title, side),
                          "args": [args[k] for k in sorted(args)], "gold": gold})
    return sides


def agreement(sides: list[dict], score: dict[str, float], seed: int = 1) -> dict:
    """Gold-pair agreement and weighted within-side Spearman, with intervals resampling arguments in each side."""
    def measures(weights: list[dict[str, int]]) -> tuple[float, float]:
        right = total = 0.0
        rhos, ns = [], []
        for side, w in zip(sides, weights, strict=True):
            for win, lose in side["gold"]:
                k = w[win] * w[lose]
                if k:
                    d = score[win] - score[lose]
                    right += k * (1.0 if d > 0 else 0.5 if d == 0 else 0.0)
                    total += k
            xs = [a for a in side["args"] for _ in range(w[a["id"]])]
            if len(xs) > 2:
                rhos.append(spearman([score[a["id"]] for a in xs], [a["rank"] for a in xs]))
                ns.append(len(xs))
        return right / total, float(np.average(rhos, weights=ns))

    full = [{a["id"]: 1 for a in s["args"]} for s in sides]
    pair, rho = measures(full)
    rng = random.Random(seed)
    boot = []
    for _ in range(1000):
        weights = []
        for s in sides:
            w = {a["id"]: 0 for a in s["args"]}
            for _ in s["args"]:
                w[rng.choice(s["args"])["id"]] += 1
            weights.append(w)
        boot.append(measures(weights))
    p_lo, p_hi = np.quantile([b[0] for b in boot], [0.05, 0.95])
    r_lo, r_hi = np.quantile([b[1] for b in boot], [0.05, 0.95])
    return {"gold_pairs": round(pair, 3), "gold_pairs_lo": round(float(p_lo), 3), "gold_pairs_hi": round(float(p_hi), 3),
            "spearman": round(rho, 3), "spearman_lo": round(float(r_lo), 3), "spearman_hi": round(float(r_hi), 3)}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--judge", default="claude_cli")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    debates = sorted(DEBATES)
    assert len(DEBATES) == 2  # drawn as random.Random(SEED).sample(sorted(all 16 debate stems), 2); see the note
    sides = load()
    if args.fresh or not VERDICTS.exists():
        provider = provider_for(args.judge, args.model)
        recorded: dict = {"source": "%s/%s, engine defaults, seed %d" % (args.judge, args.model, SEED), "sides": {}}
        for side in sides:
            items = [Item(id=a["id"], text=a["text"]) for a in side["args"]]
            tag = "%s-%s-%s" % (args.judge, args.model, side["key"][-40:])
            recorded["sides"][side["key"]] = {
                "pairwise": await judge_pairs(items, side["rubric"], provider, RankConfig(seed=SEED),
                                              CACHE / "runs.db", tag, args.concurrency),
                "score_alone_1to10": await score_alone(items, side["rubric"], provider, 10,
                                                       CACHE / ("score10-%s.json" % tag), args.concurrency)}
        await provider.close()
        if not VERDICTS.exists() and args.judge == "claude_cli":
            VERDICTS.write_text(json.dumps(recorded, indent=0) + "\n")
    else:
        recorded = json.loads(VERDICTS.read_text())

    pairwise: dict[str, float] = {}
    alone: dict[str, float] = {}
    for side in sides:
        rec = recorded["sides"][side["key"]]
        items = [Item(id=a["id"], text=a["text"]) for a in side["args"]]
        pairwise.update(ratings_from(items, rec["pairwise"], SEED))
        alone.update({k: float(v) for k, v in rec["score_alone_1to10"].items()})
    length = {a["id"]: float(len(a["text"].split())) for s in sides for a in s["args"]}
    report = {"dataset": "UKPConvArg1, debates %s, %d arguments, %d gold pairs" % (
                  debates, sum(len(s["args"]) for s in sides), sum(len(s["gold"]) for s in sides)),
              "pairwise_calls": sum(len(recorded["sides"][s["key"]]["pairwise"]) for s in sides),
              "results": {"pairwise (engine)": agreement(sides, pairwise),
                          "score each alone, 1 to 10": agreement(sides, alone),
                          "word count": agreement(sides, length)}}
    r = report["results"]
    p, a, w = r["pairwise (engine)"], r["score each alone, 1 to 10"], r["word count"]
    report["positive_by_preregistered_rule"] = all(
        p[m] > a[m] and p[m] > w[m] and p[m + "_lo"] > w[m] for m in ("gold_pairs", "spearman"))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
