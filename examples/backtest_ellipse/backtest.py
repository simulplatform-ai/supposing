"""Essay benchmark: rank 100 learner essays and compare with two trained human raters (ELLIPSE corpus).

    uv run python examples/backtest_ellipse/backtest.py                 # replays the recorded verdicts: no calls
    uv run python examples/backtest_ellipse/backtest.py --fresh         # judges again with Claude Haiku (claude_cli)

Data: the ELLIPSE corpus (Crossley et al., 2023), https://github.com/scrosseye/ELLIPSE-Corpus, under CC BY 4.0:
essays by English language learners in grades 8 to 12, each scored 1 to 5 for overall proficiency by two raters.
The test split and the raw rater scores are downloaded from that repository and checked by SHA-256; no essay text
is stored here. ``rubric.txt`` is the corpus's holistic rubric (its five level descriptions) with two lines added.

The sample is the one of the published result: 100 essays drawn with seed 20261121 from the 192 test-split essays
on the prompt "Distance learning". ``verdicts-haiku-2026-11-21.json`` holds that run's 800 pairwise verdicts
(Claude Haiku via claude_cli, both orders, 8 rounds, Swiss pairing) and the per-essay scores, so the numbers
reproduce without a judge call. The engine refits the ratings from those verdicts.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import agreement, fetch, judge_pairs, provider_for, ratings_from, score_alone, spearman  # noqa: E402

from supposing import Item, RankConfig  # noqa: E402

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache" / "supposing" / "backtest_ellipse"
REPO = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/"
FILES = {"ELLIPSE_Final_github_test.zip": "9ecddfcda83f6a99c8a24dda47fb1c30673fedeaf578a0fadac6645fd151a2f8",
         "ellipsis_raw_rater_scores_anon_all_essay.zip": "f9c3cede2d54144225e0651af6415137f7da07f2d2955e8f8587896c293d5e85"}
# The corpus's README publishes these passwords; the zips are locked to keep the test set out of web crawls.
PASSWORDS = {"ELLIPSE_Final_github_test.zip": b"ellipse_test",
             "ellipsis_raw_rater_scores_anon_all_essay.zip": b"ellipse_raw_data"}
PROMPT, N, SEED = "Distance learning", 100, 20261121
VERDICTS = HERE / "verdicts-haiku-2026-11-21.json"


def read_csv(zip_path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        return list(csv.DictReader(io.StringIO(z.read(name, pwd=PASSWORDS[zip_path.name]).decode("utf-8"))))


def load_sample() -> list[dict]:
    """Exactly the published sample: the prompt's test essays that have rater rows, in file order, seeded draw."""
    test = read_csv(fetch(REPO + "ELLIPSE_Final_github_test.zip", CACHE / "ELLIPSE_Final_github_test.zip",
                          FILES["ELLIPSE_Final_github_test.zip"]))
    raters = {r["text_id_kaggle"]: r for r in read_csv(fetch(
        REPO + "ellipsis_raw_rater_scores_anon_all_essay.zip", CACHE / "ellipsis_raw_rater_scores_anon_all_essay.zip",
        FILES["ellipsis_raw_rater_scores_anon_all_essay.zip"]))}
    pool = [r for r in test if r["prompt"] == PROMPT and r["text_id_kaggle"] in raters]
    return [{"id": r["text_id_kaggle"], "text": r["full_text"].strip(), "overall": float(r["Overall"]),
             "r1": float(raters[r["text_id_kaggle"]]["Overall_1"]), "r2": float(raters[r["text_id_kaggle"]]["Overall_2"])}
            for r in random.Random(SEED).sample(pool, N)]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="judge again instead of replaying the recorded verdicts")
    ap.add_argument("--judge", default="claude_cli")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--one-order", action="store_true", help="with --fresh: the engine's default, one order per pair")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    rows = load_sample()
    items = [Item(id=r["id"], text=r["text"]) for r in rows]
    rubric = (HERE / "rubric.txt").read_text()
    if args.fresh:
        provider = provider_for(args.judge, args.model)
        cfg = RankConfig(rounds=8, both_orders=not args.one_order, seed=SEED)
        tag = "%s-%s-%s" % (args.judge, args.model, "one" if args.one_order else "both")
        recorded = {"pairwise": await judge_pairs(items, rubric, provider, cfg, CACHE / "runs.db", tag,
                                                  args.concurrency),
                    "score_alone_1to5": await score_alone(items, rubric, provider, 5, CACHE / ("score5-%s.json" % tag),
                                                          args.concurrency)}
        await provider.close()
    else:
        recorded = json.loads(VERDICTS.read_text())
        if recorded["sample"] != [r["id"] for r in rows]:
            raise SystemExit("The sample drawn here differs from the recorded one.")

    human = [r["overall"] for r in rows]
    rating = ratings_from(items, recorded["pairwise"], SEED)
    preds = {"pairwise (engine fit to the verdicts)": [rating[r["id"]] for r in rows]}
    for key, label in (("score_alone_1to10", "score each alone, 1 to 10"), ("score_alone_1to5", "score each alone, 1 to 5")):
        if key in recorded:
            preds[label] = [float(recorded[key][r["id"]]) for r in rows]
    preds["word count"] = [float(len(r["text"].split())) for r in rows]
    report = {"dataset": "ELLIPSE test split, prompt %r, %d essays, seed %d" % (PROMPT, N, SEED),
              "verdicts": "fresh run" if args.fresh else VERDICTS.name, "pairwise_calls": len(recorded["pairwise"]),
              "human": "mean of two raters ('Overall'), pairs with different scores only",
              "rater_vs_rater_spearman": round(spearman([r["r1"] for r in rows], [r["r2"] for r in rows]), 3),
              "results": {k: agreement(v, human) for k, v in preds.items()}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
