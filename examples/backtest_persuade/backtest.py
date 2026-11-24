"""Essay back-test on a second corpus: rank 100 persuasive essays and compare with trained raters' holistic scores.

    uv run python examples/backtest_persuade/backtest.py             # replays the recorded verdicts: no calls
    uv run python examples/backtest_persuade/backtest.py --fresh     # judges again with Claude Haiku (claude_cli)

Data: the PERSUADE 2.0 corpus (Crossley et al., 2024), https://github.com/scrosseye/persuade_corpus_2.0, under
CC BY 4.0: persuasive essays by students in grades 6 to 12 in the United States, each with a holistic score from
1 to 6. The script downloads the corpus's test set (a zip on Google Drive, linked from that README and locked with
the password the README gives) and checks it by SHA-256; no essay text is stored here.

Design, fixed before the first run on 2026-11-24:
- the prompt "Phones and driving" (467 test essays; the prompt does not occur in ELLIPSE), 100 essays drawn with
  seed 20261124;
- ``rubric.txt``, written by us from the criteria of the corpus's holistic rating form (point of view and
  evidence, organization, language, mechanics), plus "do not reward length";
- the engine's defaults: 8 Swiss rounds, each pair judged once with a random essay first (400 calls), and the
  same judge scoring each essay alone from 1 to 6 (100 calls);
- agreement with the human score: the share of differently scored pairs in the same order, and Spearman, each with
  a 90% bootstrap interval over essays; word count as a baseline.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "backtest_ellipse"))
from common import agreement, fetch, judge_pairs, provider_for, ratings_from, score_alone  # noqa: E402

from supposing import Item, RankConfig  # noqa: E402

HERE = Path(__file__).parent
CACHE = Path.home() / ".cache" / "supposing" / "backtest_persuade"
TEST_ZIP = "https://drive.usercontent.google.com/download?id=1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1&export=download&confirm=t"
TEST_SHA256 = "51c907f90b1303d610e8fcd8c0cbf3656ec88116bc34b60b93794affa77facb2"
PASSWORD = b"persuade_test"  # published in the corpus's README
PROMPT, N, SEED = "Phones and driving", 100, 20261124
VERDICTS = HERE / "verdicts-haiku-2026-11-24.json"


def load_sample() -> list[dict]:
    path = fetch(TEST_ZIP, CACHE / "persuade_corpus_2.0_test.zip", TEST_SHA256)
    csv.field_size_limit(10**9)
    with zipfile.ZipFile(path) as z:
        rows = csv.DictReader(io.StringIO(z.read("persuade_corpus_2.0_test.csv", pwd=PASSWORD).decode("utf-8")))
        essays: dict[str, dict] = {}
        for r in rows:  # one row per discourse element; the essay fields repeat
            if r["prompt_name"] == PROMPT:
                essays.setdefault(r["essay_id"], {"id": r["essay_id"], "text": r["full_text"].strip(),
                                                  "score": float(r["holistic_essay_score"])})
    pool = [essays[k] for k in sorted(essays)]
    return random.Random(SEED).sample(pool, N)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="judge instead of replaying the recorded verdicts")
    ap.add_argument("--judge", default="claude_cli")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    rows = load_sample()
    items = [Item(id=r["id"], text=r["text"]) for r in rows]
    rubric = (HERE / "rubric.txt").read_text()
    if args.fresh or not VERDICTS.exists():
        provider = provider_for(args.judge, args.model)
        tag = "%s-%s" % (args.judge, args.model)
        recorded = {"source": "%s/%s, engine defaults, seed %d" % (args.judge, args.model, SEED),
                    "sample": [r["id"] for r in rows],
                    "pairwise": await judge_pairs(items, rubric, provider, RankConfig(seed=SEED), CACHE / "runs.db",
                                                  tag, args.concurrency),
                    "score_alone_1to6": await score_alone(items, rubric, provider, 6, CACHE / ("score6-%s.json" % tag),
                                                          args.concurrency)}
        await provider.close()
        if not VERDICTS.exists() and args.judge == "claude_cli":
            VERDICTS.write_text(json.dumps(recorded, indent=0) + "\n")
    else:
        recorded = json.loads(VERDICTS.read_text())
        if recorded["sample"] != [r["id"] for r in rows]:
            raise SystemExit("The sample drawn here differs from the recorded one.")

    human = [r["score"] for r in rows]
    rating = ratings_from(items, recorded["pairwise"], SEED)
    preds = {"pairwise (engine)": [rating[r["id"]] for r in rows],
             "score each alone, 1 to 6": [float(recorded["score_alone_1to6"][r["id"]]) for r in rows],
             "word count": [float(len(r["text"].split())) for r in rows]}
    levels = sorted(set(human))
    report = {"dataset": "PERSUADE 2.0 test set, prompt %r, %d essays, seed %d" % (PROMPT, N, SEED),
              "human": "holistic score 1-6; counts %s" % {int(v): human.count(v) for v in levels},
              "pairwise_calls": len(recorded["pairwise"]),
              "results": {k: agreement(v, human) for k, v in preds.items()}}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
