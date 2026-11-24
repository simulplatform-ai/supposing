"""Back-test: rank real Wikimedia Rapid Fund applications and compare with the real funding decisions.

    uv run python examples/backtest_rapid_fund/backtest.py            # Claude Haiku via the local claude CLI
    uv run python examples/backtest_rapid_fund/backtest.py --judge mock --no-pointwise   # free, for the plumbing
    uv run python examples/backtest_rapid_fund/backtest.py --manifest examples/backtest_rapid_fund/manifest-all.json \
        --rubrics neutral     # all 184 decided applications, each read as it was just before its decision

The applications are FY 2026-27 cycles 1 and 2 of the Wikimedia Foundation's Rapid Fund, submitted and decided
from June 2026 on, after the judge's training data. Their pages on Meta-Wiki are public under CC BY-SA 4.0
with a funded or not-funded category. ``manifest.json`` (committed) pins the sample: page ids, the revision
the judge reads (each page's first revision: the application as submitted, before any decision) and the
outcome. Texts are fetched from the public MediaWiki API on first use and cached outside the repository.
``--resample`` draws a new manifest from the categories (seeded).

The judge reads the application's main proposal, learning and financial sections: the applicant's details
and the endorsements and feedback are removed. It reports, against the decisions: AUC with a 90% interval,
the share funded in the ranking's top 20%, and the same for two baselines: longer first, and the judge
scoring each application alone (1 to 10). Each score is also given within region (the Rapid Fund decides
by region), and on the applications that passed the eligibility screen.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import supposing
from supposing import Item, RankConfig, SQLiteStore, backtest, make_provider, run_ranking

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest.json"
CACHE = Path.home() / ".cache" / "supposing" / "backtest_rapid_fund"
API = "https://meta.wikimedia.org/w/api.php"
UA = "supposing-backtest/0.1 (research back-test; https://supposing.app)"
FY = "FY 2026-27"
CYCLES = (1, 2)
REGIONS = ("CEECA", "ESEAP", "LAC", "MENA", "NWE", "SA", "SSA", "USCA")
RUBRICS = {
    "fund": "Which application would the Wikimedia Foundation's Rapid Fund be more likely to fund?",
    "neutral": "Which application is stronger?",
}
POINTWISE = ("Score this application from 1 to 10 for how likely the Wikimedia Foundation's Rapid Fund is to "
             "fund it. Reply with the number only.")


# Fetching -------------------------------------------------------------------------------------------------


def api(**params: str) -> dict:
    """One MediaWiki API call, cached on disk by its parameters, throttled to stay well under the limits."""
    params = {**params, "format": "json", "formatversion": "2"}
    query = urllib.parse.urlencode(sorted(params.items()))
    path = CACHE / "api" / (re.sub(r"[^A-Za-z0-9]+", "_", query)[:180] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    time.sleep(1.0)
    req = urllib.request.Request(API + "?" + query, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def members(category: str) -> list[dict]:
    out, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle="Category:" + category, cmlimit="500", **cont)
        out += d["query"]["categorymembers"]
        if "continue" not in d:
            return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def population() -> list[dict]:
    """Every decided application of the cycles: page id, cycle, region, outcome (1 funded, 0 not)."""
    cycle_of = {p["pageid"]: c for c in CYCLES
                for p in members("Rapid Fund proposals in %s (Round %d)" % (FY, c))}
    rows = {}
    for region in REGIONS:
        for label, outcome in (("Funded", 1), ("Not funded", 0)):
            for p in members("%s Rapid Funds for %s in %s" % (label, region, FY)):
                if p["pageid"] in cycle_of:
                    rows[p["pageid"]] = {"pageid": p["pageid"], "title": p["title"], "cycle": cycle_of[p["pageid"]],
                                         "region": region, "funded": outcome}
    return sorted(rows.values(), key=lambda r: r["pageid"])


def first_revision(pageid: int) -> dict:
    d = api(action="query", prop="revisions", pageids=str(pageid), rvlimit="1", rvdir="newer",
            rvprop="ids|timestamp", rvslots="main")
    rev = d["query"]["pages"][0]["revisions"][0]
    return {"revid": rev["revid"], "submitted": rev["timestamp"]}


# Statuses seen before a decision. Anything else (rejected, pending_grantee_reply, pending_grant_promotion,
# pending_agreement_signatures, back_to_applicant, granted, ...) is on or after the decision.
PRE_DECISION = {"draft", "prescreening", "new", "processing", "awaiting_requirements", "back_to_draft",
                "revision_required_prescreening", "community_committee_review", "pending_program_approval"}
REVIEW = {"pending_program_approval", "community_committee_review"}  # past the eligibility screen


def history(pageid: int) -> list[dict]:
    """Every revision of a page, oldest first, with the Fluxx status its status box shows."""
    d = api(action="query", prop="revisions", pageids=str(pageid), rvlimit="50", rvdir="newer",
            rvprop="ids|timestamp|content", rvslots="main")
    out = []
    for rev in d["query"]["pages"][0]["revisions"]:
        m = re.search(r"Fluxx status\|([^}]*)\}\}", rev["slots"]["main"]["content"])
        out.append({"revid": rev["revid"], "timestamp": rev["timestamp"], "status": m.group(1) if m else None})
    return out


def pre_decision_revision(pageid: int) -> dict:
    """The last revision before the status box first shows a decision (granted, rejected, ...), and when that was.
    The page is synced from the Foundation's grants system by a bot, so this is the application as the reviewers
    had it when they decided."""
    revs = history(pageid)
    decided = next((i for i, r in enumerate(revs) if r["status"] not in PRE_DECISION), len(revs))
    before = revs[max(0, decided - 1)]
    return {"revid": before["revid"], "read_status": before["status"], "read_at": before["timestamp"],
            "reviewed": any(r["status"] in REVIEW for r in revs),
            "decided_at": revs[decided]["timestamp"] if decided < len(revs) else None,
            "decision_status": revs[decided]["status"] if decided < len(revs) else None}


def revision_text(revid: int) -> str:
    d = api(action="query", prop="revisions", revids=str(revid), rvprop="content", rvslots="main")
    return d["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]


def sample(n: int, seed: int, revision: str) -> list[dict]:
    """Proportional to cycle and outcome, seeded; ``n`` of 0 takes every decided application."""
    rows = population()
    read = first_revision if revision == "first" else pre_decision_revision
    if not n:
        return [{**r, **read(r["pageid"])} for r in rows]
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault((r["cycle"], r["funded"]), []).append(r)
    keys = sorted(strata)
    quota = {k: int(n * len(strata[k]) / len(rows)) for k in keys}
    for k in sorted(keys, key=lambda k: -(n * len(strata[k]) / len(rows) % 1))[:n - sum(quota.values())]:
        quota[k] += 1  # largest remainders, so the sample is exactly n
    picked = []
    for key in keys:
        group = strata[key]
        rng.shuffle(group)
        picked += group[:quota[key]]
    return [{**r, **read(r["pageid"])} for r in sorted(picked, key=lambda r: r["pageid"])]


# Cleaning -------------------------------------------------------------------------------------------------

DROP_SECTIONS = re.compile(r"^==\s*(Applicant Details|Endorsements and Feedback)\s*==", re.I)


def _strip_templates(text: str) -> str:
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{{", i):
            depth, i = depth + 1, i + 2
        elif text.startswith("}}", i) and depth:
            depth, i = depth - 1, i + 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def clean(wikitext: str) -> str:
    """The application as a reader sees it, without the applicant's details, feedback or page furniture."""
    keep, dropping = [], False
    for line in wikitext.splitlines():
        if line.startswith("=="):
            dropping = bool(DROP_SECTIONS.match(line.strip()))
        if not dropping:
            keep.append(line)
    text = _strip_templates("\n".join(keep))
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", text)
    text = re.sub(r"__[A-Z]+__", "", text).replace("&nbsp;", " ").replace("''", "")
    text = re.sub(r"^;", "", text, flags=re.M)
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# Scoring --------------------------------------------------------------------------------------------------


def precision_top(order: list[str], outcomes: dict[str, int], share: float = 0.2, seed: int = 0) -> dict:
    """Share funded among the top ``share`` of a best-first order, with a 90% bootstrap interval over items."""
    k = max(1, round(share * len(order)))
    rank = {item: i for i, item in enumerate(order)}
    ids = np.array(order)
    point = float(np.mean([outcomes[i] for i in order[:k]]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(1000):
        s = ids[rng.integers(0, len(ids), len(ids))]
        s = sorted(s, key=lambda i: rank[str(i)])[:k]
        boot.append(np.mean([outcomes[i] for i in s]))
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {"k": k, "precision": round(point, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
            "base_rate": round(float(np.mean(list(outcomes.values()))), 3)}


def within_region(order: list[str], outcomes: dict[str, int], region: dict[str, str], seed: int = 0) -> dict:
    """AUC over funded / not-funded pairs from the same region, pooled, with a 90% interval from resampling
    applications within each region; and per region, the share funded in that region's top 20%."""
    rank = {item: i for i, item in enumerate(order)}
    groups: dict[str, list[str]] = {}
    for item in order:
        groups.setdefault(region[item], []).append(item)

    def pooled(sample_groups: list[list[str]]) -> float | None:
        right = total = 0.0
        for g in sample_groups:
            yes = [rank[i] for i in g if outcomes[i]]
            no = [rank[i] for i in g if not outcomes[i]]
            for a in yes:
                for b in no:
                    right += 1.0 if a < b else 0.5 if a == b else 0.0
                    total += 1
        return right / total if total else None

    point = pooled(list(groups.values()))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(1000):
        resampled = [[g[j] for j in rng.integers(0, len(g), len(g))] for g in groups.values()]
        v = pooled(resampled)
        if v is not None:
            boot.append(v)
    lo, hi = np.quantile(boot, [0.05, 0.95])
    regions = {}
    for name, g in sorted(groups.items()):
        k = max(1, round(0.2 * len(g)))
        regions[name] = {"n": len(g), "funded": sum(outcomes[i] for i in g),
                         "top20_k": k, "top20_funded": sum(outcomes[i] for i in g[:k])}
    contributing = {n: r for n, r in regions.items() if 0 < r["funded"] < r["n"]}
    return {"auc": round(float(point or 0), 3), "auc_lo": round(float(lo), 3), "auc_hi": round(float(hi), 3),
            "pairs": sum(r["funded"] * (r["n"] - r["funded"]) for r in contributing.values()),
            "top20_funded": sum(r["top20_funded"] for r in regions.values()),
            "top20_k": sum(r["top20_k"] for r in regions.values()), "regions": regions}


def score_order(order: list[str], outcomes: dict[str, int], region: dict[str, str] | None = None) -> dict:
    board = [{"item_id": item, "rank": i + 1} for i, item in enumerate(order)]
    bt = backtest(board, outcomes)
    out = {"auc": bt["auc"], "auc_lo": bt["auc_lo"], "auc_hi": bt["auc_hi"], "top20": precision_top(order, outcomes)}
    if region:
        out["within_region"] = within_region(order, outcomes, region)
    return out


async def pointwise(items: list[Item], provider, key: str) -> dict[str, float]:
    path = CACHE / ("pointwise_%s_%s.json" % (getattr(provider, "model", "judge"), key))
    scores = json.loads(path.read_text()) if path.exists() else {}
    gate = asyncio.Semaphore(4)

    async def one(it: Item) -> None:
        if it.id in scores:
            return
        async with gate:
            c = await provider.complete("You assess grant applications.", "%s\n\n<application>\n%s\n</application>"
                                        % (POINTWISE, it.text))
        m = re.search(r"\d+(\.\d+)?", c.text)
        scores[it.id] = float(m.group()) if m else 0.0
        path.write_text(json.dumps(scores))

    await asyncio.gather(*(one(it) for it in items))
    return scores


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="claude_cli", help="claude_cli (default, model haiku) or mock")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=60, help="sample size; 0 takes every decided application")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--revision", choices=("first", "pre-decision"), default="first",
                    help="which revision the judge reads, when drawing a new manifest")
    ap.add_argument("--rubrics", default="fund,neutral", help="comma-separated: %s" % ", ".join(RUBRICS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample", action="store_true", help="draw a new manifest.json from the categories")
    ap.add_argument("--no-pointwise", action="store_true", help="skip the score-each-alone baseline")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    manifest_path: Path = args.manifest
    if args.resample or not manifest_path.exists():
        pop = population()
        manifest_path.write_text(json.dumps({"source": API, "fiscal_year": FY, "cycles": list(CYCLES), "seed": args.seed,
                                             "revision": args.revision,
                                             "population": {"decided": len(pop), "funded": sum(r["funded"] for r in pop)},
                                             "items": sample(args.n, args.seed, args.revision)}, indent=1) + "\n")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["items"]
    items = [Item(id=str(r["pageid"]), text=clean(revision_text(r["revid"])), title=r["title"].rsplit("/", 1)[-1])
             for r in rows]
    outcomes = {str(r["pageid"]): r["funded"] for r in rows}
    sample_key = hashlib.sha256(" ".join(sorted("%s@%s" % (r["pageid"], r["revid"]) for r in rows)).encode()).hexdigest()[:12]
    region = {str(r["pageid"]): r["region"] for r in rows}
    reviewed = {str(r["pageid"]) for r in rows if r.get("reviewed")}

    def reviewed_only(order: list[str]) -> dict:
        """The same scores on applications that passed the eligibility screen, where the decision is a judgement."""
        if not reviewed:
            return {}
        sub_order = [i for i in order if i in reviewed]
        return {"reviewed": {"n": len(sub_order), "funded": sum(outcomes[i] for i in sub_order),
                             **score_order(sub_order, {i: outcomes[i] for i in sub_order}, region)}}
    words = {it.id: len(it.text.split()) for it in items}
    read_at = [r.get("read_at") or r["submitted"] for r in rows]
    print("%d applications, %d funded; median %d words; revisions read from %s to %s" % (
        len(items), sum(outcomes.values()), int(np.median(list(words.values()))), min(read_at)[:10],
        max(read_at)[:10]), file=sys.stderr)

    spec = {"provider": args.judge, "model": args.model} if args.judge == "claude_cli" else {"provider": "mock"}
    provider = make_provider(spec)
    store = SQLiteStore(CACHE / "runs.db")
    report: dict = {"judge": spec, "engine": supposing.__version__, "n": len(items),
                    "funded": sum(outcomes.values()), "rankings": {}, "baselines": {}}
    for key in args.rubrics.split(","):
        rubric = RUBRICS[key]
        def progress(done: int, planned: int, key: str = key) -> None:
            print("\r%s: %d/%d judge calls" % (key, done, planned), end="", file=sys.stderr)

        ranking = await run_ranking(items, rubric, provider, RankConfig(), store,
                                    run_id="%s:%s:%s:%s" % (args.judge, args.model, key, sample_key),
                                    concurrency=args.concurrency, on_progress=progress)
        print(file=sys.stderr)
        order = [e["item_id"] for e in ranking.leaderboard]
        report["rankings"][key] = {"rubric": rubric, "calls": len(ranking.comparisons),
                                   "reliability": ranking.diagnostics["reliability"]["line"],
                                   **score_order(order, outcomes, region), **reviewed_only(order)}
    report["baselines"]["longer_first"] = {**score_order(sorted(words, key=lambda i: -words[i]), outcomes, region),
                                           **reviewed_only(sorted(words, key=lambda i: -words[i]))}
    if not args.no_pointwise:
        scores = await pointwise(items, provider, sample_key)
        by_score = sorted(scores, key=lambda i: (-scores[i], i))
        report["baselines"]["pointwise"] = {"calls": len(scores), **score_order(by_score, outcomes, region),
                                            **reviewed_only(by_score)}
    await provider.close()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
