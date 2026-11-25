"""Results in plain words: how far the judge can be trusted, and the verdict sentence.

Computed here once, so the command line, the API, the MCP tools and the web app all say the same thing.
"""
from __future__ import annotations

import bisect
from typing import Any, Literal

Grade = Literal["good", "caution", "poor"]


def _grade(good: bool, ok: bool) -> Grade:
    return "good" if good else "caution" if ok else "poor"


def _pct(x: float | None) -> str:
    return "-" if x is None else "%d%%" % round(100 * x)


def judge_checks(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Grade the judge from a run's diagnostics: ``{"grade", "line", "checks"}``.

    Each check has ``key``, ``label``, ``value`` (the number, or None when not measured yet), ``shown`` (the
    number as text), ``grade`` (``"good"``, ``"caution"``, ``"poor"`` or None), ``note`` (what the number
    means) and ``warning`` (one clause to say when the grade is poor). The ``order`` check is present only
    when pairs were judged in both orders. ``grade`` is the worst graded check, and
    ``line`` says it in one plain sentence. Thresholds are rules of thumb, not statistical tests.
    """
    d = diagnostics
    order = d.get("order_consistency")
    first = d.get("first_position_win_rate")
    split = d.get("split_half_reliability")
    length = d.get("length_correlation")
    agreement = d.get("ranking_agreement")
    first_both = d.get("first_shown_won_both")
    second_both = d.get("second_shown_won_both")
    bias = None if first is None else abs(first - 0.5)
    # A gap from 50% that chance alone gives often is not graded: on 16 verdicts a fair judge is off by 25 points
    # one time in twenty. 1.96 and 2.58 standard errors are the 95% and 99% two-sided bounds.
    se = 0.5 / max(1, d.get("decided") or 0) ** 0.5

    checks = []
    if order is not None:  # only measured when each pair is judged in both orders
        order_note = "Across %d pairs. The rest flipped when the texts swapped places, and count as draws." % (
            d.get("pairs_both_orders") or 0)
        if first_both is not None and second_both is not None and order < 1:
            order_note += " In %s of pairs the first-shown text won both times; in %s the second-shown did." % (
                _pct(first_both), _pct(second_both))
        checks.append({"key": "order", "label": "Same winner in both orders", "value": order, "shown": _pct(order),
                       "grade": _grade(order >= 0.85, order >= 0.7), "note": order_note,
                       "warning": "the judge often changed its mind when the two texts swapped places"})
    checks += [
        {"key": "position", "label": "First-shown text wins", "value": first, "shown": _pct(first),
         "grade": None if bias is None else _grade(bias <= 0.05 or bias < 1.96 * se, bias <= 0.12 or bias < 2.58 * se),
         "note": "50% means no position bias. Each text is shown first about half the time, so the bias "
                 "cancels out of the ranking, but verdicts swayed by position carry less information.",
         "warning": "the judge strongly favoured whichever text it read %s" % (
             "first" if (first or 0.5) >= 0.5 else "second")},
        {"key": "split", "label": "Split-half reliability", "value": split,
         "shown": "-" if split is None else "%.2f" % split,
         "grade": None if split is None else _grade(split >= 0.8, split >= 0.6),
         "note": "Needs more comparisons." if split is None else
                 "Rank correlation between two rankings built from random halves of the comparisons. Low means: "
                 "add rounds.",
         "warning": "rankings built from two halves of the comparisons disagree, so more rounds would help"},
        {"key": "agreement", "label": "Verdicts the ranking explains", "value": agreement, "shown": _pct(agreement),
         "grade": None,
         "note": "Share of verdicts where the higher-rated item won. The rest are upsets: noise, or preferences "
                 "that don't fit one scale.",
         "warning": ""},
        {"key": "length", "label": "Length vs rating", "value": length,
         "shown": "-" if length is None else "%+.2f" % length,
         # With fewer than 10 items a strong rank correlation happens by chance too often to grade.
         "grade": None if length is None or (d.get("n_items") or 0) < 10 else _grade(abs(length) < 0.3,
                                                                                   abs(length) < 0.5),
         "note": "Rank correlation between word count and rating. A strong value may be real, or the judge "
                 "rewarding %s texts." % ("shorter" if (length or 0) < 0 else "longer"),
         "warning": "ratings follow length closely, which may mean the judge rewards %s texts" % (
             "shorter" if (length or 0) < 0 else "longer")},
    ]
    graded = [c for c in checks if c["grade"]]
    poor = [c for c in graded if c["grade"] == "poor"]
    caution = [c for c in graded if c["grade"] == "caution"]
    if not graded:
        grade, line = None, "Too few comparisons to check the judge yet."
    elif poor:
        grade = "poor"
        more = len(poor) - 1
        line = "Treat with care: %s (%s %s%s). Read some comparisons before relying on this ranking." % (
            poor[0]["warning"], poor[0]["label"].lower(), poor[0]["shown"],
            "" if not more else "; %d more check%s poor" % (more, " is" if more == 1 else "s are"))
    elif caution:
        grade = "caution"
        line = "Mostly consistent, with %d caution%s: %s." % (
            len(caution), "" if len(caution) == 1 else "s",
            ", ".join("%s %s" % (c["label"].lower(), c["shown"]) for c in caution))
    else:
        grade = "good"
        line = "The judge was consistent: %s." % ", ".join(
            "%s %s" % (c["label"].lower(), c["shown"]) for c in graded)
    return {"grade": grade, "line": line, "checks": checks}


def tiers(leaderboard: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a best-first leaderboard into groups the judge cannot reliably order within.

    A new tier starts before a row when every row above it has a rank range ending before every rank range
    from that row down begins. A tier of one item is settled in its place.
    """
    n = len(leaderboard)
    suffix_min = [0] * (n + 1)
    suffix_min[n] = n + 1
    for i in range(n - 1, -1, -1):
        suffix_min[i] = min(suffix_min[i + 1], leaderboard[i]["rank_lo"])
    out: list[list[dict[str, Any]]] = []
    top = 0
    for i, row in enumerate(leaderboard):
        if not out or top < suffix_min[i]:
            out.append([])
        out[-1].append(row)
        top = max(top, row["rank_hi"])
    return out


IN, OUT = 0.95, 0.05  # a top-N place is settled in at >= 95% and settled out at <= 5%


def display_groups(leaderboard: list[dict[str, Any]], top_n: int | None = None, live: bool = False
                   ) -> list[dict[str, Any]]:
    """The leaderboard in display groups, best first: ``[{key, label, tier, close, item_ids}]``.

    Top-N runs (``top_n`` below the item count, with ``p_top_n`` on every row) get at most three groups, in
    rank order. **Close calls** is the contiguous span of ranks from the first to the last item whose chance
    of a top-N place is between 5% and 95%. **Settled** is every item above that span (each has a chance of
    at least 95%). Everything below the span is one **out** group. With no item in between, settled is the
    leading run of items at 95% or more and the rest are out. Labels read "2 settled in the top 5",
    "3 places left · 5 contenders" and "33 out".

    Full rankings get one group per clear tier (``tiers``): a new group starts where every rank range above
    ends before every rank range below begins. A set the judge cannot separate is honestly one group.

    ``tier`` is a colour step from 1 (best) to 4; ``close`` marks the contested group.

    ``live=True`` is for a run still going, whose groups will change: a full ranking gets at most three groups,
    "1st so far" (or "Top 2 so far" when the judge cannot yet separate them), "Close behind" (the next tier)
    and "Still sorting" (every tier below, as one group); a top-N run's labels become "2 in the top 5 so far",
    "Close behind" and "Still sorting". Keys and tiers are those of the finished groups.
    """
    n = len(leaderboard)
    if not n:
        return []
    if top_n and top_n < n and all(r.get("p_top_n") is not None for r in leaderboard):
        open_ = [i for i, r in enumerate(leaderboard) if OUT < r["p_top_n"] < IN]
        if open_:
            start, end = open_[0], open_[-1] + 1
        else:
            start = next((i for i, r in enumerate(leaderboard) if r["p_top_n"] < IN), n)
            end = start
        settled, close, out = leaderboard[:start], leaderboard[start:end], leaderboard[end:]
        left = max(0, top_n - len(settled))
        groups = [
            ("settled", "%d in the top %d so far" % (len(settled), top_n) if live else
             "%d settled in the top %d" % (len(settled), top_n), 1, False, settled),
            ("close", "Close behind" if live else "%d place%s left · %d contender%s" % (
                left, "" if left == 1 else "s", len(close), "" if len(close) == 1 else "s"), 2, True, close),
            ("out", "Still sorting" if live else "%d out" % len(out), 3, False, out),
        ]
    else:
        found = tiers(leaderboard)
        if live and len(found) > 3:
            found = found[:2] + [[r for rows in found[2:] for r in rows]]
        k = len(found)
        groups = []
        for i, rows in enumerate(found):
            lo, hi = rows[0]["rank"], rows[-1]["rank"]
            if live:
                label = (("1st so far" if len(rows) == 1 else "Top %d so far" % len(rows)) if i == 0 else
                         "Close behind" if i == 1 else "Still sorting")
            else:
                label = "Rank %d" % lo if lo == hi else "Ranks %d-%d" % (lo, hi)
            groups.append(("tier-%d" % (i + 1), label, 2 if k == 1 else 1 + round(3 * i / (k - 1)), False, rows))
    return [{"key": key, "label": label, "tier": tier, "close": close, "item_ids": [r["item_id"] for r in rows]}
            for key, label, tier, close, rows in groups if rows]


def _name(row: dict[str, Any]) -> str:
    return str(row.get("title") or row["item_id"])


def _names(rows: list[dict[str, Any]]) -> str:
    names = [_name(r) for r in rows]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def summarize(leaderboard: list[dict[str, Any]], diagnostics: dict[str, Any], final: bool = True,
              top_n: int | None = None, caution: bool = True) -> str:
    """The verdict in plain words, for showing above the leaderboard.

    Says how sure the ranking is without bare percentages: who is first and whether that is clear, or for a
    top-N run which places are settled; then how many clear tiers there are. ``final=False`` prefixes "So far,".
    When a judge check is poor the sentence never says "clearly", and with ``caution`` it adds the warning
    (``diagnostics["reliability"]["line"]``): tight intervals from an inconsistent judge are not trustworthy.
    """
    if not leaderboard or not diagnostics.get("decided"):
        return "No calls have been decided yet, so there is no ranking."
    n = len(leaderboard)
    groups = tiers(leaderboard)
    top = leaderboard[0]
    reliability = diagnostics.get("reliability") or judge_checks(diagnostics)
    trusted = reliability["grade"] != "poor"  # the intervals measure sampling noise, not the judge's quality
    phrase = True  # the sentence starts with our words, not an item's name
    # Mostly tied: fewer than one pair in five is reliably ordered (their rank ranges do not overlap).
    los = sorted(r["rank_lo"] for r in leaderboard)
    ordered = sum(n - bisect.bisect_right(los, r["rank_hi"]) for r in leaderboard)
    shown_groups = display_groups(leaderboard, top_n)
    by_key = {g["key"]: g for g in shown_groups}
    top_n_run = top_n is not None and not shown_groups[0]["key"].startswith("tier-")
    if top_n_run and top_n:  # say exactly what the groups say
        settled = by_key.get("settled", {}).get("item_ids", [])
        contenders = len(by_key.get("close", {}).get("item_ids", []))
        rows = {r["item_id"]: r for r in leaderboard}
        if not contenders and len(settled) >= top_n:
            shown = _names([rows[i] for i in settled[:top_n]]) if top_n <= 5 else "see below"
            sentence = "The top %d are settled: %s." % (top_n, shown)
        elif not settled:
            sentence = "None of the top %d places is settled yet; %d items are contesting them." % (
                top_n, contenders)
        else:
            left = top_n - len(settled)
            sentence = "%d of the top %d %s settled; %d item%s contesting the last %d place%s." % (
                len(settled), top_n, "is" if len(settled) == 1 else "are", contenders,
                " is" if contenders == 1 else "s are", left, "" if left == 1 else "s")
    elif n > 2 and ordered < 0.2 * n * (n - 1) / 2:
        sentence = "The judge can't reliably separate most of these items. Add rounds or sharpen the rubric."
    elif top["rank_hi"] == 1:
        phrase = False
        sentence = "%s is first, clearly." % _name(top) if trusted else "%s ranks first." % _name(top)
    else:
        phrase = False
        k = sum(r["rank_lo"] == 1 for r in leaderboard)
        sentence = "%s ranks first, but any of the top %d could be first." % (_name(top), max(k, 2))
    if len(groups) > 1 and not sentence.startswith("The judge can't") and not top_n_run:
        sentence += " %d clear tiers." % len(groups)
    few = sum(1 for r in leaderboard if r.get("few_verdicts"))
    if few and few < n:
        sentence += " %d item%s too few verdicts to place yet." % (few, " has" if few == 1 else "s have")
    if not final:
        sentence = "So far, " + (sentence[0].lower() + sentence[1:] if phrase else sentence)
    if caution and not trusted:
        sentence += " " + reliability["line"]
    return sentence
