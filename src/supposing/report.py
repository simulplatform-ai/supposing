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
    means) and ``warning`` (one clause to say when the grade is poor). ``grade`` is the worst graded check, and
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

    if order is None:
        order_note = "Only measured when each pair is judged in both orders."
    else:
        order_note = "Across %d pairs. The rest flipped when the texts swapped places, and count as draws." % (
            d.get("pairs_both_orders") or 0)
        if first_both is not None and second_both is not None and order < 1:
            order_note += " In %s of pairs the first-shown text won both times; in %s the second-shown did." % (
                _pct(first_both), _pct(second_both))
    checks = [
        {"key": "order", "label": "Same winner in both orders", "value": order, "shown": _pct(order),
         "grade": None if order is None else _grade(order >= 0.85, order >= 0.7), "note": order_note,
         "warning": "the judge often changed its mind when the two texts swapped places"},
        {"key": "position", "label": "First-shown text wins", "value": first, "shown": _pct(first),
         "grade": None if bias is None else _grade(bias <= 0.05, bias <= 0.12),
         "note": "50% means no position bias. Judging both orders cancels the bias out of the ranking, but "
                 "verdicts swayed by position carry less information.",
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
    if n > 2 and ordered < 0.2 * n * (n - 1) / 2:
        sentence = "The judge can't reliably separate most of these items. Add rounds or sharpen the rubric."
    elif top_n and top_n < n and all(r.get("p_top_n") is not None for r in leaderboard):
        n_top = top_n
        settled = [r for r in leaderboard if r["p_top_n"] >= 0.95]
        contested = [r for r in leaderboard if 0.05 < r["p_top_n"] < 0.95]
        if len(settled) >= n_top:
            shown = _names(settled[:n_top]) if n_top <= 5 else "see below"
            sentence = "The top %d are settled: %s." % (n_top, shown)
        elif not settled:
            sentence = "None of the top %d places is settled yet; %d items are contesting them." % (
                n_top, len(contested))
        else:
            sentence = "%d of the top %d are settled; %d items are contesting the last %d place%s." % (
                len(settled), n_top, len(contested), n_top - len(settled), "" if n_top - len(settled) == 1 else "s")
    elif top["rank_hi"] == 1:
        phrase = False
        sentence = "%s is first, clearly." % _name(top) if trusted else "%s ranks first." % _name(top)
    else:
        phrase = False
        k = sum(r["rank_lo"] == 1 for r in leaderboard)
        sentence = "%s ranks first, but any of the top %d could be first." % (_name(top), max(k, 2))
    if len(groups) > 1 and not sentence.startswith("The judge can't"):
        sentence += " %d clear tiers." % len(groups)
    if not final:
        sentence = "So far, " + (sentence[0].lower() + sentence[1:] if phrase else sentence)
    if caution and not trusted:
        sentence += " " + reliability["line"]
    return sentence
