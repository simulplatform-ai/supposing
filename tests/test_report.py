import numpy as np
import pytest
from test_statistics import judge_run

from supposing import RankConfig, results
from supposing.report import judge_checks

CLEAN = {"decided": 200, "order_consistency": 0.92, "pairs_both_orders": 100, "first_position_win_rate": 0.51,
         "first_shown_won_both": 0.05, "second_shown_won_both": 0.03, "split_half_reliability": 0.9,
         "ranking_agreement": 0.85, "length_correlation": 0.1, "n_items": 20}


def by_key(report):
    return {c["key"]: c for c in report["checks"]}


def test_consistent_judge_is_good():
    report = judge_checks(CLEAN)
    assert report["grade"] == "good"
    assert report["line"].startswith("The judge was consistent")
    assert by_key(report)["order"]["shown"] == "92%"


def test_position_gaps_within_chance_are_not_poor():
    """16 verdicts from a fair judge are 25 points off 50% one time in twenty; 200 verdicts almost never."""
    few = by_key(judge_checks({**CLEAN, "decided": 16, "first_position_win_rate": 0.25}))["position"]
    many = by_key(judge_checks({**CLEAN, "decided": 200, "first_position_win_rate": 0.25}))["position"]
    assert few["grade"] == "caution" and many["grade"] == "poor"
    assert by_key(judge_checks({**CLEAN, "decided": 16, "first_position_win_rate": 0.4}))["position"]["grade"] == "good"


def test_one_order_runs_have_no_order_check():
    report = judge_checks({**CLEAN, "order_consistency": None, "pairs_both_orders": 0})
    assert "order" not in by_key(report)
    assert report["grade"] == "good" and "both orders" not in report["line"]


def test_nothing_measured_yet():
    report = judge_checks({"decided": 0})
    assert report["grade"] is None
    assert all(c["grade"] is None for c in report["checks"])


@pytest.mark.parametrize(("key", "value", "grade"), [
    ("order_consistency", 0.85, "good"), ("order_consistency", 0.75, "caution"), ("order_consistency", 0.6, "poor"),
    ("first_position_win_rate", 0.45, "good"), ("first_position_win_rate", 0.4, "caution"),
    ("first_position_win_rate", 0.7, "poor"),
    ("split_half_reliability", 0.7, "caution"), ("split_half_reliability", 0.3, "poor"),
    ("length_correlation", -0.4, "caution"), ("length_correlation", 0.6, "poor"),
])
def test_thresholds(key, value, grade):
    report = judge_checks({**CLEAN, key: value})
    assert report["grade"] == grade
    if grade == "poor":
        assert report["line"].startswith("Treat with care: ")


def test_length_is_not_graded_on_few_items():
    assert by_key(judge_checks({**CLEAN, "length_correlation": 0.9, "n_items": 6}))["length"]["grade"] is None


def test_several_poor_checks_name_the_first_and_count_the_rest():
    report = judge_checks({**CLEAN, "order_consistency": 0.4, "first_position_win_rate": 0.8})
    assert report["line"] == ("Treat with care: the judge often changed its mind when the two texts swapped "
                              "places (same winner in both orders 40%; 1 more check is poor). Read some "
                              "comparisons before relying on this ranking.")


def test_planted_position_bias_is_reported():
    """A judge with a first-position bias of 1.5 units, on items close in quality: most both-order pairs
    flip toward the first-shown text, and the report says so."""
    items, comps = judge_run(np.linspace(0, 1, 12), RankConfig(rounds=8, both_orders=True), seed=0, bias=1.5)
    diag = results(items, comps)["diagnostics"]
    assert diag["first_shown_won_both"] > 0.4
    assert diag["second_shown_won_both"] < 0.1
    reliability = diag["reliability"]
    assert reliability["grade"] == "poor"
    assert by_key(reliability)["position"]["grade"] == "poor"
    assert "first-shown text won both times" in by_key(reliability)["order"]["note"]


def test_unbiased_judge_flips_both_ways_equally():
    items, comps = judge_run(np.linspace(0, 1, 12), RankConfig(rounds=8, both_orders=True), seed=0)
    diag = results(items, comps)["diagnostics"]
    assert abs(diag["first_shown_won_both"] - diag["second_shown_won_both"]) < 0.1
    assert by_key(diag["reliability"])["position"]["grade"] == "good"


# The verdict sentence ------------------------------------------------------------------------------

from supposing import rank, summarize, tiers  # noqa: E402


def row(item, lo, hi, p=None):
    out = {"item_id": item, "title": item, "rank": lo, "rank_lo": lo, "rank_hi": hi}
    if p is not None:
        out["p_top_n"] = p
    return out


def test_tiers_break_where_rank_ranges_stop_overlapping():
    board = [row("a", 1, 1), row("b", 2, 3), row("c", 2, 4), row("d", 3, 4), row("e", 5, 5)]
    assert [[r["item_id"] for r in t] for t in tiers(board)] == [["a"], ["b", "c", "d"], ["e"]]


def test_clear_winner():
    board = [row("Alpha", 1, 1), row("Beta", 2, 3), row("Gamma", 2, 3), row("Delta", 4, 5), row("Eps", 4, 5)]
    assert summarize(board, CLEAN) == "Alpha is first, clearly. 3 clear tiers."


def test_contested_first_place():
    board = [row("Alpha", 1, 2), row("Beta", 1, 3), row("Gamma", 2, 3), row("Delta", 4, 4), row("Eps", 5, 5)]
    assert summarize(board, CLEAN) == "Alpha ranks first, but any of the top 2 could be first. 3 clear tiers."


def test_mostly_tied():
    board = [row(x, 1, 5) for x in "abcde"]
    assert summarize(board, CLEAN) == ("The judge can't reliably separate most of these items. Add rounds or "
                                       "sharpen the rubric.")


def test_top_n_settled_and_contested():
    board = [row("a", 1, 1, 1.0), row("b", 2, 2, 0.99), row("c", 3, 5, 0.6), row("d", 3, 6, 0.5),
             row("e", 4, 6, 0.3), row("f", 6, 6, 0.01)]
    assert summarize(board, CLEAN, top_n=3).startswith("2 of the top 3 are settled; 3 items are contesting the "
                                                       "last 1 place.")
    settled = [row("a", 1, 1, 1.0), row("b", 2, 2, 0.99), row("c", 3, 3, 0.97), row("d", 4, 6, 0.02),
               row("e", 4, 6, 0.0), row("f", 4, 6, 0.0)]
    assert summarize(settled, CLEAN, top_n=3) == "The top 3 are settled: a, b and c."  # groups, not tiers


def test_provisional_and_caution():
    board = [row("Alpha", 1, 1), row("Beta", 2, 2)]
    poor = {**CLEAN, "order_consistency": 0.4}
    assert summarize(board, poor, final=False) == (
        "So far, Alpha ranks first. 2 clear tiers. Treat with care: the judge often changed its mind when "
        "the two texts swapped places (same winner in both orders 40%). Read some comparisons before relying on "
        "this ranking.")
    assert summarize(board, poor, caution=False) == "Alpha ranks first. 2 clear tiers."
    assert summarize(board, CLEAN) == "Alpha is first, clearly. 2 clear tiers."
    assert summarize([row(x, 1, 5) for x in "abcde"], CLEAN, final=False).startswith("So far, the judge can't")


def test_nothing_decided():
    assert summarize([], {"decided": 0}) == "No calls have been decided yet, so there is no ranking."


def test_ranking_carries_summary_and_interval():
    ranking = rank({"a": "[[q=0.9]] a", "b": "[[q=0.1]] b", "c": "[[q=0.5]] c"}, "Better?", provider="mock")
    assert ranking.summary == summarize(ranking.leaderboard, ranking.diagnostics)
    assert ranking.interval == "90%" and ranking.n_bootstrap > 0
