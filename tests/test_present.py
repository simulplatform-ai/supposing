from supposing import present, rank


def test_results_payload_leads_with_the_summary_and_cuts_the_leaderboard():
    ranking = rank({"d%d" % i: "Doc %d [[q=%.1f]]" % (i, i / 10) for i in range(8)}, "Better?", provider="mock")
    res = {"run_id": "r1", "type": "ranking", "final": True, "summary": ranking.summary,
           "ranking": {"leaderboard": ranking.leaderboard, "diagnostics": ranking.diagnostics, "interval": "90%",
                       "n_bootstrap": ranking.n_bootstrap}}
    run = {"id": "r1", "type": "ranking", "status": "done", "planned_calls": 64, "progress": {"done": 64}}
    text, data = present.results_payload(res, run, top=3, texts={"d7": "x" * 500})
    assert text == ranking.summary
    assert len(data["ranking"]["leaderboard"]) == 3 and data["n_items"] == 8
    assert "summary" not in data and data["run"] is run
    assert all(len(r["text"]) <= 200 for r in data["ranking"]["leaderboard"])


def test_provisional_results_say_what_to_do_next():
    res = {"type": "ranking", "final": False, "summary": "So far, A ranks first.", "ranking": {"leaderboard": []}}
    run = {"id": "r2", "type": "ranking", "status": "running", "planned_calls": 40, "progress": {"done": 10}}
    text, _ = present.results_payload(res, run, top=5, waited=30)
    assert text == ("So far, A ranks first. Provisional: 10 of 40 judge calls done after 30s; call "
                    "get_results(run_id=r2) in a little while.")


def test_run_line_and_errors():
    run = {"id": "r3", "type": "ranking", "status": "running", "planned_calls": 40, "ranking": {"rounds": 8},
           "progress": {"done": 12, "round": 2, "cost_usd": 0.01}}
    assert present.run_line(run).startswith("Run r3 is running: round 2 of 8, 12 of 40 judge calls done")
    assert "code: run_not_found" in present.error_text("run_not_found", "No such run")
    assert present.error_text("invalid_request", "bad", details=[{"field": "top", "message": "too big"}]) == (
        "Invalid arguments: top: too big. (code: invalid_request) Fix those arguments and call again.")


def test_estimate_line():
    assert present.estimate_line({"n_items": 10, "min_calls": 60, "max_calls": 160, "est_cost_usd": 0.5,
                                  "est_cost_usd_min": 0.2, "est_seconds": 120}) == (
        "10 items, 60 to 160 judge calls, $0.20 to $0.50, about 2 minutes")
    assert present.estimate_line({"n_items": 3, "max_calls": 12, "est_cost_usd": 0}) == "3 items, 12 judge calls, free"


def test_billed_estimate_shows_price_per_call_and_tier():
    est = {"n_items": 8, "min_calls": 8, "max_calls": 32, "est_cost_usd": 0.128, "est_cost_usd_min": 0.032,
           "billed": True, "price_per_call_usd": 0.004, "price_tier": "economy", "est_seconds": 12}
    assert present.estimate_line(est) == (
        "8 items, 8 to 32 judge calls, up to $0.13 of credit at $0.004 a call (economy tier), about 12 seconds")
    assert "$0.0125 a call" in present.estimate_line({**est, "price_per_call_usd": 0.0125})
