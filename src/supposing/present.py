"""Results worded for agents and terminals, from the API's own objects.

The CLI, the stdio MCP server and the hosted MCP server all call these, so a run reads the same everywhere:
one sentence first, then the data. Everything here takes and returns plain dicts and strings; turning them
into an MCP tool result is the server's job.
"""
from __future__ import annotations

from typing import Any

NOT_FOUND_HINT = ("Use the %s_id that rank_documents or compare_options returned. Runs and datasets are visible "
                  "only to the workspace that owns the API key, and in local mode only to this server process.")


def error_text(code: str, message: str, hint: str | None = None, details: list[dict[str, Any]] | None = None) -> str:
    """One line for the agent: what went wrong, the API's stable code, then what to do next, in MCP terms."""
    out = message.rstrip().rstrip(".") + "."
    fields = [d for d in details or [] if isinstance(d, dict) and d.get("field")]
    if fields and code == "invalid_request":
        out = "Invalid arguments: %s." % "; ".join("%s: %s" % (d["field"], d.get("message")) for d in fields)
        hint = "Fix those arguments and call again."
    elif code.endswith("_not_found") and code[:-10] in ("run", "dataset", "item", "comparison"):
        hint = NOT_FOUND_HINT % code[:-10]
    return "%s (code: %s)%s" % (out, code, " " + hint if hint else "")


def run_line(run: dict[str, Any]) -> str:
    """One sentence on where a run stands, from the API's Run object."""
    p = run.get("progress") or {}
    rid, status = run["id"], run["status"]
    counts = "%d of %s judge calls done, $%.4f spent" % (p.get("done", 0), run.get("planned_calls"),
                                                       float(p.get("cost_usd") or 0))
    if status == "queued":
        return "Run %s is queued; no judge calls have finished yet. Call get_run again in a few seconds." % rid
    if status == "running":
        return ("Run %s is running: round %d of %s, %s. get_results already shows a provisional ranking."
                % (rid, p.get("round", 0), (run.get(run["type"]) or {}).get("rounds"), counts))
    if status == "done":
        note = " %s" % run["note"] if run.get("note") else ""
        return "Run %s is done: %s.%s Call get_results(run_id) for the ranking." % (rid, counts, note)
    if status == "failed":
        return "Run %s failed: %s. Fix the cause and start a new run." % (rid, run.get("error") or "unknown error")
    return "Run %s was canceled after %s." % (rid, counts)


def results_payload(res: dict[str, Any], run: dict[str, Any], top: int, texts: dict[str, str] | None = None,
                    waited: int | None = None) -> tuple[str, dict[str, Any]]:
    """The API's Results object as-is (leaderboard cut to ``top`` rows), plus the Run and a reading guide.

    Returns the text line (the summary, plus what to do next while the run is going) and the data.
    """
    kind = res.get("type", "ranking")
    body = dict(res.get(kind) or {})
    board = body.get("leaderboard") or []
    rows = [dict(e, text=texts.get(str(e["item_id"]), "")[:200]) if texts is not None else e for e in board[:top]]
    diag = dict(body.get("diagnostics") or {})
    if isinstance(diag.get("upsets"), list):
        diag["upsets"] = diag["upsets"][:5]
    data = {k: v for k, v in res.items() if k != "summary"}
    data[kind] = {**body, "leaderboard": rows, "diagnostics": diag}
    data.update(n_items=len(board), run=run, how_to_read=(
        "rank_lo-rank_hi is the %s interval on rank: items whose intervals overlap may be tied. rating is on an "
        "Elo-like scale where only differences matter; wins and losses count decided judge calls."
        % body.get("interval", "90%")))
    text = res["summary"]
    if not res.get("final"):
        p = run.get("progress") or {}
        text += " Provisional: %d of %s judge calls done" % (p.get("done", 0), run.get("planned_calls"))
        text += ("%s; call get_results(run_id=%s) in a little while." % (
            " after %ds" % waited if waited is not None else "", run["id"]))
    return text, data


def _money(usd: float, exact: bool = False) -> str:
    if exact and usd < 0.1:
        return "$%.3f" % usd if round(usd, 3) == usd else "$%.4f" % usd
    return "$%.2f" % usd if usd >= 0.01 else "under $0.01"


def _duration(seconds: float) -> str:
    if seconds < 90:
        n = max(1, round(seconds))
        return "%d second%s" % (n, "" if n == 1 else "s")
    if seconds < 5400:
        return "%d minutes" % round(seconds / 60)
    return "%.1f hours" % (seconds / 3600)


def estimate_line(est: dict[str, Any]) -> str:
    """Items, calls, money and time in one line, from the API's Estimate object. The one wording for the CLI
    and both MCP servers: billed runs show the credit at the per-call price and tier."""
    lo, hi = est.get("min_calls"), est["max_calls"]
    calls = "%d judge calls" % hi if lo in (0, None, hi) else "%d to %d judge calls" % (lo, hi)
    top, low = est.get("est_cost_usd"), est.get("est_cost_usd_min")
    if top is None:
        cost = "cost unknown for this model"
    elif est.get("billed") and est.get("price_per_call_usd") is not None:
        cost = "up to %s of credit at %s a call (%s tier)" % (
            _money(top), _money(est["price_per_call_usd"], exact=True), est.get("price_tier") or "?")
    elif top == 0:
        cost = "free"
    elif low is not None and _money(low) != _money(top):
        cost = "%s to %s" % (_money(low), _money(top))
    else:
        cost = "up to %s" % _money(top)
    line = "%d items, %s, %s" % (est["n_items"], calls, cost)
    if est.get("est_seconds"):
        line += ", about %s" % _duration(est["est_seconds"])
    return line
