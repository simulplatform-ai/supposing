"""Printing: JSON for programs on stdout, tables and progress for people, errors that say what to do next."""
from __future__ import annotations

import json
import sys
from typing import Any

# Exit codes. Documented in `supposing --help` and docs/cli.md; keep the three in step.
OK = 0
RUN_FAILED = 1  # the run finished but failed or was canceled
USAGE = 2  # bad arguments, or a folder with nothing to rank
AUTH = 3  # no API key, or the key was refused
NOT_FOUND = 4
REFUSED = 5  # the API refused the request (limits, provider unavailable, invalid input)
UNAVAILABLE = 6  # could not reach the API, or it failed
NEEDS_CONFIRMATION = 7  # a run would spend money and --yes was not given
NEEDS_CREDIT = 8  # the workspace's credit doesn't cover the run: top up, lower the cost, or use your own key
INTERRUPTED = 130

EXIT_CODES_HELP = """exit codes:
  0  success
  1  the run finished but failed or was canceled
  2  usage error: bad arguments, or nothing to rank
  3  no API key, or the key was refused (run `supposing login`)
  4  not found
  5  the API refused the request (see the error code and hint)
  6  could not reach the API, or it failed; safe to retry
  7  the run would spend money: rerun with --yes to confirm
  8  not enough credit: `supposing billing topup`, or use your own OpenRouter key
  130 interrupted (a cloud run keeps going; `supposing runs watch <id>`)"""


def plural(n: int, word: str) -> str:
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


class CliError(Exception):
    def __init__(self, message: str, *, code: str, exit_code: int, hint: str | None = None,
                 extra: dict[str, Any] | None = None):
        super().__init__(message)
        self.message, self.code, self.exit_code, self.hint = message, code, exit_code, hint
        self.extra = extra or {}  # more fields for the JSON error, e.g. topup_url


def emit_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def info(msg: str = "") -> None:
    """For people only: goes to stderr, so stdout stays clean for pipes and --json."""
    print(msg, file=sys.stderr)


def error(err: CliError, as_json: bool) -> int:
    if as_json:
        emit_json({"error": {"code": err.code, "message": err.message, "hint": err.hint, **err.extra}})
    else:
        info("error: %s" % err.message)
        if err.hint:
            info("  %s" % err.hint)
        if err.extra.get("topup_command"):
            info("  Add credit now: %s" % err.extra["topup_command"])
        if err.extra.get("topup_url") and err.extra["topup_url"] not in (err.hint or ""):
            info("  Or open %s" % err.extra["topup_url"])
    return err.exit_code


class ProgressLine:
    """A one-line progress counter on stderr, redrawn in place on a terminal and silent otherwise."""

    def __init__(self) -> None:
        self.live = sys.stderr.isatty()
        self.shown = False

    def update(self, text: str) -> None:
        if self.live:
            print("\r\033[K" + text, end="", file=sys.stderr, flush=True)
            self.shown = True

    def end(self) -> None:
        if self.shown:
            print(file=sys.stderr)
            self.shown = False


def table(rows: list[list[str]], header: list[str], right: set[int] | None = None) -> str:
    right = right or set()
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]

    def line(r: list[str]) -> str:
        cells = [str(c).rjust(w) if i in right else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))]
        return "  ".join(cells).rstrip()

    return "\n".join([line(header), *(line(r) for r in rows)])


def _item_label(e: dict[str, Any]) -> str:
    """The title, and your id for the item when it says something the title doesn't."""
    title = str(e.get("title") or e["item_id"])
    sid = e.get("source_id")
    return "%s [%s]" % (title, sid) if sid and sid != title and sid not in title else title


def print_results(res: dict[str, Any]) -> None:
    """The human view of a Results body: the verdict sentence, then the leaderboard, then what intervals mean."""
    body = res.get(res.get("type") or "ranking") or {}
    board = body.get("leaderboard") or []
    print(res.get("summary") or "")
    if not board:
        return
    interval = body.get("interval") or "90%"
    print()
    rows = []
    for e in board:
        lo, hi = e.get("rank_lo", e["rank"]), e.get("rank_hi", e["rank"])
        rows.append([str(e["rank"]), str(lo) if lo == hi else "%d-%d" % (lo, hi), "%.0f" % e["rating"],
                     "%.0f to %.0f" % (e["rating_lo"], e["rating_hi"]),
                     "%d-%d" % (e.get("wins", 0), e.get("losses", 0)), _item_label(e)])
    text = table(rows, ["rank", "could be", "rating", interval + " interval", "won-lost", "item"], right={0, 2})
    groups = body.get("groups") or []
    if len(groups) > 1:  # the engine's display groups (settled / close calls / out, or clear tiers)
        lines = text.split("\n")
        starts = {}
        index = {e["item_id"]: i for i, e in enumerate(board)}
        for g in groups:
            if g["item_ids"] and g["item_ids"][0] in index:
                starts[index[g["item_ids"][0]]] = g["label"]
        text = "\n".join([lines[0]] + [part for i, line in enumerate(lines[1:])
                                         for part in ((["", "-- %s" % starts[i]] if i else ["-- %s" % starts[i]]) if i in starts else []) + [line]])
    print(text)
    print()
    print("\"could be\" is the range of ranks each item holds with %s confidence. Items whose ranges overlap "
          "are not reliably different." % interval)
    diag = body.get("diagnostics") or {}
    reliability = diag.get("reliability") or {}
    if reliability.get("line"):  # the engine's graded judge checks, worded once for every surface
        if reliability["line"] not in (res.get("summary") or ""):  # the summary may already carry the warning
            print("Judge: %s" % reliability["line"])
        return
    checks = []
    if diag.get("order_consistency") is not None:
        checks.append("same winner in both orders %.0f%% of the time" % (100 * diag["order_consistency"]))
    if diag.get("first_position_win_rate") is not None:
        checks.append("item shown first won %.0f%%" % (100 * diag["first_position_win_rate"]))
    if checks:
        print("Judge checks: %s." % "; ".join(checks))
