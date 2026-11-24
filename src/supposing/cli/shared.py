"""What `rank` and `run` share: reading the run flags, asking before spending, the result document."""
from __future__ import annotations

import argparse
import sys
from typing import Any

from ..cloud import JudgeError, parse_judge
from ..rubric import Rubric
from .output import NEEDS_CONFIRMATION, USAGE, CliError, info

FREE_PROVIDERS = ("mock", "claude_cli")


def rubric_text(rubric: str | Rubric) -> str:
    """The rubric as the API takes it and the result shows it."""
    return rubric.text if isinstance(rubric, Rubric) else rubric


def cli_judge(args: argparse.Namespace, local: bool) -> dict[str, Any]:
    """``{provider, model, level}`` from --judge and --model; a usage error says which values work."""
    try:
        return parse_judge(args.judge, args.model, local=local)
    except JudgeError as exc:
        raise CliError(str(exc), code="invalid_judge", exit_code=USAGE,
                       hint="Try --judge quick (the default), balanced or thorough; `supposing models` lists the rest."
                       ) from exc


def judge_label(judge: dict[str, Any]) -> str:
    """quick (anthropic/claude-haiku-5.5), openai/gpt-6-luna, or claude_cli (haiku)."""
    if judge.get("level"):
        return "%s (%s)" % (judge["level"], judge["model"])
    if judge["provider"] == "openrouter":
        return judge["model"]
    return judge["provider"] if judge["model"] == judge["provider"] else "%s (%s)" % (judge["provider"], judge["model"])


def ranking_options(args: argparse.Namespace) -> dict[str, Any]:
    """The ranking settings the flags set; unset flags are left out so the defaults apply. ``both_orders`` is
    always sent, so a run judges each pair once unless --both-orders is given."""
    out: dict[str, Any] = {"rounds": args.rounds, "both_orders": args.both_orders, "pairing": args.pairing,
                           "max_chars": args.max_chars, "seed": args.seed}
    if args.top_n:
        out.update(goal="top_n", top_n=args.top_n)
    return {k: v for k, v in out.items() if v is not None}


def confirm(args: argparse.Namespace, estimate_line: str, free: bool, rerun_hint: str) -> None:
    """Show the estimate, then ask before anything is spent. Free runs and --yes go straight ahead."""
    if not args.json:
        info("Estimate: %s" % estimate_line)
    if args.yes or free:
        return
    if not sys.stdin.isatty() or args.json:
        raise CliError("This run would spend money: %s." % estimate_line, code="confirmation_required",
                       exit_code=NEEDS_CONFIRMATION, hint="Rerun with --yes to start it. %s" % rerun_hint)
    try:
        answer = input("Start it? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise CliError("Not started.", code="not_confirmed", exit_code=NEEDS_CONFIRMATION, hint=None)


def result_document(run: dict[str, Any], estimate: dict[str, Any] | None, results: dict[str, Any] | None,
                    skipped: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """What `rank --json`, `run --json` and `runs watch --json` print: the run, its estimate and its results,
    each in the API's own shape, and the files a folder upload left out (``[{path, reason}]``)."""
    return {"run": run, "estimate": estimate, "results": results, "skipped": skipped or []}
