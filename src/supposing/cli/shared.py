"""What `rank` and `run` share: reading the run flags, asking before spending, the result document."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
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


def judge_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """--context, --examples and --system as rank()'s context, examples and prompt; usage errors say what to fix."""
    from ..judge import JudgePrompt, load_examples  # only when a run needs them
    from ..ranking import ConfigError

    out: dict[str, Any] = {}
    if getattr(args, "context", None):
        path = Path(args.context)
        if path.suffix.lower() in (".md", ".txt") and not path.is_file():
            raise CliError("Can't read the context file %s." % path, code="invalid_file", exit_code=USAGE,
                           hint="Pass the path of a .md or .txt file, or the context itself in quotes.")
        out["context"] = path.read_text(encoding="utf-8") if path.suffix.lower() in (".md", ".txt") else args.context
    try:
        if getattr(args, "examples", None):
            out["examples"] = load_examples(args.examples)
        if getattr(args, "system", None):
            out["prompt"] = JudgePrompt.load(args.system)
    except OSError as exc:
        raise CliError("Can't read %s (%s)." % (exc.filename, exc.strerror), code="invalid_file", exit_code=USAGE,
                       hint="Check the path.") from exc
    except (ConfigError, ValueError) as exc:
        flag = "--system" if "prompt" in str(exc) or "placeholder" in str(exc) else "--examples"
        raise CliError(str(exc).rstrip(".") + ".", code="invalid_request", exit_code=USAGE,
                       hint="See `supposing rank --help` for the %s format." % flag) from exc
    return out


def ranking_options(args: argparse.Namespace) -> dict[str, Any]:
    """The ranking settings the flags set; unset flags are left out so the defaults apply. ``both_orders`` is
    always sent, so a run judges each pair once unless --both-orders is given."""
    out: dict[str, Any] = {"rounds": args.rounds, "both_orders": args.both_orders, "pairing": args.pairing,
                           "max_chars": args.max_chars, "seed": args.seed}
    if args.top_n:
        out.update(goal="top_n", top_n=args.top_n)
    return {k: v for k, v in out.items() if v is not None}


COMPARISON_FIELDS = ("id", "round", "item_a", "item_b", "title_a", "title_b", "status", "winner_item", "reasoning",
                     "error")


def comparison_row(c: Any, titles: Mapping[str, str | None]) -> dict[str, Any]:
    """A local judge call in the API's comparison shape: item_a was shown first."""
    return {"id": c.id, "round": c.round, "item_a": c.item_a, "item_b": c.item_b,
            "title_a": titles.get(c.item_a) or c.item_a, "title_b": titles.get(c.item_b) or c.item_b,
            "status": c.status, "winner_item": c.winner_item, "reasoning": c.reasoning, "error": c.error}


def print_comparisons(rows: list[dict[str, Any]]) -> None:
    """Each judge call: who met whom, the winner, and the judge's reasoning."""
    if not rows:
        print("No calls yet.")
    for c in rows:
        won = c["title_a"] if c["winner_item"] == c["item_a"] else c["title_b"] if c["winner_item"] == c["item_b"] \
            else None
        print("round %d · %s vs %s · %s" % (c["round"], c["title_a"], c["title_b"],
                                            "winner %s" % won if won else c["status"]))
        if c.get("reasoning"):
            print("  " + c["reasoning"].strip().replace("\n", "\n  "))
        if c.get("error"):
            print("  error: %s" % c["error"])


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
                    skipped: list[dict[str, str]] | None = None,
                    comparisons: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """What `rank --json`, `run --json` and `runs watch --json` print: the run, its estimate and its results,
    each in the API's own shape, and the files a folder upload left out (``[{path, reason}]``). With
    --with-comparisons, also every judge call with its reasoning."""
    out = {"run": run, "estimate": estimate, "results": results, "skipped": skipped or []}
    if comparisons is not None:
        out["comparisons"] = comparisons
    return out
