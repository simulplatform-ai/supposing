"""The ``supposing`` command. ``supposing rank`` runs on this machine; ``supposing run`` runs in the cloud. Both
take the same flags and print the same result shape. Every command takes ``--json``."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from .. import __version__
from ..cloud import CloudError
from ..rubric import Rubric
from . import billing, cloud, local
from .output import (
    AUTH,
    EXIT_CODES_HELP,
    INTERRUPTED,
    NEEDS_CREDIT,
    NOT_FOUND,
    REFUSED,
    UNAVAILABLE,
    USAGE,
    CliError,
    error,
)

Handler = Callable[[argparse.Namespace], int]


class Parser(argparse.ArgumentParser):
    """argparse, except that with --json anywhere on the line a usage error is the usual JSON error object."""

    json_errors = False

    def error(self, message: str) -> NoReturn:
        if Parser.json_errors:
            raise CliError(message[0].upper() + message[1:] + ".", code="usage", exit_code=USAGE,
                           hint="See `%s --help`." % self.prog)
        super().error(message)


def _rubric(value: str) -> str | Rubric:
    """A rubric in plain words, or @file: Markdown, text or JSON read by ``Rubric.load`` (criteria and weights)."""
    if value.startswith("@"):
        try:
            return Rubric.load(value[1:])
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                "can't read the rubric file %s (%s). Create it, or pass the rubric in quotes: --rubric \"Which is "
                "clearer?\"" % (value[1:], exc.strerror)) from exc
        except ValueError as exc:
            raise argparse.ArgumentTypeError("%s is not a valid rubric: %s" % (value[1:], exc)) from exc
    return value


def _source_flags(p: argparse.ArgumentParser) -> None:
    """For a JSONL, JSON or CSV file: which fields make each item. Names match the API's import."""
    g = p.add_argument_group("reading a JSONL, JSON or CSV file (one item per row)")
    g.add_argument("--text-field", dest="text_fields", action="append", metavar="FIELD",
                   help="the field with the text; repeat to join several, each under a heading (default: a field "
                   "named text, output, response, answer or content)")
    g.add_argument("--id-field", metavar="FIELD", help="each row's id, kept in every output (default: a field named "
                   "id, else the row number)")
    g.add_argument("--title-field", metavar="FIELD", help="the title to show (default: the id)")
    g.add_argument("--metadata-field", dest="metadata_fields", action="append", metavar="FIELD",
                   help="keep this field on the item but never show it to the judge; repeatable")


def _run_flags(p: argparse.ArgumentParser) -> None:
    """Flags shared by `rank` (local) and `run` (cloud)."""
    p.add_argument("--rubric", type=_rubric, required=True,
                   help="what the judge should prefer, in plain words, or @file to read it from a file")
    p.add_argument("--provider", help="judge provider: openrouter (default), mock (free, fake), claude_cli; "
                   "locally also anthropic and openai_compat")
    p.add_argument("--model", help="judge model, e.g. anthropic/claude-haiku-5.5")
    p.add_argument("--rounds", type=int, help="opponents per item (default 8); more rounds, tighter intervals")
    p.add_argument("--top-n", type=int, help="only settle the top N, spending fewer calls on the rest")
    p.add_argument("--one-order", action="store_true", help="judge each pair once instead of in both orders")
    p.add_argument("--pairing", choices=["swiss", "random"])
    p.add_argument("--max-chars", type=int, help="cut items longer than this before judging")
    p.add_argument("--max-cost", type=float, metavar="USD", help="hard spend cap: no judge call starts that could take the cost past USD")
    p.add_argument("--seed", type=int)
    p.add_argument("-y", "--yes", action="store_true", help="start without asking, even if it costs money")
    p.add_argument("--dry-run", action="store_true", help="show the estimate and stop")


def build_parser() -> argparse.ArgumentParser:
    common = Parser(add_help=False)
    common.add_argument("--json", action="store_true", help="print JSON on stdout (errors too)")
    common.add_argument("--api-url", help="API base URL (default: SUPPOSING_API_URL, the saved login, or "
                        "https://api.supposing.app)")

    parser = Parser(
        prog="supposing", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Rank documents with LLM judges: pairwise tournaments with confidence intervals.",
        epilog="quick start:\n  supposing rank ./docs --rubric \"Which is clearest?\" --provider mock\n"
               "  supposing login\n  supposing run ./docs --rubric \"Which is clearest?\"\n\n" + EXIT_CODES_HELP)
    parser.add_argument("--version", action="version", version="supposing %s" % __version__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, handler: Handler, help: str, parent=sub) -> argparse.ArgumentParser:
        p = parent.add_parser(name, help=help, description=help, parents=[common])
        p.set_defaults(handler=handler)
        return p

    r = add("rank", local.rank, "rank a folder of documents, or the rows of a JSONL or CSV file, on this machine")
    r.add_argument("folder", type=Path, metavar="source",
                   help="a folder of .md, .txt and .html files (one document each), or a .jsonl, .json or .csv file "
                   "(one item per row)")
    _run_flags(r)
    _source_flags(r)
    r.add_argument("--base-url", help="for openai_compat, e.g. http://localhost:11434/v1")
    r.add_argument("--concurrency", type=int, default=8, help="judge calls at once (default 8)")
    r.add_argument("--state", type=Path, help="SQLite file to save progress in; rerun with it to resume")

    r = add("run", cloud.run, "rank a folder or an uploaded dataset in the cloud")
    r.add_argument("source", help="a folder or a .jsonl, .json or .csv file to upload, or the id of a dataset "
                   "already uploaded")
    _run_flags(r)
    _source_flags(r)
    r.add_argument("--name", help="name for the run")
    r.add_argument("--detach", action="store_true", help="start the run and print it, without waiting")

    r = add("login", cloud.login, "save an API key for this machine (create one in the web app under Developers)")
    r.add_argument("--api-key", help="the key; if left out, it is read from stdin or asked for")
    add("logout", cloud.logout, "forget the saved API key")
    add("whoami", cloud.whoami, "show which workspace the API key belongs to")

    ds = sub.add_parser("datasets", help="upload and manage datasets").add_subparsers(
        dest="sub", required=True, metavar="command")
    r = add("push", cloud.datasets_push, "upload a folder (one item per file) or a JSONL, JSON or CSV file (one item "
            "per row) as a new dataset", ds)
    r.add_argument("folder", type=Path, metavar="source")
    r.add_argument("--name", help="dataset name (default: the folder's or file's name)")
    _source_flags(r)
    r = add("list", cloud.datasets_list, "list datasets, newest first", ds)
    r.add_argument("--limit", type=int, default=50)
    r = add("get", cloud.datasets_get, "show a dataset and its items", ds)
    r.add_argument("dataset_id")
    r = add("delete", cloud.datasets_delete, "delete a dataset and its runs", ds)
    r.add_argument("dataset_id")

    runs = sub.add_parser("runs", help="list, watch and cancel runs").add_subparsers(
        dest="sub", required=True, metavar="command")
    r = add("list", cloud.runs_list, "list runs, newest first", runs)
    r.add_argument("--dataset", dest="dataset_id")
    r.add_argument("--limit", type=int, default=20)
    r = add("get", cloud.runs_get, "show a run's status and progress", runs)
    r.add_argument("run_id")
    r = add("watch", cloud.runs_watch, "follow a run until it finishes, then print its results", runs)
    r.add_argument("run_id")
    r = add("cancel", cloud.runs_cancel, "stop a run; calls already made are kept", runs)
    r.add_argument("run_id")

    r = add("results", cloud.results, "print a run's results: the verdict, then the leaderboard")
    r.add_argument("run_id")
    # `supposing billing` alone shows the balance; its subcommands do the rest.
    bl = add("billing", billing.show, "credit balance, ledger and top-ups").add_subparsers(
        dest="sub", required=False, metavar="command")
    r = add("ledger", billing.ledger, "every credit movement, newest first", bl)
    r.add_argument("--limit", type=int, default=30)
    r = add("topup", billing.topup, "print a payment page for adding credit (nothing is charged until you pay)", bl)
    r.add_argument("amount", type=billing.check_amount, help="whole US dollars, e.g. 10")
    add("portal", billing.portal, "print a link to receipts and payment details", bl)
    add("pricing", billing.pricing, "the price of a judge call by model tier (no login needed)")
    add("usage", billing.usage, "this month's runs, judge calls and spend, and the credit balance")

    r = add("comparisons", cloud.comparisons, "list a run's judge calls, with each verdict's reasoning")
    r.add_argument("run_id")
    r.add_argument("--item", dest="item_id", help="only calls involving this item")
    r.add_argument("--status", choices=["pending", "claimed", "done", "failed"])
    r.add_argument("--limit", type=int, default=50)

    r = add("mcp", _mcp, "serve the MCP tools over stdio for a local coding agent (needs supposing[mcp])")
    r.add_argument("--local", action="store_true", help="rank with the engine on this machine even if logged in")
    r.add_argument("mcp_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _mcp(args: argparse.Namespace) -> int:
    """Start the stdio MCP server: cloud tools when an API key resolves, the local engine otherwise."""
    try:
        from ..mcp_server import main as mcp_main  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CliError("The MCP server needs the mcp package (%s)." % exc, code="mcp_not_installed",
                       exit_code=USAGE, hint="pip install 'supposing[mcp]'") from exc
    rest = list(args.mcp_args)
    if args.local:
        rest.insert(0, "--local")
    return int(mcp_main(rest) or 0)


_CLOUD_EXIT = {"insufficient_credit": NEEDS_CREDIT, "no_api_key": AUTH, "unauthorized": AUTH, "invalid_api_key": AUTH, "invalid_session": AUTH,
               "connection_error": UNAVAILABLE, "server_error": UNAVAILABLE, "internal_error": UNAVAILABLE,
               "bad_response": UNAVAILABLE, "no_items": USAGE}


def _from_cloud(exc: CloudError) -> CliError:
    code = _CLOUD_EXIT.get(exc.code)
    if code is None:
        code = (NOT_FOUND if exc.status == 404 or exc.code.endswith("_not_found")
                else UNAVAILABLE if (exc.status or 0) >= 500 or exc.status == 429 else REFUSED)
    return CliError(exc.message, code=exc.code, exit_code=code, hint=exc.hint)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    Parser.json_errors = "--json" in argv
    try:
        args = build_parser().parse_args(argv)
    except CliError as exc:
        return error(exc, True)
    try:
        return args.handler(args)
    except CliError as exc:
        return error(exc, args.json)
    except CloudError as exc:
        err = _from_cloud(exc)
        if exc.code == "insufficient_credit":  # say where to top up, in words and in --json
            from ..cloud import Client

            with Client(api_url=getattr(args, "api_url", None), source="cli") as sp:
                err.extra = billing.topup_hint(sp)
        return error(err, args.json)
    except KeyboardInterrupt:
        hint = getattr(args, "interrupt_hint", None)
        return error(CliError("Interrupted.", code="interrupted", exit_code=INTERRUPTED, hint=hint), args.json)


if __name__ == "__main__":
    sys.exit(main())
