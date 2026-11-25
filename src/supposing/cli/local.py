"""``supposing rank``: a ranking on this machine, with no account and no network beyond the judge."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from ..cloud.folder import read_folder
from ..cloud.records import RecordsError, is_table, read_records
from ..providers import ProviderError, make_provider
from ..ranking import ConfigError, Item, RankConfig, fewest_calls, planned_calls
from ..runner import RunError, run_ranking
from ..samples import items as sample_items
from ..samples import list_samples
from ..samples import load as load_sample
from ..store import SQLiteStore
from . import preflight, telemetry
from .output import NOT_FOUND, OK, RUN_FAILED, USAGE, CliError, ProgressLine, emit_json, info, print_results
from .shared import (
    cli_judge,
    comparison_row,
    confirm,
    judge_inputs,
    judge_label,
    print_comparisons,
    ranking_options,
    result_document,
    rubric_text,
)

FREE_JUDGES_HINT = ("or judge free on this machine: --judge claude_cli (your Claude subscription), --judge "
                    "openai_compat --base-url http://localhost:11434/v1 --model llama3.2 (Ollama), or --judge mock "
                    "(made-up verdicts, to try the flow)")
PROVIDER_HINTS = {
    "openrouter": "Set OPENROUTER_API_KEY (https://openrouter.ai/settings/keys), " + FREE_JUDGES_HINT + ".",
    "anthropic": "Set ANTHROPIC_API_KEY, " + FREE_JUDGES_HINT + ".",
    "openai_compat": "Pass --base-url, e.g. http://localhost:11434/v1 for Ollama.",
    "claude_cli": "Install Claude Code and log in, or use --judge mock to try it free.",
}


def samples(args: argparse.Namespace) -> int:
    """The built-in samples: id, items and rubric."""
    rows = list_samples()
    if args.json:
        emit_json({"data": rows})
        return OK
    for s in rows:
        print("%s  (%d items)  %s" % (s["id"], s["n_items"], s["description"]))
        print("    rubric: %s" % s["rubric"])
    print()
    print("Try one: supposing rank --sample %s --judge mock" % rows[0]["id"])
    return OK


def read_items(args: argparse.Namespace) -> tuple[list[Item], list[dict[str, str]]]:
    """Items from the source argument: a folder (one per file, id = path) or a table file (one per row, id = the
    row's id), or a built-in sample. Returns the items and what was skipped as [{path, reason}]."""
    if getattr(args, "sample", None):
        if args.folder:
            raise CliError("Pass a source or --sample, not both.", code="usage", exit_code=USAGE)
        try:
            sample = load_sample(args.sample)
        except KeyError as exc:
            raise CliError(str(exc.args[0]), code="sample_not_found", exit_code=NOT_FOUND,
                           hint="List them: supposing samples") from exc
        if args.rubric is None:
            args.rubric = sample["rubric"]
        return sample_items(args.sample), []
    if args.folder is None:
        raise CliError("Nothing to rank.", code="usage", exit_code=USAGE,
                       hint="Pass a folder or a .jsonl/.csv file, or try a sample: supposing rank --sample stories "
                       "--judge mock")
    if args.rubric is None:
        raise CliError("The following arguments are required: --rubric.", code="usage", exit_code=USAGE,
                       hint="Say what the judge should prefer: --rubric \"Which is clearest?\"")
    source = args.folder
    if source.suffix.lower() in (".xlsx", ".xls"):
        raise CliError("supposing rank can't read %s on this machine." % source.suffix, code="invalid_file",
                       exit_code=USAGE, hint="Save it as CSV, or rank it in the cloud: supposing run %s ..." % source.name)
    if getattr(args, "sheet", None):
        raise CliError("--sheet is for an .xlsx file in the cloud.", code="usage", exit_code=USAGE)
    if is_table(source):
        try:
            recs = read_records(source, args.text_fields, args.id_field, args.title_field, args.metadata_fields)
        except RecordsError as exc:
            raise CliError(str(exc), code="invalid_file", exit_code=USAGE, hint=exc.hint) from exc
        items = [Item(id=r.source_id, text=r.text, title=r.title) for r in recs.items]
        return items, [{"path": "%s#%s" % (source.name, sid), "reason": why} for sid, why in recs.skipped]
    if args.text_fields or args.id_field or args.title_field or args.metadata_fields:
        raise CliError("--text-field, --id-field, --title-field and --metadata-field are for a JSONL, JSON or CSV "
                       "file, and %s is not one." % source, code="usage", exit_code=USAGE,
                       hint="Pass a .jsonl, .json or .csv file, or drop those flags for a folder.")
    try:
        found = read_folder(source)
    except FileNotFoundError as exc:
        raise CliError(str(exc), code="not_a_folder", exit_code=USAGE,
                       hint="Pass a folder of .md, .txt or .html files, or a .jsonl, .json or .csv file.") from exc
    return ([Item(id=title, text=text, title=title) for title, text in found.items],
            [{"path": p, "reason": why} for p, why in found.skipped])


def too_few(args: argparse.Namespace, n: int, skipped: list[dict[str, str]]) -> CliError:
    """No ranking with fewer than two items: say how many were found, and what was left out and why."""
    where = args.folder or args.sample
    found = "No items with text" if n == 0 else "Only 1 item with text"
    hint = "A ranking needs at least two. Put one document in each .md, .txt or .html file, or one row per item " \
           "in a .jsonl, .json or .csv file."
    if skipped:
        hint += " Skipped %d: %s%s." % (len(skipped), ", ".join("%s (%s)" % (s["path"], s["reason"]) for s in skipped[:5]),
                                       " ..." if len(skipped) > 5 else "")
        if any(s["path"].lower().endswith((".pdf", ".docx")) for s in skipped):
            hint += " PDF and Word files are read in the cloud: supposing run %s ..." % where
    return CliError("%s in %s." % (found, where), code="too_few_items", exit_code=USAGE, hint=hint)


def duplicate_groups(items: list[Item]) -> list[list[str]]:
    """Ids of items whose texts are identical (ignoring surrounding whitespace), in groups of two or more."""
    seen: dict[str, list[str]] = {}
    for it in items:
        seen.setdefault(it.text.strip(), []).append(it.id)
    return [ids for ids in seen.values() if len(ids) > 1]


OUTPUT_TOKENS_PER_CALL = 150
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal")


def per_call_tokens(items: list[Item], rubric: str, max_chars: int | None) -> int:
    """About how many input tokens one call reads: two items, the rubric and the instructions (4 chars a token)."""
    chars = sum(min(len(i.text), max_chars or 10**9) for i in items) / max(1, len(items))
    return int(2 * chars / 4 + 450 + len(rubric) / 4)


def is_local_url(url: str | None) -> bool:
    """A judge on this machine or your own network: loopback, a .local name, or a private address."""
    host = (urlparse(url or "").hostname or "").lower()
    if host in LOCAL_HOSTS or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def cost_basis(provider: str, base_url: str | None) -> tuple[bool, str]:
    """Whether a run costs money, and what the estimate rests on, in words. Only a run that costs money asks
    before it starts."""
    if provider == "mock":
        return True, "free: made-up verdicts, no model is called"
    if provider == "claude_cli":
        return True, "no charge: your Claude subscription, within its usage limits"
    if provider == "openai_compat" and is_local_url(base_url):
        return True, "no charge: a model on your own machine or network (%s)" % (urlparse(base_url or "").hostname)
    who = {"openrouter": "OpenRouter", "anthropic": "Anthropic"}.get(provider, "the endpoint at %s" % base_url)
    return False, "billed by %s at the model's price, which supposing does not know" % who


def rank(args: argparse.Namespace) -> int:
    items, skipped_rows = read_items(args)
    if len(items) < 2:
        raise too_few(args, len(items), skipped_rows)
    for same in duplicate_groups(items):
        info("Same text: %s. Each is ranked; remove copies you don't want judged twice." % " = ".join(same[:5]))
    judge = cli_judge(args, local=True)
    provider_kind, model = judge["provider"], judge["model"]
    options = ranking_options(args)
    try:
        cfg = RankConfig(**options, max_cost_usd=args.max_cost, max_calls=args.max_calls)
    except ConfigError as exc:
        raise CliError(str(exc), code="invalid_request", exit_code=USAGE) from exc
    lo, hi = fewest_calls(len(items), cfg), planned_calls(len(items), cfg)
    telemetry.note(local=True, provider=provider_kind, base_url=args.base_url, model=model, level=judge["level"],
                   items=len(items), rounds=cfg.rounds, both_orders=cfg.both_orders, sample=args.sample)
    extra = judge_inputs(args)
    told = len(extra.get("context") or "") + sum(len(e.a) + len(e.b) + len(e.why or "") for e in extra.get("examples") or [])
    if extra.get("prompt"):
        told += len(extra["prompt"].system)
    tokens_in = hi * (per_call_tokens(items, rubric_text(args.rubric), cfg.max_chars) + told // 4)
    free, basis = cost_basis(provider_kind, args.base_url)
    estimate = {"type": "ranking", "n_items": len(items), "min_calls": lo, "max_calls": hi,
                "est_input_tokens": tokens_in, "est_output_tokens": hi * OUTPUT_TOKENS_PER_CALL,
                "est_cost_usd": 0.0 if free else None, "cost_basis": basis, "judge": judge}
    calls = "%d judge calls" % hi if lo == hi else "%d to %d judge calls" % (lo, hi)
    if cfg.max_calls is not None and hi == cfg.max_calls:
        calls += " (at most, --max-calls)"
    if free:
        cost = basis
    else:
        cost = "%s; about %s input and %s output tokens%s" % (
            basis, f"{tokens_in:,}", f"{hi * OUTPUT_TOKENS_PER_CALL:,}",
            ", at most $%.2f (--max-cost)" % args.max_cost if args.max_cost else "")
    line = "%d items, %s with %s, %s" % (len(items), calls, judge_label(judge), cost)
    if skipped_rows and not args.json:
        info("Skipped %d: %s" % (len(skipped_rows), ", ".join(s["path"] for s in skipped_rows[:5])
                                 + (" ..." if len(skipped_rows) > 5 else "")))
    skipped = skipped_rows
    if args.dry_run:
        if args.json:
            emit_json({"estimate": estimate, "skipped": skipped})
        else:
            info("Estimate: %s" % line)
        return OK
    spec = {"provider": provider_kind, "model": model}
    if args.base_url:
        spec["base_url"] = args.base_url
    try:  # before asking to spend: a run that can't start shouldn't ask for --yes
        provider = make_provider(spec)
    except (ProviderError, KeyError) as exc:
        raise CliError(str(exc).rstrip(".") + ".", code="provider_unavailable", exit_code=USAGE,
                       hint=PROVIDER_HINTS.get(provider_kind, "Use --judge mock to try it free.")) from exc
    if provider_kind != "mock":
        preflight.check(provider_kind, model, args.base_url)
    confirm(args, line, free, "Add --max-cost to cap the spend, or --dry-run to see the estimate first.")
    args.interrupt_hint = ("Run the same command again to continue from %s." % args.state if args.state else
                           "Nothing was kept. Add --state run.db to make a run you can continue.")
    store = SQLiteStore(args.state) if args.state else None
    bar = ProgressLine()

    async def go():
        try:
            return await run_ranking(items, args.rubric, provider, cfg, store, run_id=str(args.folder.resolve()) if args.folder else "sample:%s" % args.sample,
                                     concurrency=args.concurrency,
                                     on_progress=lambda done, total: bar.update("%d/%d judged" % (done, total)),
                                     **extra)
        finally:
            await provider.close()

    try:
        ranking = asyncio.run(go())
    except RunError as exc:
        bar.end()
        raise CliError(str(exc), code=exc.code, exit_code=USAGE if exc.code == "state_mismatch" else RUN_FAILED,
                       hint=exc.hint or ("Rerun with the same --state to continue where it stopped." if args.state
                                         else None)) from exc
    bar.end()
    titles = {it.id: it.title for it in items}
    board = [{**e, "title": titles.get(e["item_id"], e["item_id"])} for e in ranking.leaderboard]
    body = {"leaderboard": board, "diagnostics": ranking.diagnostics, "interval": ranking.interval,
            "n_bootstrap": ranking.n_bootstrap, "groups": ranking.groups}
    results = {"run_id": None, "type": "ranking", "final": True, "summary": ranking.summary, "ranking": body}
    run = {"id": None, "type": "ranking", "status": "done", "judge": judge,
           "ranking": {"rubric": rubric_text(args.rubric), **options}, "note": ranking.note,
           "progress": {"done": sum(c.status == "done" for c in ranking.comparisons),
                        "failed": sum(c.status == "failed" for c in ranking.comparisons), "cost_usd": ranking.cost_usd}}
    rows = [comparison_row(c, titles) for c in ranking.comparisons] if args.with_comparisons else None
    if args.json:
        emit_json(result_document(run, estimate, results, skipped, rows))
        return OK
    print_results(results)
    if ranking.note:
        print(ranking.note)
    if rows is not None:
        print()
        print_comparisons(rows)
    elif args.state:
        print("Every verdict and its reasoning: supposing comparisons --state %s" % args.state)
    else:
        print("Every verdict and its reasoning: add --with-comparisons")
    return OK


def _state_runs(path: Path) -> list[str]:
    """The run ids a --state file holds, oldest first."""
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        return [r[0] for r in con.execute("SELECT run_id FROM comparisons GROUP BY run_id ORDER BY MIN(rowid)")]
    except sqlite3.DatabaseError:
        return []
    finally:
        con.close()


def local_comparisons(args: argparse.Namespace) -> int:
    """`supposing comparisons --state FILE`: the judge calls of a local run, with their reasoning."""
    path = Path(args.state)
    runs = _state_runs(path) if path.is_file() else []
    if not runs:
        raise CliError("%s holds no ranking." % path, code="not_found", exit_code=NOT_FOUND,
                       hint="Pass the file you gave `supposing rank --state`.")
    run_id = args.run_id or (runs[0] if len(runs) == 1 else None)
    if run_id not in runs:
        raise CliError("%s holds %d runs; say which." % (path, len(runs)) if not args.run_id
                       else "%s holds no run %s." % (path, args.run_id), code="usage", exit_code=USAGE,
                       hint="Runs in this file: %s" % "; ".join(runs[:10]))
    titles: dict[str, str] = {}
    if run_id.startswith("sample:"):
        with contextlib.suppress(KeyError):
            titles = {i.id: i.title or i.id for i in sample_items(run_id[len("sample:"):])}
    store = SQLiteStore(path)
    try:
        comps = store.load(run_id)
    finally:
        store.close()
    rows = [comparison_row(c, titles) for c in comps
            if (not args.item_id or args.item_id in (c.item_a, c.item_b)) and (not args.status or c.status == args.status)]
    rows = rows[:args.limit] if args.limit else rows
    if args.json:
        emit_json({"run_id": run_id, "data": rows, "next_cursor": None})
        return OK
    print_comparisons(rows)
    return OK

