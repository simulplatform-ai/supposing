"""``supposing rank``: a ranking on this machine, with no account and no network beyond the judge."""
from __future__ import annotations

import argparse
import asyncio

from ..cloud.folder import read_folder
from ..cloud.records import RecordsError, is_table, read_records
from ..providers import KINDS, ProviderError, make_provider
from ..ranking import ConfigError, Item, RankConfig, fewest_calls, planned_calls
from ..runner import DEFAULT_MODEL, RunError, run_ranking
from ..store import SQLiteStore
from .output import OK, RUN_FAILED, USAGE, CliError, ProgressLine, emit_json, info, print_results
from .shared import FREE_PROVIDERS, confirm, ranking_options, result_document, rubric_text

PROVIDER_HINTS = {
    "openrouter": "Set OPENROUTER_API_KEY, or use --provider mock to try it free.",
    "anthropic": "Set ANTHROPIC_API_KEY, or use --provider mock to try it free.",
    "openai_compat": "Pass --base-url, e.g. http://localhost:11434/v1 for Ollama.",
    "claude_cli": "Install Claude Code and log in, or use --provider mock to try it free.",
}


def read_items(args: argparse.Namespace) -> tuple[list[Item], list[dict[str, str]]]:
    """Items from the source argument: a folder (one per file, id = path) or a table file (one per row, id = the
    row's id). Returns the items and what was skipped as [{path, reason}]."""
    source = args.folder
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


def rank(args: argparse.Namespace) -> int:
    items, skipped_rows = read_items(args)
    if len(items) < 2:
        raise CliError("Need at least two items with text in %s." % args.folder, code="too_few_items",
                       exit_code=USAGE, hint="One document per file in a folder, or one row per item in a file.")
    provider_kind = args.provider or "openrouter"
    if provider_kind not in KINDS:
        raise CliError("Unknown provider %s." % provider_kind, code="unknown_provider", exit_code=USAGE,
                       hint="Use one of %s." % ", ".join(KINDS))
    options = ranking_options(args)
    try:
        cfg = RankConfig(**options, max_cost_usd=args.max_cost)
    except ConfigError as exc:
        raise CliError(str(exc), code="invalid_request", exit_code=USAGE) from exc
    model = args.model or ("mock" if provider_kind == "mock" else DEFAULT_MODEL)
    lo, hi = fewest_calls(len(items), cfg), planned_calls(len(items), cfg)
    free = provider_kind in FREE_PROVIDERS
    estimate = {"type": "ranking", "n_items": len(items), "min_calls": lo, "max_calls": hi,
                "est_cost_usd": 0.0 if free else None, "provider": provider_kind, "model": model}
    calls = "%d judge calls" % hi if lo == hi else "%d to %d judge calls" % (lo, hi)
    cost = "free" if free else ("at most $%.2f (--max-cost)" % args.max_cost if args.max_cost
                                else "cost depends on %s's price" % model)
    line = "%d items, %s with %s, %s" % (len(items), calls, provider_kind, cost)
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
                       hint=PROVIDER_HINTS.get(provider_kind, "Use --provider mock to try it free.")) from exc
    confirm(args, line, free, "Add --max-cost to cap the spend.")
    store = SQLiteStore(args.state) if args.state else None
    bar = ProgressLine()

    async def go():
        try:
            return await run_ranking(items, args.rubric, provider, cfg, store, run_id=str(args.folder.resolve()),
                                     concurrency=args.concurrency,
                                     on_progress=lambda done, total: bar.update("%d/%d judged" % (done, total)))
        finally:
            await provider.close()

    try:
        ranking = asyncio.run(go())
    except RunError as exc:
        bar.end()
        raise CliError(str(exc), code="run_failed", exit_code=RUN_FAILED,
                       hint="Rerun with the same --state to continue where it stopped." if args.state else None
                       ) from exc
    bar.end()
    titles = {it.id: it.title for it in items}
    board = [{**e, "title": titles.get(e["item_id"], e["item_id"])} for e in ranking.leaderboard]
    body = {"leaderboard": board, "diagnostics": ranking.diagnostics, "interval": ranking.interval,
            "n_bootstrap": ranking.n_bootstrap}
    results = {"run_id": None, "type": "ranking", "final": True, "summary": ranking.summary, "ranking": body}
    run = {"id": None, "type": "ranking", "status": "done", "judge": {"provider": provider_kind, "model": model},
           "ranking": {"rubric": rubric_text(args.rubric), **options}, "note": ranking.note,
           "progress": {"cost_usd": ranking.cost_usd}}
    if args.json:
        emit_json(result_document(run, estimate, results, skipped))
        return OK
    print_results(results)
    if ranking.note:
        print(ranking.note)
    return OK
