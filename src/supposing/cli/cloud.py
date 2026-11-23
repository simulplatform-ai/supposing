"""Cloud commands: login, datasets, run, runs, results, comparisons. They talk to the API over HTTPS only."""
from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path
from typing import Any

from ..cloud import Client, CloudError, Run, config_path, remove_config, resolve, write_config
from ..cloud.config import KEY_PREFIX
from ..cloud.records import is_table
from . import billing
from .output import (
    AUTH,
    NEEDS_CREDIT,
    OK,
    REFUSED,
    RUN_FAILED,
    USAGE,
    CliError,
    ProgressLine,
    emit_json,
    info,
    plural,
    print_results,
    table,
)
from .shared import confirm, ranking_options, result_document, rubric_text

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _client(args: argparse.Namespace) -> Client:
    return Client(api_url=args.api_url, source="cli")


def _short_time(v: Any) -> str:
    return str(v or "")[:16].replace("T", " ")


# --- account --------------------------------------------------------------------------------------------------

def login(args: argparse.Namespace) -> int:
    key = args.api_key
    if not key and not sys.stdin.isatty():
        key = sys.stdin.readline().strip()
        if not key:  # piped or scripted, with nothing on stdin: there is no one to ask
            raise CliError("No API key on stdin.", code="no_api_key", exit_code=AUTH,
                           hint="Pipe the key in (supposing login < key.txt) or pass --api-key.")
    if not key:
        info("Create an API key in the web app under Developers, then paste it here.")
        key = getpass.getpass("API key: ").strip()
    if not key.startswith(KEY_PREFIX):
        raise CliError("That doesn't look like an API key: keys start with %s." % KEY_PREFIX, code="invalid_api_key",
                       exit_code=AUTH, hint="Create one in the web app under Developers.")
    url = resolve(api_url=args.api_url).api_url
    with Client(api_key=key, api_url=url, source="cli") as sp:
        me = sp.me()  # check the key works before saving it
    path = write_config(key, url)
    if args.json:
        emit_json({"org_id": me.org_id, "org_name": me.org_name, "api_url": url, "config": str(path),
                   "key_prefix": key[:10]})
    else:
        print("Logged in to %s at %s. Key saved to %s (readable only by you)." % (me.org_name, url, path))
    return OK


def logout(args: argparse.Namespace) -> int:
    removed = remove_config()
    if args.json:
        emit_json({"removed": removed, "config": str(config_path())})
    else:
        print("Removed %s." % config_path() if removed else "Not logged in on this machine.")
    return OK


def whoami(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        me = sp.me()
        source = {"env": "SUPPOSING_API_KEY", "file": str(config_path()), "argument": "--api-key"}.get(
            sp.key_source, sp.key_source)
        try:
            bill = sp.billing.get()
        except CloudError:  # an older server, or billing not reachable: whoami still answers
            bill = None
        if args.json:
            emit_json({**me.to_dict(), "api_url": sp.api_url, "key_from": source,
                       "balance_usd": bill.balance_usd if bill else None})
        else:
            print("%s (%s), key from %s, API %s" % (me.org_name, me.org_id, source, sp.api_url))
            if bill:
                print("Credit balance $%.2f. Details: supposing billing" % bill.balance_usd)
    return OK


# --- datasets -------------------------------------------------------------------------------------------------

def _push(sp: Client, folder: Path, name: str | None, quiet: bool, reuse: bool = False,
          args: argparse.Namespace | None = None) -> Any:
    fields = {k: getattr(args, k, None) for k in ("text_fields", "id_field", "title_field", "metadata_fields")}
    try:
        ds = sp.datasets.push(folder, name, reuse=reuse, **fields)
    except FileNotFoundError as exc:
        raise CliError(str(exc), code="not_a_folder", exit_code=USAGE,
                       hint="Pass a folder of .md, .txt or .html files.") from exc
    if not quiet:
        if ds.reused:
            info("Using dataset \"%s\" (%s): the same %s were uploaded before." % (
                ds.name, ds.id, plural(ds.n_items, "file")))
        else:
            info("Uploaded %s from %s as dataset \"%s\" (%s)." % (plural(ds.n_items, "item"), folder, ds.name, ds.id))
        if ds.skipped:
            info("Skipped %d: %s" % (len(ds.skipped), ", ".join(s["path"] for s in ds.skipped[:5])
                                            + (" ..." if len(ds.skipped) > 5 else "")))
    return ds


def datasets_push(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        ds = _push(sp, args.folder, args.name, args.json, args=args)
    if args.json:
        emit_json({**ds.to_dict(), "skipped": ds.skipped})
    else:
        print(ds.id)
    return OK


def datasets_list(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        page = sp.datasets.list(limit=args.limit)
    if args.json:
        emit_json({"data": [d.to_dict() for d in page.data], "next_cursor": page.next_cursor})
    elif not page.data:
        print("No datasets yet. Upload one: supposing datasets push ./folder")
    else:
        print(table([[d.id, str(d.n_items), _short_time(d.raw.get("created_at")), d.name] for d in page.data],
                    ["id", "items", "created", "name"], right={1}))
    return OK


def datasets_get(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        ds = sp.datasets.get(args.dataset_id)
    if args.json:
        emit_json(ds.to_dict())
        return OK
    print("%s  %d items  %s" % (ds.name, ds.n_items, ds.id))
    if ds.items:
        print()
        print(table([[i.id, str(i.n_words), i.title] for i in ds.items], ["item id", "words", "title"], right={1}))
    return OK


def datasets_delete(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        sp.datasets.delete(args.dataset_id)
    if args.json:
        emit_json({"deleted": args.dataset_id})
    else:
        print("Deleted dataset %s." % args.dataset_id)
    return OK


# --- runs -----------------------------------------------------------------------------------------------------

def _progress_text(run: Run, leader: str | None = None) -> str:
    p = run.progress
    total = max(p.total, run.planned_calls)
    text = "%s · round %d · %d/%d calls · $%.4f" % (run.status, p.round, p.done, total, p.cost_usd)
    return text + (" · leading: %s" % leader if leader else "")


def _follow(sp: Client, args: argparse.Namespace, run: Run, estimate: dict[str, Any] | None,
            skipped: list[dict[str, str]] | None = None) -> int:
    """Wait for a run, then print its results. Shared by `run` and `runs watch`."""
    args.interrupt_hint = "The run keeps going on the server. Follow it again: supposing runs watch %s" % run.id
    bar = ProgressLine()
    run = sp.runs.wait(run.id, on_progress=lambda r: bar.update(_progress_text(r)), poll_interval=1.0,
                       on_event=lambda e: bar.update(_progress_text(e.run, e.leaders[0].title if e.leaders else None)))
    bar.end()
    results = sp.runs.results(run.id)
    if args.json:
        emit_json(result_document(run.to_dict(), estimate, results.to_dict(), skipped))
    else:
        if run.status != "done":
            info("Run %s %s%s" % (run.id, run.status, ": %s" % run.error if run.error else "."))
        print_results(results.to_dict())
        if run.note:
            print(run.note)
        print("Cost $%.4f. Every verdict: supposing comparisons %s" % (run.progress.cost_usd, run.id))
    return OK if run.status == "done" else RUN_FAILED


def run(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        source = Path(args.source)
        upload = source.is_dir() or is_table(source)
        if not upload and not UUID_RE.match(args.source):
            raise CliError("%s is neither a folder, a JSONL/JSON/CSV file, nor a dataset id." % args.source,
                           code="not_a_folder", exit_code=USAGE,
                           hint="Pass a folder of documents, a .jsonl or .csv file, or an id from "
                           "`supposing datasets list`.")
        provider = args.provider or "openrouter"  # the API's default judge
        available = sp.settings().get("providers") or []  # before uploading: a refused run leaves no dataset
        if provider not in available:
            raise CliError("Provider %s is not available in this workspace." % provider,
                           code="provider_unavailable", exit_code=REFUSED,
                           hint="Available: %s. Add an OpenRouter key in Settings to use openrouter, or pass "
                           "--provider mock to try it free." % ", ".join(available))
        # A folder reuses an identical earlier upload, so retrying (e.g. with --yes after exit 7) adds no copy.
        pushed = _push(sp, source, None, args.json, reuse=True, args=args) if upload else None
        dataset_id = pushed.id if pushed else args.source
        skipped = pushed.skipped if pushed else []
        rubric = rubric_text(args.rubric)
        model = args.model or ("mock" if args.provider == "mock" else None)
        kw: dict[str, Any] = dict(provider=args.provider, model=model, max_cost_usd=args.max_cost,
                                  **ranking_options(args))
        est = sp.runs.estimate(dataset_id, rubric, **kw)
        line = est.describe()
        credit = billing.balance_after(sp, est)  # billed runs: what the credit will be after
        if credit:
            line += "; " + credit
        if args.dry_run:
            if args.json:
                emit_json({"estimate": est.to_dict(), "dataset_id": dataset_id, "skipped": skipped})
            else:
                info("Estimate: %s" % line)
                info("Start it: supposing run %s --rubric ... (same flags)" % dataset_id)
            return OK
        if credit and "short by" in credit:  # the API would refuse it: say so before asking to spend
            raise CliError("This run needs up to $%.2f of credit; %s." % (est.est_cost_usd or 0, credit),
                           code="insufficient_credit", exit_code=NEEDS_CREDIT,
                           hint="Add credit, lower the cost (fewer rounds, fewer items, or a cheaper judge model), "
                           "or use your own OpenRouter key in Settings (never billed).", extra=billing.topup_hint(sp))
        confirm(args, line, est.est_cost_usd == 0, "Add --max-cost to cap the spend.")
        created = sp.runs.create(dataset_id, rubric, name=args.name, **kw)
        if not args.json:
            info("Started run %s." % created.id)
        if args.detach:
            if args.json:
                emit_json(result_document(created.to_dict(), est.to_dict(), None, skipped))
            else:
                print(created.id)
                info("Follow it: supposing runs watch %s" % created.id)
            return OK
        return _follow(sp, args, created, est.to_dict(), skipped)


def runs_list(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        page = sp.runs.list(dataset_id=args.dataset_id, limit=args.limit)
    if args.json:
        emit_json({"data": [r.to_dict() for r in page.data], "next_cursor": page.next_cursor})
    elif not page.data:
        print("No runs yet. Start one: supposing run ./folder --rubric \"...\"")
    else:
        print(table([[r.id, r.status, "%d/%d" % (r.progress.done, max(r.progress.total, r.planned_calls)),
                      "$%.4f" % r.progress.cost_usd, _short_time(r.raw.get("created_at")), r.name] for r in page.data],
                    ["id", "status", "calls", "cost", "created", "name"], right={2, 3}))
    return OK


def runs_get(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        r = sp.runs.get(args.run_id)
    if args.json:
        emit_json(r.to_dict())
        return OK
    print("%s  %s" % (r.name, r.id))
    print(_progress_text(r))
    print("dataset %s (%s) · judge %s %s" % (r.dataset_name, r.dataset_id, r.judge.get("provider"),
                                             r.judge.get("model")))
    print("rubric: %s" % r.config.get("rubric"))
    if r.error:
        print("error: %s" % r.error)
    if r.note:
        print(r.note)
    return OK


def runs_watch(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        return _follow(sp, args, sp.runs.get(args.run_id), None)


def runs_cancel(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        r = sp.runs.cancel(args.run_id)
    if args.json:
        emit_json(r.to_dict())
    else:
        print("Run %s is %s; %d calls were made." % (r.id, r.status, r.progress.done))
    return OK


def results(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        res = sp.runs.results(args.run_id)
    if args.json:
        emit_json(res.to_dict())
    else:
        print_results(res.to_dict())
    return OK


def comparisons(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        page = sp.runs.comparisons(args.run_id, item_id=args.item_id, status=args.status, limit=args.limit)
    if args.json:
        emit_json({"data": [c.to_dict() for c in page.data], "next_cursor": page.next_cursor})
        return OK
    if not page.data:
        print("No calls yet.")
    for c in page.data:
        won = c.title_a if c.winner_item == c.item_a else c.title_b if c.winner_item == c.item_b else None
        print("round %d · %s vs %s · %s" % (c.round, c.title_a, c.title_b,
                                            "winner %s" % won if won else c.status))
        if c.reasoning:
            print("  " + c.reasoning.strip().replace("\n", "\n  "))
        if c.error:
            print("  error: %s" % c.error)
    if page.next_cursor:
        print("More: --json shows next_cursor.")
    return OK


