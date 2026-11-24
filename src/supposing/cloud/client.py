"""HTTP client for the Supposing cloud API (``/v1``). Sync (:class:`Client`) and async (:class:`AsyncClient`)
share every request definition below; only how a request is sent differs.

    from supposing.cloud import Client
    with Client() as sp:                      # SUPPOSING_API_KEY, or the key saved by `supposing login`
        ds = sp.datasets.push("./proposals")
        print(sp.runs.estimate(ds.id, "Which proposal is most fundable?").describe())
        run = sp.runs.create(ds.id, "Which proposal is most fundable?")
        run = sp.runs.wait(run.id, on_progress=lambda r: print(r.progress.done, "/", r.progress.total))
        print(sp.runs.results(run.id).summary)
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .. import __version__
from .config import resolve
from .errors import CloudError, connection_error
from .folder import read_folder
from .models import (
    Billing,
    Checkout,
    Comparison,
    Dataset,
    Estimate,
    ImportPreview,
    Item,
    LedgerEntry,
    Me,
    Page,
    Pricing,
    Results,
    Run,
    RunEvent,
)
from .records import RecordsError, is_table, read_records

RETRY_STATUS = (502, 503, 504)
RANKING_OPTIONS = ("rounds", "both_orders", "pairing", "goal", "top_n", "max_chars", "seed")


@dataclass
class _Call[T]:
    method: str
    path: str
    parse: Callable[[Any], T]
    json: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    auth: bool = True  # False for the few public routes, which work without an API key
    files: list[tuple[str, tuple[str, bytes]]] | None = None  # multipart uploads: (field, (filename, content))
    data: dict[str, str] | None = None  # multipart form fields

    @property
    def idempotent(self) -> bool:
        return self.method in ("GET", "DELETE")


def _page[T](parse: Callable[[dict[str, Any]], T]) -> Callable[[Any], Page[T]]:
    def go(body: Any) -> Page[T]:
        return Page([parse(x) for x in body.get("data") or []], body.get("next_cursor"))
    return go


def _none(_: Any) -> None:
    return None


def _params(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def run_body(dataset_id: str, rubric: str, *, type: str = "ranking", name: str | None = None,
             judge: str | dict[str, Any] | None = None, max_cost_usd: float | None = None,
             ephemeral: bool = False, **options: Any) -> dict[str, Any]:
    """The JSON body of a run request. ``judge`` is quick (the default), balanced, thorough, an OpenRouter model
    id, mock or claude_cli, or a ``{provider, model}`` object. ``options`` are the run type's own (for a ranking:
    rounds, goal, ...)."""
    unknown = set(options) - set(RANKING_OPTIONS)
    if unknown:
        raise TypeError("Unknown ranking options: %s" % ", ".join(sorted(unknown)))
    body: dict[str, Any] = {"type": type, "dataset_id": dataset_id,
                            type: {"rubric": rubric, **{k: v for k, v in options.items() if v is not None}}}
    if name:
        body["name"] = name
    if judge is not None:
        body["judge"] = judge
    if max_cost_usd is not None:
        body["max_cost_usd"] = max_cost_usd
    if ephemeral:  # a one-off: kept out of lists and deleted after 30 days
        body["ephemeral"] = True
    return body


class _Requests:
    """Every API request, as data."""

    @staticmethod
    def me() -> _Call[Me]:
        return _Call("GET", "/v1/me", Me.parse)

    @staticmethod
    def settings() -> _Call[dict[str, Any]]:
        return _Call("GET", "/v1/settings", dict)

    @staticmethod
    def usage() -> _Call[dict[str, Any]]:
        return _Call("GET", "/v1/usage", dict)

    @staticmethod
    def models(params: dict[str, Any]) -> _Call[dict[str, Any]]:
        return _Call("GET", "/v1/models", dict, params=_params(**params))

    @staticmethod
    def samples_run(sample_id: str, judge: str | dict[str, Any] | None) -> _Call[Run]:
        body = {"judge": judge} if judge is not None else None
        return _Call("POST", "/v1/samples/%s/run" % sample_id, Run.parse, json=body)

    @staticmethod
    def pricing() -> _Call[Pricing]:
        return _Call("GET", "/v1/pricing", Pricing.parse, auth=False)

    @staticmethod
    def billing_get() -> _Call[Billing]:
        return _Call("GET", "/v1/billing", Billing.parse)

    @staticmethod
    def billing_ledger(limit: int | None, cursor: str | None) -> _Call[Page[LedgerEntry]]:
        return _Call("GET", "/v1/billing/ledger", _page(LedgerEntry.parse), params=_params(limit=limit, cursor=cursor))

    @staticmethod
    def billing_checkout(amount_usd: int, success_url: str | None, cancel_url: str | None) -> _Call[Checkout]:
        body: dict[str, Any] = {"amount_usd": amount_usd, **_params(success_url=success_url, cancel_url=cancel_url)}
        return _Call("POST", "/v1/billing/checkout", Checkout.parse, json=body)

    @staticmethod
    def billing_portal() -> _Call[Checkout]:
        return _Call("POST", "/v1/billing/portal", lambda d: Checkout.parse({"id": "", **d}))

    @staticmethod
    def datasets_list(limit: int | None, cursor: str | None) -> _Call[Page[Dataset]]:
        return _Call("GET", "/v1/datasets", _page(Dataset.parse), params=_params(limit=limit, cursor=cursor))

    @staticmethod
    def datasets_get(dataset_id: str) -> _Call[Dataset]:
        return _Call("GET", "/v1/datasets/%s" % dataset_id, Dataset.parse)

    @staticmethod
    def datasets_create(name: str, items: list[dict[str, Any]], ephemeral: bool = False) -> _Call[Dataset]:
        body: dict[str, Any] = {"name": name, "items": items}
        if ephemeral:
            body["ephemeral"] = True
        return _Call("POST", "/v1/datasets", Dataset.parse, json=body)

    @staticmethod
    def datasets_upload(name: str, files: list[tuple[str, bytes]], redact: bool, ephemeral: bool) -> _Call[Dataset]:
        data = {"name": name, **({"redact": "true"} if redact else {}), **({"ephemeral": "true"} if ephemeral else {})}
        return _Call("POST", "/v1/datasets/upload", Dataset.parse, files=[("files", f) for f in files], data=data)

    @staticmethod
    def imports_create(filename: str, content: bytes, sheet: str | None) -> _Call[ImportPreview]:
        return _Call("POST", "/v1/imports", ImportPreview.parse, files=[("file", (filename, content))],
                     data={"sheet": sheet} if sheet else None)

    @staticmethod
    def imports_commit(import_id: str, name: str, mapping: dict[str, Any], sheet: str | None,
                       ephemeral: bool) -> _Call[Dataset]:
        body: dict[str, Any] = {"name": name, "mapping": mapping, **_params(sheet=sheet)}
        if ephemeral:
            body["ephemeral"] = True
        return _Call("POST", "/v1/imports/%s/commit" % import_id, Dataset.parse, json=body)

    @staticmethod
    def datasets_add_items(dataset_id: str, items: list[dict[str, str]]) -> _Call[Dataset]:
        return _Call("POST", "/v1/datasets/%s/items" % dataset_id, Dataset.parse, json=items)

    @staticmethod
    def datasets_delete(dataset_id: str) -> _Call[None]:
        return _Call("DELETE", "/v1/datasets/%s" % dataset_id, _none)

    @staticmethod
    def items_get(item_id: str) -> _Call[Item]:
        return _Call("GET", "/v1/items/%s" % item_id, Item.parse)

    @staticmethod
    def runs_estimate(body: dict[str, Any]) -> _Call[Estimate]:
        return _Call("POST", "/v1/runs/estimate", Estimate.parse, json=body)

    @staticmethod
    def runs_create(body: dict[str, Any]) -> _Call[Run]:
        return _Call("POST", "/v1/runs", Run.parse, json=body)

    @staticmethod
    def runs_list(dataset_id: str | None, limit: int | None, cursor: str | None) -> _Call[Page[Run]]:
        return _Call("GET", "/v1/runs", _page(Run.parse),
                     params=_params(dataset_id=dataset_id, limit=limit, cursor=cursor))

    @staticmethod
    def runs_get(run_id: str) -> _Call[Run]:
        return _Call("GET", "/v1/runs/%s" % run_id, Run.parse)

    @staticmethod
    def runs_cancel(run_id: str) -> _Call[Run]:
        return _Call("POST", "/v1/runs/%s/cancel" % run_id, Run.parse)

    @staticmethod
    def runs_delete(run_id: str) -> _Call[None]:
        return _Call("DELETE", "/v1/runs/%s" % run_id, _none)

    @staticmethod
    def runs_results(run_id: str) -> _Call[Results]:
        return _Call("GET", "/v1/runs/%s/results" % run_id, Results.parse)

    @staticmethod
    def runs_comparisons(run_id: str, item_id: str | None, status: str | None, limit: int | None,
                         cursor: str | None) -> _Call[Page[Comparison]]:
        return _Call("GET", "/v1/runs/%s/comparisons" % run_id, _page(Comparison.parse),
                     params=_params(item_id=item_id, status=status, limit=limit, cursor=cursor))


R = _Requests


SPREADSHEETS = (".csv", ".tsv", ".xlsx", ".xls")  # parsed by the server's import, with its column mapping
DOCUMENTS = (".pdf", ".docx")  # converted to text by the server's upload


@dataclass
class _Push:
    """How a source becomes a dataset: items sent as JSON, files uploaded, or a spreadsheet imported."""
    kind: str  # "items" | "upload" | "import"
    name: str
    items: list[dict[str, Any]] = field(default_factory=list)
    files: list[tuple[str, bytes]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    path: Path | None = None
    mapping: dict[str, Any] = field(default_factory=dict)  # the caller's fields; the server's guess fills the rest


def _plan_push(source: str | Path, name: str | None, redact: bool, text_fields: list[str] | None,
               id_field: str | None, title_field: str | None, metadata_fields: list[str] | None) -> _Push:
    path = Path(source)
    fields = {"text_fields": text_fields, "id_field": id_field, "title_field": title_field,
              "metadata_fields": metadata_fields}
    given = {k: v for k, v in fields.items() if v}
    name = name or path.resolve().name
    if path.is_file() and path.suffix.lower() in SPREADSHEETS:
        return _Push("import", name, path=path, mapping=given)
    if is_table(path):  # JSONL and JSON: read here, sent as items with their source ids
        try:
            recs = read_records(path, text_fields, id_field, title_field, metadata_fields)
        except RecordsError as exc:
            raise CloudError(str(exc), code="invalid_file", hint=exc.hint) from exc
        items = []
        for r in recs.items:
            item: dict[str, Any] = {"title": r.title, "text": r.text, "source_id": r.source_id}
            if r.metadata:
                item["metadata"] = r.metadata
            items.append(item)
        if not items:
            raise CloudError("No rows of %s have text." % path.name, code="no_items",
                             hint="Check --text-field names the field with the text.")
        return _Push("items", name, items=items,
                     skipped=[{"path": "%s#%s" % (path.name, sid), "reason": why} for sid, why in recs.skipped])
    if given:
        raise CloudError("text_fields, id_field, title_field and metadata_fields are for a JSONL, JSON, CSV or XLSX "
                         "file, and %s is not one." % source, code="usage",
                         hint="Pass a table file, or leave the fields out for a folder.")
    if not path.is_dir():
        raise FileNotFoundError("%s is not a folder or a .jsonl, .json, .csv or .xlsx file" % source)
    # A folder is uploaded as files, as the web app does: the server turns PDF and DOCX into text, titles each item
    # from its first heading (source_id from the file name), redacts on request and reports duplicates.
    found = read_folder(path)
    docs = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in DOCUMENTS
                  and not any(part.startswith(".") for part in p.relative_to(path).parts))
    files = [(t, (path / t).read_bytes()) for t, _ in found.items]
    files += [(str(p.relative_to(path)), p.read_bytes()) for p in docs]
    if not files:
        raise CloudError("No .md, .txt, .html, .pdf or .docx files with text in %s." % source, code="no_items",
                         hint="Point at a folder of documents, one per file, or at a JSONL, CSV or XLSX file.")
    skipped = [{"path": p, "reason": "not .md, .txt, .html, .pdf or .docx"} for p, _ in found.skipped
               if not p.lower().endswith(DOCUMENTS) and _ != "empty"]
    skipped += [{"path": p, "reason": why} for p, why in found.skipped if why == "empty"]
    # What the server will hold, to spot an identical earlier upload: each file's name, and its length for plain
    # text, which the server keeps as it is.
    compare = [{"filename": t.rsplit("/", 1)[-1], "n_chars": len(x) if t.lower().endswith((".md", ".txt")) else None}
               for t, x in found.items]
    compare += [{"filename": p.name, "n_chars": None} for p in docs]
    return _Push("upload", name, files=files, skipped=skipped, items=compare)


def _import_mapping(preview: ImportPreview, given: dict[str, Any]) -> dict[str, Any]:
    """The server's guess, with whatever the caller named taking its place."""
    mapping = {k: preview.mapping.get(k) for k in ("id_field", "title_field", "text_fields", "metadata_fields")}
    mapping.update(given)
    if "id_field" in given and "title_field" not in given:
        mapping["title_field"] = None  # the title defaults to the id the caller chose, not the server's guess
    mapping["metadata_fields"] = [f for f in mapping.get("metadata_fields") or []
                                  if f not in (mapping.get("text_fields") or [])]
    return mapping


def _local_import_items(push: _Push, mapping: dict[str, Any]) -> list[dict[str, Any]] | None:
    """For a CSV or TSV, the items the import will make, to spot an identical earlier upload. None for XLSX."""
    if push.path is None or push.path.suffix.lower() not in (".csv", ".tsv"):
        return None
    try:
        recs = read_records(push.path, mapping.get("text_fields"), mapping.get("id_field"),
                            mapping.get("title_field"), mapping.get("metadata_fields"))
    except RecordsError:
        return None
    return [{"title": r.title, "text": r.text} for r in recs.items]


def _finish(ds: Dataset, push: _Push) -> Dataset:
    """The skipped list in one shape, {path, reason}, whichever route made the dataset."""
    skipped = list(push.skipped)
    for s in ds.raw.get("skipped") or []:
        if "row" in s:
            skipped.append({"path": "%s#row %s" % (push.path.name if push.path else "", s["row"]),
                            "reason": s.get("reason", "")})
        else:
            skipped.append({"path": s.get("filename") or s.get("path") or "", "reason": s.get("reason", "")})
    ds.skipped = skipped
    return ds


def _event(line: str) -> RunEvent | None:
    """A RunEvent from one SSE line, or None for comments, blank lines and other fields."""
    if not line.startswith("data:"):
        return None
    return RunEvent.parse(json.loads(line[5:].strip()))


def _stream_error(exc: Exception) -> CloudError:
    return CloudError("The run's event stream broke (%s)." % type(exc).__name__, code="stream_error",
                      hint="It is safe to follow the run again; the run itself keeps going.")


def _same_items(ds: Dataset, items: list[dict[str, Any]]) -> bool:
    """Whether a dataset already holds exactly these items: for uploaded files, the same file names (and lengths
    where known); otherwise the same titles, in order, with the same lengths."""
    if items and "filename" in items[0]:
        held = sorted((i.raw.get("filename") or "", i.n_chars) for i in ds.items or [])
        want = sorted((it["filename"], it["n_chars"]) for it in items)
        return len(held) == len(want) and all(h[0] == w[0] and (w[1] is None or h[1] == w[1])
                                              for h, w in zip(held, want, strict=True))
    held_titles = [(i.title, i.n_chars) for i in ds.items or []]
    return held_titles == [(it["title"], len(it["text"])) for it in items]


def _parse_response[T](call: _Call[T], resp: httpx.Response) -> T:
    if resp.status_code >= 400:
        raise CloudError.from_response(resp)
    if resp.status_code == 204 or not resp.content:
        return call.parse(None)
    try:
        body = resp.json()
    except ValueError as exc:
        raise CloudError("The server sent a response that is not JSON.", code="bad_response",
                         status=resp.status_code, hint="Check SUPPOSING_API_URL points at the API.") from exc
    return call.parse(body)


class _Base:
    def __init__(self, api_key: str | None, api_url: str | None, timeout: float, retries: int,
                 source: str | None = None):
        creds = resolve(api_key, api_url)
        self.api_url = creds.api_url
        self.key_source = creds.source
        self.retries = retries
        self.timeout = timeout
        self._headers = {"User-Agent": "supposing-python/%s" % __version__, "Accept": "application/json"}
        if creds.api_key:
            self._headers["Authorization"] = "Bearer " + creds.api_key
        if source:  # recorded as the source of datasets and runs this client creates (web, api, cli, mcp)
            self._headers["Supposing-Source"] = source

    def _no_key(self) -> CloudError:
        return CloudError("No API key.", code="no_api_key",
                          hint="Run `supposing login`, or set SUPPOSING_API_KEY. Create a key in the web app under Developers.")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * 2 ** attempt, 8.0)


class Client(_Base):
    """Blocking client. Use as a context manager, or call :meth:`close`."""

    def __init__(self, api_key: str | None = None, api_url: str | None = None, *, timeout: float = 30.0,
                 retries: int = 2, transport: httpx.BaseTransport | None = None, source: str | None = None):
        super().__init__(api_key, api_url, timeout, retries, source)
        self._http = httpx.Client(base_url=self.api_url, headers=self._headers, timeout=timeout, transport=transport)
        self.datasets = Datasets(self)
        self.items = Items(self)
        self.runs = Runs(self)
        self.billing = BillingApi(self)

    def send[T](self, call: _Call[T]) -> T:
        if call.auth and "Authorization" not in self._headers:
            raise self._no_key()
        attempt = 0
        while True:
            try:
                resp = self._http.request(call.method, call.path, json=call.json, params=call.params or None,
                                                   files=call.files, data=call.data)
            except httpx.TransportError as exc:
                if call.idempotent and attempt < self.retries:
                    time.sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                raise connection_error(self.api_url, exc) from exc
            if resp.status_code in RETRY_STATUS and call.idempotent and attempt < self.retries:
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue
            return _parse_response(call, resp)

    def me(self) -> Me:
        return self.send(R.me())

    def settings(self) -> dict[str, Any]:
        """The workspace's settings, including ``providers``: the judge providers it can use now."""
        return self.send(R.settings())

    def usage(self) -> dict[str, Any]:
        """This calendar month so far: runs, judge calls, tokens and spend, with the plan's limits."""
        return self.send(R.usage())

    def pricing(self) -> Pricing:
        """The price of a judge call on our model key, by tier. Needs no API key."""
        return self.send(R.pricing())

    def models(self, dataset_id: str | None = None, *, filter: str | None = None, q: str | None = None,
               **run: Any) -> dict[str, Any]:
        """The judges this workspace can pick: ``levels`` (quick, balanced, thorough), then ``data`` (OpenRouter
        models) and ``local`` (mock, claude_cli; empty when hosted). With a ``dataset_id`` each is priced for that
        run; ``run`` takes its goal, top_n, rounds, both_orders and max_cost_usd. ``filter``: suggested, zdr or
        cheapest. Pass an option's ``id`` as a run's judge."""
        return self.send(R.models({"dataset_id": dataset_id, "filter": filter, "q": q, **run}))

    def open_sample(self, sample_id: str, judge: str | dict[str, Any] | None = None) -> Run:
        """A built-in sample copied into the workspace. Without a judge, its stored ranking as a finished run,
        free and instant; with one, a new run, paid like any run."""
        return self.send(R.samples_run(sample_id, judge))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncClient(_Base):
    """Async client. Use as an async context manager, or await :meth:`close`."""

    def __init__(self, api_key: str | None = None, api_url: str | None = None, *, timeout: float = 30.0,
                 retries: int = 2, transport: httpx.AsyncBaseTransport | None = None, source: str | None = None):
        super().__init__(api_key, api_url, timeout, retries, source)
        self._http = httpx.AsyncClient(base_url=self.api_url, headers=self._headers, timeout=timeout,
                                       transport=transport)
        self.datasets = AsyncDatasets(self)
        self.items = AsyncItems(self)
        self.runs = AsyncRuns(self)
        self.billing = AsyncBillingApi(self)

    async def send[T](self, call: _Call[T]) -> T:
        if call.auth and "Authorization" not in self._headers:
            raise self._no_key()
        attempt = 0
        while True:
            try:
                resp = await self._http.request(call.method, call.path, json=call.json, params=call.params or None,
                                                   files=call.files, data=call.data)
            except httpx.TransportError as exc:
                if call.idempotent and attempt < self.retries:
                    await asyncio.sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                raise connection_error(self.api_url, exc) from exc
            if resp.status_code in RETRY_STATUS and call.idempotent and attempt < self.retries:
                await asyncio.sleep(self._backoff(attempt))
                attempt += 1
                continue
            return _parse_response(call, resp)

    async def me(self) -> Me:
        return await self.send(R.me())

    async def settings(self) -> dict[str, Any]:
        return await self.send(R.settings())

    async def usage(self) -> dict[str, Any]:
        return await self.send(R.usage())

    async def pricing(self) -> Pricing:
        return await self.send(R.pricing())

    async def models(self, dataset_id: str | None = None, *, filter: str | None = None, q: str | None = None,
                     **run: Any) -> dict[str, Any]:
        return await self.send(R.models({"dataset_id": dataset_id, "filter": filter, "q": q, **run}))

    async def open_sample(self, sample_id: str, judge: str | dict[str, Any] | None = None) -> Run:
        return await self.send(R.samples_run(sample_id, judge))

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# --- resources ------------------------------------------------------------------------------------------------

class Datasets:
    def __init__(self, c: Client):
        self._c = c

    def list(self, limit: int | None = None, cursor: str | None = None) -> Page[Dataset]:
        return self._c.send(R.datasets_list(limit, cursor))

    def iter(self) -> Iterator[Dataset]:
        cursor = None
        while True:
            page = self.list(cursor=cursor)
            yield from page.data
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    def get(self, dataset_id: str) -> Dataset:
        return self._c.send(R.datasets_get(dataset_id))

    def create(self, name: str, items: list[dict[str, Any]] | None = None, ephemeral: bool = False) -> Dataset:
        """``items`` are ``{"title": ..., "text": ...}``. ``ephemeral`` datasets stay out of lists and are deleted
        after 30 days: use it for one-off comparisons."""
        return self._c.send(R.datasets_create(name, items or [], ephemeral))

    def push(self, source: str | Path, name: str | None = None, reuse: bool = False, *,
             text_fields: list[str] | None = None, id_field: str | None = None, title_field: str | None = None,
             metadata_fields: list[str] | None = None, sheet: str | None = None, redact: bool = False,
             ephemeral: bool = False) -> Dataset:
        """Create a dataset from a folder or a table file.

        - A folder: every .md, .txt, .html, .pdf and .docx file is one item, titled by its path.
        - A JSONL or JSON file: every row is one item, read here.
        - A CSV, TSV or XLSX file: imported by the server, which guesses the columns; ``sheet`` picks a sheet.

        For a table file, ``text_fields`` name the field(s) the judge reads (joined under a heading each when
        several), ``id_field`` the row's id (kept as ``source_id`` in every output), ``title_field`` the title
        (default: the id) and ``metadata_fields`` fields kept but never shown to the judge. ``redact`` removes
        names, emails and phone numbers from what the judge reads (folders only). ``Dataset.skipped`` lists what
        was left out and ``Dataset.duplicates`` likely duplicates. With ``reuse``, an existing dataset of the same
        name holding the same items is returned instead (``Dataset.reused`` is True)."""
        push = _plan_push(source, name, redact, text_fields, id_field, title_field, metadata_fields)
        compare = push.items if push.kind in ("items", "upload") else None
        preview = None
        if push.kind == "import":
            assert push.path is not None
            preview = self._c.send(R.imports_create(push.path.name, push.path.read_bytes(), sheet))
            push.mapping = _import_mapping(preview, push.mapping)
            compare = _local_import_items(push, push.mapping)
        if reuse and compare is not None:
            for d in self.iter():
                if d.name == push.name and d.n_items == len(compare):
                    full = self.get(d.id)
                    if _same_items(full, compare):
                        full.reused = True
                        return _finish(full, push)
        if preview is not None:
            ds = self._c.send(R.imports_commit(preview.id, push.name, push.mapping, sheet, ephemeral))
        elif push.kind == "upload":
            ds = self._c.send(R.datasets_upload(push.name, push.files, redact, ephemeral))
        else:
            ds = self._c.send(R.datasets_create(push.name, push.items, ephemeral))
        return _finish(ds, push)

    def add_items(self, dataset_id: str, items: list[dict[str, str]]) -> Dataset:
        return self._c.send(R.datasets_add_items(dataset_id, items))

    def delete(self, dataset_id: str) -> None:
        self._c.send(R.datasets_delete(dataset_id))


class BillingApi:
    """Prepaid credit. Runs on our model key are paid from it; runs on your own key, mock and claude_cli are not."""

    def __init__(self, c: Client):
        self._c = c

    def get(self) -> Billing:
        """Balance, what running runs hold, and the top-up options."""
        return self._c.send(R.billing_get())

    def ledger(self, limit: int | None = None, cursor: str | None = None) -> Page[LedgerEntry]:
        """Every credit movement, newest first."""
        return self._c.send(R.billing_ledger(limit, cursor))

    def checkout(self, amount_usd: int, success_url: str | None = None, cancel_url: str | None = None) -> Checkout:
        """A payment page for ``amount_usd`` of credit. Nothing is charged until someone pays on that page."""
        return self._c.send(R.billing_checkout(amount_usd, success_url, cancel_url))

    def portal(self) -> Checkout:
        """A link to receipts and payment details."""
        return self._c.send(R.billing_portal())


class Items:
    def __init__(self, c: Client):
        self._c = c

    def get(self, item_id: str) -> Item:
        return self._c.send(R.items_get(item_id))


class Runs:
    def __init__(self, c: Client):
        self._c = c

    def estimate(self, dataset_id: str, rubric: str, **kw: Any) -> Estimate:
        """What a run would cost, without starting it. Takes the same arguments as :meth:`create`."""
        return self._c.send(R.runs_estimate(run_body(dataset_id, rubric, **kw)))

    def create(self, dataset_id: str, rubric: str, *, type: str = "ranking", name: str | None = None,
               judge: str | dict[str, Any] | None = None, max_cost_usd: float | None = None,
               ephemeral: bool = False, **options: Any) -> Run:
        """Start a run. ``judge`` is quick (the default), balanced, thorough, an OpenRouter model id or mock.
        ``options`` are the type's own: for a ranking, rounds, both_orders, pairing, goal, top_n, max_chars,
        seed."""
        return self._c.send(R.runs_create(run_body(dataset_id, rubric, type=type, name=name, judge=judge,
                                                   max_cost_usd=max_cost_usd, ephemeral=ephemeral, **options)))

    def list(self, dataset_id: str | None = None, limit: int | None = None, cursor: str | None = None
             ) -> Page[Run]:
        return self._c.send(R.runs_list(dataset_id, limit, cursor))

    def get(self, run_id: str) -> Run:
        return self._c.send(R.runs_get(run_id))

    def cancel(self, run_id: str) -> Run:
        return self._c.send(R.runs_cancel(run_id))

    def delete(self, run_id: str) -> None:
        self._c.send(R.runs_delete(run_id))

    def events(self, run_id: str) -> Iterator[RunEvent]:
        """Follow a run live: a ``progress`` event whenever calls complete, then one ``finished`` event."""
        c = self._c
        if "Authorization" not in c._headers:
            raise c._no_key()
        path = "/v1/runs/%s/events" % run_id
        try:
            with c._http.stream("GET", path, headers={"Accept": "text/event-stream"},
                                timeout=httpx.Timeout(c.timeout, read=None)) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise CloudError.from_response(resp)
                for line in resp.iter_lines():
                    ev = _event(line)
                    if ev:
                        yield ev
                        if ev.event == "finished":
                            return
        except httpx.TransportError as exc:
            raise _stream_error(exc) from exc

    def wait(self, run_id: str, on_progress: Callable[[Run], None] | None = None, poll_interval: float = 1.0,
             timeout: float | None = None, on_event: Callable[[RunEvent], None] | None = None) -> Run:
        """Block until the run is done, failed or canceled. Follows the event stream, and polls every
        ``poll_interval`` seconds if the stream is unavailable. ``on_progress`` gets the run at each update;
        ``on_event`` also gets the current leaders."""
        deadline = time.monotonic() + timeout if timeout else None
        if not timeout:
            try:
                for ev in self.events(run_id):
                    if on_progress:
                        on_progress(ev.run)
                    if on_event:
                        on_event(ev)
                    if ev.run.finished:
                        return ev.run
            except CloudError as exc:
                if exc.code not in ("stream_error", "not_found") and exc.status not in (404, 405, 501):
                    raise
        while True:
            run = self.get(run_id)
            if on_progress:
                on_progress(run)
            if run.finished:
                return run
            if deadline and time.monotonic() > deadline:
                raise CloudError("Run %s is still %s." % (run_id, run.status), code="wait_timeout",
                                 hint="It keeps running on the server. Check again with `supposing runs watch`.")
            time.sleep(poll_interval)

    def results(self, run_id: str) -> Results:
        return self._c.send(R.runs_results(run_id))

    def comparisons(self, run_id: str, item_id: str | None = None, status: str | None = None,
                    limit: int | None = None, cursor: str | None = None) -> Page[Comparison]:
        return self._c.send(R.runs_comparisons(run_id, item_id, status, limit, cursor))


class AsyncDatasets:
    def __init__(self, c: AsyncClient):
        self._c = c

    async def list(self, limit: int | None = None, cursor: str | None = None) -> Page[Dataset]:
        return await self._c.send(R.datasets_list(limit, cursor))

    async def iter(self) -> AsyncIterator[Dataset]:
        cursor = None
        while True:
            page = await self.list(cursor=cursor)
            for d in page.data:
                yield d
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    async def get(self, dataset_id: str) -> Dataset:
        return await self._c.send(R.datasets_get(dataset_id))

    async def create(self, name: str, items: list[dict[str, Any]] | None = None, ephemeral: bool = False
                     ) -> Dataset:
        return await self._c.send(R.datasets_create(name, items or [], ephemeral))

    async def push(self, source: str | Path, name: str | None = None, reuse: bool = False, *,
                   text_fields: list[str] | None = None, id_field: str | None = None,
                   title_field: str | None = None, metadata_fields: list[str] | None = None,
                   sheet: str | None = None, redact: bool = False, ephemeral: bool = False) -> Dataset:
        push = _plan_push(source, name, redact, text_fields, id_field, title_field, metadata_fields)
        compare = push.items if push.kind in ("items", "upload") else None
        preview = None
        if push.kind == "import":
            assert push.path is not None
            preview = await self._c.send(R.imports_create(push.path.name, push.path.read_bytes(), sheet))
            push.mapping = _import_mapping(preview, push.mapping)
            compare = _local_import_items(push, push.mapping)
        if reuse and compare is not None:
            async for d in self.iter():
                if d.name == push.name and d.n_items == len(compare):
                    full = await self.get(d.id)
                    if _same_items(full, compare):
                        full.reused = True
                        return _finish(full, push)
        if preview is not None:
            ds = await self._c.send(R.imports_commit(preview.id, push.name, push.mapping, sheet, ephemeral))
        elif push.kind == "upload":
            ds = await self._c.send(R.datasets_upload(push.name, push.files, redact, ephemeral))
        else:
            ds = await self._c.send(R.datasets_create(push.name, push.items, ephemeral))
        return _finish(ds, push)

    async def add_items(self, dataset_id: str, items: list[dict[str, str]]) -> Dataset:
        return await self._c.send(R.datasets_add_items(dataset_id, items))

    async def delete(self, dataset_id: str) -> None:
        await self._c.send(R.datasets_delete(dataset_id))


class AsyncBillingApi:
    def __init__(self, c: AsyncClient):
        self._c = c

    async def get(self) -> Billing:
        return await self._c.send(R.billing_get())

    async def ledger(self, limit: int | None = None, cursor: str | None = None) -> Page[LedgerEntry]:
        return await self._c.send(R.billing_ledger(limit, cursor))

    async def checkout(self, amount_usd: int, success_url: str | None = None, cancel_url: str | None = None
                       ) -> Checkout:
        return await self._c.send(R.billing_checkout(amount_usd, success_url, cancel_url))

    async def portal(self) -> Checkout:
        return await self._c.send(R.billing_portal())


class AsyncItems:
    def __init__(self, c: AsyncClient):
        self._c = c

    async def get(self, item_id: str) -> Item:
        return await self._c.send(R.items_get(item_id))


class AsyncRuns:
    def __init__(self, c: AsyncClient):
        self._c = c

    async def estimate(self, dataset_id: str, rubric: str, **kw: Any) -> Estimate:
        return await self._c.send(R.runs_estimate(run_body(dataset_id, rubric, **kw)))

    async def create(self, dataset_id: str, rubric: str, *, type: str = "ranking", name: str | None = None,
                     judge: str | dict[str, Any] | None = None, max_cost_usd: float | None = None,
                     ephemeral: bool = False, **options: Any) -> Run:
        return await self._c.send(R.runs_create(run_body(dataset_id, rubric, type=type, name=name, judge=judge,
                                                         max_cost_usd=max_cost_usd, ephemeral=ephemeral, **options)))

    async def list(self, dataset_id: str | None = None, limit: int | None = None, cursor: str | None = None
                   ) -> Page[Run]:
        return await self._c.send(R.runs_list(dataset_id, limit, cursor))

    async def get(self, run_id: str) -> Run:
        return await self._c.send(R.runs_get(run_id))

    async def cancel(self, run_id: str) -> Run:
        return await self._c.send(R.runs_cancel(run_id))

    async def delete(self, run_id: str) -> None:
        await self._c.send(R.runs_delete(run_id))

    async def events(self, run_id: str) -> AsyncIterator[RunEvent]:
        c = self._c
        if "Authorization" not in c._headers:
            raise c._no_key()
        path = "/v1/runs/%s/events" % run_id
        try:
            async with c._http.stream("GET", path, headers={"Accept": "text/event-stream"},
                                      timeout=httpx.Timeout(c.timeout, read=None)) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise CloudError.from_response(resp)
                async for line in resp.aiter_lines():
                    ev = _event(line)
                    if ev:
                        yield ev
                        if ev.event == "finished":
                            return
        except httpx.TransportError as exc:
            raise _stream_error(exc) from exc

    async def wait(self, run_id: str, on_progress: Callable[[Run], None] | None = None, poll_interval: float = 1.0,
                   timeout: float | None = None, on_event: Callable[[RunEvent], None] | None = None) -> Run:
        deadline = time.monotonic() + timeout if timeout else None
        if not timeout:
            try:
                async for ev in self.events(run_id):
                    if on_progress:
                        on_progress(ev.run)
                    if on_event:
                        on_event(ev)
                    if ev.run.finished:
                        return ev.run
            except CloudError as exc:
                if exc.code not in ("stream_error", "not_found") and exc.status not in (404, 405, 501):
                    raise
        while True:
            run = await self.get(run_id)
            if on_progress:
                on_progress(run)
            if run.finished:
                return run
            if deadline and time.monotonic() > deadline:
                raise CloudError("Run %s is still %s." % (run_id, run.status), code="wait_timeout",
                                 hint="It keeps running on the server. Check again with `supposing runs watch`.")
            await asyncio.sleep(poll_interval)

    async def results(self, run_id: str) -> Results:
        return await self._c.send(R.runs_results(run_id))

    async def comparisons(self, run_id: str, item_id: str | None = None, status: str | None = None,
                          limit: int | None = None, cursor: str | None = None) -> Page[Comparison]:
        return await self._c.send(R.runs_comparisons(run_id, item_id, status, limit, cursor))
