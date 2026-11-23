"""Reading a JSONL, JSON or CSV file into items: one row per item, the text from named fields.

Shared by `supposing rank` (local), `supposing run`/`datasets push` and `Client.datasets.push`. Option names
follow the API's import: ``text_fields`` (one or more, joined in order under a "## <field>" heading each when
there are several), ``id_field`` (kept as each item's source id), ``title_field`` (defaults to ``id_field``) and
``metadata_fields`` (kept on the item, never shown to the judge). Other fields are dropped.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TABLE_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".tsv")
LIKELY_TEXT = ("text", "output", "response", "answer", "completion", "content", "body", "document")
LIKELY_ID = ("id", "uid", "key", "name", "submission id", "submission_id")


class RecordsError(ValueError):
    """A file that can't be read as items. ``hint`` says what to pass instead."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


@dataclass
class Record:
    source_id: str  # the row's id: id_field's value, or its row number
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Records:
    items: list[Record] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (row id, reason)
    fields: list[str] = field(default_factory=list)  # every field seen, for error messages


def is_table(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() in TABLE_SUFFIXES


def _rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for n, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RecordsError("%s line %d is not JSON: %s." % (path.name, n, exc.msg)) from exc
        return rows
    if suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):  # {"items": [...]} or {"data": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if not isinstance(data, list):
            raise RecordsError("%s is not a JSON list of objects." % path.name, "Use one object per item, or JSONL.")
        return data
    dialect = "excel-tab" if suffix == ".tsv" else "excel"
    return list(csv.DictReader(raw.splitlines(), dialect=dialect))


def _pick(fields: list[str], wanted: tuple[str, ...]) -> str | None:
    lower = {f.lower(): f for f in fields}
    return next((lower[w] for w in wanted if w in lower), None)


def read_records(path: str | Path, text_fields: list[str] | None = None, id_field: str | None = None,
                 title_field: str | None = None, metadata_fields: list[str] | None = None) -> Records:
    """Items from a table file. Without ``text_fields`` a single obvious column (text, output, response, ...) is
    used; without ``id_field`` an obvious id column, else the row number."""
    p = Path(path)
    rows = [r for r in _rows(p) if isinstance(r, dict)]
    if not rows:
        raise RecordsError("%s has no rows." % p.name)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = Records(fields=fields)
    listed = ", ".join(fields[:12]) + (" ..." if len(fields) > 12 else "")
    if not text_fields:
        guess = _pick(fields, LIKELY_TEXT)
        if not guess:
            raise RecordsError("Which field of %s holds the text? It has: %s." % (p.name, listed),
                               "Name it: --text-field <field> (repeat it to join several fields).")
        text_fields = [guess]
    missing = [f for f in [*text_fields, id_field, title_field, *(metadata_fields or [])] if f and f not in fields]
    if missing:
        raise RecordsError("%s has no field %s. It has: %s." % (p.name, ", ".join(repr(m) for m in missing), listed),
                           "Field names are case-sensitive.")
    id_field = id_field or _pick(fields, LIKELY_ID)
    seen: set[str] = set()
    for n, row in enumerate(rows, 1):
        sid = str(row.get(id_field) if id_field and row.get(id_field) not in (None, "") else n).strip()
        if sid in seen:
            raise RecordsError("%s repeats the id %r (row %d)." % (p.name, sid, n),
                               "Pick a field whose values are unique with --id-field, or leave it out to use row "
                               "numbers.")
        seen.add(sid)
        parts = [(f, "" if row.get(f) is None else str(row.get(f)).strip()) for f in text_fields]
        parts = [(f, v) for f, v in parts if v]
        if not parts:
            out.skipped.append((sid, "no text in %s" % ", ".join(text_fields)))
            continue
        text = parts[0][1] if len(text_fields) == 1 else "\n\n".join("## %s\n\n%s" % (f, v) for f, v in parts)
        title = str(row.get(title_field) or "").strip() if title_field else ""
        meta = {f: str(row[f]) for f in metadata_fields or [] if row.get(f) not in (None, "")}
        out.items.append(Record(sid, (title or sid)[:300], text, meta))
    return out
