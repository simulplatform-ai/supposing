"""Typed views of API responses. Each keeps the full response in ``raw``, so fields the client does not know yet
are still reachable, and ``to_dict()`` returns exactly what the API sent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

FINISHED = ("done", "failed", "canceled")


def _dt(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class _Model:
    raw: dict[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass
class Page[T]:
    data: list[T]
    next_cursor: str | None


@dataclass
class Me(_Model):
    org_id: str
    org_name: str
    email: str | None
    via: str

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Me:
        return cls(d, str(d.get("org_id")), d.get("org_name") or "", d.get("email"), d.get("via") or "")


@dataclass
class ItemSummary(_Model):
    id: str
    title: str
    n_words: int
    n_chars: int
    source_id: str | None = None  # your id for the item: the import's id column, else from the file name

    @classmethod
    def parse(cls, d: dict[str, Any]) -> ItemSummary:
        return cls(d, str(d["id"]), d.get("title") or "", int(d.get("n_words") or 0), int(d.get("n_chars") or 0),
                   d.get("source_id"))


@dataclass
class Item(ItemSummary):
    text: str = ""

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Item:
        s = ItemSummary.parse(d)
        return cls(d, s.id, s.title, s.n_words, s.n_chars, s.source_id, d.get("text") or "")


@dataclass
class Dataset(_Model):
    id: str
    name: str
    n_items: int
    created_at: datetime | None
    items: list[ItemSummary] | None = None  # present on a single dataset, not in lists
    skipped: list[dict[str, Any]] = field(default_factory=list)  # what a push left out, as {path, reason}
    reused: bool = False  # push(reuse=True) found the same items already uploaded
    duplicates: list[dict[str, Any]] = field(default_factory=list)  # likely duplicates the server found

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Dataset:
        items = [ItemSummary.parse(i) for i in d["items"]] if isinstance(d.get("items"), list) else None
        return cls(d, str(d["id"]), d.get("name") or "", int(d.get("n_items") or 0), _dt(d.get("created_at")),
                   items, list(d.get("skipped") or []), False, list(d.get("duplicates") or []))


@dataclass
class Estimate(_Model):
    type: str
    n_items: int
    min_calls: int
    max_calls: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float | None  # upper bound; None when the model's price is unknown
    est_cost_usd_min: float | None
    est_seconds: float | None
    billed: bool = False  # paid from the workspace's credit (our model key); own-key, mock and claude_cli are not
    price_tier: str | None = None
    price_per_call_usd: float | None = None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Estimate:
        return cls(d, d.get("type") or "ranking", int(d.get("n_items") or 0), int(d.get("min_calls") or 0),
                   int(d.get("max_calls") or 0), int(d.get("est_input_tokens") or 0),
                   int(d.get("est_output_tokens") or 0), d.get("est_cost_usd"), d.get("est_cost_usd_min"),
                   d.get("est_seconds"), bool(d.get("billed")), d.get("price_tier"), d.get("price_per_call_usd"))

    def describe(self) -> str:
        """One line for a person: items, calls and money."""
        calls = ("%d judge calls" % self.max_calls if self.min_calls in (0, self.max_calls)
                 else "%d to %d judge calls" % (self.min_calls, self.max_calls))
        hi, lo = self.est_cost_usd, self.est_cost_usd_min
        if hi is None:
            cost = "cost unknown for this model"
        elif self.billed and self.price_per_call_usd is not None:
            cost = "up to %s of credit at %s a call (%s tier)" % (
                _money(hi), _money(self.price_per_call_usd, exact=True), self.price_tier or "?")
        elif hi == 0:
            cost = "free"
        elif lo is not None and _money(lo) != _money(hi):
            cost = "%s to %s" % (_money(lo), _money(hi))
        else:
            cost = "up to %s" % _money(hi)
        line = "%d items, %s, %s" % (self.n_items, calls, cost)
        if self.est_seconds:
            line += ", about %s" % _duration(self.est_seconds)
        return line


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


@dataclass
class ImportPreview(_Model):
    """A spreadsheet the server has parsed, waiting for its column mapping. ``mapping`` is the server's guess."""
    id: str
    filename: str
    format: str
    sheets: list[str]
    sheet: str | None
    n_rows: int
    columns: list[dict[str, Any]]  # name, filled, avg_chars, max_chars, unique, distinct, samples
    mapping: dict[str, Any]  # id_field, title_field, text_fields, metadata_fields
    warnings: list[str]

    @classmethod
    def parse(cls, d: dict[str, Any]) -> ImportPreview:
        return cls(d, str(d["id"]), d.get("filename") or "", d.get("format") or "", list(d.get("sheets") or []),
                   d.get("sheet"), int(d.get("n_rows") or 0), list(d.get("columns") or []),
                   dict(d.get("mapping") or {}), list(d.get("warnings") or []))


@dataclass
class Billing(_Model):
    """The workspace's prepaid credit. Runs on our model key are paid from it; runs on your own key are not."""
    enabled: bool  # False when the server has no payment provider: no top-ups, no billed runs
    balance_usd: float  # available for new runs (what running runs hold is already taken out)
    held_usd: float  # held by queued and running runs, not yet spent
    topup_options_usd: list[int]
    min_topup_usd: int
    max_topup_usd: int
    free_grant_usd: float  # welcome credit every new workspace gets
    topup_url: str | None  # the web page to add credit, when the server says

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Billing:
        return cls(d, bool(d.get("enabled")), float(d.get("balance_usd") or 0), float(d.get("held_usd") or 0),
                   [int(x) for x in d.get("topup_options_usd") or []], int(d.get("min_topup_usd") or 0),
                   int(d.get("max_topup_usd") or 0), float(d.get("free_grant_usd") or 0), d.get("topup_url"))


@dataclass
class PriceTier(_Model):
    name: str
    price_per_call_usd: float
    examples: list[str]

    @classmethod
    def parse(cls, d: dict[str, Any]) -> PriceTier:
        return cls(d, d.get("name") or "", float(d.get("price_per_call_usd") or 0), list(d.get("examples") or []))


@dataclass
class Pricing(_Model):
    """What a judge call costs on our model key, by tier. ``provisional``: prices may still change."""
    provisional: bool
    unit_chars: int  # one call is up to this many characters of all the judge reads: both items, rubric, instructions
    free_grant_usd: float
    min_topup_usd: int
    max_topup_usd: int
    tiers: list[PriceTier]

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Pricing:
        return cls(d, bool(d.get("provisional")), int(d.get("unit_chars") or 0), float(d.get("free_grant_usd") or 0),
                   int(d.get("min_topup_usd") or 0), int(d.get("max_topup_usd") or 0),
                   [PriceTier.parse(t) for t in d.get("tiers") or []])


@dataclass
class LedgerEntry(_Model):
    """One credit movement: grant, topup, hold (a run reserves its estimate), trueup (unused hold returned),
    refund or adjustment. Positive adds credit."""
    id: str
    kind: str
    amount_usd: float
    run_id: str | None
    note: str | None
    created_at: datetime | None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(d, str(d["id"]), d.get("kind") or "", float(d.get("amount_usd") or 0),
                   str(d["run_id"]) if d.get("run_id") else None, d.get("note"), _dt(d.get("created_at")))


@dataclass
class Checkout(_Model):
    """A payment page: send the customer to ``url``. Credit arrives once the payment is confirmed."""
    id: str
    url: str

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Checkout:
        return cls(d, str(d.get("id") or ""), d["url"])


@dataclass
class Progress(_Model):
    total: int
    done: int
    failed: int
    in_flight: int
    pending: int
    round: int
    cost_usd: float

    @classmethod
    def parse(cls, d: dict[str, Any] | None) -> Progress:
        d = d or {}
        n = {k: int(d.get(k) or 0) for k in ("total", "done", "failed", "in_flight", "pending", "round")}
        return cls(d, n["total"], n["done"], n["failed"], n["in_flight"], n["pending"], n["round"],
                   float(d.get("cost_usd") or 0))


@dataclass
class Run(_Model):
    id: str
    type: str
    name: str
    status: str  # queued | running | done | failed | canceled
    dataset_id: str
    dataset_name: str
    judge: dict[str, Any]
    config: dict[str, Any]  # the run type's settings, sent under its type: for a ranking, rubric, rounds, ...
    max_cost_usd: float | None
    planned_calls: int
    note: str | None
    error: str | None
    progress: Progress
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Run:
        kind = d.get("type") or "ranking"
        return cls(d, str(d["id"]), kind, d.get("name") or "", d.get("status") or "", str(d.get("dataset_id")),
                   d.get("dataset_name") or "", dict(d.get("judge") or {}), dict(d.get(kind) or {}),
                   d.get("max_cost_usd"), int(d.get("planned_calls") or 0),
                   d.get("note"), d.get("error"), Progress.parse(d.get("progress")), _dt(d.get("created_at")),
                   _dt(d.get("started_at")), _dt(d.get("finished_at")))


@dataclass
class LeaderboardEntry(_Model):
    item_id: str
    title: str
    rank: int
    rank_lo: int
    rank_hi: int
    rating: float
    rating_lo: float
    rating_hi: float
    wins: int
    losses: int
    p_top_n: float | None
    source_id: str | None = None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> LeaderboardEntry:
        return cls(d, str(d["item_id"]), d.get("title") or str(d["item_id"]), int(d["rank"]),
                   int(d.get("rank_lo", d["rank"])), int(d.get("rank_hi", d["rank"])), float(d["rating"]),
                   float(d.get("rating_lo", d["rating"])), float(d.get("rating_hi", d["rating"])),
                   int(d.get("wins") or 0), int(d.get("losses") or 0), d.get("p_top_n"), d.get("source_id"))


@dataclass
class Results(_Model):
    run_id: str | None
    type: str
    final: bool
    summary: str  # one plain sentence: the verdict
    leaderboard: list[LeaderboardEntry]
    diagnostics: dict[str, Any]
    interval: str | None
    n_bootstrap: int | None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Results:
        kind = d.get("type") or "ranking"
        body: dict[str, Any] = d.get(kind) or {}
        board = [LeaderboardEntry.parse(e) for e in body.get("leaderboard") or []]
        return cls(d, d.get("run_id"), kind, bool(d.get("final")), d.get("summary") or "", board,
                   dict(body.get("diagnostics") or {}), body.get("interval"), body.get("n_bootstrap"))


@dataclass
class RunEvent(_Model):
    """One event from a run's live stream: ``progress`` whenever calls complete, then ``finished`` once."""
    event: str
    run: Run
    leaders: list[LeaderboardEntry]  # the current top 3; empty until a call is decided

    @classmethod
    def parse(cls, d: dict[str, Any]) -> RunEvent:
        return cls(d, d.get("event") or "progress", Run.parse(d["run"]),
                   [LeaderboardEntry.parse(e) for e in d.get("leaders") or []])


@dataclass
class Comparison(_Model):
    id: str
    round: int
    item_a: str
    item_b: str
    title_a: str
    title_b: str
    status: str
    winner_item: str | None
    reasoning: str | None
    error: str | None
    cost_usd: float | None
    source_id_a: str | None = None
    source_id_b: str | None = None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> Comparison:
        return cls(d, str(d["id"]), int(d.get("round") or 0), str(d.get("item_a")), str(d.get("item_b")),
                   d.get("title_a") or "", d.get("title_b") or "", d.get("status") or "",
                   str(d["winner_item"]) if d.get("winner_item") else None, d.get("reasoning"), d.get("error"),
                   d.get("cost_usd"), d.get("source_id_a"), d.get("source_id_b"))
