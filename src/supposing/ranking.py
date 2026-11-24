"""Pairwise-ranking tournaments as a pure state machine.

Nothing here does I/O. A run is a list of items, a config and the comparisons made so far. From that
state, ``next_action`` says what to do next: judge the pending comparisons, plan another round, or stop.
``plan_round`` produces the next round's comparisons deterministically, so any process that holds the
state can continue a run: the local runner in ``runner.py``, or a fleet of workers sharing a database.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from . import rating, report, scheduler

WARMUP_ROUNDS = 3  # top-N runs rank everything this long before narrowing
PLAN_BOOTSTRAP = 100  # resamples behind each top-N planning decision: steps of 1% in the chance of a place
SETTLED = 0.05  # top-N: an item is settled in (out) when its chance of a top-N place is >= 95% (<= 5%)
MAX_ATTEMPTS = 3  # a comparison that fails this often is given up on

Status = Literal["pending", "done", "failed"]


class ConfigError(ValueError):
    """A run setting is missing, unknown or out of range. The message names the setting."""


@dataclass
class RankConfig:
    """How a ranking run spends its judge calls. Every field has a usable default.

    Values are coerced and clamped on construction (``rounds`` to 1-200, for example); a value that cannot be
    read raises ``ConfigError``. ``from_dict`` and ``to_dict`` round-trip the settings as plain JSON.

    - ``rounds``: opponents per item. Each round pairs every item once.
    - ``both_orders``: judge every pair twice, once in each order. Twice the calls; adds the order check. Off,
      each pair is judged once with a random text first, which cancels position bias across the ranking.
    - ``pairing``: ``"swiss"`` meets items of similar current rating; ``"random"`` pairs uniformly;
      ``"round_robin"`` plans every pair in each round, so all calls can run at once. For small sets:
      n items make n(n-1)/2 pairs per round, so ``rounds=1`` is usually enough.
    - ``goal``: ``"full"`` ranks everything; ``"top_n"`` ranks everything for a warm-up of three rounds,
      then judges only the items whose side of the cut is still open (with one settled item either side as
      a reference) and stops when none is open.
    - ``max_chars``: longer texts are cut before judging, with a note to the judge. 0 means no limit.
    - ``max_cost_usd``: a hard cap on recorded spend. No call starts that could take spend past it, judging
      each call as dear as the dearest so far, and the run's note says when the cap ended it. Providers that
      do not report a price (``claude_cli``) cannot be capped.
    - ``seed``: pairing and the bootstrap are deterministic given the seed.
    """

    rounds: int = 8  # opponents per item
    both_orders: bool = False  # judge every pair twice, once in each order
    pairing: Literal["swiss", "random", "round_robin"] = "swiss"
    goal: Literal["full", "top_n"] = "full"
    top_n: int = 10
    max_chars: int = 60000  # per item; longer texts are cut
    max_cost_usd: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.goal not in ("full", "top_n"):
            raise ConfigError("goal must be 'full' or 'top_n'")
        if self.pairing not in ("swiss", "random", "round_robin"):
            raise ConfigError("pairing must be 'swiss', 'random' or 'round_robin'")
        if self.pairing == "round_robin" and self.goal == "top_n":
            raise ConfigError("round_robin pairing judges every pair, so it cannot narrow to a top N")
        try:
            self.rounds = max(1, min(200, int(self.rounds)))
            self.top_n = max(1, int(self.top_n))
            self.max_chars = max(0, int(self.max_chars))
            self.seed = int(self.seed)
            self.both_orders = bool(self.both_orders)
            if self.max_cost_usd in ("", 0):
                self.max_cost_usd = None
            if self.max_cost_usd is not None:
                self.max_cost_usd = float(self.max_cost_usd)
        except (TypeError, ValueError) as exc:
            raise ConfigError("invalid run setting: %s" % exc) from exc

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> RankConfig:
        """Build from plain settings, rejecting unknown keys. ``None`` values take the default."""
        raw = raw or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError("unknown run setting: %s" % ", ".join(sorted(unknown)))
        kwargs: dict[str, Any] = {k: v for k, v in raw.items() if v is not None or k == "max_cost_usd"}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """The settings as a JSON-ready dict."""
        return asdict(self)


@dataclass
class Item:
    """One document to rank. ``id`` must be unique within a run; ``title`` is for display."""

    id: str
    text: str
    title: str | None = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def meta(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "n_words": self.n_words}


@dataclass
class Comparison:
    """One judge call: ``item_a`` is shown first, ``item_b`` second.

    ``plan_round`` creates comparisons as ``pending``; ``apply_verdict`` and ``apply_error`` move them to
    ``done`` (with ``winner_item``) or, after ``MAX_ATTEMPTS`` tries, to ``failed``. The two orders of one
    pair share a ``pair_id``. A run's whole state is its list of comparisons, so store them as rows.
    """

    id: str
    round: int
    pair_id: int
    item_a: str
    item_b: str
    status: Status = "pending"
    winner_item: str | None = None
    reasoning: str | None = None
    raw: str | None = None
    error: str | None = None
    attempts: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_rating_row(self) -> dict[str, Any]:
        """The fields the rating model reads."""
        return {"id": self.id, "round": self.round, "pair_id": self.pair_id, "item_a": self.item_a,
                "item_b": self.item_b, "winner_item": self.winner_item}


@dataclass
class Action:
    """What ``next_action`` says to do: judge ``pending``, plan round ``round``, or stop (with an optional
    ``note`` saying why the run stopped early)."""

    kind: Literal["judge", "plan", "done"]
    pending: list[Comparison] = field(default_factory=list)
    round: int = 0
    note: str | None = None


def planned_calls(n_items: int, cfg: RankConfig) -> int:
    """Upper bound on judge calls. Top-N runs narrow after the warm-up and use fewer."""
    pairs = n_items * (n_items - 1) // 2 if cfg.pairing == "round_robin" else n_items // 2
    return cfg.rounds * pairs * (2 if cfg.both_orders else 1)


def fewest_calls(n_items: int, cfg: RankConfig) -> int:
    """Lower bound. A top-N run can stop as soon as the warm-up is over, if the cut is already settled."""
    narrows = cfg.goal == "top_n" and cfg.top_n < n_items and cfg.rounds > WARMUP_ROUNDS
    if not narrows:
        return planned_calls(n_items, cfg)
    return WARMUP_ROUNDS * (n_items // 2) * (2 if cfg.both_orders else 1)


def decided(comparisons: list[Comparison]) -> list[Comparison]:
    """Comparisons with a verdict."""
    return [c for c in comparisons if c.status == "done" and c.winner_item]


def spent(comparisons: list[Comparison]) -> float:
    """Recorded cost in US dollars. Providers that do not report a price count as zero."""
    return sum(c.cost_usd or 0.0 for c in comparisons)


def call_cost(comparisons: list[Comparison]) -> float | None:
    """The dearest single judge call so far, in US dollars, or None before any call has been made.

    Providers that do not report a price count as zero, so a spend cap cannot bind on them.
    """
    costs = [(c.cost_usd or 0.0) / c.attempts for c in comparisons if c.attempts]
    return max(costs) if costs else None


def affordable(comparisons: list[Comparison], cfg: RankConfig, in_flight: int = 0) -> bool:
    """Whether one more call can start without risking the spend cap, with ``in_flight`` calls already running.

    Each call, finished or running, is assumed to cost as much as the dearest one so far. Until the first
    price is known only one call may run at a time.
    """
    if cfg.max_cost_usd is None:
        return True
    per_call = call_cost(comparisons)
    if per_call is None:
        return in_flight == 0 and spent(comparisons) < cfg.max_cost_usd
    return spent(comparisons) + (in_flight + 1) * per_call <= cfg.max_cost_usd + 1e-9  # float sums


def next_action(comparisons: list[Comparison], cfg: RankConfig, n_items: int) -> Action:
    """What a run should do next, given everything recorded so far.

    Pure: the answer depends only on the arguments. Call it again after every batch of verdicts. With a
    spend cap, check ``affordable`` before starting each call, counting the calls already running.
    """
    if n_items < 2:
        return Action("done", note="A ranking needs at least two items.")
    rounds_done = max((c.round for c in comparisons), default=0)
    if not affordable(comparisons, cfg):
        return Action("done", note="Stopped by the spend cap: $%.4f spent of $%.2f, and another call could pass "
                      "it (after %d of %d rounds)." % (spent(comparisons), cfg.max_cost_usd or 0, rounds_done,
                                                       cfg.rounds))
    pending = [c for c in comparisons if c.status == "pending"]
    if pending:
        return Action("judge", pending=pending)
    if rounds_done >= cfg.rounds:
        return Action("done")
    if rounds_done and not decided(comparisons):
        return Action("done", note="No comparison produced a readable verdict.")
    return Action("plan", round=rounds_done + 1)


def _swiss_or_random(items: list[Item], done: list[Comparison], cfg: RankConfig, round_no: int,
                     rng: random.Random) -> tuple[list[tuple[str, str]], str | None]:
    """Swiss or random pairs, after narrowing a top-N run to the items near the cut. Empty, with a note,
    when the cut is settled."""
    ids = [it.id for it in items]
    played = {frozenset((c.item_a, c.item_b)) for c in done}
    games: dict[str, int] = {}
    for c in done:
        games[c.item_a] = games.get(c.item_a, 0) + 1
        games[c.item_b] = games.get(c.item_b, 0) + 1
    ratings: dict[str, float] | None = None
    pool = ids
    if done:
        rows = [c.as_rating_row() for c in done]
        narrowing = cfg.goal == "top_n" and round_no > WARMUP_ROUNDS and cfg.top_n < len(ids)
        if narrowing:
            status, ratings = cut_status(items, done, cfg)
            pool = _band_and_anchors(status, ratings)
            if not pool:
                return [], "Stopped early after %d rounds: the top %d is settled." % (round_no - 1, cfg.top_n)
        else:
            log = rating.MatchLog(ids, rows)
            elo = rating.to_elo(log.fit())
            ratings = {item_id: float(elo[i]) for i, item_id in enumerate(ids)}
    return scheduler.make_round(pool, ratings, played, games, rng, cfg.pairing), None


def plan_round(items: list[Item], comparisons: list[Comparison], cfg: RankConfig, round_no: int
               ) -> tuple[list[Comparison], str | None]:
    """The next round's comparisons, or none and a note when nothing is left worth judging.

    Deterministic in (items, decided comparisons, cfg, round_no), so it can be re-run safely after a crash.
    """
    done = decided(comparisons)
    ids = [it.id for it in items]
    rng = random.Random(cfg.seed * 1000 + round_no)
    if cfg.pairing == "round_robin":
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        rng.shuffle(pairs)
    else:
        pairs, note = _swiss_or_random(items, done, cfg, round_no, rng)
        if note:
            return [], note

    flip = random.Random(cfg.seed * 7919 + round_no)
    next_pair = max((c.pair_id for c in comparisons), default=0) + 1
    out: list[Comparison] = []
    for offset, (a, b) in enumerate(pairs):
        if flip.random() < 0.5:
            a, b = b, a
        pair_id = next_pair + offset
        orders = [(a, b), (b, a)] if cfg.both_orders else [(a, b)]
        for k, (first, second) in enumerate(orders):
            out.append(Comparison(id="r%d-p%d-%s" % (round_no, pair_id, "ab"[k]), round=round_no,
                                  pair_id=pair_id, item_a=first, item_b=second))
    if not out:
        return [], "Nothing left to pair."
    return out, None


def cut_status(items: list[Item], done: list[Comparison], cfg: RankConfig
               ) -> tuple[dict[str, str], dict[str, float]]:
    """For a top-N run: each item's side of the cut (``"in"``, ``"out"`` or ``"open"``), and its rating.

    An item is settled when its chance of a top-N place has been at least 95% (in) or at most 5% (out) both
    after the last round and after the one before. Asking twice guards against stopping on one lucky round.
    """
    metas = [it.meta() for it in items]
    last = max((c.round for c in done), default=0)

    def sides(rows: list[Comparison]) -> tuple[dict[str, str], dict[str, float]]:
        board = rating.compute_results(metas, [c.as_rating_row() for c in rows], top_n=cfg.top_n,
                                       seed=cfg.seed, max_bootstrap=PLAN_BOOTSTRAP)["leaderboard"]
        side = {e["item_id"]: "in" if (e["p_top_n"] or 0) >= 1 - SETTLED else
                "out" if e["p_top_n"] is not None and e["p_top_n"] <= SETTLED else "open" for e in board}
        return side, {e["item_id"]: e["rating"] for e in board}

    now, ratings = sides(done)
    before, _ = sides([c for c in done if c.round < last])
    return {i: s if before.get(i) == s else "open" for i, s in now.items()}, ratings


def _band_and_anchors(status: dict[str, str], ratings: dict[str, float]) -> list[str]:
    """The open items, plus the weakest settled-in and the strongest settled-out item as reference points.
    Empty when nothing is open."""
    band = [i for i, s in status.items() if s == "open"]
    if not band:
        return []
    ins = [i for i, s in status.items() if s == "in"]
    outs = [i for i, s in status.items() if s == "out"]
    anchors = ([min(ins, key=lambda i: ratings[i])] if ins else []) + (
        [max(outs, key=lambda i: ratings[i])] if outs else [])
    return band + anchors


def results(items: list[Item], comparisons: list[Comparison], top_n: int | None = None, seed: int = 0,
            live: bool = False) -> dict[str, Any]:
    """The ranking so far: ``{"leaderboard", "diagnostics", "n_bootstrap", "interval", "groups"}``.

    Each leaderboard row has ``item_id``, ``title``, ``n_words``, ``rating`` (Elo scale: mean 1500, and a
    400-point gap means 10:1 odds), ``rating_lo``/``rating_hi`` (90% interval), ``rank`` and
    ``rank_lo``/``rank_hi`` (90% range), ``wins``, ``losses``, and ``p_top_n`` when ``top_n`` is given.
    Rows are sorted best first. Undecided and failed comparisons are ignored.

    ``groups`` splits the leaderboard for display (``supposing.display_groups``): settled, close calls and out
    for a top-N run, clear tiers otherwise; ``live=True`` labels them for a run still going ("1st so far").
    ``diagnostics`` holds the judge checks as numbers, and ``diagnostics["reliability"]`` grades them in words
    (see ``supposing.report.judge_checks``).

    Intervals come from resampling pairs, so they reflect how much the verdicts would move with a different
    draw of the same judge. An item that wins or loses nearly every comparison is pulled toward the middle
    and its rating interval may sit too close to the middle; its rank range is still reliable.
    """
    out = rating.compute_results([it.meta() for it in items], [c.as_rating_row() for c in decided(comparisons)],
                                 top_n=top_n, seed=seed)
    out["diagnostics"]["reliability"] = report.judge_checks(out["diagnostics"])
    out["groups"] = report.display_groups(out["leaderboard"], top_n, live=live)
    return out


def timeline(items: list[Item], comparisons: list[Comparison]) -> list[dict[str, Any]]:
    """Ratings and ranks after each round, for playback: a list of ``{"round", "comparisons", "ratings",
    "ranks"}`` frames."""
    rows = [{**c.as_rating_row()} for c in decided(comparisons)]
    return rating.timeline([it.meta() for it in items], rows)
