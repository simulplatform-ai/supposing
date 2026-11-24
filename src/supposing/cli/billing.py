"""Credit and prices: `supposing billing`, `supposing pricing`, `supposing usage`.

Runs on our model key are paid from the workspace's prepaid credit. Runs on your own OpenRouter key, the mock
judge and claude_cli are never billed. Every amount comes from the API; nothing here knows a price.
"""
from __future__ import annotations

import argparse
from typing import Any

from ..cloud import Billing, Client, CloudError, Estimate
from .output import OK, emit_json, info, table


def _client(args: argparse.Namespace) -> Client:
    return Client(api_url=args.api_url, source="cli")


def _usd(v: float) -> str:
    return "$%.2f" % v if abs(v) >= 0.01 or v == 0 else "$%.4f" % v


def topup_hint(sp: Client) -> dict[str, Any]:
    """Where to add credit, for an insufficient_credit error: the web page and the CLI command. Best effort."""
    out: dict[str, Any] = {}
    try:
        b = sp.billing.get()
    except CloudError:
        return out
    if b.topup_url:
        out["topup_url"] = b.topup_url
    if b.enabled and b.topup_options_usd:
        out["topup_command"] = "supposing billing topup %d" % b.topup_options_usd[0]
    out["balance_usd"] = b.balance_usd
    return out


def balance_after(sp: Client, est: Estimate) -> str | None:
    """For a billed run: 'balance $2.00, $1.74 after at most'. None for runs that are not billed."""
    if not est.billed or est.est_cost_usd is None:
        return None
    try:
        b = sp.billing.get()
    except CloudError:
        return None
    after = b.balance_usd - est.est_cost_usd
    if after < 0:
        return "balance %s, short by %s" % (_usd(b.balance_usd), _usd(-after))
    return "balance %s, %s after at most" % (_usd(b.balance_usd), _usd(after))


def show(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        b = sp.billing.get()
    if args.json:
        emit_json(b.to_dict())
        return OK
    print("Balance %s%s" % (_usd(b.balance_usd), ", %s held by running runs" % _usd(b.held_usd) if b.held_usd else ""))
    if not b.enabled:
        print("Billing is off on this server: runs on your own OpenRouter key and the mock judge still work.")
        return OK
    print("Add credit: supposing billing topup <%s>%s" % (
        "|".join(str(x) for x in b.topup_options_usd), " or %s" % b.topup_url if b.topup_url else ""))
    print("Runs on your own OpenRouter key (Settings) are never billed. Prices: supposing pricing")
    return OK


def ledger(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        page = sp.billing.ledger(limit=args.limit)
    if args.json:
        emit_json({"data": [e.to_dict() for e in page.data], "next_cursor": page.next_cursor})
        return OK
    if not page.data:
        print("No credit movements yet.")
        return OK
    print(table([[str(e.created_at or "")[:16], e.kind, ("+" if e.amount_usd > 0 else "") + _usd(e.amount_usd),
                  e.run_id or "", e.note or ""] for e in page.data],
                ["when", "kind", "amount", "run", "note"], right={2}))
    return OK


def topup(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        session = sp.billing.checkout(args.amount)
    if args.json:
        emit_json({"amount_usd": args.amount, **session.to_dict()})
        return OK
    info("Open this page to pay %s. The credit arrives as soon as the payment is confirmed." % _usd(args.amount))
    print(session.url)
    return OK


def portal(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        session = sp.billing.portal()
    if args.json:
        emit_json(session.to_dict())
    else:
        print(session.url)
    return OK


def pricing(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        p = sp.pricing()
    if args.json:
        emit_json(p.to_dict())
        return OK
    print("Price of one judge call on our model key%s:" % (" (provisional)" if p.provisional else ""))
    print()
    print(table([[t.name, "$%.4f" % t.price_per_call_usd if t.price_per_call_usd < 0.01 else "$%.3f" % t.price_per_call_usd,
                  ", ".join(t.examples)] for t in p.tiers], ["tier", "per call", "for example"], right={1}))
    print()
    print("One call is up to %s characters of everything the judge reads: both items, the rubric and our "
          "instructions. A longer call counts as more than one." % f"{p.unit_chars:,}")
    line = "Runs on your own OpenRouter key are never billed."
    if p.min_topup_usd:
        line += " Credit is added in whole dollars, from %s." % _usd(p.min_topup_usd)
    if p.free_grant_usd > 0:
        line += " New workspaces get %s of credit." % _usd(p.free_grant_usd)
    print(line)
    return OK


def usage(args: argparse.Namespace) -> int:
    with _client(args) as sp:
        u = sp.usage()
        try:
            b: Billing | None = sp.billing.get()
        except CloudError:
            b = None
    if args.json:
        emit_json({**u, "billing": b.to_dict() if b else None})
        return OK
    print("%s so far: %d runs, %d judge calls, %s of model spend (%s on our key)." % (
        str(u.get("month", ""))[:7], u.get("runs", 0), u.get("judge_calls", 0), _usd(u.get("cost_usd") or 0),
        _usd(u.get("platform_cost_usd") or 0)))
    if b:
        print("Credit balance %s." % _usd(b.balance_usd))
    return OK


def check_amount(value: str) -> int:
    try:
        amount = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("the amount is whole US dollars, e.g. 10") from exc
    if amount <= 0:
        raise argparse.ArgumentTypeError("the amount must be positive")
    return amount

