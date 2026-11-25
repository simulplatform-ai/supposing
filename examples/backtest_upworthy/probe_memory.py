"""Memorisation probe (pre-registered addendum A): ask each model cold whether it has seen 30 sampled headlines.

    uv run python examples/backtest_upworthy/probe_memory.py            # replays the recorded replies
    uv run python examples/backtest_upworthy/probe_memory.py --fresh    # asks Haiku and Opus again (claude_cli)

Weak evidence either way: naming Upworthy can come from the headline's style, and not naming it does not rule
out knowing which headline won.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path

from supposing import make_provider

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("upworthy", HERE / "backtest.py")
assert spec and spec.loader
bt = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(HERE.parent / "backtest_ellipse"))
spec.loader.exec_module(bt)
RECORDED = HERE / "probe-2026-11-24.json"
QUESTION = "Have you seen this headline before, and which publication ran it? Headline: %s"
SYSTEM = "Answer briefly."


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    tests = sorted(random.Random(bt.SEED).sample(bt.eligible(), bt.N_TESTS), key=lambda t: t["test"])
    heads = {a["id"]: a["headline"] for t in tests for a in t["arms"]}
    ids = random.Random(bt.SEED).sample(sorted(heads), 30)
    if args.fresh or not RECORDED.exists():
        replies: dict = {}
        for model in ("haiku", "opus"):
            provider = make_provider({"provider": "claude_cli", "model": model})
            gate = asyncio.Semaphore(4)

            async def ask(i: str, provider=provider, gate=gate) -> tuple[str, str]:
                async with gate:
                    return i, (await provider.complete(SYSTEM, QUESTION % heads[i])).text

            replies[model] = dict(await asyncio.gather(*(ask(i) for i in ids)))
        RECORDED.write_text(json.dumps({"ids": ids, "replies": replies}, indent=1) + "\n")
    recorded = json.loads(RECORDED.read_text())
    out = {m: {"asked": len(r), "names_upworthy": sum("upworthy" in t.lower() for t in r.values())}
           for m, r in recorded["replies"].items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
