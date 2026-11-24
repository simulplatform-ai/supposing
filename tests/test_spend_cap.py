import pytest

from supposing import Item, RankConfig, next_action, plan_round, run_ranking
from supposing.providers import MockProvider


class Priced(MockProvider):
    """The mock judge, billed at a fixed price per call."""

    def __init__(self, price: float, **kw):
        super().__init__(**kw)
        self.price = price

    async def complete(self, system, user):
        out = await super().complete(system, user)
        out.cost_usd = self.price
        return out


ITEMS = [Item(id="d%02d" % i, text="Doc %d [[q=%.2f]]" % (i, i / 20)) for i in range(20)]


@pytest.mark.parametrize("concurrency", [1, 8, 32])
async def test_spend_never_passes_the_cap(concurrency):
    provider = Priced(0.01)
    ranking = await run_ranking(ITEMS, "Better?", provider, RankConfig(rounds=8, max_cost_usd=0.255),
                                concurrency=concurrency)
    assert ranking.cost_usd <= 0.255
    assert provider.calls == 25  # every affordable call was made, and no more
    assert ranking.note and ranking.note.startswith("Stopped by the spend cap")


async def test_free_providers_are_not_capped():
    ranking = await run_ranking(ITEMS[:6], "Better?", MockProvider(), RankConfig(rounds=2, max_cost_usd=0.01))
    assert ranking.note is None and len(ranking.comparisons) == 6


def test_next_action_stops_pending_calls_at_the_cap():
    cfg = RankConfig(rounds=2, max_cost_usd=0.05)
    comps, _ = plan_round(ITEMS[:12], [], cfg, 1)
    for c in comps[:5]:
        c.status, c.winner_item, c.attempts, c.cost_usd = "done", c.item_a, 1, 0.01
    action = next_action(comps, cfg, 12)
    assert action.kind == "done" and "spend cap" in (action.note or "")
    cfg_more = RankConfig(rounds=2, max_cost_usd=0.06)
    assert next_action(comps, cfg_more, 12).kind == "judge"
