"""Rank five short pitches with the free mock judge. No API key, no network.

    python examples/quickstart.py

The mock judge is a fake: it prefers whichever text carries the higher hidden ``[[q=...]]`` score, with some
noise, so you can see the machinery work. Swap ``provider="mock"`` for a real one (see the README) to rank
by an actual model's judgment.
"""
from supposing import rank

pitches = {
    "bakery": "A neighbourhood bakery that sells yesterday's bread at half price. [[q=0.35]]",
    "tutoring": "Peer tutoring for first-year students, run by second-years for credit. [[q=0.70]]",
    "repair-cafe": "A monthly repair cafe where volunteers fix appliances for free. [[q=0.80]]",
    "app": "An app that reminds you to drink water. [[q=0.15]]",
    "library": "A tool library: borrow a drill instead of buying one. [[q=0.60]]",
}

ranking = rank(pitches, "Which project would do the most good for the neighbourhood per dollar?",
               provider="mock", rounds=6)

print(ranking.summary)
print()
print("rank  item          rating   90% interval     rank range")
for row in ranking.leaderboard:
    print("%4d  %-12s %7.0f   [%5.0f, %5.0f]   %d-%d" % (
        row["rank"], row["item_id"], row["rating"], row["rating_lo"], row["rating_hi"], row["rank_lo"],
        row["rank_hi"]))
print()
print("judge calls:", len(ranking.comparisons))
print("judge checks:", ranking.diagnostics["reliability"]["line"])
best = ranking.comparisons[0]
print("one verdict: %s beat %s. %s" % (best.winner_item, best.item_b if best.winner_item == best.item_a
                                       else best.item_a, best.reasoning))
