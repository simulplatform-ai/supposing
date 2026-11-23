"""Deciding who meets whom in the next round."""
from __future__ import annotations

import random

# Rating points of noise added before the Swiss sort, so opponents are near in rating but not always adjacent.
# Strict neighbour pairing (40 points) inflated the rating scale by about 1.4 times when items were close in
# quality, and the 90% intervals then covered the planted ratings only 67-69% of the time. At 250 points
# coverage was back to 84-87%, with the same rank accuracy (tests/test_statistics.py).
JITTER = 250.0
SEARCH_BUDGET = 10_000  # steps spent looking for a round with no rematches before settling for fewer


def make_round(item_ids: list, ratings: dict | None, played: set, games: dict, rng: random.Random,
               pairing: str = "swiss") -> list:
    """Pair every item at most once. Returns a list of (a, b) tuples.

    swiss: items with similar current ratings meet, which is where a comparison is most
    informative. random: uniform pairing. Either way, a pair that has already met is only
    paired again when no round without rematches exists (or none is found within
    ``SEARCH_BUDGET`` steps). With an odd count, one of the items that have played most sits out.
    """
    ids = list(item_ids)
    if len(ids) < 2:
        return []
    rng.shuffle(ids)
    if pairing == "swiss" and ratings:
        ids.sort(key=lambda i: -(ratings.get(i, 1500.0) + rng.gauss(0, JITTER)))

    # Odd count: the bye goes to one of those who have played most, preferring one that leaves a round with
    # no rematches.
    byes: list = [None]
    if len(ids) % 2:
        most = max(games.get(i, 0) for i in ids)
        byes = [i for i in ids if games.get(i, 0) == most]
    budget = [SEARCH_BUDGET]
    for bye in byes:
        order = [i for i in ids if i != bye]
        pairs = _fresh_matching(order, played, budget)
        if pairs is not None:
            return pairs
        if budget[0] <= 0:
            break
    order = [i for i in ids if i != byes[0]]
    return _fewest_rematches(order, played)


def _fresh_matching(order: list, played: set, budget: list) -> list | None:
    """Pair ``order`` with no rematches, each item taking the earliest fresh partner that still lets the
    rest be paired. Depth-first with backtracking; None if impossible or the budget runs out."""
    # Each frame: (items still to pair, index of the next candidate partner for the first of them).
    stack: list[tuple[list, int]] = [(order, 1)]
    chosen: list[tuple] = []
    while stack:
        budget[0] -= 1
        if budget[0] < 0:
            return None
        rest, k = stack[-1]
        if not rest:
            return chosen
        a = rest[0]
        while k < len(rest) and frozenset((a, rest[k])) in played:
            k += 1
        if k >= len(rest):
            stack.pop()
            if chosen:
                chosen.pop()
            continue
        stack[-1] = (rest, k + 1)
        b = rest[k]
        chosen.append((a, b))
        stack.append(([i for i in rest[1:] if i != b], 1))
    return None


def _fewest_rematches(order: list, played: set) -> list:
    """Greedy pairing that takes a fresh partner where it can, then trades partners to undo stranded rematches."""
    pairs = []
    remaining = order
    while len(remaining) >= 2:
        a = remaining[0]
        partner = next((b for b in remaining[1:] if frozenset((a, b)) not in played), remaining[1])
        remaining = [i for i in remaining[1:] if i != partner]
        pairs.append((a, partner))

    def fresh(x, y):
        return frozenset((x, y)) not in played

    for i, (a, b) in enumerate(pairs):
        if fresh(a, b):
            continue
        for j, (c, d) in enumerate(pairs):
            if j == i:
                continue
            if fresh(a, c) and fresh(b, d):
                pairs[i], pairs[j] = (a, c), (b, d)
                break
            if fresh(a, d) and fresh(b, c):
                pairs[i], pairs[j] = (a, d), (b, c)
                break
    return pairs
