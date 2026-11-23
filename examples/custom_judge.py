"""Plug in your own judge by implementing the two-method ``Provider`` protocol.

    python examples/custom_judge.py

Anything that turns a system prompt and a user prompt into text can judge: a model behind your own gateway,
a rules engine, or a human in a loop. This one prefers the shorter text, and answers in the JSON shape the
judge prompt asks for.
"""
import json
import re

from supposing import Completion, rank


class ShorterIsBetter:
    name = "shorter"
    model = "rule"

    async def complete(self, system: str, user: str) -> Completion:
        a, b = re.findall(r"<text_[ab]>\n(.*?)\n</text_[ab]>", user, re.S)
        winner = "A" if len(a) <= len(b) else "B"
        return Completion(text=json.dumps({"reasoning": "Text %s is shorter." % winner, "winner": winner}))

    async def close(self) -> None:
        pass


answers = {
    "terse": "Use a list.",
    "medium": "A list works well here because order matters.",
    "long": "There are several options, but on balance a list is the best fit, because order matters and "
            "the collection is small.",
}
ranking = rank(answers, "Which answer is more concise?", provider=ShorterIsBetter(), rounds=3)
print([row["item_id"] for row in ranking.leaderboard])
assert ranking.leaderboard[0]["item_id"] == "terse"
