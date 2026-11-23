"""Save a run to a file, stop it, and continue it later without judging anything twice.

    python examples/resume.py

``SQLiteStore`` commits every verdict as it arrives. Running again with the same file and run id picks up
where the last process stopped, whether it finished its rounds, crashed or was killed.
"""
import tempfile
from pathlib import Path

from supposing import SQLiteStore, rank

docs = {"doc%d" % i: "Document %d. [[q=%.2f]]" % (i, i / 10) for i in range(10)}
rubric = "Which document is better?"
path = Path(tempfile.mkdtemp()) / "run.db"

# First session: two rounds only.
first = rank(docs, rubric, provider="mock", rounds=2, store=SQLiteStore(path), run_id="demo")
print("after 2 rounds: %d calls, leader %s" % (len(first.comparisons), first.leaderboard[0]["item_id"]))

# Later, perhaps in another process: ask for six rounds. The first two are reused from the file.
second = rank(docs, rubric, provider="mock", rounds=6, store=SQLiteStore(path), run_id="demo")
print("after 6 rounds: %d calls, leader %s" % (len(second.comparisons), second.leaderboard[0]["item_id"]))
assert all(c.attempts == 1 for c in second.comparisons), "nothing is judged twice"
