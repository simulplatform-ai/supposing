# supposing

<!-- mcp-name: app.supposing/supposing -->

Rank documents with LLM judges, with a confidence interval on every place. You write what "better" means in plain words; a model compares the documents two at a time and the verdicts become a ranking.

```sh
uvx supposing rank --sample stories --rubric "Which is the better short story?" --judge claude_cli --model haiku
```

```
The Lighthouse Keeper's Ledger ranks first, but any of the top 3 could be first. Treat with care: ratings follow length closely, which may mean the judge rewards longer texts (length vs rating +0.69). Read some comparisons before relying on this ranking.

rank  could be  rating  90% interval  won-lost  item
   1  1-4         1950  1771 to 2144  7-1       The Lighthouse Keeper's Ledger
   2  1-4         1948  1740 to 2222  7-1       Salt
   3  1-6         1820  1570 to 2130  6-2       The Census Taker
   4  2-7         1704  1484 to 1899  5-3       The Translator
   5  3-7         1676  1484 to 1833  5-3       Quarterly Review
   6  4-8         1566  1360 to 1712  4-4       An Inheritance of Clocks
   7  4-8         1548  1392 to 1722  5-3       Notes Toward a Second Moon
   8  6-10        1371  1220 to 1523  3-5       The Battle of Zorgon Prime
   9  7-10        1353  1187 to 1478  3-5       My Cat
  10  8-11        1196  1015 to 1366  2-6       Rain
  11  10-12       1034  894 to 1210   1-7       The Most Beautiful Sunset
  12  11-12        833  730 to 1066   0-8       The Meeting That Could Have Been An Email
```

Twelve built-in stories, 48 judge calls to Claude Haiku through a Claude subscription, 29 seconds. **could be** is the range of places each story holds with 90% confidence; where two ranges overlap, the judge has not told those stories apart.

Rank your own: `uvx supposing rank ./folder --rubric "..."` on a folder of `.md`, `.txt` or `.html` files, or a JSONL or CSV file. The default judge runs on OpenRouter (`OPENROUTER_API_KEY`); `--judge claude_cli` uses your Claude subscription, and `--judge mock` is free and fake, for trying things. `--dry-run` shows the calls first. In a project, `uv add supposing`.

**Coding agents:** read [llms.txt](https://app.supposing.app/llms.txt). To give an agent the tools, connect the hosted MCP server at `https://api.supposing.app/mcp` ([how](https://app.supposing.app/docs/connect/)), or run it locally with `uvx "supposing[mcp]" mcp`.

[Documentation](https://app.supposing.app/docs) · [Developers](https://supposing.app/developers) · [Supposing Cloud](https://app.supposing.app) · [Discord](https://discord.gg/PJQGNPXCZ6)

## Try it in a minute, with no API key

The `mock` provider is a free, deterministic fake judge. It prefers whichever text carries the higher hidden `[[q=...]]` score, with some noise, so you can see the whole thing work before spending anything.

```python
from supposing import rank

ranking = rank(
    {"a.md": "First draft. [[q=0.2]]", "b.md": "Second draft. [[q=0.9]]", "c.md": "Third draft. [[q=0.5]]"},
    "Which draft makes the strongest case for funding?",
    provider="mock",
)
for row in ranking.leaderboard:
    print(row["rank"], row["item_id"], row["rating"], row["rank_lo"], row["rank_hi"])
```

Or from the command line, on a folder with one document per `.md`, `.txt` or `.html` file:

```sh
supposing rank ./proposals --rubric "Which proposal makes the strongest case for funding?" --judge mock
```

```
p3.md ranks first, but any of the top 2 could be first. 2 clear tiers.

rank  could be  rating  90% interval  won-lost  item
-- Ranks 1-3
   1  1-2         1850  1683 to 2021  7-1       p3.md
   2  2-3         1617  1348 to 1721  3-5       p4.md
   3  2-3         1496  1392 to 1706  6-2       p2.md

-- Rank 4
   4  4           1037  960 to 1168   0-8       p1.md

"could be" is the range of ranks each item holds with 90% confidence. Items whose ranges overlap are not reliably different.
Judge: Mostly consistent, with 2 cautions: first-shown text wins 25%, split-half reliability 0.65.
```

The mock judge is noisy on purpose, and the judge checks on the last line say so, as they would for a real judge that inconsistent.

No documents to hand? `supposing.samples` has two datasets to try: `samples.items("stories")` (twelve short pieces) and `samples.items("grants")` (thirty invented applications to a small community fund, from vague to well evidenced), each with a suggested rubric in `samples.load(id)["rubric"]`.

More in [`examples/`](https://github.com/simulplatform-ai/supposing/tree/main/examples): resuming a run, driving it step by step, and writing your own judge. Each one runs with `uv run examples/<name>.py` and no key.

## Real judges

| `provider` | Needs |
|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY`; `model` is any OpenRouter model id |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai_compat` | `base_url`, for Ollama, vLLM, LM Studio or OpenAI |
| `claude_cli` | Claude Code installed and logged in; for local use |
| `mock` | Nothing |

```python
ranking = rank(docs, rubric, model="anthropic/claude-haiku-5.5")   # OpenRouter
```

Know the cost before you start. A run makes at most `planned_calls(n_items, RankConfig(...))` judge calls: by default 8 rounds × (n ÷ 2) pairs, so 4 calls per document (8 with `both_orders=True`). `supposing rank ... --dry-run` prints the estimate and stops, and `max_cost_usd` (`--max-cost`) is a hard cap: no call starts that could take recorded spend past it (each call is assumed to cost as much as the dearest so far), and the run's note says when the cap ended it. Providers that don't report prices, such as `claude_cli`, can't be capped.

## Rubrics

A rubric is a prompt: what the judge should prefer, in plain words. It goes into every comparison as written. Say what matters more, and what to ignore, in sentences. A string is enough; to keep one in a file, use a Markdown or text file, where a first line `# Name` is its name:

```markdown
# Grant proposals

Which proposal makes the strongest case for funding? Weigh impact (how many people benefit, and how much) about twice as much as feasibility (can this team do it with this budget?). Ignore formatting and spelling.
```

```python
from supposing import Rubric, rank

ranking = rank(docs, Rubric.load("grants.md"))
```

On the command line, `--rubric` takes the text or the path of a `.md` or `.txt` file.

## Reading the result

`ranking.groups` splits the leaderboard for display: for a top-N run, the items settled in the top N, the close calls (every rank from the first to the last item whose chance of a top-N place is between 5% and 95%), and the rest; for a full ranking, its clear tiers. `ranking.summary` is the verdict in one plain sentence, such as "repair-cafe is first, clearly. 4 clear tiers." A tier is a group of items whose rank ranges overlap, so the judge cannot reliably order them; `supposing.tiers(leaderboard)` returns them. When a judge check is poor, the summary adds one sentence of warning.

`ranking.leaderboard` is sorted best first. Each row has:

- `rating`: on an Elo-like scale. The average is 1500, and a 400-point gap means the judge prefers the higher item about 10 times out of 11.
- `rating_lo`, `rating_hi`: the 90% interval for the rating.
- `rank_lo`, `rank_hi`: the ranks the item holds with 90% confidence. **Items whose rank ranges overlap are not reliably different.** Report them as a tie, or add rounds.
- `wins`, `losses`, and `p_top_n` (the chance of being in the top N) for top-N runs.

`ranking.diagnostics` says how far the judge can be trusted:

- `order_consistency` (with `both_orders=True` only): the share of pairs with the same winner in both orders. Low means the judge often cannot tell the two apart, or is swayed by position.
- `first_position_win_rate`: near 0.5 is good. Each document is shown first about half the time, so this bias cancels out of the ranking, but a high value still means noisy verdicts. A gap that chance alone often gives is not graded.
- `split_half_reliability`: how well two halves of the verdicts agree on the order (Spearman). Near 1 is stable.
- `length_correlation`: whether longer documents tend to rank higher.
- `upsets`: verdicts that contradict the final ranking most strongly. Read a few.
- `first_shown_won_both`, `second_shown_won_both` (with `both_orders=True` only): the share of pairs where the same position won both times. A gap between the two is position bias.
- `reliability`: the checks above graded Good, Caution or Poor, with one plain sentence (`reliability["line"]`) to show beside the ranking. Thresholds are rules of thumb, not tests.

These checks say whether the judge is consistent. They cannot say whether its taste is right. Read some verdicts (`ranking.comparisons`, each with `reasoning`) before trusting a ranking.

## Check a judge against past decisions

If you have real decisions for some documents (selected or rejected, a panel's scores), check whether a judge and rubric would have made them, and pick the rubric that predicts them best on decisions it was not chosen on. `outcomes` maps item ids to a number, higher is better.

```python
from supposing import backtest, tune

backtest(ranking.leaderboard, {"a.md": 1, "b.md": 0, ...})["line"]
# "Against 40 past decisions, the ranking puts the better outcome above the worse 78% of the time (90% interval 66-88%; chance is 50%). ..."

tuning = await tune(items, outcomes, [base_rubric, candidate_1, candidate_2], provider)
tuning.chosen, tuning.line
```

`tune` splits the items into train and test (stratified by outcome; `groups` keeps resubmissions together). Every rubric judges the same pairs of train items with different outcomes. A candidate replaces the base (the first rubric) only if its one-sided 95% lower bound on the gain in agreement, corrected for the number of candidates, is above zero. Only then do the base and the choice judge pairs of test items, once. Intervals come from resampling items, not pairs, because pairs that share an item are not independent. The candidates are yours to write; nothing proposes them yet. Calls: at most `planned_tune_calls(len(rubrics), train_pairs, test_pairs)`.

In `tests/test_backtest.py`, with a judge that scores what its rubric names and a panel that selected on one feature: a rubric naming that feature is chosen and wins held out; with shuffled outcomes a candidate was chosen in 2 of 100 runs (5 of 100 with 200 train pairs), against 9 of 30 when intervals resampled pairs.

## How it works

- **Swiss rounds.** Each round pairs every document once, with an opponent of similar current rating, where a verdict is informative. A pair meets again only when no round without rematches exists.
- **Round robin** (`pairing="round_robin"`) is for a handful of options: every pair is planned in the first round, so all the calls can run at once. n options make n(n-1)/2 pairs (28 for 8), so one round is usually enough.
- **Random first position.** Each pair is judged once, with a random document shown first, so the judge's tendency to favour one position cancels out across the ranking. `both_orders=True` judges every pair twice, once in each order: twice the calls, and an extra check of how often the judge changes its mind.
- **Bradley-Terry ratings** fitted to all verdicts at once, so the result does not depend on the order the calls finished in. A weak prior (half a win and half a loss against an average opponent) keeps an unbeaten document's rating finite.
- **Intervals** come from a bootstrap over pairs: refit on resampled pairs a few hundred times and take the middle 90%.
- **Top-N mode** (`goal="top_n", top_n=10`) ranks everything for three rounds. After that it judges only the documents whose side of the cut is still open (a chance of making the top N between 5% and 95%, checked after each of the last two rounds), plus the nearest settled document on either side as a reference, and stops when none is open. `ranking.calls_saved` says how many calls that saved against the fixed plan. In simulations with 30 to 100 documents it used 45-53% of a full ranking's calls and found the true top N as often.
- **Resumable.** With a `SQLiteStore` (`--state run.db` on the command line), every verdict is saved as it arrives. Run the same command again and it continues; nothing already decided is judged again.

### What the tests show

`tests/test_statistics.py` plants a known order, simulates a judge that follows it with noise, and checks what comes back (the many-trial coverage and top-N simulations are marked `stats`; run them with `pytest -m stats`):

- The planted order is recovered, and with enough verdicts the planted ratings too.
- The 90% intervals contain the planted rating 80-97% of the time on evenly spread and on closely matched collections, and the rank ranges at least 80-85%, from about six verdicts per document. With three per document (20 documents, 3 rounds) rating coverage falls to about 77%.
- A judge that favours the first text strongly (82% first-position wins between equals) still gives the planted order, judged once per pair or in both orders, and the diagnostics report the bias. Judged once, its intervals still contain the planted ratings at about the stated rate.
- Pairing never repeats a pair while a round without rematches exists, and byes rotate.
- A run stopped at any point and resumed from its file gives exactly the same result as an uninterrupted one.

**Known limit.** When a document wins or loses nearly every comparison, the data cannot say how far ahead it is, and the prior pulls its rating toward the middle. Its rating interval may then sit too close to the middle. The error is in the safe direction (the gap is understated), and its rank range is still reliable.

## Building on the engine

`supposing.ranking` is a pure state machine with no I/O. `next_action(comparisons, config, n_items)` says whether to judge the pending comparisons, plan the next round, or stop; `plan_round(...)` produces the next round deterministically; `results(...)` fits the ranking. Any process holding the comparison rows can continue a run, so you can run it from a job queue or across many workers ([`examples/state_machine.py`](https://github.com/simulplatform-ai/supposing/blob/main/examples/state_machine.py)).

- Keep runs anywhere by implementing the three-method `RunStore` protocol (`load`, `add`, `update`).
- Use any judge by implementing the two-method `Provider` protocol (`complete`, `close`; [`examples/custom_judge.py`](https://github.com/simulplatform-ai/supposing/blob/main/examples/custom_judge.py)).
- `SYSTEM_PROMPT` is exactly what the judge is told.

Public names are those in `supposing.__all__`; everything else may change.

## Licence

Apache License 2.0. See [LICENSE](https://github.com/simulplatform-ai/supposing/blob/main/LICENSE) and [NOTICE](https://github.com/simulplatform-ai/supposing/blob/main/NOTICE).
