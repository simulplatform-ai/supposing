# supposing

Rank documents with LLM judges. A model compares the documents two at a time against a rubric you write in plain words, and the verdicts are combined into a ranking with confidence intervals.

```sh
pip install supposing
```

Documentation: [app.supposing.app/docs](https://app.supposing.app/docs). The same package has a client and command line for Supposing Cloud (`supposing run`, `supposing.cloud`) and a stdio MCP server for coding agents (`pip install "supposing[mcp]"`, then `supposing mcp`).

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
supposing rank ./proposals --rubric "Which proposal makes the strongest case for funding?" --provider mock
```

```
p3.md ranks first, but any of the top 2 could be first. 3 clear tiers. Treat with care: the judge often changed its mind when the two texts swapped places (same winner in both orders 69%; 1 more check is poor). Read some comparisons before relying on this ranking.

rank  could be  rating  90% interval  won-lost  item
   1  1-2         1826  1746 to 1875  11-5      p3.md
   2  1-2         1819  1738 to 1858  11-5      p4.md
   3  3           1444  1362 to 1547  10-6      p2.md
   4  4            911  857 to 1027   0-16      p1.md
```

The mock judge is noisy on purpose, and the result says so: the warning comes from the judge checks, and appears whenever a real judge is that inconsistent too.

No documents to hand? `supposing.samples` has two datasets to try: `samples.items("stories")` (twelve short pieces) and `samples.items("grants")` (thirty invented applications to a small community fund, from vague to well evidenced), each with a suggested rubric in `samples.load(id)["rubric"]`.

More in [`examples/`](https://github.com/simulplatform-ai/supposing/tree/main/examples): resuming a run, driving it step by step, and writing your own judge. Each one runs with `python examples/<name>.py` and no key.

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

Know the cost before you start. A run makes at most `planned_calls(n_items, RankConfig(...))` judge calls: by default 8 rounds × (n ÷ 2) pairs × 2 orders, so 8 calls per document. `supposing rank ... --dry-run` prints the estimate and stops, and `max_cost_usd` (`--max-cost`) is a hard cap: no call starts that could take recorded spend past it (each call is assumed to cost as much as the dearest so far), and the run's note says when the cap ended it. Providers that don't report prices, such as `claude_cli`, can't be capped.

## Rubrics

A rubric is what the judge should prefer, in plain words. A string is enough. To keep one in a file, weight its parts, or add instructions for the judge, use a `Rubric`:

```markdown
# Grant proposals

Which proposal makes the strongest case for funding?

## Criteria

- Impact (weight 2): how many people benefit, and how much.
- Feasibility: can this team do it with this budget?

## Judge instructions

Ignore formatting and spelling.
```

```python
from supposing import Rubric, rank

ranking = rank(docs, Rubric.load("grants.md"))
```

Every part but the text is optional, so any Markdown or text file is a rubric. `Rubric.save` writes Markdown, or JSON for a `.json` path. The judge sees the criteria and their weights in its prompt and still returns one verdict per comparison; weights say what matters more, they do not score criteria separately.

## Reading the result

`ranking.summary` is the verdict in one plain sentence, such as "repair-cafe is first, clearly. 4 clear tiers." A tier is a group of items whose rank ranges overlap, so the judge cannot reliably order them; `supposing.tiers(leaderboard)` returns them. When a judge check is poor, the summary adds one sentence of warning.

`ranking.leaderboard` is sorted best first. Each row has:

- `rating`: on an Elo-like scale. The average is 1500, and a 400-point gap means the judge prefers the higher item about 10 times out of 11.
- `rating_lo`, `rating_hi`: the 90% interval for the rating.
- `rank_lo`, `rank_hi`: the ranks the item holds with 90% confidence. **Items whose rank ranges overlap are not reliably different.** Report them as a tie, or add rounds.
- `wins`, `losses`, and `p_top_n` (the chance of being in the top N) for top-N runs.

`ranking.diagnostics` says how far the judge can be trusted:

- `order_consistency`: the share of pairs with the same winner in both orders. Low means the judge often cannot tell the two apart, or is swayed by position.
- `first_position_win_rate`: near 0.5 is good. Both-order judging cancels this bias in the ranking, but a high value still means noisy verdicts.
- `split_half_reliability`: how well two halves of the verdicts agree on the order (Spearman). Near 1 is stable.
- `length_correlation`: whether longer documents tend to rank higher.
- `upsets`: verdicts that contradict the final ranking most strongly. Read a few.
- `first_shown_won_both`, `second_shown_won_both`: among pairs judged in both orders, the share where the same position won both times. A gap between the two is position bias.
- `reliability`: the checks above graded Good, Caution or Poor, with one plain sentence (`reliability["line"]`) to show beside the ranking. Thresholds are rules of thumb, not tests.

These checks say whether the judge is consistent. They cannot say whether its taste is right. Read some verdicts (`ranking.comparisons`, each with `reasoning`) before trusting a ranking.

## How it works

- **Swiss rounds.** Each round pairs every document once, with an opponent of similar current rating, where a verdict is informative. A pair meets again only when no round without rematches exists.
- **Round robin** (`pairing="round_robin"`) is for a handful of options: every pair is planned in the first round, so all the calls can run at once. n options make n(n-1)/2 pairs (28 for 8), so one round is usually enough.
- **Both orders.** Each pair is judged twice with the order swapped, so every document is shown first exactly as often as second. This cancels the judge's tendency to favour one position.
- **Bradley-Terry ratings** fitted to all verdicts at once, so the result does not depend on the order the calls finished in. A weak prior (half a win and half a loss against an average opponent) keeps an unbeaten document's rating finite.
- **Intervals** come from a bootstrap over pairs: refit on resampled pairs a few hundred times and take the middle 90%.
- **Top-N mode** (`goal="top_n", top_n=10`) ranks everything for three rounds. After that it judges only the documents whose side of the cut is still open (a chance of making the top N between 5% and 95%, checked after each of the last two rounds), plus the nearest settled document on either side as a reference, and stops when none is open. `ranking.calls_saved` says how many calls that saved against the fixed plan. In simulations with 30 to 100 documents it used 45-53% of a full ranking's calls and found the true top N as often.
- **Resumable.** With a `SQLiteStore` (`--state run.db` on the command line), every verdict is saved as it arrives. Run the same command again and it continues; nothing already decided is judged again.

### What the tests show

`tests/test_statistics.py` plants a known order, simulates a judge that follows it with noise, and checks what comes back (the many-trial coverage and top-N simulations are marked `stats`; run them with `pytest -m stats`):

- The planted order is recovered, and with enough verdicts the planted ratings too.
- The 90% intervals contain the planted rating 80-97% of the time on evenly spread and on closely matched collections, and the rank ranges at least 80-85%.
- A judge that favours the first text strongly (82% first-position wins between equals) still gives the planted order, and the diagnostics report the bias.
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
