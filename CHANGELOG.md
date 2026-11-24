# Changelog

## 0.2.0 (2026-11-23)

### Breaking

- **The judge is one `--judge` value.** `supposing rank` and `supposing run` take `--judge quick|balanced|thorough|<OpenRouter model id>`, default `quick`; on your own machine also `mock`, `claude_cli`, `anthropic` and `openai_compat`. `--provider` is gone, and `--model` only picks the model of a `claude_cli`, `anthropic` or `openai_compat` judge.
- **`supposing.cloud`:** `runs.create` and `runs.estimate` take `judge=` (the same values, or a `{provider, model}` object) instead of `provider=` and `model=`.
- **Each pair is judged once by default** (`RankConfig.both_orders=False`), with a random text shown first, so position bias averages out of the ranking. Runs make half the calls: 8 rounds are 4 calls per document. `both_orders=True` (`--both-orders`, MCP `both_orders`) judges every pair twice, as before; `--one-order` is gone.
- The judge check "Same winner in both orders" appears only when pairs were judged in both orders; one-order runs have no `order` check.
- **`Rubric` is a name and a text.** `Criterion`, criteria, weights, `instructions` and JSON rubric files are gone. The text goes into the prompt as written; write weights and instructions into it as sentences. `--rubric` takes the text or the path of a `.md` or `.txt` file (no `@`).
- **MCP:** `judge` is a string (a level, a model id, or locally `mock` or `claude_cli`), not a `{provider, model}` object.

### New

- **Judge levels** `quick`, `balanced` and `thorough`, the same as the web app; `supposing.cloud.LEVELS` and `parse_judge`. `supposing models` (and `Client.models()`) lists the judges with their price per call and, with `--dataset`, for a run.
- **Samples:** `supposing samples`, and `supposing rank --sample <id>` ranks a built-in sample with its own rubric. MCP: `list_samples`, and `rank_documents(sample=...)`, which in cloud mode opens the sample's stored ranking, free. `Client.open_sample()`.
- **Backtest:** `backtest(leaderboard, outcomes)` says how well a ranking agrees with past decisions (AUC and top-k overlap, each with a 90% interval over items, and one sentence). `tune(items, outcomes, rubrics, provider)` picks the rubric that best predicts past decisions on a train split, keeping the base unless a candidate beats it by more than chance, then measures base and choice once on held-out items. With `train_test_split`, `decided_pairs` and `planned_tune_calls`.
- `examples/backtest_rapid_fund/`: a reproducible back-test against 60 real Wikimedia Rapid Fund decisions (FY 2026-27 cycles 1 and 2, after the judge's training cutoff). The pinned sample is `manifest.json`; texts are fetched from Meta-Wiki and cached.
- `display_groups(leaderboard, top_n)` and `results()["groups"]` (also `Ranking.groups`): one rule for showing a ranking. Top-N runs split into settled (a top-N chance of 95% or more), close calls (the contiguous span from the first to the last item with a 5-95% chance) and one out group; full rankings split into clear tiers. `supposing rank` prints the groups as headings.
- `supposing rank`, `run` and `datasets push` read JSONL, JSON and CSV files, one item per row (`--text-field`, `--id-field`, `--title-field`, `--metadata-field`); `run` and `datasets push` also send XLSX through the API's import and upload folders of PDF and Word files (`--redact`).
- `supposing --version`. MCP tools on `rank_documents` and `compare_options`: `both_orders`.

### Changed

- The top-N verdict sentence counts the same groups as `display_groups`, and no longer appends a tier count. It fixes a display rule, used elsewhere, that let close calls swallow 38 of 40 items.
- "First-shown text wins" is no longer graded caution or poor for a gap from 50% that chance alone often gives (within 1.96 or 2.58 standard errors).
- `supposing run --dry-run` prices a run even before the workspace can pay for it, and says what is missing (`can_start`, `blocked_by` in `--json`). Hosted hints never suggest the mock judge.
- `supposing pricing`: a call unit is everything the judge reads (both items, the rubric and our instructions).

## 0.1.0 (2026-11-22)

First public release.

- `rank` and `run_ranking`: pairwise tournaments with an LLM judge, Bradley-Terry ratings and 90% bootstrap intervals on rating and rank.
- Swiss, random and round-robin pairing; both-order judging to cancel position bias; top-N runs that stop when the cut is settled.
- `supposing.ranking`: the run as a pure state machine (`next_action`, `plan_round`, `results`) for job queues and many workers.
- Plain-words results: `summarize` (the verdict sentence), `tiers`, and graded judge checks in `diagnostics["reliability"]`.
- `Rubric`: criteria with weights and judge instructions, saved as Markdown or JSON.
- Providers: OpenRouter, Anthropic, any OpenAI-compatible API, the local Claude Code CLI, and a free mock judge.
- Resumable runs in one SQLite file (`SQLiteStore`).
- `max_cost_usd` is a hard spend cap: no call starts that could take spend past it.
- `supposing.samples`: two sample datasets we wrote (twelve short stories; thirty invented grant applications).
- `supposing` command line: `rank` a folder locally, or `run` it on Supposing Cloud.
- `supposing.cloud`: a Python client for the Supposing Cloud API.
- `supposing mcp`: a stdio MCP server for coding agents (`uvx "supposing[mcp]" mcp`).
