# Changelog

## 0.4.0 (2026-11-25)

### Breaking

- **Saved runs from 0.3.0 and earlier are refused.** A state file (`--state`) or store made before 0.4.0 has no run fingerprint, so opening it raises `RunError` with `code="state_mismatch"` (the CLI exits with that code and the hint "Use a new state file or run id to start fresh") before any call. Nothing is reused or changed in the file; start a new state file, or rerun the old one with 0.3.0.
- **A saved run is refused under a different rubric, prompt or judge** (the same `state_mismatch`), where 0.3.0 silently reused the old verdicts under the new label. Every comparison carries the run's fingerprint (`run_fingerprint`: rubric text, system prompt and prompt layout, context and examples, judge name and model, order mode, text cut), and `Ranking.fingerprint` reports it. `RunError` has `code` and `hint`.
- **Naming the judge in Python:** `rank(judge=...)` is the preferred argument and takes the same strings as the command line (`quick`, `balanced`, `thorough`, an OpenRouter model id, `mock`, `claude_cli`, ...), a `Provider` or a judge function; `provider=` and `model=` still work and `judge=` wins when both are given. `rank(model=)` now defaults to none: OpenRouter still uses `DEFAULT_MODEL`, and `claude_cli` gets its own default (before, it was handed the OpenRouter id). An unknown judge string raises `RunError` (`invalid_judge`) with a suggestion.
- **Ranges for items with fewer than three verdicts are the whole leaderboard** (rank 1 to n, the whole rating span, `few_verdicts: true`), where before an item with no verdicts could show the narrowest range on the board. Two items with two verdicts no longer read "clearly first".
- **Connection failures to a local judge are final.** A refused or timed-out connection to a judge on this machine or a private network (5-second connect timeout) is no longer retried, so a missing Ollama server fails in seconds, with one line saying what to check; public services (OpenRouter, Anthropic) still retry a failed connection, now with the same 5-second connect timeout. `claude_cli` stops at once on a model it does not know.
- No command-line flag was renamed. New flags are listed below.

### Removed

- Removed the PERSUADE, Rapid Fund and UKPConvArg back-test examples (`examples/backtest_persuade/`, `examples/backtest_rapid_fund/`, `examples/backtest_ukpconvarg/`) and their recorded verdicts; ELLIPSE and the Upworthy headline back-tests remain.

### New

- `results(..., max_bootstrap=N)` caps the resamples behind the intervals, for fast, coarser live views of large runs; `n_bootstrap` says how many were used.
- **Customising the judge, in layers.** Each is one optional argument to `rank` and `run_ranking`; the one-line call is unchanged.
  - `context=` (a brief shared by every comparison) and `examples=` (past judgements: `Example(a, b, winner, why=None)` or dicts; `load_examples(path)` reads JSONL) are shown in their own blocks after the rubric. `Item(..., context=)` shows a document's own context beside its text.
  - `prompt=JudgePrompt(system, extra_fields=None, rules=True)` replaces the system prompt with a template using `{rules}` (the built-in rules) and `{output_format}` (the reply contract). It is checked before any call: `{output_format}` is required, unknown placeholders are refused, and leaving out `{rules}` (and so the guard against instructions inside documents) needs `rules=False`. `JudgePrompt.load(path)` reads a template. `SYSTEM_PROMPT` is unchanged and is now built from `INTRO`, `RULES` and `output_format()` in `supposing.judge`.
  - `judge=` takes your own function `fn(a: Item, b: Item, rubric)`, plain or `async`, returning `"A"`, `"B"`, a `Verdict` or a dict with `winner`; it gets the items in the order the library chose, with no prompt to scrape. `judge=` also takes a provider name, an OpenRouter model id or a `Provider`, as `provider=` does.
  - The library still picks the order, parses the verdict and runs the judge checks; the run fingerprint now covers context, examples and the rendered prompt.
- **Extra verdict fields are kept.** Any field of the judge's reply besides winner and reasoning lands in `Comparison.extra` (and `Verdict.extra`); with `JudgePrompt.extra_fields` they are requested and converted to the given type.
- **A `Provider`'s `complete` may be a plain function**: it runs in a worker thread. Before, it had to be `async` and the docs did not say so.
- **`rank()` input:** dicts with `text` (and optionally `id`, `title`, `context`) are accepted; fewer than two items, an empty text or an unknown kind of item raise `RunError` with a clear message (`too_few_items`, `invalid_items`) instead of a numpy error.
- **Adding documents to a ranking works.** Run it again with the same store and the bigger set: `next_action(..., item_ids=...)` keeps planning catch-up rounds until every document has been planned into as many pairs as the others (`games_target`, `behind`), so a new document is judged before it is ranked. The summary says how many documents have too few verdicts to place. Comparisons that name a document no longer in the set are ignored instead of failing.
- `examples/backtest_upworthy/`: `--model` (verdicts kept per model, recorded for Haiku and Opus), `--tests-limit` for running part of a back-test and resuming from the cache, `compare_models.py` for paired differences between models, and `probe_memory.py`, a pre-registered memorisation probe.
- **The judge's reasoning for local runs.** `supposing rank --with-comparisons` prints every judge call with its reasoning after the ranking (or adds `comparisons` to `--json`); `supposing comparisons --state run.db` reads a saved local run. `supposing run --with-comparisons` does the same for a cloud run.
- `supposing pricing` counts a call in bytes (24,000 bytes, about 24,000 characters of English). `--pairing round_robin`, and help for `--pairing` and `--seed`. Ctrl-C says how to continue (with `--state`) or how to make a run resumable. Local `--json` progress counts calls. Both MCP servers report the package version.
- `--max-calls N` and `RankConfig.max_calls`: a hard cap on judge calls for any judge, including ones that report no price (`claude_cli`, a local model). The estimate counts it, and the run's note says when the cap ended it.
- `supposing rank --context`, `--examples` and `--system`, and `context` and `examples` on the local MCP server's tools: the engine's layers for telling the judge more. The cloud doesn't take them yet; `supposing run` refuses them with code `unsupported`.

### Changed

- **Free local judges start without `--yes`.** `mock`, `claude_cli` and an `openai_compat` model on your own machine or network (localhost, a `.local` name or a private address) no longer ask before starting. The estimate says what it rests on, and for priced judges gives the input and output tokens.
- **Judge problems fail in seconds, with a hint.** Before a local run, `supposing rank` checks the judge: an OpenRouter or Anthropic key that is refused, a model id that doesn't exist, an `openai_compat` server that can't be reached ("Is Ollama running?") or that doesn't have the model (it lists the ones it has). With no OpenRouter key, the hint names the free judges (`claude_cli`, Ollama, `mock`). `--model` examples fit the judge, and a mistyped `--judge` gets "Did you mean ...?". A saved run refused under a different rubric or judge (`state_mismatch`) is a usage error (exit 2).
- **Input errors say what was found.** "Only 1 item with text in ...", with every skipped file and why (PDF and Word files are read by `supposing run`). Items with identical text are flagged.

## 0.3.0 (2026-11-24)

### New

- **Anonymous usage statistics from the command line, on by default, easy to turn off.** After each `supposing` command, one small event goes to Supposing. It says which command ran, whether it worked, roughly how long it took and how many items it had, which kind of judge it used, and the package, Python and OS versions, with a random id made on your machine. It never contains your texts, titles, file names, paths, rubric, model output, keys, custom model names, hostnames or command-line arguments. The first time, the command prints a short notice and sends nothing. Turn it off with `supposing telemetry off`, `SUPPOSING_TELEMETRY=0` or `DO_NOT_TRACK=1`; it is also off in CI, under pytest, and when output is piped before you have seen the notice. `supposing telemetry status` shows the id and exactly what the last event contained. Using `supposing` as a library, and the MCP server, never send anything. Details: https://app.supposing.app/docs/telemetry/. The exact fields are in `supposing.cli.telemetry_schema`.
- `display_groups(..., live=True)` and `results(..., live=True)` label the groups of a run still going: "1st so far" (or "Top 2 so far"), "Close behind" and one "Still sorting" group for everything below; for top-N runs "2 in the top 5 so far", "Close behind", "Still sorting". The group shape is unchanged.
- `examples/backtest_ellipse/`: the essay benchmark (100 ELLIPSE essays, two trained raters), reproducible from the public engine; the recorded verdicts replay with no judge calls.
- `examples/backtest_upworthy/`: a pre-registered back-test on 60 Upworthy headline A/B tests (CC BY 4.0, fetched from OSF): the engine ranks each test's headlines and is compared with real click-through, against length, scoring each headline alone and ranking the whole list in one prompt. Recorded verdicts replay with no judge calls.

### Changed

- An empty reply is no longer retried inside the OpenRouter, OpenAI-compatible and Anthropic providers: it was a paid call. `judge_pair` makes at most two calls, the second only when the first reply is empty or unreadable (`READ_ATTEMPTS`). `Verdict.attempts` counts them, and `Verdict.completion` carries the usage and cost summed over all of them.

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
