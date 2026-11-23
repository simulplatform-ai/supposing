# Changelog

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
- `supposing mcp`: a stdio MCP server for coding agents (`pip install "supposing[mcp]"`).
