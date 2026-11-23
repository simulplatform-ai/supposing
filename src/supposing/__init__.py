"""Supposing: rank documents with LLM judges.

    from supposing import rank
    ranking = rank({"a.md": text_a, "b.md": text_b, ...}, "Which proposal is most fundable?", provider="mock")
    ranking.leaderboard

Three layers, each usable on its own:

- ``rank`` and ``run_ranking`` run a whole ranking in this process and return a ``Ranking``.
- ``next_action``, ``plan_round`` and ``results`` are the run as a pure state machine over ``Comparison``
  rows, for running it from a job queue or across workers. ``judge_pair``, ``apply_verdict`` and
  ``apply_error`` make and record one judge call.
- ``supposing.samples`` holds sample datasets for a first ranking with no files of your own.
- ``supposing.present`` words runs, results, estimates and errors for agents and terminals.
- ``Provider`` (built with ``make_provider``) is the judge; ``RunStore`` (``MemoryStore``, ``SQLiteStore``)
  is where a run's comparisons are kept.
"""
from . import present, samples
from .judge import SYSTEM_PROMPT, apply_error, apply_verdict, judge_pair
from .providers import Completion, Provider, ProviderError, make_provider
from .ranking import (
    Action,
    Comparison,
    ConfigError,
    Item,
    RankConfig,
    affordable,
    fewest_calls,
    next_action,
    plan_round,
    planned_calls,
    results,
    timeline,
)
from .report import judge_checks, summarize, tiers
from .rubric import Criterion, Rubric
from .runner import Ranking, RunError, rank, run_ranking
from .store import MemoryStore, RunStore, SQLiteStore

__version__ = "0.1.0"

__all__ = [
    # Run a ranking
    "rank", "run_ranking", "Ranking", "RunError", "Item", "Rubric", "Criterion", "RankConfig", "ConfigError", "planned_calls",
    "fewest_calls",
    # The state machine
    "Comparison", "Action", "next_action", "affordable", "plan_round", "results", "timeline",
    # Results in words
    "summarize", "tiers", "judge_checks", "present", "samples",
    # One judge call
    "judge_pair", "apply_verdict", "apply_error", "SYSTEM_PROMPT",
    # Judges and stores
    "Provider", "Completion", "ProviderError", "make_provider", "RunStore", "MemoryStore", "SQLiteStore",
]
