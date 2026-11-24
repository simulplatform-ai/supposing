"""The judge, in the words the web app, the API, the CLI and the MCP tools share (DESIGN.md 4.13).

A judge is a string: a level (``quick``, the default, ``balanced`` or ``thorough``), an OpenRouter model id
(it contains a ``/``), ``mock`` (free, made-up verdicts) or ``claude_cli`` (your Claude subscription, on your
own machine only). The levels name the same models as the API's ``LEVEL_MODELS``.
"""
from __future__ import annotations

from typing import Any

LEVELS: dict[str, str] = {
    "quick": "anthropic/claude-haiku-5.5",
    "balanced": "google/gemini-3.8-flash",
    "thorough": "anthropic/claude-sonnet-5.5",
}
DEFAULT_JUDGE = "quick"
PROVIDER_JUDGES = ("mock", "claude_cli")  # provider names that are judges on their own, in the API too
LOCAL_PROVIDERS = ("anthropic", "openai_compat")  # judges only `supposing rank` can use
PROVIDER_DEFAULT_MODEL = {"mock": "mock", "claude_cli": "haiku"}

JUDGE_HELP = ("quick (the default), balanced, thorough, or an OpenRouter model id such as "
              "anthropic/claude-opus-5.5. On your own machine only: mock (free, made-up verdicts, for trying "
              "things) and claude_cli (your Claude subscription).")


class JudgeError(ValueError):
    """A judge value no surface accepts. The message says which values work."""


def parse_judge(value: str | None, model: str | None = None, local: bool = False) -> dict[str, Any]:
    """``{provider, model, level}`` for a judge string. ``model`` picks the model of a provider judge
    (``claude_cli``, and locally ``anthropic`` or ``openai_compat``); ``local`` allows the local-only providers."""
    v = (value or DEFAULT_JUDGE).strip()
    providers = PROVIDER_JUDGES + (LOCAL_PROVIDERS if local else ())
    if v in providers:
        chosen = model or PROVIDER_DEFAULT_MODEL.get(v)
        if not chosen:
            raise JudgeError("The %s judge needs a model: add --model, e.g. --model llama3." % v)
        return {"provider": v, "model": chosen, "level": None}
    if model:
        raise JudgeError("--model goes with a provider judge (%s). For OpenRouter, pass the model as the judge: "
                         "--judge %s." % (", ".join(providers), model if "/" in model else "anthropic/claude-opus-5.5"))
    if v in LEVELS:
        return {"provider": "openrouter", "model": LEVELS[v], "level": v}
    if "/" in v:
        return {"provider": "openrouter", "model": v, "level": next((k for k, m in LEVELS.items() if m == v), None)}
    if v in LOCAL_PROVIDERS:
        raise JudgeError("The %s judge works only on this machine: use supposing rank." % v)
    raise JudgeError("Unknown judge %r. Use quick, balanced, thorough, an OpenRouter model id like "
                     "anthropic/claude-haiku-5.5, mock or claude_cli." % v)


def api_judge(value: str | None, model: str | None = None) -> str | dict[str, str] | None:
    """What a run request sends as ``judge``: the short form, or ``{provider, model}`` for a provider judge
    with its own model. None leaves the API's default (quick)."""
    if value is None and model is None:
        return None
    j = parse_judge(value, model)
    if model:
        return {"provider": j["provider"], "model": j["model"]}
    return (value or DEFAULT_JUDGE).strip()
