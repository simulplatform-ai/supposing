"""Anonymous usage telemetry for the ``supposing`` command line: one event per command, opt-out.

Only the CLI's ``main`` calls this; ``import supposing``, the library API and the MCP server never do. What an event
may contain is fixed by ``telemetry_schema`` (an allow-list of enums and buckets); anything else is never sent.

Off when any of these holds: ``SUPPOSING_TELEMETRY=0``, ``DO_NOT_TRACK=1``, ``supposing telemetry off``, a CI
environment, a test run, or no terminal on stdout before the first-run notice was ever shown. The first run that
could send prints the notice instead and sends nothing. Sending happens at exit, in a background thread capped at
1.5 seconds; every failure is silent and nothing here changes a command's output or exit code.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..cloud.config import config_path
from . import telemetry_schema as schema

TIMEOUT_SECONDS = 1.5
NOTICE = (
    "supposing collects anonymous usage data: the command, its outcome and duration, the item count range and the "
    "judge kind. Never your texts, rubric, file names or keys.\n"
    "Turn it off: supposing telemetry off (or SUPPOSING_TELEMETRY=0). Details: %s\n" % schema.DOCS_URL)
OFF_VALUES = ("0", "false", "off", "no")
# Environment variables that mean a CI or build system is running the command.
CI_VARS = ("CI", "CONTINUOUS_INTEGRATION", "BUILD_NUMBER", "RUN_ID", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI",
           "TRAVIS", "BUILDKITE", "JENKINS_URL", "JENKINS_HOME", "TEAMCITY_VERSION", "TF_BUILD",
           "BITBUCKET_BUILD_NUMBER", "CODEBUILD_BUILD_ID", "DRONE", "APPVEYOR", "SEMAPHORE", "BUILDKITE_BUILD_ID",
           "HEROKU_TEST_RUN_ID", "NETLIFY", "VERCEL", "CF_PAGES", "GOOGLE_CLOUD_BUILD", "BUILD_ID", "WOODPECKER")
EXIT_OUTCOMES = {0: "ok", 1: "run_failed", 2: "usage", 3: "auth", 4: "not_found", 5: "refused", 6: "unavailable",
                 7: "needs_confirmation", 8: "needs_credit", 130: "interrupted"}
SILENT_COMMANDS = ("telemetry", "mcp")  # managing telemetry, and the long-lived MCP server, never send

_context: dict[str, Any] = {}


# --- state: <config dir>/telemetry.json ----------------------------------------------------------------------
def state_path() -> Path:
    return config_path().parent / "telemetry.json"


def load_state() -> dict[str, Any]:
    try:
        data = json.loads(state_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    try:
        p = state_path()
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(p)
    except OSError:
        pass


def _install_id(state: dict[str, Any]) -> str:
    """A random uuid4 made on this machine, not derived from anything about it."""
    if not schema._UUID4.fullmatch(str(state.get("install_id") or "")):
        state["install_id"] = str(uuid.uuid4())
    return state["install_id"]


# --- when it is off ------------------------------------------------------------------------------------------
def _stdout_is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def off_reason(env: dict[str, str], stdout_tty: bool, state: dict[str, Any], under_pytest: bool) -> str | None:
    """Why nothing may be sent, or None. Checked in this order, so status names the strongest reason."""
    if (env.get("SUPPOSING_TELEMETRY") or "").strip().lower() in OFF_VALUES:
        return "SUPPOSING_TELEMETRY=%s" % env["SUPPOSING_TELEMETRY"].strip()
    if (env.get("DO_NOT_TRACK") or "").strip().lower() not in ("",) + OFF_VALUES:
        return "DO_NOT_TRACK is set"
    if state.get("enabled") is False:
        return "turned off with `supposing telemetry off`"
    ci = next((v for v in CI_VARS if (env.get(v) or "").strip().lower() not in ("",) + OFF_VALUES), None)
    if ci:
        return "a CI environment (%s is set)" % ci
    if under_pytest:
        return "running under pytest"
    if not state.get("notice_shown") and not stdout_tty:
        return "no terminal, and the first-run notice has not been shown"
    return None


# --- what a command adds to its event ------------------------------------------------------------------------
def judge_kind(provider: str | None, base_url: str | None = None) -> str | None:
    if provider is None:
        return None
    if provider == "openai_compat":
        url = (base_url or "").lower()
        return "ollama" if ("11434" in url or "ollama" in url) else "openai" if "api.openai.com" in url else "other"
    return provider if provider in schema.JUDGES else "other"


def model_kind(model: str | None, level: str | None = None, provider: str | None = None) -> str | None:
    """A short name for a known model; any other model string becomes "other"."""
    if level in ("quick", "balanced", "thorough"):
        return level
    if provider == "mock":
        return "mock"
    if not model:
        return None
    m = model.lower()
    anthropic = provider in ("claude_cli", "anthropic") or m.startswith("anthropic/") or m.startswith("claude")
    for name in ("haiku", "sonnet", "opus"):
        if anthropic and name in m:
            return name
    return "other"


def sample_kind(sample_id: str | None) -> str | None:
    if not sample_id:
        return None
    return sample_id if sample_id in schema.SAMPLES else "other"


def note(*, local: bool | None = None, provider: str | None = None, base_url: str | None = None,
         model: str | None = None, level: str | None = None, items: int | None = None, rounds: int | None = None,
         both_orders: bool | None = None, sample: str | None = None) -> None:
    """Called by a command with what it did. Raw values are reduced to the schema's enums here; none is kept."""
    _context.update(local=local, judge=judge_kind(provider, base_url), model=model_kind(model, level, provider),
                    items=schema.items_bucket(items),
                    rounds=rounds if isinstance(rounds, int) and rounds in range(1, 201) else None,
                    both_orders=both_orders if isinstance(both_orders, bool) else None,
                    sample=sample_kind(sample))


def command_name(args: Any) -> str:
    if getattr(args, "command", None) in SILENT_COMMANDS:
        return args.command
    name = "_".join(p for p in (getattr(args, "command", None), getattr(args, "sub", None)) if p)
    return name if name in schema.COMMANDS else "other"


def os_family() -> str:
    s = sys.platform
    return "linux" if s.startswith("linux") else "darwin" if s == "darwin" else "windows" if s in ("win32", "cygwin") \
        else "other"


def build_event(install_id: str, command: str, exit_code: int | None, seconds: float) -> dict[str, Any]:
    event: dict[str, Any] = {k: None for k in schema.FIELDS}
    event.update({k: v for k, v in _context.items() if k in schema.FIELDS})
    event.update(install_id=install_id, command=command, version=__version__,
                 python="%d.%d" % sys.version_info[:2], os=os_family(),
                 duration=schema.duration_bucket(seconds),
                 outcome=EXIT_OUTCOMES.get(exit_code, "error") if exit_code is not None else "error")
    return event


# --- sending -------------------------------------------------------------------------------------------------
def post(body: bytes) -> None:
    req = urllib.request.Request(schema.ENDPOINT, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "supposing-cli"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS):
        pass


def _send_quietly(body: bytes) -> None:
    with contextlib.suppress(Exception):  # any failure, offline or a server error, is silent
        post(body)


def finish(command: str, exit_code: int | None, seconds: float, stdout_tty: bool | None = None) -> None:
    """At the end of a command: the first-run notice, or one event, or nothing. Never raises."""
    try:
        if command in SILENT_COMMANDS:
            return
        tty = _stdout_is_tty() if stdout_tty is None else stdout_tty
        state = load_state()
        if off_reason(dict(os.environ), tty, state, _under_pytest()):
            return
        if not state.get("notice_shown"):
            sys.stderr.write("\n" + NOTICE)  # this run sends nothing
            state["notice_shown"] = datetime.now(UTC).date().isoformat()
            _install_id(state)
            save_state(state)
            return
        event = build_event(_install_id(state), command, exit_code, seconds)
        body = json.dumps(event, separators=(",", ":")).encode()
        if not schema.valid(event) or len(body) > schema.MAX_BYTES:
            return
        state["last_event"] = event
        save_state(state)
        t = threading.Thread(target=_send_quietly, args=(body,), daemon=True)
        t.start()
        t.join(TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001  telemetry never breaks a command
        pass
    finally:
        _context.clear()


# --- `supposing telemetry off|on|status` ------------------------------------------------------------------------
def set_enabled(on: bool) -> dict[str, Any]:
    state = load_state()
    state["enabled"] = on
    if on:  # turning it on is consent: no notice needed
        state.setdefault("notice_shown", datetime.now(UTC).date().isoformat())
        _install_id(state)
    save_state(state)
    return status(state)


def status(state: dict[str, Any] | None = None, stdout_tty: bool | None = None) -> dict[str, Any]:
    state = load_state() if state is None else state
    tty = _stdout_is_tty() if stdout_tty is None else stdout_tty
    reason = off_reason(dict(os.environ), tty, state, under_pytest=False)
    return {"enabled": reason is None, "off_because": reason, "install_id": state.get("install_id"),
            "notice_shown": state.get("notice_shown"), "last_event": state.get("last_event"),
            "endpoint": schema.ENDPOINT, "docs": schema.DOCS_URL, "state_file": str(state_path())}
