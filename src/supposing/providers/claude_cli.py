"""The local ``claude`` CLI in headless mode, which uses a Claude subscription login. For local use only."""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from typing import Any

from .base import Completion, ProviderError


class ClaudeCLIProvider:
    """Runs ``claude -p`` once per call, with no tools, from an empty directory so no project context leaks in."""

    name = "claude_cli"

    def __init__(self, model: str = "haiku", binary: str | None = None):
        found = binary or shutil.which("claude")
        if not found:
            raise ProviderError("The `claude` CLI was not found on PATH.", retryable=False)
        self.binary: str = found
        self.model = model
        self._cwd = tempfile.mkdtemp(prefix="supposing-claude-")

    def args(self, system: str) -> list[str]:
        return [
            self.binary, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]

    async def complete(self, system: str, user: str) -> Completion:
        last_error = "unknown error"
        for attempt in range(3):
            started = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *self.args(system),
                cwd=self._cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(user.encode()), timeout=600)
            except TimeoutError:
                proc.kill()
                last_error = "claude CLI timed out"
                continue
            except asyncio.CancelledError:
                proc.kill()
                raise
            try:
                data: dict[str, Any] = json.loads(out.decode() or "{}")
            except json.JSONDecodeError:
                data = {}
            if proc.returncode == 0 and data.get("result") and not data.get("is_error"):
                usage = data.get("usage") or {}
                prompt_tokens = sum(
                    usage.get(k) or 0
                    for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                )
                return Completion(
                    text=str(data["result"]).strip(),
                    prompt_tokens=prompt_tokens or None,
                    completion_tokens=usage.get("output_tokens"),
                    cost_usd=None,  # a subscription is not billed per call
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
            last_error = (str(data.get("result") or "") or err.decode() or out.decode() or "claude CLI failed")[:400]
            await asyncio.sleep(3 * (attempt + 1))
        raise ProviderError(last_error)

    async def close(self) -> None:
        pass
