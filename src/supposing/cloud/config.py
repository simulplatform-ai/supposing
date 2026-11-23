"""Where the client finds its API key and URL.

Order: explicit arguments, then ``SUPPOSING_API_KEY`` / ``SUPPOSING_API_URL``, then the config file written by
``supposing login`` (``$XDG_CONFIG_HOME/supposing/config.json``, default ``~/.config/supposing/config.json``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_URL = "https://api.supposing.app"
KEY_PREFIX = "sp_"


def config_path() -> Path:
    if os.environ.get("SUPPOSING_CONFIG"):
        return Path(os.environ["SUPPOSING_CONFIG"])
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "supposing" / "config.json"


@dataclass
class Credentials:
    api_key: str | None
    api_url: str
    source: str  # "argument" | "env" | "file" | "none"


def read_config(path: Path | None = None) -> dict[str, str]:
    p = path or config_path()
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


def write_config(api_key: str, api_url: str, path: Path | None = None) -> Path:
    """Save the key readable only by you (mode 600)."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = {**read_config(p), "api_key": api_key, "api_url": api_url.rstrip("/")}
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(p, 0o600)
    return p


def remove_config(path: Path | None = None) -> bool:
    p = path or config_path()
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


def resolve(api_key: str | None = None, api_url: str | None = None) -> Credentials:
    file = read_config()
    url = (api_url or os.environ.get("SUPPOSING_API_URL") or file.get("api_url") or DEFAULT_API_URL).rstrip("/")
    if api_key:
        return Credentials(api_key, url, "argument")
    if os.environ.get("SUPPOSING_API_KEY"):
        return Credentials(os.environ["SUPPOSING_API_KEY"], url, "env")
    if file.get("api_key"):
        return Credentials(file["api_key"], url, "file")
    return Credentials(None, url, "none")
