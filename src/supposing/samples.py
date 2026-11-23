"""Sample datasets for a first ranking with no files of your own: the CLI, the MCP servers and the API all
read this one copy.

    from supposing import rank, samples
    s = samples.load("stories")
    rank(samples.items("stories"), s["rubric"], provider="mock")

``stories`` are twelve short pieces written for Supposing. ``grants`` are thirty short applications to an
invented community fund, from vague to well evidenced; every person and organisation in them is fictional.
Both are covered by the package's Apache-2.0 licence.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

from .ranking import Item


@cache
def _all() -> dict[str, dict[str, Any]]:
    data = json.loads(resources.files(__package__).joinpath("samples.json").read_text(encoding="utf-8"))
    return {s["id"]: s for s in data}


def list_samples() -> list[dict[str, Any]]:
    """Each sample's ``id``, ``name``, ``description``, ``rubric``, ``credit`` (None for our own) and
    ``n_items``."""
    return [{k: s[k] for k in ("id", "name", "description", "rubric", "credit")} | {"n_items": len(s["items"])}
            for s in _all().values()]


def load(sample_id: str) -> dict[str, Any]:
    """A whole sample: the fields of ``list_samples`` plus ``items``, each with ``title``, ``filename`` and
    ``text``. Raises ``KeyError`` naming the samples that exist."""
    s = _all().get(sample_id)
    if s is None:
        raise KeyError("No sample called %r. Use one of: %s." % (sample_id, ", ".join(_all())))
    return json.loads(json.dumps(s))  # a copy, so callers cannot change the shared data


def items(sample_id: str) -> list[Item]:
    """A sample's items, ready for ``rank`` or ``run_ranking``."""
    return [Item(id=i["filename"], text=i["text"], title=i["title"]) for i in load(sample_id)["items"]]
