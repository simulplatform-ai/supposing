"""Reading a folder of documents into items, shared by ``supposing rank`` and ``datasets.push``."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXTENSIONS = (".md", ".txt", ".html", ".htm")


@dataclass
class FolderItems:
    items: list[tuple[str, str]] = field(default_factory=list)  # (title, text); title is the relative path
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (relative path, reason)


def read_folder(folder: str | Path) -> FolderItems:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError("%s is not a folder" % root)
    out = FolderItems()
    for p in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(p.relative_to(root))
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() not in EXTENSIONS:
            out.skipped.append((rel, "not .md, .txt or .html"))
            continue
        text = p.read_text(errors="replace").strip()
        if not text:
            out.skipped.append((rel, "empty"))
            continue
        out.items.append((rel, text))
    return out
