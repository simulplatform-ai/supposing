"""What the judge should prefer: a prompt, in plain words.

A plain string works everywhere a ``Rubric`` does. A ``Rubric`` adds an optional name, and reads from and
writes to a Markdown or text file. A first line ``# Name`` is the name; everything else is the text the judge
reads::

    # Grant proposals

    Which proposal makes the strongest case for funding? Weigh impact (how many people benefit, and how much)
    about twice as much as feasibility (can this team do it with this budget?). Ignore formatting and spelling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rubric:
    """What the judge should prefer. ``text`` goes into every prompt as written; ``name`` is a label."""

    text: str
    name: str | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("A rubric needs some text: say what the judge should prefer.")

    def to_markdown(self) -> str:
        return ("# %s\n\n%s\n" % (self.name, self.text)) if self.name else self.text + "\n"

    @classmethod
    def from_markdown(cls, source: str, name: str | None = None) -> Rubric:
        """A leading ``# Name`` line is the name; the rest is the text."""
        lines = source.strip().splitlines()
        if lines and lines[0].startswith("# "):
            return cls(text="\n".join(lines[1:]), name=lines[0][2:].strip())
        return cls(text=source, name=name)

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        """Read a Markdown or text file. Without a ``# Name`` line the file's stem is the name."""
        p = Path(path)
        return cls.from_markdown(p.read_text(), name=p.stem)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown())


def as_text(rubric: str | Rubric) -> str:
    """What goes in the prompt for either form."""
    return rubric.text if isinstance(rubric, Rubric) else rubric
