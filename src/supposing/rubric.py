"""What the judge should prefer, as an object you can keep in a file and reuse.

A rubric is plain words, optionally broken into named criteria with relative weights, plus optional extra
instructions for the judge. A plain string works everywhere a ``Rubric`` does; a ``Rubric`` is for when
you want to save it, share it, or weight its parts.

On disk a rubric is Markdown that reads well on its own::

    # Grant proposals

    Which proposal makes the strongest case for funding?

    ## Criteria

    - Impact (weight 2): how many people benefit, and how much.
    - Feasibility: can this team do it with this budget?

    ## Judge instructions

    Ignore formatting and spelling.

Every part but the text is optional, so any Markdown or text file is a valid rubric. ``.json`` files hold the
same fields as ``to_dict``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_CRITERION = re.compile(r"^[-*]\s+(?P<name>[^:(]+?)\s*(?:\(weight\s+(?P<weight>[0-9.]+)\))?\s*:\s*(?P<text>.*)$")
_HEADING = re.compile(r"^##\s+(criteria|judge instructions)\s*$", re.I)


@dataclass
class Criterion:
    """One part of a rubric. ``weight`` is relative to the other criteria; the judge is told the weights."""

    name: str
    description: str = ""
    weight: float = 1.0


@dataclass
class Rubric:
    """What the judge should prefer. ``text`` is required; ``criteria`` and ``instructions`` are optional."""

    text: str
    name: str | None = None
    criteria: list[Criterion] = field(default_factory=list)
    instructions: str | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        self.criteria = [c if isinstance(c, Criterion) else Criterion(**c) for c in self.criteria]
        if not self.text and not self.criteria:
            raise ValueError("A rubric needs some text: say what the judge should prefer.")
        for c in self.criteria:
            c.weight = float(c.weight)
            if c.weight <= 0:
                raise ValueError("Criterion %r has weight %g; weights must be positive." % (c.name, c.weight))

    def render(self) -> str:
        """The rubric as the judge reads it, inside the prompt's criteria block."""
        parts = [self.text] if self.text else []
        if self.criteria:
            weighted = len({c.weight for c in self.criteria}) > 1
            lines = []
            for c in self.criteria:
                label = "%s (weight %g)" % (c.name, c.weight) if weighted else c.name
                lines.append("- %s: %s" % (label, c.description) if c.description else "- %s" % label)
            head = "Criteria, with their relative weights:" if weighted else "Criteria:"
            parts.append(head + "\n" + "\n".join(lines))
        if self.instructions:
            parts.append("Instructions from the rubric's author: " + self.instructions.strip())
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Rubric:
        return cls(text=raw.get("text") or "", name=raw.get("name"), criteria=list(raw.get("criteria") or []),
                   instructions=raw.get("instructions"))

    def to_markdown(self) -> str:
        out = []
        if self.name:
            out.append("# %s" % self.name)
        if self.text:
            out.append(self.text)
        if self.criteria:
            lines = []
            for c in self.criteria:
                label = c.name if c.weight == 1 else "%s (weight %g)" % (c.name, c.weight)
                lines.append("- %s: %s" % (label, c.description))
            out.append("## Criteria\n\n" + "\n".join(lines))
        if self.instructions:
            out.append("## Judge instructions\n\n" + self.instructions.strip())
        return "\n\n".join(out) + "\n"

    @classmethod
    def from_markdown(cls, source: str, name: str | None = None) -> Rubric:
        """Parse the format in this module's docstring. Text with no headings is all rubric text."""
        lines = source.strip().splitlines()
        if lines and lines[0].startswith("# "):
            name = lines[0][2:].strip()
            lines = lines[1:]
        sections: dict[str, list[str]] = {"text": [], "criteria": [], "judge instructions": []}
        current = "text"
        for line in lines:
            heading = _HEADING.match(line.strip())
            if heading:
                current = heading.group(1).lower()
                continue
            sections[current].append(line)
        criteria = []
        for line in sections["criteria"]:
            m = _CRITERION.match(line.strip())
            if m:
                criteria.append(Criterion(m["name"].strip(), m["text"].strip(), float(m["weight"] or 1)))
            elif line.strip() and criteria:  # a wrapped description line
                criteria[-1].description += " " + line.strip()
        instructions = "\n".join(sections["judge instructions"]).strip() or None
        return cls(text="\n".join(sections["text"]), name=name, criteria=criteria, instructions=instructions)

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        """Read a rubric from ``.json``, or from Markdown or plain text (any other extension)."""
        p = Path(path)
        source = p.read_text()
        if p.suffix.lower() == ".json":
            return cls.from_dict(json.loads(source))
        return cls.from_markdown(source, name=p.stem)

    def save(self, path: str | Path) -> None:
        """Write to ``.json``, or to Markdown for any other extension."""
        p = Path(path)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        else:
            p.write_text(self.to_markdown())


def as_text(rubric: str | Rubric) -> str:
    """What goes in the prompt for either form. A plain string is used as given."""
    return rubric.render() if isinstance(rubric, Rubric) else rubric
