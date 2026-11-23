import pytest

from supposing import Criterion, Rubric, rank
from supposing.judge import build_prompt
from supposing.providers import MockProvider

GRANTS = """# Grant proposals

Which proposal makes the strongest case for funding?

## Criteria

- Impact (weight 2): how many people benefit, and how much.
- Feasibility: can this team do it
  with this budget?

## Judge instructions

Ignore formatting and spelling.
"""


def test_markdown_parses_every_part():
    r = Rubric.from_markdown(GRANTS)
    assert r.name == "Grant proposals"
    assert r.text == "Which proposal makes the strongest case for funding?"
    assert r.criteria == [Criterion("Impact", "how many people benefit, and how much.", 2.0),
                          Criterion("Feasibility", "can this team do it with this budget?", 1.0)]
    assert r.instructions == "Ignore formatting and spelling."


@pytest.mark.parametrize("suffix", [".md", ".json", ".txt"])
def test_save_and_load_round_trip(tmp_path, suffix):
    r = Rubric.from_markdown(GRANTS)
    path = tmp_path / ("rubric" + suffix)
    r.save(path)
    assert Rubric.load(path) == r


def test_any_text_file_is_a_rubric(tmp_path):
    path = tmp_path / "clarity.txt"
    path.write_text("Which explanation would a new engineer understand faster?\n")
    r = Rubric.load(path)
    assert r.name == "clarity" and r.text.startswith("Which explanation") and not r.criteria


def test_render_names_weights_only_when_they_differ():
    even = Rubric("Better?", criteria=[Criterion("Clarity"), Criterion("Accuracy", "facts are right")])
    assert "weight" not in even.render()
    assert "- Accuracy: facts are right" in even.render()
    weighted = Rubric.from_markdown(GRANTS).render()
    assert "Impact (weight 2)" in weighted and "Feasibility (weight 1)" in weighted
    assert weighted.endswith("Instructions from the rubric's author: Ignore formatting and spelling.")


def test_plain_string_prompt_is_unchanged():
    assert build_prompt(Rubric("Which is clearer?"), "a", "b") == build_prompt("Which is clearer?", "a", "b")


@pytest.mark.parametrize("bad", [dict(text=""), dict(text="x", criteria=[Criterion("c", weight=0)])])
def test_invalid_rubrics_are_refused(bad):
    with pytest.raises(ValueError):
        Rubric(**bad)


def test_rank_accepts_a_rubric():
    seen = []

    class Recording(MockProvider):
        async def complete(self, system, user):
            seen.append(user)
            return await super().complete(system, user)

    rank({"a": "[[q=0.9]] a", "b": "[[q=0.1]] b"}, Rubric.from_markdown(GRANTS), provider=Recording(), rounds=1)
    assert seen and all("Impact (weight 2)" in u for u in seen)
