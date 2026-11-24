import pytest

from supposing import Rubric, rank
from supposing.judge import build_prompt
from supposing.providers import MockProvider

GRANTS = """# Grant proposals

Which proposal makes the strongest case for funding?
Ignore formatting and spelling.
"""


def test_markdown_heading_is_the_name_and_the_rest_is_the_text():
    r = Rubric.from_markdown(GRANTS)
    assert r.name == "Grant proposals"
    assert r.text == "Which proposal makes the strongest case for funding?\nIgnore formatting and spelling."


@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_save_and_load_round_trip(tmp_path, suffix):
    r = Rubric.from_markdown(GRANTS)
    path = tmp_path / ("rubric" + suffix)
    r.save(path)
    assert Rubric.load(path) == r


def test_any_text_file_is_a_rubric(tmp_path):
    path = tmp_path / "clarity.txt"
    path.write_text("Which explanation would a new engineer understand faster?\n")
    r = Rubric.load(path)
    assert r.name == "clarity" and r.text == "Which explanation would a new engineer understand faster?"


def test_a_rubric_and_its_text_give_the_same_prompt():
    assert build_prompt(Rubric("Which is clearer?", name="x"), "a", "b") == build_prompt("Which is clearer?", "a", "b")


def test_empty_rubrics_are_refused():
    with pytest.raises(ValueError):
        Rubric("  ")


def test_rank_accepts_a_rubric():
    seen = []

    class Recording(MockProvider):
        async def complete(self, system, user):
            seen.append(user)
            return await super().complete(system, user)

    rank({"a": "[[q=0.9]] a", "b": "[[q=0.1]] b"}, Rubric.from_markdown(GRANTS), provider=Recording(), rounds=1)
    assert seen and all("Ignore formatting and spelling." in u and "Grant proposals" not in u for u in seen)
