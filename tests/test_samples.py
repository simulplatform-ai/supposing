import pytest

from supposing import rank, samples


def test_samples_list_and_load():
    listed = {s["id"]: s for s in samples.list_samples()}
    assert set(listed) == {"stories", "grants"}
    assert listed["stories"]["n_items"] == 12 and listed["grants"]["n_items"] == 30
    assert all(s["credit"] is None for s in listed.values())  # all our own writing, under the package licence
    grants = samples.load("grants")
    assert all(i["text"].strip() and i["title"] for i in grants["items"])
    assert all(110 <= len(i["text"].split()) <= 260 for i in grants["items"])
    grants["items"].clear()
    assert len(samples.load("grants")["items"]) == 30


def test_unknown_sample_names_the_real_ones():
    with pytest.raises(KeyError, match="stories"):
        samples.load("cats")


def test_a_sample_ranks_with_the_mock_judge():
    s = samples.load("stories")
    ranking = rank(samples.items("stories"), s["rubric"], provider="mock", rounds=2)
    assert len(ranking.leaderboard) == 12
