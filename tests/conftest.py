import pytest

from supposing import rating


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: spawns subprocesses")
    config.addinivalue_line("markers", "stats: many-trial statistical simulations; run with `make check-stats`")


@pytest.fixture
def fewer_resamples(monkeypatch):
    """100 bootstrap resamples instead of up to 300: coverage estimates stay honest, the tests stay quick."""
    monkeypatch.setattr(rating, "n_bootstrap_for", lambda n_items, n_matches: 100)
