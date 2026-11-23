def pytest_configure(config):
    config.addinivalue_line("markers", "slow: spawns subprocesses")
    config.addinivalue_line("markers", "stats: many-trial statistical simulations; run with `make check-stats`")
