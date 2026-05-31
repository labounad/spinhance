"""
Root pytest configuration.

Slow tests (marked ``@pytest.mark.slow``) are DESELECTED by default so the
everyday suite stays fast. Run them explicitly with ``--runslow``:

    pytest                 # fast suite (slow tests skipped)
    pytest --runslow       # everything, including slow tests
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked @pytest.mark.slow")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow test, deselected unless --runslow is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
