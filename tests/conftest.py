"""
Pytest configuration file.

No sys.path manipulation is needed: tests/ is a package (tests/__init__.py),
so pytest inserts the repository root into sys.path, making the ``kinetix``
package importable when pytest is invoked from the repository root
(e.g. ``python -m pytest tests/``).

It also registers the ``--runslow`` flag: tests marked ``@pytest.mark.slow``
(heavy real-MACE CI-NEB pathway sweeps that can take many hours or even a
day) are skipped by default and only run when pytest is invoked with
``--runslow``.
"""
import pytest



def pytest_addoption(parser):
    """Add the `--runslow` command-line flag (opt-in for long-running tests)."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run tests marked `slow` (real MACE CI-NEB pathway sweeps; "
        "can take many hours or even a day).",
    )


def pytest_configure(config):
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: tests that run many real MACE CI-NEB calculations (can take "
        "many hours or even a day); they are SKIPPED by default and only run "
        "when pytest is invoked with `--runslow`",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `slow` tests unless `--runslow` is passed on the command line."""
    if config.getoption("--runslow", default=False):
        return
    skip_slow = pytest.mark.skip(
        reason="slow test: needs `--runslow` (runs many real MACE CI-NEB "
        "calculations; can take many hours or even a day)"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)