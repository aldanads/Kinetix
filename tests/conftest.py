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
    """Add the `--runslow` and `--update-pzt-trace` command-line flags."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run tests marked `slow` (real MACE CI-NEB pathway sweeps; "
        "can take many hours or even a day).",
    )
    parser.addoption(
        "--update-pzt-trace",
        action="store_true",
        default=False,
        help="Regenerate the miniature-PZT golden-trace fixtures consumed by "
        "tests/test_pzt_golden_trace.py: rebuild tests/fixtures/"
        "grid_PZT_trace.pkl.gz from the production lattice builder, record "
        "2000 seeded kMC events, and overwrite tests/fixtures/"
        "pzt_golden_trace.json. Review the fixture diff before committing.",
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

@pytest.fixture
def kinetix_log_collector():
    """Collect log records from kinetix.* loggers (immune to propagate=False)."""
    import logging
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    
    logger = logging.getLogger("kinetix")
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    yield records
    
    logger.removeHandler(handler)
    logger.setLevel(original_level)