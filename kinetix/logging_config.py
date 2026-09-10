# kinetix/logging_config.py
"""Minimal logging setup for Kinetix. Call once at startup."""
import logging
import sys
import warnings


def setup_logging(level=logging.INFO, log_file=None):
    """Configure the 'kinetix' logger hierarchy.

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
        log_file: optional path to also write logs to a file.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger("kinetix")
    root.setLevel(level)

    # Prevent duplicate output: child loggers (kinetix.lattice.crystal, ...)
    # propagate up to 'kinetix' where our handler emits once. Without stopping
    # propagation here, records would keep climbing to the real root logger ""
    # and, if a third-party library installed a handler there (e.g. via
    # logging.basicConfig()), every message would be printed a second time
    # with Python's default '%(levelname)s:%(name)s:%(message)s' format.
    root.propagate = False

    root.handlers.clear()  # Prevent duplicate handlers when called multiple times (or across MPI ranks)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy external library warnings (torch, mace, e3nn)
    warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
    warnings.filterwarnings("ignore", message=".*weights_only.*", module="mace")


def get_logger(name):
    """Get a logger for the given module name.

    Usage in other modules:
        from kinetix.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)