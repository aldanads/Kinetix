# tests/test_simulator_rename.py
"""
Contract spec for the Phase 7 global rename (pure rename, no logic change):

  * ``Crystal_Lattice`` -> ``KMCSimulator`` (class + every reference)
  * ``kinetix/lattice/crystal.py`` -> ``kinetix/lattice/simulator.py``
  * ``System_state`` -> ``simulator`` (cli/analysis/superbasin/island/tests)
  * collaborators read/write the simulator through ``self.simulator``

BEHAVIOR NOTES pinned below:
  * ``KMCSimulator.__module__`` is ``kinetix.lattice.simulator``; the old
    module path is gone (``git mv``), so ``find_spec`` returns None
  * the top-level package exports ``KMCSimulator`` (the stale name is NOT kept
    in ``kinetix.__all__``)
  * the one legacy name that survives is the documented backward-compat alias
    ``Crystal_Lattice = KMCSimulator`` at the bottom of ``simulator.py`` - the
    grids in ``data/grids/`` reference only ``kinetix.lattice.site``, so they
    keep loading; the alias exists for external callers, not for the grids
  * a pre-rename ``variables.pkl`` (``save_variables`` output, written by
    ``cli.py``) stores the OLD module path and therefore no longer unpickles -
    pinned as a known limitation so adding a shim stays a deliberate decision

This file is deliberately excluded from the self-scan below: the rename
contract has to spell out the old identifiers it forbids.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

import kinetix
from kinetix.lattice.simulator import KMCSimulator

PACKAGE = Path(kinetix.__file__).resolve().parent
REPO_ROOT = PACKAGE.parent
THIS_FILE = Path(__file__).resolve()

# Identifiers the rename removes, and the one file allowed to spell one of them
# (the backward-compat alias line).
FORBIDDEN_ANYWHERE = ("System_state",)
FORBIDDEN_OUTSIDE_SIMULATOR = ("Crystal_Lattice",)
ALIAS_LINE = "Crystal_Lattice = KMCSimulator"
SIMULATOR_PATH = PACKAGE / "lattice" / "simulator.py"
OLD_MODULE_PATHS = (
    "kinetix.lattice.crystal",
    "kinetix/lattice/crystal.py",
    "lattice/crystal.py",
)


def _iter_python_files():
  """Every first-party module: kinetix/ package + tests/."""
  for root in (PACKAGE, REPO_ROOT / "tests"):
    for path in sorted(root.rglob("*.py")):
      if "__pycache__" in path.parts or path.resolve() == THIS_FILE:
        continue
      yield path


# =============================================================================
# Module / class identity
# =============================================================================

def test_class_lives_in_simulator_module():
  """``KMCSimulator`` is defined in ``kinetix/lattice/simulator.py``."""
  assert KMCSimulator.__name__ == "KMCSimulator"
  assert KMCSimulator.__module__ == "kinetix.lattice.simulator"
  assert Path(kinetix.lattice.simulator.__file__).name == "simulator.py"


def test_legacy_alias_is_the_same_class():
  """The documented alias keeps ``from ... import Crystal_Lattice`` working."""
  from kinetix.lattice.simulator import Crystal_Lattice

  assert Crystal_Lattice is KMCSimulator


def test_old_module_path_is_gone():
  """``git mv`` removed the old module: no crystal.py, no importable path."""
  assert not (PACKAGE / "lattice" / "crystal.py").exists()
  assert importlib.util.find_spec("kinetix.lattice.crystal") is None


def test_top_level_export_uses_new_name():
  """``kinetix.KMCSimulator`` is the public name; the stale one is dropped."""
  assert kinetix.KMCSimulator is KMCSimulator
  assert "KMCSimulator" in kinetix.__all__


# =============================================================================
# No stale identifiers anywhere (the acceptance criterion, pinned)
# =============================================================================

def test_no_stale_identifiers_in_package_or_tests():
  """``System_state`` anywhere, ``Crystal_Lattice`` outside the alias: none."""
  offenders: list[str] = []
  for path in _iter_python_files():
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      rel = path.relative_to(REPO_ROOT)
      for token in FORBIDDEN_ANYWHERE:
        if token in line:
          offenders.append(f"{rel}:{lineno}: {token}")
      for token in FORBIDDEN_OUTSIDE_SIMULATOR:
        if token in line and not (
          path.resolve() == SIMULATOR_PATH.resolve() and line.strip() == ALIAS_LINE
        ):
          offenders.append(f"{rel}:{lineno}: {token}")
      for token in OLD_MODULE_PATHS:
        if token in line:
          offenders.append(f"{rel}:{lineno}: {token}")
  assert offenders == []


def test_alias_is_the_last_statement_of_simulator():
  """The alias is an appended compatibility shim, not a surviving definition."""
  lines = [ln.strip() for ln in SIMULATOR_PATH.read_text(encoding="utf-8").splitlines()]
  code_lines = [ln for ln in lines if ln and not ln.startswith("#")]
  assert code_lines[-1] == ALIAS_LINE


# =============================================================================
# Pickle safety - grids are unaffected, result artefacts are not
# =============================================================================

def test_grid_pickles_reference_only_the_site_class():
  """Grids store ``Site`` objects (3,456 of them), never the simulator class."""
  grid_dir = REPO_ROOT / "data" / "grids"
  grids = sorted(grid_dir.glob("*.pkl"))
  if not grids:
    pytest.skip(f"no cached grids in {grid_dir}")
  for grid in grids:
    raw = grid.read_bytes()
    assert b"kinetix.lattice.site" in raw, grid.name
    assert b"Crystal_Lattice" not in raw, grid.name
    assert b"lattice.crystal" not in raw, grid.name


def test_legacy_result_pickle_module_path_no_longer_resolves():
  """A pre-rename ``variables.pkl`` stores the old module path -> ModuleNotFound.

  Known limitation of the pure rename (documented in AGENTS.md): the grids load
  fine, but pickled *result* artefacts written by ``save_variables`` before the
  rename name ``kinetix.lattice.crystal`` and cannot be unpickled any more.
  """
  legacy = b"ckinetix.lattice.crystal\nCrystal_Lattice\n."
  with pytest.raises(ModuleNotFoundError):
    pickle.loads(legacy)
