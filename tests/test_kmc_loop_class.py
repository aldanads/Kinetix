# tests/test_kmc_loop_class.py
"""
Behavioral spec for kinetix/lattice/kmc_loop.py (KMCLoop).

Phase 4 of the simulator.py split: the BKL step orchestration (step_kmc,
_kmc_step, superbasin search/invalidation/activation policy) moved out of
KMCSimulator behind a lazy ``kmc_loop`` property.

Global delegate cleanup: only ``step_kmc`` remains as a facade method (the
core public API). The eight granular loop delegates were DELETED - callers
reach ``simulator.kmc_loop.<name>`` directly.

BEHAVIOR NOTES pinned below:
  * only ``step_kmc`` is a one-line facade method; the other eight names must
    NOT exist on KMCSimulator
  * KMCLoop is stateless: ``system`` is the only instance attribute; every
    simulation field (time, rank, mpi_ctx, superbasin_dict, ...) is read and
    written through ``self.simulator``
  * ``_kmc_step`` reaches ``processes`` through the retained SYSTEM facade
    delegate (``self.simulator.processes``) - the golden trace wraps the
    *instance* attribute, so bypassing it would void the physics contract
  * the extracted module contains no MPI logic beyond the VERBATIM pre-existing
    ``system.rank`` guard + ``system.mpi_ctx.bcast`` of ``time`` inside
    ``step_kmc`` (serial runs have mpi_ctx=None, so the bcast is skipped)
  * INTEGRATION: 5 steps through ``crystal.step_kmc(rng)`` reproduce the first
    5 golden-fixture steps (time and processes-call counts, rel=1e-9)
  * the loop reaches its collaborators directly:
    ``self.simulator.solver_coordinator._evaluate_fields_for_kmc`` /
    ``get_timestep_limit`` and ``self.simulator.event_handler._update_rates_lazily``

Construction reuses the golden trace's own loaders/builder
(``tests/test_golden_trace.py``) so this run is bit-identical to the trace.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kinetix.lattice.simulator import KMCSimulator
from kinetix.lattice.kmc_loop import KMCLoop
from tests.test_golden_trace import (
    FIXTURE_PATH,
    GRID_PATH,
    RATE_ABS_TOL,
    RATE_REL_TOL,
    _build_lattice as _golden_build_lattice,
    _load_act_e,
    _load_vcm_config,
)

# The nine methods extracted in Phase 4 (only step_kmc stays on the facade).
DELEGATES = (
    "step_kmc",
    "_kmc_step",
    "_search_superbasin",
    "update_superbasin",
    "should_activate_superbasin",
    "is_filament_percolating",
    "_check_event_based_superbasin",
    "_check_time_based_superbasin",
    "_slow_timesteps",
)

INTEGRATION_STEPS = 5  # first N golden-fixture steps re-run through the delegate


# =============================================================================
# Fixtures - golden trace construction (real configs, production loaders)
# =============================================================================

@pytest.fixture(scope="module")
def grid_path():
    """The cached production grid, or skip (building one needs network)."""
    if not GRID_PATH.exists():
        pytest.skip(f"production grid not found: {GRID_PATH}")
    return GRID_PATH


@pytest.fixture(scope="module")
def system(grid_path):
    """Fresh, UNTOUCHED lattice (golden construction) for structure tests."""
    config = _load_vcm_config()
    defects = config.defects.to_dict()
    return _golden_build_lattice(config, defects, _load_act_e(config, defects))


@pytest.fixture(scope="module")
def trajectory(grid_path):
    """Fresh lattice with defects + rates, plus an INSTANCE-level ``processes``
    recorder installed exactly like ``tests/test_golden_trace.py::_run_trace``.

    Returns ``(crystal, recorder)``; ``recorder["calls"]`` accumulates every
    event tuple executed through the instance attribute.
    """
    config = _load_vcm_config()
    defects = config.defects.to_dict()
    crystal = _golden_build_lattice(config, defects, _load_act_e(config, defects))
    crystal.defect_gen()
    crystal.event_handler._update_rates_lazily({}, {})  # materialize rates

    recorder = {"calls": []}
    original_processes = crystal.processes

    def recording_processes(chosen_event):
        recorder["calls"].append(chosen_event)
        original_processes(chosen_event)

    crystal.processes = recording_processes  # INSTANCE attribute (as trace does)
    try:
        yield crystal, recorder
    finally:
        crystal.processes = original_processes

# =============================================================================
# Instantiation & delegation contract
# =============================================================================

def test_kmc_loop_instantiated_lazily_via_property(system):
    """``KMCSimulator.kmc_loop`` builds and caches one loop per system."""
    loop = system.kmc_loop
    assert isinstance(loop, KMCLoop)
    assert loop.simulator is system
    assert system.kmc_loop is loop            # cached on the instance
    assert system._kmc_loop is loop


def test_lazy_property_uses_local_import():
    """The property must import KMCLoop lazily (no top-level cycle, Phase 4
    pattern shared with ``solver_coordinator``)."""
    src = inspect.getsource(KMCSimulator.kmc_loop.fget)
    assert "from kinetix.lattice.kmc_loop import KMCLoop" in src
    assert "hasattr(self, '_kmc_loop')" in src


def test_package_and_module_class_identity():
    from kinetix.lattice.kmc_loop import KMCLoop as Direct
    assert Direct is KMCLoop


def test_only_step_kmc_stays_on_the_facade():
    """Global delegate cleanup: the eight granular loop delegates are gone;
    ``step_kmc`` is the core public API and stays a one-line facade."""
    assert not hasattr(KMCSimulator, "_kmc_step")
    for name in DELEGATES[1:]:
        assert not hasattr(KMCSimulator, name), name
    src = inspect.getsource(KMCSimulator.step_kmc)
    assert "self.kmc_loop." in src
    assert src.count("return") == 1
    assert src.count("\n") <= 3


def test_bodies_live_on_kmc_loop_not_crystal():
    """The algorithm (catalog/tree/BKL/superbasin) exists only in kmc_loop.py."""
    assert "TR_catalog" in inspect.getsource(KMCLoop._kmc_step)
    assert "search_value" in inspect.getsource(KMCLoop._kmc_step)
    assert "Superbasin(" in inspect.getsource(KMCLoop._search_superbasin)
    assert "nothing_happen_count" in inspect.getsource(
        KMCLoop._check_event_based_superbasin)
    # ... and nowhere on the facade
    facade_src = inspect.getsource(KMCSimulator)
    assert "TR_catalog" not in facade_src
    assert "Superbasin(" not in facade_src


def test_loop_is_stateless(system):
    """The loop holds ONLY the simulator reference (all state stays on it)."""
    loop = KMCLoop(system)
    assert set(vars(loop)) == {"simulator"}
    for attr in ("time", "rank", "mpi_ctx", "superbasin_dict", "grid_crystal",
                 "events_tracking", "active_event_sites"):
        assert not hasattr(loop, attr), attr


def test_loop_module_has_no_runtime_crystal_import():
    """The module must load stand-alone (crystal imports it, not the reverse)."""
    import kinetix.lattice.kmc_loop as mod
    tree = ast.parse(Path(mod.__file__).read_text())
    runtime_imports = set()
    for node in tree.body:  # module level only; TYPE_CHECKING block is an If
        if isinstance(node, ast.Import):
            runtime_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            runtime_imports.add((node.module or "").split(".")[0])
    assert runtime_imports == {"__future__", "logging", "time", "typing",
                               "numpy", "kinetix"}, runtime_imports
    guard = next(n for n in tree.body if isinstance(n, ast.If))
    assert "TYPE_CHECKING" in ast.unparse(guard.test)
    assert "KMCSimulator" in ast.unparse(guard)


def test_no_mpi_logic_in_extracted_code(system):
    """MPI concern of ``step_kmc`` is the VERBATIM pre-existing guard+bcast of
    ``system.time``; no MPI import, no new collective, state on the system."""
    import kinetix.lattice.kmc_loop as mod
    src = inspect.getsource(mod)
    assert "mpi4py" not in src
    assert "self.rank" not in src and "self.mpi_ctx" not in src  # routed
    assert "self.simulator.rank" in src          # guard moved verbatim
    assert "self.simulator.mpi_ctx.bcast" in src # bcast moved verbatim
    assert "gather" not in src and "scatter" not in src
    # serial construction is the documented default (initialization.py):
    # mpi_ctx=None => the bcast branch is skipped and rank is 0
    assert system.mpi_ctx is None and system.rank == 0


# =============================================================================
# Superbasin policy delegation (no filament / empty tracker -> deterministic)
# =============================================================================

def test_superbasin_policy_runs_in_the_loop(system):
    loop = system.kmc_loop
    # fresh lattice: no clusters attached across the cell, empty tracker
    assert loop.is_filament_percolating() is False
    assert loop._slow_timesteps() is False
    # re-running the policy is deterministic (pure reads of simulator state)
    assert loop.is_filament_percolating() is False
    assert loop._slow_timesteps() is False
    # not percolating => activation refused before touching the tracker
    assert loop.should_activate_superbasin(1e-30) is False


def test_search_superbasin_is_a_noop_without_percolating_filament(system):
    before = dict(system.superbasin_dict)
    loop = system.kmc_loop
    loop._search_superbasin(1e-30)          # direct component call; must not raise
    assert system.superbasin_dict == before



@pytest.fixture(scope="module")
def golden_fixture():
    """The shipped 30-step golden fixture (the physics contract)."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# Integration: the delegate reproduces the golden trace
# =============================================================================

def test_step_kmc_matches_golden_fixture_through_delegate(trajectory,
                                                          golden_fixture):
    """5 steps through ``crystal.step_kmc`` reproduce the first 5 fixture steps.

    This exercises the WHOLE Phase-4 chain: ``KMCSimulator.step_kmc``
    (delegate) -> ``KMCLoop.step_kmc`` -> ``_kmc_step`` -> balanced-tree BKL ->
    ``self.simulator.processes`` (EventHandler delegate) -> time bookkeeping.
    """
    crystal, recorder = trajectory
    expected = golden_fixture["steps"][:INTEGRATION_STEPS]
    assert len(recorder["calls"]) == 0, (
        "trajectory must still be pristine - this test steps it first; "
        "reorder so it runs before any other stepping test"
    )
    calls_before = 0
    for block in expected:
        crystal.step_kmc(crystal.rng)   # <- the delegate -> KMCLoop.step_kmc
        assert crystal.time == pytest.approx(
            block["time"], rel=RATE_REL_TOL, abs=RATE_ABS_TOL), block["step"]
        assert len(recorder["calls"]) - calls_before == block["n_processes_calls"], (
            block["step"], len(recorder["calls"]) - calls_before,
            block["n_processes_calls"])
        calls_before = len(recorder["calls"])
    assert calls_before >= 1            # at least one event executed


def test_kmc_loop_reaches_processes_via_instance_delegate(trajectory,
                                                          golden_fixture):
    """``_kmc_step`` must call ``self.simulator.processes(...)`` - the SYSTEM
    delegate - so an INSTANCE-level wrapper (what the golden trace wraps) sees
    every executed event. Routing directly to ``event_handler.processes``
    would silently bypass the wrapper."""
    crystal, recorder = trajectory
    if not recorder["calls"]:
        # Not stepped yet (e.g. -k selection): take exactly the fixture's
        # first step so this test is self-sufficient.
        crystal.step_kmc(crystal.rng)
        assert len(recorder["calls"]) == golden_fixture["steps"][0]["n_processes_calls"]
    assert len(recorder["calls"]) >= 1
    # chosen tuples are (rate, dest, label, origin) - the shape the golden
    # trace and test_kmc_loop also consume (barrier lives in the catalog only)
    assert all(isinstance(call, tuple) and len(call) >= 4
               for call in recorder["calls"])
    assert crystal.time > 0             # the wrapper really called through


def test_bkl_time_advance_formula(trajectory):
    """dt == -log(u)/total_rate for an executed event, else dt == the
    timestep limit - computed through the delegate on the live catalog."""
    crystal, _ = trajectory
    crystal.event_handler._update_rates_lazily({}, {})   # current rates; clears the dirty set
    total_rate = sum(
        event.rate
        for idx in crystal.active_event_sites + crystal.generation_sites
        if idx not in crystal.superbasin_dict
        for event in crystal.grid_crystal[idx].defect.events
    )
    assert total_rate > 0

    rng = crystal.rng                       # peek the draw _kmc_step will use
    state = rng.bit_generator.state
    u = rng.random()
    rng.bit_generator.state = state

    dt, chosen = crystal.kmc_loop._kmc_step(rng, {}, {})   # delegate
    if chosen is not None:
        assert dt == pytest.approx(-np.log(u) / total_rate, rel=1e-9)
    else:
        # No event within timestep_limits: the loop advanced by the limit.
        assert dt == pytest.approx(crystal.solver_coordinator.get_timestep_limit(), rel=1e-12)

