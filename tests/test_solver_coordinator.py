# tests/test_solver_coordinator.py
"""
Behavioral spec for kinetix/solvers/coordinator.py (SolverCoordinator).

Phase 2 of the crystal.py split: solver orchestration extracted from
Crystal_Lattice; Crystal_Lattice keeps one-line delegates (original names),
so cli.py and the kMC loop are unchanged.

No dolfinx/mesh required: the coordinator only *orchestrates* — solver
instances are plain doubles injected as ``system._poisson_solver`` /
``system._heat_solver`` (exactly how cli.py attaches the real ones).

BEHAVIOR NOTES pinned below:
  * all mutable scheduling state lives on the SYSTEM (last_field_solve_time,
    time, V) — test_kmc_loop.py and the save_variables pickle path depend on
    it; the coordinator itself is stateless
  * no solver attached -> _evaluate_fields_for_kmc returns ({}, {}) on rank 0
    (the golden-trace contract, test_golden_trace.py:261)
  * ``from kinetix.solvers import SolverCoordinator`` must work without the
    FEM stack (subprocess test with dolfinx/petsc4py/ufl blocked)
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import constants

from kinetix.lattice.crystal import Crystal_Lattice
from kinetix.solvers import SolverCoordinator
from kinetix.solvers.coordinator import SolverCoordinator as SolverCoordinatorDirect

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bare_system(**overrides) -> Crystal_Lattice:
    """Uninitialized Crystal_Lattice carrying only the orchestration state."""
    system = Crystal_Lattice.__new__(Crystal_Lattice)  # skip __init__/physics
    system.rank = 0
    system.mpi_ctx = None
    system.time = 0.0
    system.grid_crystal = []
    system.active_event_sites = []
    system.generation_sites = []
    system._fields_changed = False
    system._dirty_sites = set()
    system.clusters = {}
    system.screening_factor = 1.0
    for key, value in overrides.items():
        setattr(system, key, value)
    return system


class _FakePoisson:
    """Duck-typed stand-in for PoissonSolver (FEM stack not required)."""

    def __init__(self):
        self.evaluation_points = None

    def evaluate_electric_field_at_points(self, points):
        self.evaluation_points = points
        return {(0.0, 0.0, 0.0): 1.0}  # sentinel payload


# =============================================================================
# Instantiation & extraction contract
# =============================================================================

def test_solver_coordinator_instantiated_lazily_via_delegate():
    system = _bare_system()
    coordinator = system.solver_coordinator
    assert isinstance(coordinator, SolverCoordinator)
    assert coordinator.system is system
    assert system.solver_coordinator is coordinator  # cached on the instance
    assert system._solver_coordinator is coordinator


def test_package_and_module_imports_agree():
    assert SolverCoordinator is SolverCoordinatorDirect


def test_crystal_methods_are_one_line_delegates():
    """Every extracted method survives on Crystal_Lattice as a delegate only."""
    for name in ('save_electric_bias', 'get_evaluation_points',
                 'prepare_clusters_for_bcs', '_evaluate_fields_for_kmc',
                 'get_timestep_limit', 'should_solve_fields_now'):
        src = inspect.getsource(getattr(Crystal_Lattice, name))
        assert 'self.solver_coordinator.' in src, name
        assert src.count('return') == 1, name
        assert src.count('\n') <= 3, name


def test_private_extractors_moved_off_crystal():
    assert not hasattr(Crystal_Lattice, '_extract_particles_charges')
    assert not hasattr(Crystal_Lattice, '_extract_generation_site_location')
    assert hasattr(SolverCoordinator, '_extract_particles_charges')
    assert hasattr(SolverCoordinator, '_extract_generation_site_location')


def test_save_electric_bias_delegate_sets_state_on_system():
    system = _bare_system()
    system.save_electric_bias(0.7)
    assert system.V == 0.7  # physics reads crystal.V (scavenging, top electrode)
    assert not hasattr(system.solver_coordinator, 'V')  # coordinator stateless


# =============================================================================
# Scheduling — state stays on the system
# =============================================================================

def test_should_solve_fields_now_schedule_and_state():
    elec = SimpleNamespace(voltage_update_time=1.0e-4)
    system = _bare_system(time=0.0)

    should, snap = system.should_solve_fields_now(elec)
    assert should and snap
    assert system.last_field_solve_time == 0.0  # written on the SYSTEM

    system.time = 0.9e-4
    assert system.should_solve_fields_now(elec) == (False, False)

    system.time = 1.0e-4
    should, snap = system.should_solve_fields_now(elec)
    assert should and snap
    assert system.last_field_solve_time == pytest.approx(1.0e-4)
    assert system.time == pytest.approx(1.0e-4)  # snapped to the schedule
    # coordinator never caches scheduling state itself
    assert not hasattr(system.solver_coordinator, 'last_field_solve_time')


def test_get_timestep_limit_caps_to_field_deadline():
    system = _bare_system(time=0.4e-4, timestep_limits=1.0e-4,
                          last_field_solve_time=0.0)
    assert system.get_timestep_limit() == pytest.approx(0.6e-4)

    # Deadline within tolerance -> time snaps forward, nothing left
    system.time = 1.0000001e-4
    assert system.get_timestep_limit() == 0.0
    assert system.time == pytest.approx(1.0e-4)


# =============================================================================
# Field evaluation for kMC (golden-trace contract + mocked FEM solver)
# =============================================================================

def test_evaluate_fields_without_solver_returns_empty_dicts():
    """No _poisson_solver attached -> ({}, {}) on rank 0 (golden trace)."""
    system = _bare_system()
    E_field, T_field = system._evaluate_fields_for_kmc()
    assert E_field == {}
    assert T_field == {}


def test_evaluate_fields_uses_faked_poisson_solver():
    site = SimpleNamespace(position=np.array([1.0, 2.0, 3.0]))
    fake = _FakePoisson()
    system = _bare_system(
        _poisson_solver=fake,
        _fields_changed=True,
        active_event_sites=[0],
        generation_sites=[],
        grid_crystal=[site],
    )
    E_field, T_field = system._evaluate_fields_for_kmc()
    assert E_field == {(0.0, 0.0, 0.0): 1.0}
    assert T_field == {}  # no _heat_solver attached
    np.testing.assert_array_equal(fake.evaluation_points, [[1.0, 2.0, 3.0]])


def test_evaluate_fields_dirty_sites_branch():
    """_fields_changed=False -> only _dirty_sites are re-evaluated."""
    sites = [
        SimpleNamespace(position=np.array([0.0, 0.0, 0.0])),
        SimpleNamespace(position=np.array([5.0, 5.0, 5.0])),
    ]
    fake = _FakePoisson()
    system = _bare_system(
        _poisson_solver=fake,
        _fields_changed=False,
        active_event_sites=[0],
        generation_sites=[],
        grid_crystal=sites,
        _dirty_sites={1},
    )
    system._evaluate_fields_for_kmc()
    np.testing.assert_array_equal(fake.evaluation_points, [[5.0, 5.0, 5.0]])


# =============================================================================
# Solver inputs: evaluation points & BC clusters
# =============================================================================

def test_get_evaluation_points_combines_charges_and_gen_sites():
    system = _bare_system(
        screening_factor=2.0,
        active_event_sites=[0],
        generation_sites=[1],
        grid_crystal=[
            SimpleNamespace(position=np.array([0.0, 0.0, 0.0]),
                            defect=SimpleNamespace(charge=-1.0)),
            SimpleNamespace(position=np.array([1.0, 1.0, 1.0])),
        ],
    )
    locations, charges, evaluation_points = system.get_evaluation_points()
    # charge * elementary charge * screening factor (production formula)
    np.testing.assert_allclose(charges, [-1.0 * constants.e * 2.0])
    np.testing.assert_array_equal(locations, [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(evaluation_points,
                                  [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_get_evaluation_points_empty_system():
    system = _bare_system()
    locations, charges, evaluation_points = system.get_evaluation_points()
    assert locations.shape == (0, 3)
    assert charges.shape == (0,)
    assert evaluation_points.shape == (0, 3)


def test_prepare_clusters_for_bcs_delegates_to_clusters():
    calls = []

    class _FakeCluster:
        def prepare_cluster_for_bcs(self, grid, size):
            calls.append((grid, size))

    cluster = _FakeCluster()
    system = _bare_system(clusters={0: cluster}, grid_crystal=['grid'],
                          crystal_size=(1, 2, 3))
    out = system.prepare_clusters_for_bcs()
    assert out is system.clusters  # serial rank 0 returns its own dict
    assert calls == [(['grid'], (1, 2, 3))]  # grid_crystal forwarded whole


# =============================================================================
# FEM stack is optional for the coordinator import
# =============================================================================

def test_coordinator_importable_without_dolfinx():
    """``from kinetix.solvers import SolverCoordinator`` must not need dolfinx."""
    script = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('dolfinx', 'petsc4py', 'ufl'):\n"
        "            raise ModuleNotFoundError('blocked for test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from kinetix.solvers import SolverCoordinator\n"
        "print('OK', SolverCoordinator.__name__)\n"
    )
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert 'OK SolverCoordinator' in result.stdout