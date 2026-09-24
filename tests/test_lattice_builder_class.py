# tests/test_lattice_builder_class.py
"""
Behavioral spec for kinetix/lattice/lattice_builder.py (LatticeBuilder).

Phase 5 of the crystal.py split: the 36 lattice-construction/initialization
methods moved out of Crystal_Lattice; Crystal_Lattice keeps thin one-line
delegates plus a lazy ``lattice_builder`` property, so initialization.py,
cli.py, metadata.py and the golden trace are unchanged.

BEHAVIOR NOTES pinned below:
  * delegates are one-liners that forward to ``self.lattice_builder.<name>``,
    all parameters - defaults included - passed positionally
  * delegate signatures (parameter names + defaults) are IDENTICAL to the
    builder's; callers rely on the defaults
  * LatticeBuilder is stateless: ``system`` is the only instance attribute;
    grid_crystal, structure, basis_vectors, coord_cache, the k-d tree and
    rank/mpi_ctx are read/written through ``self.system``
  * construction collaborators STAY on Crystal_Lattice and are reached as
    ``self.system._is_active_site`` / ``self.system._minimum_image_vector``
  * no module outside crystal.py references ``lattice_builder`` (delegate
    indirection intact)
  * INTEGRATION: the golden-trace lattice (built through the delegates with the
    production loaders) has the fixture's 3456 sites, populated neighbours and
    interface flags - and the Phase-6 live ``defects_config`` binding still
    holds for every site although ``crystal_grid`` now lives on the builder

Pre-existing quirks pinned so a future fix is deliberate (NOT introduced by
Phase 5 - both reproduce at HEAD):
  * ``_validate_migration_network()``'s default ``radius=None`` crashes in
    ``_generate_periodic_images`` (``None / float``); the in-package caller
    passes a real radius (``_initialize_migration_pathways``)
  * ``_process_batch_sites_worker`` forwards SIX positional arguments to
    ``Site.neighbors_analysis`` (which declares five), and
    ``_parallel_neighbors_analysis`` has no in-package callers
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kinetix.lattice.crystal import Crystal_Lattice
from kinetix.lattice.lattice_builder import LatticeBuilder
from tests.test_golden_trace import (
    FIXTURE_PATH,
    GRID_PATH,
    _build_lattice as _golden_build_lattice,
    _load_act_e,
    _load_vcm_config,
)

# The 36 methods extracted in Phase 5 (every one has a thin delegate).
DELEGATES = (
    # structure & Materials-Project model (8)
    "_load_mp_cache",
    "_save_mp_cache",
    "lattice_model",
    "_is_inside_supercell",
    "_apply_miller_orientation",
    "_get_rotation_matrix",
    "_create_supercell",
    "_compute_basis_vectors",
    # migration pathways & network validation (2)
    "_initialize_migration_pathways",
    "_validate_migration_network",
    # neighbour search & radius analysis (7)
    "_build_kdtree",
    "_get_neighbors_for_site",
    "_generate_periodic_images",
    "_check_percolation_at_radius",
    "find_optimal_radius",
    "diagnose_steep_down",
    "diagnose_interstitial_presence",
    # grid assembly (1)
    "crystal_grid",
    # site initialization helpers + interstitial generation (10)
    "_efficient_act_e_copy",
    "_get_applicable_defects_for_site",
    "_compute_interface_flags",
    "_generate_interstitial_sites",
    "_find_interstitials_voronoi",
    "_refine_interstitial_positions",
    "_cluster_and_average",
    "_validate_interstitial_positions",
    "create_ovito_xyz_file",
    "_handle_missing_neighbors",
    # neighbour analysis (4)
    "_parallel_neighbors_analysis",
    "_sequencial_neighbors_analysis",
    "get_num_cores",
    "_process_batch_sites_worker",
    # coordinates, Wulff shape & edges (3)
    "get_idx_coords",
    "Wulff_Shape",
    "create_edges",
    # cluster tracking init (1)
    "_initialize_cluster_tracking",
)


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
  """Pristine golden-trace lattice, built entirely through the delegates."""
  config = _load_vcm_config()
  defects = config.defects.to_dict()
  return _golden_build_lattice(config, defects, _load_act_e(config, defects))


@pytest.fixture(scope="module")
def golden_meta():
  """The shipped fixture's meta block (site count, hashes, versions)."""
  with open(FIXTURE_PATH, encoding="utf-8") as fh:
    return json.load(fh)["meta"]


def _param_shape(func):
  """(name, default) pairs - annotations deliberately ignored."""
  return [(p.name, p.default) for p in inspect.signature(func).parameters.values()]


def _lattice_digest(system):
  """Cheap content digest: rounded positions + neighbour counts per site."""
  return {idx: (tuple(np.round(site.position, 6)), len(site.nearest_neighbors_idx))
          for idx, site in sorted(system.grid_crystal.items())}


# =============================================================================
# Instantiation & delegation contract
# =============================================================================

def test_lattice_builder_instantiated_lazily_via_property(system):
  """``Crystal_Lattice.lattice_builder`` builds and caches one builder."""
  builder = system.lattice_builder
  assert isinstance(builder, LatticeBuilder)
  assert builder.system is system
  assert system.lattice_builder is builder       # cached on the instance
  assert system._lattice_builder is builder


def test_lazy_property_uses_local_import():
  """The property must import LatticeBuilder lazily (no top-level cycle,
  Phase 5 pattern shared with ``solver_coordinator``/``kmc_loop``)."""
  src = inspect.getsource(Crystal_Lattice.lattice_builder.fget)
  assert "from kinetix.lattice.lattice_builder import LatticeBuilder" in src
  assert "hasattr(self, '_lattice_builder')" in src


def test_delegates_are_thin_and_forward_to_builder():
  """Every extracted method survives on Crystal_Lattice as a delegate only,
  and the real body exists on LatticeBuilder."""
  assert len(DELEGATES) == 36 and len(set(DELEGATES)) == 36
  for name in DELEGATES:
    src = inspect.getsource(getattr(Crystal_Lattice, name))
    assert "self.lattice_builder." in src, name
    assert src.count("return") == 1, name
    assert src.count("\n") <= 3, name
    assert "def %s(" % name in src, name
    assert callable(getattr(LatticeBuilder, name, None)), name


def test_delegate_signatures_match_builder():
  """Parameter names AND defaults are identical to the builder's - external
  callers keep relying on the defaults (annotations are exempt)."""
  for name in DELEGATES:
    assert _param_shape(getattr(Crystal_Lattice, name)) == \
        _param_shape(getattr(LatticeBuilder, name)), name


def test_delegates_forward_arguments_positionally(system, monkeypatch):
  """Spot-check on a spy: defaults are filled by the delegate and every
  parameter is forwarded positionally, in order."""
  calls = {}

  class _Spy:
    def __getattr__(self, name):
      def _record(*args, **kwargs):
        calls[name] = (args, kwargs)
        return ("sentinel", name)
      return _record

  monkeypatch.setattr(system, "_lattice_builder", _Spy())
  assert system.find_optimal_radius() == ("sentinel", "find_optimal_radius")
  assert calls["find_optimal_radius"] == (("interstitial", 1.5, 6.0, 0.25, 0.5), {})
  assert system._generate_interstitial_sites() == ("sentinel", "_generate_interstitial_sites")
  assert calls["_generate_interstitial_sites"] == ((None,), {})
  assert system._cluster_and_average([1, 2]) == ("sentinel", "_cluster_and_average")
  assert calls["_cluster_and_average"] == (([1, 2], 0.7), {})
  assert system._load_mp_cache("k") == ("sentinel", "_load_mp_cache")
  assert calls["_load_mp_cache"] == (("k",), {})


def test_bodies_live_on_builder_not_crystal():
  """Construction internals exist only in lattice_builder.py."""
  assert "Rodrigues" in inspect.getsource(LatticeBuilder._get_rotation_matrix)
  assert "min_non_zero_element" in inspect.getsource(LatticeBuilder._compute_basis_vectors)
  assert "coord_cache" in inspect.getsource(LatticeBuilder.get_idx_coords)
  assert "neighbors_analysis" in inspect.getsource(LatticeBuilder._process_batch_sites_worker)
  for name in DELEGATES:
    src = inspect.getsource(getattr(Crystal_Lattice, name))
    assert "Rodrigues" not in src, name
    assert "min_non_zero_element" not in src, name
    assert "coord_cache" not in src, name


def test_builder_is_stateless(system):
  """The builder holds ONLY the system reference (state stays on the system)."""
  builder = LatticeBuilder(system)
  assert set(vars(builder)) == {"system"}
  for attr in ("grid_crystal", "structure", "structure_basic", "basis_vectors",
               "Act_E_dict", "defects_config", "coord_cache", "_kdtree",
               "_kdtree_indices", "radius_neighbors", "crystal_size", "cache_dir"):
    assert not hasattr(builder, attr), attr


def test_builder_module_has_no_runtime_crystal_import():
  """The module must load stand-alone (crystal imports it, not the reverse)."""
  import kinetix.lattice.lattice_builder as mod
  tree = ast.parse(Path(mod.__file__).read_text())
  runtime_imports = set()
  for node in tree.body:  # module level only; TYPE_CHECKING block is an If
    if isinstance(node, ast.Import):
      runtime_imports.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
      runtime_imports.add((node.module or "").split(".")[0])
  assert runtime_imports == {"__future__", "copy", "json", "logging", "os", "time",
                             "itertools", "typing", "numpy", "pymatgen", "kinetix"}, \
      runtime_imports
  guard = next(n for n in tree.body if isinstance(n, ast.If))
  assert "TYPE_CHECKING" in ast.unparse(guard.test)
  assert "Crystal_Lattice" in ast.unparse(guard)


def test_collaborators_stay_on_crystal_and_are_reached_via_system():
  """``_is_active_site`` / ``_minimum_image_vector`` are NOT duplicated onto
  the builder - it calls them on the system."""
  for name in ("_is_active_site", "_minimum_image_vector"):
    assert callable(getattr(Crystal_Lattice, name)), name
    assert not hasattr(LatticeBuilder, name), name
  mig = inspect.getsource(LatticeBuilder._initialize_migration_pathways)
  assert "self.system._minimum_image_vector(" in mig
  assert "self.system._is_active_site(" in mig
  assert "self.system._is_active_site(" in inspect.getsource(LatticeBuilder.crystal_grid)


def test_no_mpi_logic_added_and_state_read_through_system(system):
  """MPI usage is the VERBATIM pre-existing one (rank guards + structure
  broadcast); nothing new, and no rank/mpi state on the builder."""
  import kinetix.lattice.lattice_builder as mod
  src = inspect.getsource(mod)
  assert "mpi4py" not in src
  assert "self.rank" not in src and "self.mpi_ctx" not in src   # routed
  assert "self.system.rank" in src            # guards, moved verbatim
  assert "self.system.mpi_ctx.bcast" in src   # structure broadcast
  assert "gather" not in src and "scatter" not in src
  # serial construction is the documented default (initialization.py):
  # mpi_ctx=None => every guard is trivially rank 0, the bcast is skipped
  assert system.mpi_ctx is None and system.rank == 0


def test_only_crystal_touches_lattice_builder():
  """External callers still go through Crystal_Lattice's delegates."""
  import kinetix.lattice.lattice_builder as mod
  package = Path(mod.__file__).resolve().parent.parent
  offenders = []
  for path in sorted(package.rglob("*.py")):
    if path.name in {"crystal.py", "lattice_builder.py"}:
      continue
    if "lattice_builder" in path.read_text(encoding="utf-8"):
      offenders.append(str(path.relative_to(package)))
  assert offenders == []


# =============================================================================
# Behavioural: construction helpers reached through the delegates
# =============================================================================

def test_get_rotation_matrix_rotates_onto_target(system):
  """Rodrigues rotation: proper rotation taking vec1 onto vec2 (stateless, so
  the delegate and a fresh builder agree exactly)."""
  v1 = np.array([1.0, 0.0, 0.0])
  v2 = np.array([0.0, 1.0, 0.0])
  rot = system._get_rotation_matrix(v1, v2)                 # delegate
  direct = LatticeBuilder(system)._get_rotation_matrix(v1, v2)
  assert np.allclose(rot, direct)
  assert np.allclose(rot @ v1, v2)
  assert np.allclose(rot @ rot.T, np.eye(3))                # orthogonal
  assert np.isclose(np.linalg.det(rot), 1.0)                # proper rotation
  # special cases of the formula
  assert np.allclose(system._get_rotation_matrix(v1, v1), np.eye(3))
  assert np.allclose(system._get_rotation_matrix(v1, -v1) @ v1, -v1)


def test_compute_basis_vectors_via_delegate_is_deterministic(system):
  """Re-running the delegate reproduces the same grid basis (single scalar
  rescaling of the lattice matrix, <= 1 for fractional spacing)."""
  before = np.array(system.basis_vectors, copy=True)
  assert system._compute_basis_vectors() is None            # sets on the system
  assert np.allclose(system.basis_vectors, before)
  lattice_matrix = np.array(system.structure_basic.lattice.matrix)
  mask = np.abs(lattice_matrix) > 1e-12
  ratio = np.array(system.basis_vectors)[mask] / lattice_matrix[mask]
  assert np.allclose(ratio, ratio[0])                       # one scale factor
  assert 0 < ratio[0] <= 1


def test_get_idx_coords_round_trip_and_cache_on_system(system):
  """Coordinate -> lattice-index lookup uses (and caches on) the SYSTEM's
  coord_cache, never on the builder."""
  site = next(iter(system.grid_crystal.values()))
  coords = np.array(site.position)
  basis = np.array(system.basis_vectors)

  idx = system.get_idx_coords(coords, basis)                # delegate
  assert isinstance(idx, tuple) and len(idx) == 3
  assert all(isinstance(v, (int, np.integer)) for v in idx)
  expected = tuple(np.round(np.linalg.solve(basis.transpose(), coords)).astype(int))
  assert idx == expected

  assert tuple(coords) in system.coord_cache                # cached on system
  assert system.get_idx_coords(coords, basis) is idx        # cache hit
  assert not hasattr(system.lattice_builder, "coord_cache")


def test_efficient_act_e_copy_isolates_inner_dicts(system):
  """Per-site energy copy: new outer+inner dicts, equal content, writes to the
  copy never leak back into the live Act_E_dict."""
  copy = system._efficient_act_e_copy(system.Act_E_dict)    # delegate
  assert copy == system.Act_E_dict and copy is not system.Act_E_dict
  for name, energies in copy.items():
    assert energies is not system.Act_E_dict[name]
  name = next(iter(copy))
  key = next(iter(copy[name]))
  copy[name][key] = "MUTATED"
  assert system.Act_E_dict[name][key] != "MUTATED"
  assert system._efficient_act_e_copy({}) == {}


def test_get_applicable_defects_for_site_matches_registry(system):
  """Sublattice lookup reads the live dict[str, DefectConfig] on the system."""
  site_types = sorted({st for cfg in system.defects_config.values()
                       for st in cfg.get("allowed_sublattices", [])})
  site_types.append("no_such_sublattice")
  for site_type in site_types:
    names = system._get_applicable_defects_for_site(site_type)   # delegate
    expected = [n for n, cfg in system.defects_config.items()
                if site_type in cfg.get("allowed_sublattices", [])]
    assert names == expected, site_type
    assert isinstance(names, list)


class _StubSite:
  """Minimal stand-in for a lattice site, recording neighbour-analysis calls."""

  def __init__(self, position):
    self.position = list(position)
    self.calls = []

  def neighbors_analysis(self, *args):
    self.calls.append(args)


def test_process_batch_sites_worker_call_contract(system):
  """The worker scans the FULL grid, filters by radius and calls
  ``neighbors_analysis`` in place with (grid, idxs, positions, crystal_size,
  event_labels, key).

  NOTE: the six-argument call is the pre-existing shape at HEAD (Site declares
  five parameters and ``_parallel_neighbors_analysis`` has no in-package
  callers); Phase 5 moved the line verbatim - adjust only with a deliberate fix.
  """
  grid = {
    "a": _StubSite((0.0, 0.0, 0.0)),
    "b": _StubSite((1.0, 0.0, 0.0)),     # inside radius 2.0
    "c": _StubSite((5.0, 0.0, 0.0)),     # outside
    "d": _StubSite((0.0, 0.0, 1.9)),     # inside
  }
  shared = {"crystal_size": (10.0, 10.0, 10.0),
            "event_labels": {"migration_interstitial": 0},
            "radius_neighbors": 2.0}
  assert system._process_batch_sites_worker(["a"], grid, shared) is None
  assert len(grid["a"].calls) == 1
  grid_arg, idxs, positions, size, labels, key = grid["a"].calls[0]
  assert grid_arg is grid                          # full grid, in place
  assert idxs == ["b", "d"]                        # in-radius, grid order
  assert positions == [grid["b"].position, grid["d"].position]
  assert size is shared["crystal_size"]
  assert labels is shared["event_labels"]
  assert key == "a"
  for other in ("b", "c", "d"):
    assert grid[other].calls == []                 # only batch keys processed


def test_validate_migration_network_is_read_only(system):
  """Production signature (a real radius) runs over the built grid and changes
  nothing. The default ``radius=None`` crashes in ``_generate_periodic_images``
  - pre-existing latent bug, pinned so a fix is deliberate."""
  before = _lattice_digest(system)
  assert system._validate_migration_network(system.radius_neighbors) is None
  assert _lattice_digest(system) == before
  with pytest.raises(TypeError):
    system._validate_migration_network()


def test_get_num_cores_bounds(system):
  """Core count honours its ceiling and never drops below one."""
  cores = system.get_num_cores()                    # delegate, default 6
  assert isinstance(cores, int) and 1 <= cores <= 6
  assert system.get_num_cores(local_max_cores=1) == 1
  assert system.lattice_builder.get_num_cores(local_max_cores=1) == 1


# =============================================================================
# Integration: the golden lattice was produced by the builder's code paths
# =============================================================================

def test_golden_lattice_built_through_the_builder(system, golden_meta):
  """The fixture lattice (built through the delegates with the production
  loaders) carries everything the MOVED code sets: site count, k-d tree,
  neighbours, interface flags and the Phase-6 live-registry binding."""
  grid = system.grid_crystal
  assert len(grid) == golden_meta["n_sites"] == 3456
  assert system._kdtree is not None
  assert len(system._kdtree_indices) == 3456
  assert np.asarray(system.basis_vectors).shape == (3, 3)
  assert isinstance(system.coord_cache, dict)
  # per-site invariants written during construction (crystal_grid)
  assert all(site.idx == key for key, site in grid.items())
  assert all(site.defect is not None for site in grid.values())
  assert all(site.nearest_neighbors_idx for site in grid.values())
  assert all(site.defects_config is system.defects_config for site in grid.values())
  assert sum(site.is_at_bottom_interface for site in grid.values()) > 0
  assert sum(site.is_at_top_interface for site in grid.values()) > 0




