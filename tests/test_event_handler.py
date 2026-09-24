# tests/test_event_handler.py
"""
Behavioral spec for kinetix/lattice/events.py (EventHandler).

Phase 3 of the simulator.py split: kMC event execution extracted from
KMCSimulator. KMCSimulator keeps thin delegates for the names that
production code calls from outside the module, so the golden trace (which
wraps the *instance* attribute ``crystal.processes``), superbasin.py,
state_loader.py and the deposition paths are unchanged.

What is REAL here (loaded through production loaders - no hardcoded config
literals):
  - data/parameters/presets/VCM_mock.yaml              (SimulationConfig.from_yaml)
  - data/parameters/activation_energies/VCM_HfO2.json  (load_activation_energies)
  - data/parameters/grain_boundaries/gb_vertical_planar.yaml (GrainBoundariesConfig)
  - data/grids/grid_HfO2_3nm.pkl                       (production grid fast-path)

Test-side overrides (documented, mirroring test_kmc_loop.py):
  1. config.material.formula = 'HfO2'                       (offline MP substitute)
  2. config.defects.defects['oxygen_interstitial'].initial_concentration_bulk
     = 0.05                                                  (production sweep pattern)
  3. calculator_config=None                                  (MACE not wired here)
  4. crystal.timestep_limits / last_field_solve_time set by the fixture
     (normally provided post-init by the ElectricalController loop).

BEHAVIOR NOTES pinned below:
  * the handler is stateless (only ``.simulator``); every read/write of lattice
    state goes through the system reference
  * ``processes`` MUST stay reachable as ``crystal.processes`` - the golden
    trace wraps that instance attribute to observe the event catalog
  * ``_is_active_site`` stays ON KMCSimulator (lattice construction uses
    it); the handler calls ``self.simulator._is_active_site``
  * no runtime import of kinetix.lattice.simulator (TYPE_CHECKING only) - the
    module must load stand-alone
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from kinetix.configs.config_loader import (
  get_parameters_root,
  load_activation_energies,
)
from kinetix.configs.grain_boundary_config import GrainBoundariesConfig
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.initialization import (
  _process_activation_energies,
  initialize_grid_crystal,
)
from kinetix.lattice.simulator import KMCSimulator
from kinetix.lattice.events import EventHandler
from kinetix.lattice.site import Site

# =============================================================================
# Constants
# =============================================================================

SEED = 42
N_STEPS = 25
GRID_NAME = "grid_HfO2_3nm"          # production grid matching VCM_mock
O_I_BULK_CONCENTRATION = 0.05        # production sweep override (real file: 0.0)
EVENTS_MODULE = Path(__file__).resolve().parent.parent / "kinetix" / "lattice" / "events.py"

# Names production code (cli.py, state_loader.py, superbasin.py, deposition,
# tests) uses through KMCSimulator -> keep one-line delegates.
DELEGATED = ("processes", "update_sites_topology", "_introduce_specie_site",
             "_update_rates_lazily", "_get_mobile_sites")

# Names with no external caller -> moved outright, no delegate.
MOVED_ONLY = ("_handle_migration_event", "_should_scavenge",
              "_handle_generation_event", "_handle_redox_event",
              "_handle_reaction_event", "_defect_by_name",
              "_find_empty_neighbor", "_is_at_top_electrode",
              "_get_gb_charge_state", "_install_defect_site",
              "_track_occupancy_update", "_remove_species_at_site")

# =============================================================================
# Fixtures - real configs via production loaders
# =============================================================================

@pytest.fixture(scope="module")
def vcm_config():
  """REAL VCM_mock preset + documented test-side overrides."""
  config = SimulationConfig.from_yaml(
    get_parameters_root() / "presets" / "VCM_mock.yaml"
  )
  config.material.formula = "HfO2"  # override 1
  config.defects.defects["oxygen_interstitial"].initial_concentration_bulk = (
    O_I_BULK_CONCENTRATION  # override 2
  )
  return config


@pytest.fixture(scope="module")
def registry(vcm_config):
  """Real defect registry (dict form, as production hands it to the lattice)."""
  return vcm_config.defects.to_dict()


@pytest.fixture(scope="module")
def act_e(vcm_config, registry):
  """Real activation energies through the production pipeline (CLI wiring)."""
  preset_path = get_parameters_root() / "presets" / "VCM_mock.yaml"
  ae_data = load_activation_energies(preset_path, vcm_config.settings)
  return _process_activation_energies(
    registry, ae_data, vcm_config.settings.technology
  )


@pytest.fixture(scope="module")
def reactions(vcm_config):
  """Real reaction registry (dict form)."""
  return vcm_config.reactions.to_dict()


@pytest.fixture(scope="module")
def system(vcm_config, registry, act_e, reactions):
  """Real lattice built exactly like the production electronic_device branch
  (grid fast-path load), seeded - same construction as test_kmc_loop.py."""
  gb_configurations = (
    [gb.to_dict() for gb in vcm_config.grain_boundaries]
    if vcm_config.grain_boundaries else None
  )
  crystal = initialize_grid_crystal(
    GRID_NAME,
    None,  # mpi_ctx=None -> truly serial
    vcm_config.material,
    vcm_config.experimental,
    act_e,
    vcm_config.settings.lammps_output,
    vcm_config.superbasin,
    False,  # save_data=False -> never writes grids
    settings=vcm_config.settings,
    rng=np.random.default_rng(SEED),
    cache_dir=get_parameters_root().parent / "cache",
    calculator_config=None,  # override 3
    defects_config=registry,
    reactions_config=reactions,
    gb_configurations=gb_configurations,
    simulation_type=vcm_config.settings.simulation_type,
  )
  crystal.timestep_limits = float(vcm_config.superbasin.time_step_limits)
  crystal.last_field_solve_time = 0.0
  crystal.defect_gen()                       # inject O_i (uses crystal.rng)
  crystal._update_rates_lazily({}, {})       # materialize rates
  return crystal


# =============================================================================
# Synthetic hosts (pure handler logic, no lattice build)
# =============================================================================

def _bare_system(registry, act_e, reactions=None, **overrides):
  """Uninitialized KMCSimulator carrying only the state the handler reads."""
  crystal = KMCSimulator.__new__(KMCSimulator)  # skip __init__
  crystal.rank = 0
  crystal.rng = np.random.default_rng(SEED)
  crystal.grid_crystal = {}
  crystal.active_event_sites = []
  crystal.generation_sites = []
  crystal._dirty_sites = set()
  crystal._fields_changed = False
  crystal.defects_config = registry
  # Same derivation as KMCSimulator.__init__ (registry-driven, no literals)
  crystal._active_site_types = {
    stype for cfg in registry.values()
    for stype in cfg.get("allowed_sublattices", [])
  }
  crystal.reactions_config = reactions if reactions is not None else {}
  crystal.METAL_SPECIES = ()
  crystal.affected_site = "Empty"
  crystal.V = 0.0
  crystal.scavenged_ions = {}
  crystal.migration_pathways = 0
  crystal.clusters = {}
  crystal.atom_to_cluster = {}
  crystal.gb_model = None
  crystal.poisson_config = None
  crystal.temperature = 300.0
  crystal.allow_specie_removal = False
  for key, value in overrides.items():
    setattr(crystal, key, value)
  return crystal


def _site(idx, specie, site_type, registry, act_e, neighbors=(), position=None):
  """Real Site with the grid-level wiring the handler relies on.

  ``idx`` is the grid key (tests use verbose labels); the physical position
  defaults to the index origin so distance/region helpers still work.
  """
  if position is None:
    position = tuple(float(i) for i in range(len(idx)))
  site = Site(specie, position, site_type=site_type,
              Act_E_dict=act_e, defects_config=registry)
  site.idx = idx
  site.nearest_neighbors_idx = list(neighbors)
  site.supp_by = ()  # set by neighbors_analysis during a real lattice build
  return site


def _install(crystal, *sites):
  """Register sites in the synthetic grid."""
  for site in sites:
    crystal.grid_crystal[site.idx] = site
  return crystal


# =============================================================================
# Extraction contract (structure, not physics)
# =============================================================================

def test_event_handler_instantiated_lazily_via_property():
  """KMCSimulator.event_handler builds and caches one handler per system."""
  crystal = _bare_system({}, {})
  handler = crystal.event_handler
  assert isinstance(handler, EventHandler)
  assert handler.simulator is crystal
  assert crystal.event_handler is handler       # cached on the instance
  assert crystal._event_handler is handler
  assert not hasattr(handler, "grid_crystal")   # stateless: system is the state


def test_delegates_are_thin_and_forward_to_handler():
  for name in DELEGATED:
    src = inspect.getsource(getattr(KMCSimulator, name))
    assert "self.event_handler." in src, name
    assert src.count("return") == 1, name
    assert src.count("\n") <= 3, name


def test_extracted_helpers_moved_off_crystal():
  for name in MOVED_ONLY:
    assert not hasattr(KMCSimulator, name), name
    assert hasattr(EventHandler, name), name


def test_is_active_site_stays_on_crystal():
  """Lattice construction calls it too (simulator.py), so it must not move."""
  assert hasattr(KMCSimulator, "_is_active_site")
  assert not hasattr(EventHandler, "_is_active_site")
  src = inspect.getsource(EventHandler._get_mobile_sites)
  assert "self.simulator._is_active_site(" in src


def test_delegate_reaches_handler_instance_method():
  """``crystal.processes(x)`` == ``crystal.event_handler.processes(x)``."""
  crystal = _bare_system({}, {})
  calls = []
  crystal.event_handler.processes = calls.append
  sentinel = (0.5, ("a",), 1, 0.3, ("b",))
  crystal.processes(sentinel)
  assert calls == [sentinel]


def test_events_module_has_no_runtime_crystal_import():
  """The module must load stand-alone (crystal imports it, not the reverse)."""
  tree = ast.parse(EVENTS_MODULE.read_text())
  runtime_imports = set()
  for node in tree.body:  # module level only; TYPE_CHECKING block is an If
    if isinstance(node, ast.Import):
      runtime_imports.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      runtime_imports.add((node.module or "").split(".")[0])
  assert runtime_imports == {"__future__", "numpy", "typing"}, runtime_imports
  # ... and the annotation-only import is present, guarded by TYPE_CHECKING
  guard = next(n for n in tree.body if isinstance(n, ast.If))
  assert "TYPE_CHECKING" in ast.unparse(guard.test)
  assert "KMCSimulator" in ast.unparse(guard)


# =============================================================================
# Lookups & predicates (real registries)
# =============================================================================

def test_defect_by_name_resolves_real_registry(registry, act_e):
  crystal = _bare_system(registry, act_e)
  handler = crystal.event_handler
  for name, cfg in registry.items():
    assert handler._defect_by_name(cfg["symbol"]) is cfg, name
  assert handler._defect_by_name("not_a_species") is None


def test_should_scavenge_follows_charge_times_bias(registry, act_e):
  """Scavenging needs (a) an electrode_scavenging config and (b) q*V < 0."""
  crystal = _bare_system(registry, act_e)
  handler = crystal.event_handler
  ion = _site(("i",), "O_i", "interstitial", registry, act_e)

  ion_cfg = registry[ion._get_current_defect_name()]
  charge = ion_cfg["charge"]
  expected_mass_conservation = ion_cfg["electrode_scavenging"]["mass_conservation"]
  ion.defect.charge = charge  # as introduce_specie() sets it on real lattices

  # bias pushing the ion toward the top electrode (q*V < 0)
  crystal.V = 0.5 if charge < 0 else -0.5
  assert handler._should_scavenge(ion) == (True, expected_mass_conservation)

  # reversed bias -> not driven toward the electrode
  crystal.V = -crystal.V
  assert handler._should_scavenge(ion) == (False, expected_mass_conservation)

  # a defect without the electrode_scavenging config is never scavenged
  host = _site(("h",), "O", "O", registry, act_e)
  assert handler._should_scavenge(host) == (False, False)

  # a defect without electrode_scavenging is never scavenged
  host = _site(("h",), "O", "O", registry, act_e)
  assert handler._should_scavenge(host) == (False, False)


class _StubGB:
  """GB model exposing only what the charge-state lookup reads; the real
  region classification (GrainBoundary) is covered by its own tests."""

  def __init__(self, gb_configurations, regions):
    self.gb_configurations = gb_configurations
    self._regions = regions

  def get_site_gb_region(self, position):
    return self._regions[tuple(position)]


def test_get_gb_charge_state_uses_real_gb_file(registry, act_e):
  """Values are read from the REAL gb_vertical_planar.yaml (no literals)."""
  gb_cfgs = GrainBoundariesConfig.from_yaml(
    get_parameters_root() / "grain_boundaries" / "gb_vertical_planar.yaml"
  ).to_dict()
  entries = gb_cfgs[0]["event_modifications"]["migration"]
  entry = next(e for e in entries if "charge_state" in e)
  defect_name = entry["affected_defects"][0]
  charge_state = entry["charge_state"]

  # one synthetic position per region the real config declares
  positions = {region: (float(i), 0.0, 0.0)
               for i, region in enumerate(charge_state)}
  regions = {position: region for region, position in positions.items()}
  crystal = _bare_system(registry, act_e, gb_model=_StubGB(gb_cfgs, regions))
  handler = crystal.event_handler

  # no GB model -> no modification
  assert _bare_system(registry, act_e).event_handler._get_gb_charge_state(
    defect_name, (0.0, 0.0, 0.0)) is None

  # affected defect -> the configured charge for the classified region
  for region, position in positions.items():
    assert handler._get_gb_charge_state(defect_name, position) == charge_state[region]

  # unaffected defect / missing charge_state / no entry for the event type
  assert handler._get_gb_charge_state("lattice_oxygen", positions["bulk"]) is None
  v_o = next(e for e in entries
             if "charge_state" not in e
             and defect_name not in e["affected_defects"])
  assert handler._get_gb_charge_state(v_o["affected_defects"][0],
                                      positions["bulk"]) is None
  assert handler._get_gb_charge_state(defect_name, positions["bulk"],
                                      event_type="no_such_event") is None


def test_find_empty_neighbor_uses_product_sublattice(registry, act_e):
  crystal = _bare_system(registry, act_e)
  handler = crystal.event_handler
  origin = _site(("o",), "V_O", "O", registry, act_e,
                 neighbors=[("occ",), ("vac",)])
  occupied = _site(("occ",), "O_i", "interstitial", registry, act_e)
  vacant = _site(("vac",), "Empty", "interstitial", registry, act_e)
  _install(crystal, origin, occupied, vacant)

  product = {"symbol": "O_i", "sublattice": "interstitial",
             "site_index": "neighbor"}
  assert handler._find_empty_neighbor(origin, product) == ("vac",)

  # no valid target -> reaction blocked (None), never an exception
  # (the only candidate neighbor is the wrong sublattice)
  wrong_sublattice = _site(("occ",), "O_i", registry["lattice_oxygen"]["site_type"],
                           registry, act_e)
  _install(crystal, wrong_sublattice)
  crystal.grid_crystal[("vac",)].site_type = registry["lattice_oxygen"]["site_type"]
  assert handler._find_empty_neighbor(origin, product) is None


def test_get_mobile_sites_filters_by_active_sublattice(registry, act_e):
  crystal = _bare_system(registry, act_e)
  handler = crystal.event_handler
  mobile = _site(("m",), "O_i", "interstitial", registry, act_e)
  host = _site(("h",), "O", "O", registry, act_e)
  _install(crystal, mobile, host)
  indices = [mobile.idx, host.idx]
  expected = [idx for idx in indices
              if crystal._is_active_site(crystal.grid_crystal[idx].site_type)]
  assert handler._get_mobile_sites(indices) == expected
  assert mobile.idx in expected  # the interstitial defect site is mobile


# =============================================================================
# Integration on the real lattice (VCM_mock, seed 42)
# =============================================================================

def test_kmc_step_dispatches_through_handler(system):
  """The kMC loop executes events THROUGH the handler, and the golden-trace
  monkeypatch point (crystal.processes) still intercepts them."""
  handler = system.event_handler
  handler_calls, delegate_calls = [], []
  real_processes = handler.processes
  real_delegate = system.processes
  handler.processes = lambda chosen: (handler_calls.append(chosen),
                                      real_processes(chosen))
  system.processes = lambda chosen: (delegate_calls.append(chosen),
                                     real_delegate(chosen))
  try:
    time_step, chosen = system._kmc_step(system.rng, {}, {})
  finally:
    handler.processes = real_processes
    system.processes = real_delegate

  assert chosen is not None, "lattice produced no event - degenerate setup"
  assert delegate_calls == [chosen]
  assert handler_calls == [chosen]
  assert time_step > 0


def test_step_kmc_advances_bkl_time(system):
  """dt == -log(u)/total_rate, computed from the live event catalog."""
  system._update_rates_lazily({}, {})  # consume dirty sites from earlier tests
  rng = system.rng
  state = rng.bit_generator.state
  u = rng.random()
  rng.bit_generator.state = state

  total_rate = sum(
    event.rate
    for idx in system.active_event_sites + system.generation_sites
    if idx not in system.superbasin_dict
    for event in system.grid_crystal[idx].defect.events
  )
  assert total_rate > 0
  time_step, chosen = system._kmc_step(rng, {}, {})
  assert chosen is not None
  assert time_step == pytest.approx(-np.log(u) / total_rate, rel=1e-9)


def _first_migration_event(system):
  """A real, currently-offered migration event (int label) to an empty dest."""
  for idx in list(system.active_event_sites):
    site = system.grid_crystal[idx]
    if site.defect.chemical_specie == "O_i":
      for event in site.defect.events:
        if (isinstance(event.label, int)
            and event.destination in system.grid_crystal
            and system.grid_crystal[event.destination].defect.chemical_specie
            in site.defects_config[site.defect.name]["valid_target_species"]):
          return (event.rate, event.destination, event.label, event.barrier, idx)
  return None


def test_processes_applies_migration_object_transfer(system):
  """Executing a real migration event moves the Defect object and clears the
  source (Phase-5 hop semantics), driven through the delegate."""
  chosen = _first_migration_event(system)
  assert chosen is not None, "no offered migration event - degenerate setup"
  _, dest_idx, _, _, source_idx = chosen
  source_site = system.grid_crystal[source_idx]
  defect = source_site.defect
  specie = defect.chemical_specie

  system.processes(chosen)

  assert system.grid_crystal[dest_idx].defect is defect          # by reference
  assert system.grid_crystal[dest_idx].defect.chemical_specie == specie
  assert source_site.defect.chemical_specie == system.affected_site
  assert source_site.defect is not defect                       # fresh empty
  assert dest_idx in system.active_event_sites
  assert source_idx not in system.active_event_sites
  assert system.grid_crystal[dest_idx].idx in system._dirty_sites


def test_update_rates_lazily_rebuilds_and_clears_dirty_set(system):
  marker = tuple(list(system.active_event_sites) + list(system.generation_sites))
  system._dirty_sites = set(marker)
  system._update_rates_lazily({}, {})
  assert system._dirty_sites == set()          # consumed
  rates = [event.rate for idx in marker
           for event in system.grid_crystal[idx].defect.events]
  assert rates and all(np.isfinite(r) and r >= 0 for r in rates)
  assert any(r > 0 for r in rates)

  # non-root ranks never rate-update (they only solve fields)
  system.rank = 1
  system._dirty_sites = set(marker)
  system._update_rates_lazily({}, {})
  assert system._dirty_sites == set(marker)
  system.rank = 0
  system._update_rates_lazily({}, {})


def test_introduce_specie_and_topology_refresh(system):
  """``_introduce_specie_site`` + ``update_sites_topology`` re-derive pathways
  for the dirty sites (the path state_loader.py and deposition rely on)."""
  empty_idx = next(idx for idx, site in system.grid_crystal.items()
                   if idx != system.affected_site
                   and site.defect.chemical_specie == system.affected_site
                   and system._is_active_site(site.site_type))

  support, event_sites = set(), set()
  system._introduce_specie_site(empty_idx, support, event_sites, "O_i")
  assert system.grid_crystal[empty_idx].defect.chemical_specie == "O_i"
  assert empty_idx in system.active_event_sites
  assert event_sites and support

  system.update_sites_topology(support, event_sites)
  refreshed = system.grid_crystal[empty_idx]
  assert refreshed.defect.events, "no pathways re-derived for the new occupant"

  # the loop marks the handled sites dirty (events.py, ``processes``) and the
  # next rate refresh turns the new pathways into positive rates
  system._dirty_sites |= event_sites
  system._update_rates_lazily({}, {})
  assert system._dirty_sites == set()
  assert any(event.rate > 0 for event in refreshed.defect.events)


def test_is_at_top_electrode_reads_interface_flag(registry, act_e):
  """Pins the CURRENT (pre-existing, odd) behavior: the method returns the
  site's ``is_at_bottom_interface`` flag. Byte-identical to simulator.py
  :2814-2816 before the Phase-3 move; it has no callers anywhere."""
  crystal = _bare_system(registry, act_e)
  handler = crystal.event_handler
  site = _site(("t",), "Empty", "interstitial", registry, act_e)
  _install(crystal, site)
  assert handler._is_at_top_electrode(site.idx) is False
  site.is_at_bottom_interface = True
  assert handler._is_at_top_electrode(site.idx) is True
