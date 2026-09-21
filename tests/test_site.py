"""Behavioral spec for :class:`kinetix.lattice.site.Site` (crystal.py split prep).

These tests are the regression net for the upcoming ``crystal.py`` split.  They
cover the three responsibilities of ``Site`` that the split touches:

* **Initialization** - species / position / sublattice plus the defect
  resolution helpers (``_get_applicable_defects``, ``_get_current_defect_name``).
* **Event generation** - ``available_pathways`` and its sub-handlers
  (``available_migrations``, ``available_reduction``, ``available_oxidation``,
  ``available_reactions``).
* **Rate calculation** - the Arrhenius conversion in ``transition_rates``.

No kMC loop is executed here.  Neighbour / destination sites are stubs (using
``MagicMock`` where a stub would need behaviour).

Config wiring (no hardcoded parameter literals)
-----------------------------------------------
Every parameter value asserted here is read from the *real* shipped files
through the *production* loaders, so editing a parameter file (or breaking a
loader) makes these tests fail:

* ``data/parameters/presets/PZT_ZrPbO3.yaml`` and
  ``data/parameters/presets/ECM_CeO2_cylindrical_gb.yaml`` via
  :meth:`SimulationConfig.from_yaml` (which itself pulls in the defect,
  reaction, GB and electrical component files).
* The activation-energies JSON via
  :func:`kinetix.configs.config_loader.load_activation_energies`
  (resolved from ``settings.activation_energies``).
* ``kinetix.initialization._process_activation_energies`` to turn the JSON into
  the per-defect ``Act_E_dict`` that ``Site`` consumes.

Only *boundary-condition* fixtures (empty defect config, single-site lookups)
use mocks, as required for genuinely isolated scenarios.

Pinned coupling / quirks (reported, not fixed)
---------------------------------------------
* ``supp_by``, ``energy_site``, ``destination_CN`` and ``passivation_level`` are
  **not** created by ``Site.__init__``: they are injected later by
  ``supported_by`` / ``calculate_site_energy`` during ``update_sites_topology``.
  A ``Site`` used before the topology pass therefore raises ``AttributeError``
  on the event paths - see ``TestLatentCoupling``.
* ``transition_rates`` indexes ``defects_config[current_defect]
  ['field_dependent_generation']`` directly, so a defect config lacking that key
  raises ``KeyError`` even for a purely thermal run.
* ``self.site_events`` holds ``Event`` instances (Phase 4).  ``transition_rates``
  reads ``Event.label`` / ``Event.barrier`` and writes the Arrhenius rate onto
  ``Event.rate``; the registered barrier is never overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import constants
from unittest.mock import MagicMock

from kinetix.configs.config_loader import load_activation_energies
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.initialization import _process_activation_energies
from kinetix.lattice.defect import Event
from kinetix.lattice.site import Site

KB = constants.physical_constants['Boltzmann constant in eV/K'][0]
NU0 = 7e12  # Site.NU0 - bond vibration frequency used by the Arrhenius law

# Same path-resolution pattern as tests/test_presets.py
PARAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters"

# Migration event labels are assigned by
# ``Crystal_Lattice._initialize_migration_pathways`` (one integer per unique
# minimum-image displacement vector).  These are code constants of the kMC
# event-label scheme, not parameter-file values.
PLANE_EVENT, UP_EVENT, DOWN_EVENT = 0, 1, 2


# =============================================================================
# Fixtures - real parameter files through the production loaders
# =============================================================================
PZT_PRESET_NAME = "PZT_ZrPbO3.yaml"
CERIA_PRESET_NAME = "ECM_CeO2_cylindrical_gb.yaml"


def _inject_migration_labels(act_e_dict: dict[str, dict]) -> dict[str, dict]:
  """Mirror ``Crystal_Lattice._initialize_migration_pathways``.

  When Poisson is solved, the lattice injects an int-keyed ``E_mig`` dict built
  from the directional barriers (``E_mig_plane`` / ``E_mig_upward`` /
  ``E_mig_downward``).  ``Site.available_migrations`` reads that dict for
  interstitial sites, so reproduce it here from the real values.
  """
  for energies in act_e_dict.values():
    if 'E_mig_plane' in energies:
      energies['E_mig'] = {
        PLANE_EVENT: energies['E_mig_plane'],
        UP_EVENT: energies['E_mig_upward'],
        DOWN_EVENT: energies['E_mig_downward'],
      }
  return act_e_dict


def _load_preset(name: str) -> SimulationConfig:
  path = PARAMS_DIR / "presets" / name
  assert path.exists(), f"preset not found: {path}"
  return SimulationConfig.from_yaml(path)


def _load_act_e(preset: SimulationConfig, name: str) -> dict[str, dict]:
  preset_path = PARAMS_DIR / "presets" / name
  ae_data = load_activation_energies(preset_path, preset.settings)
  act_e = _process_activation_energies(
    preset.defects.to_dict(), ae_data, preset.settings.technology
  )
  return _inject_migration_labels(act_e)


@pytest.fixture(scope="module")
def pzt_preset() -> SimulationConfig:
  """The REAL PZT preset, with all referenced component files loaded."""
  return _load_preset(PZT_PRESET_NAME)


@pytest.fixture(scope="module")
def pzt_defects(pzt_preset) -> dict[str, dict]:
  """``defects_config`` mapping produced by the production loader."""
  return pzt_preset.defects.to_dict()


@pytest.fixture(scope="module")
def pzt_reactions(pzt_preset) -> dict[str, dict]:
  """``reactions_config`` mapping produced by the production loader."""
  return pzt_preset.reactions.to_dict()


@pytest.fixture(scope="module")
def pzt_act_e(pzt_preset) -> dict[str, dict]:
  """Per-defect activation energies built from the REAL JSON file."""
  return _load_act_e(pzt_preset, PZT_PRESET_NAME)


@pytest.fixture(scope="module")
def ceria_preset() -> SimulationConfig:
  """The REAL ECM_CeO2 preset (Ag cation + redox energies)."""
  return _load_preset(CERIA_PRESET_NAME)


@pytest.fixture(scope="module")
def ceria_defects(ceria_preset) -> dict[str, dict]:
  return ceria_preset.defects.to_dict()


@pytest.fixture(scope="module")
def ceria_act_e(ceria_preset) -> dict[str, dict]:
  """Per-defect activation energies built from the REAL ECM_CeO2.json."""
  return _load_act_e(ceria_preset, CERIA_PRESET_NAME)


# =============================================================================
# Stubs / helpers
# =============================================================================
class GridSiteStub:
  """Minimal *destination / neighbour* site: only the attributes Site reads."""

  def __init__(self, site_type='interstitial', chemical_specie='Empty',
               position=(0.0, 0.0, 0.0), destination_CN=None,
               is_at_top_interface=False, is_at_bottom_interface=False,
               supp_by=(), migration_paths=None, passivation_level=0,
               ion_charge=0, current_defect=None):
    self.site_type = site_type
    self.chemical_specie = chemical_specie
    self.position = tuple(position)
    self.destination_CN = destination_CN if destination_CN is not None else {}
    self.is_at_top_interface = is_at_top_interface
    self.is_at_bottom_interface = is_at_bottom_interface
    self.supp_by = tuple(supp_by)
    self.migration_paths = migration_paths or {'Plane': [], 'Up': [], 'Down': []}
    self.passivation_level = passivation_level
    self.ion_charge = ion_charge
    self.nearest_neighbors_idx = []
    self._current_defect = current_defect

  def _get_current_defect_name(self):
    return self._current_defect


def make_site(defects_config, act_e_dict, chemical_specie, site_type,
              position=(0.0, 0.0, 0.0), reactions_config=None, supp_by=()):
  """Build a Site and inject the attributes normally set by the topology pass.

  ``supp_by`` and ``energy_site`` are written by ``supported_by()`` /
  ``calculate_site_energy()`` during ``update_sites_topology``; they are set
  here so the event logic can be exercised without a lattice.
  """
  site = Site(
    chemical_specie=chemical_specie,
    position=position,
    site_type=site_type,
    Act_E_dict=act_e_dict,
    defects_config=defects_config,
    reactions_config=reactions_config if reactions_config is not None else {},
  )
  site.supp_by = tuple(supp_by)
  current = site._get_current_defect_name()
  if current is not None and 'CN_clustering_energy' in act_e_dict[current]:
    site.energy_site = act_e_dict[current]['CN_clustering_energy'][0]
  else:
    site.energy_site = 0.0
  return site


def config_with_events(defects_config: dict[str, dict], defect_name: str,
                       events: list) -> dict[str, dict]:
  """Copy of ``defects_config`` with one defect's ``enabled_events`` replaced."""
  return {
    name: ({**cfg, 'enabled_events': list(events)} if name == defect_name
           else dict(cfg))
    for name, cfg in defects_config.items()
  }
# =============================================================================
# Initialization
# =============================================================================
class TestSiteInitialization:
  def test_position_and_species_are_stored(self):
    site = Site(chemical_specie='H', position=(1.0, 2.0, 3.0),
                site_type='interstitial')
    assert site.position == (1.0, 2.0, 3.0)
    assert site.chemical_specie == 'H'
    assert site.site_type == 'interstitial'

  def test_site_type_defaults_to_chemical_specie(self):
    site = Site(chemical_specie='V_O', position=(0.0, 0.0, 0.0))
    assert site.site_type == 'V_O'

  def test_empty_cache_and_event_containers(self):
    site = Site(chemical_specie='Empty', position=(0.0, 0.0, 0.0))
    assert site.site_events == []
    assert site.nearest_neighbors_idx == []
    assert site.migration_paths == {'Plane': [], 'Up': [], 'Down': []}
    assert site.cache_TR == {} and site.cache_planes == {}
    assert site.in_cluster_with_electrode == {'bottom_layer': False,
                                              'top_layer': False}
    assert (site.is_at_top_interface, site.is_at_bottom_interface) == (False, False)
    assert site.ion_charge == 0

  def test_applicable_defects_are_exactly_those_allowing_the_sublattice(
      self, pzt_defects, pzt_act_e):
    """``allowed_sublattices`` in the real config drives the filter."""
    for symbol, site_type in (('H', 'interstitial'), ('V_O', 'O')):
      site = Site(chemical_specie=symbol, position=(0.0, 0.0, 0.0),
                  site_type=site_type, Act_E_dict=pzt_act_e,
                  defects_config=pzt_defects)
      expected = {name for name, cfg in pzt_defects.items()
                  if site_type in cfg['allowed_sublattices']}
      assert set(site.applicable_defects) == expected
      assert expected, f"real config allows no defect on sublattice {site_type!r}"

  def test_current_defect_is_consistent_with_symbol_and_sublattice(
      self, pzt_defects, pzt_act_e):
    """The resolved defect must describe this site's species and sublattice."""
    for symbol, site_type in (('H', 'interstitial'), ('V_O', 'O')):
      site = Site(chemical_specie=symbol, position=(0.0, 0.0, 0.0),
                  site_type=site_type, Act_E_dict=pzt_act_e,
                  defects_config=pzt_defects)
      current = site._get_current_defect_name()
      assert current is not None, f"no defect resolved for {symbol!r} on {site_type!r}"
      cfg = pzt_defects[current]
      assert cfg['symbol'] == symbol
      assert cfg['site_type'] == site_type

  def test_current_defect_is_none_for_species_absent_from_config(
      self, pzt_defects, pzt_act_e):
    site = Site(chemical_specie='Xe', position=(0.0, 0.0, 0.0),
                site_type='Xe', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    assert site.applicable_defects == []
    assert site._get_current_defect_name() is None

  def test_sites_generation_layer_is_copied_from_the_real_config(
      self, pzt_defects, pzt_act_e):
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    current = site._get_current_defect_name()
    expected = pzt_defects[current]['sites_generation_layer']
    if expected is None:
      assert not hasattr(site, 'sites_generation_layer')
    else:
      assert site.sites_generation_layer == expected

  def test_passivation_level_follows_the_real_config(self, pzt_defects, pzt_act_e):
    """``passivation_level`` is set only for defects that declare it."""
    site = Site(chemical_specie='V_O', position=(0.0, 0.0, 0.0),
                site_type='O', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    current = site._get_current_defect_name()
    assert 'passivation_level' in pzt_defects[current], (
      "fixture expectation: oxygen_vacancy must declare passivation_level")
    assert site.passivation_level == pzt_defects[current]['passivation_level']

  def test_passivation_level_defaults_to_zero_when_not_declared(
      self, pzt_defects, pzt_act_e):
    """Phase 3: passivation_level is defect-carried and normalized.

    The pre-refactor Site only created the attribute when the config
    declared it (the conditional-attribute gap, report finding M1). With
    the composition model the Defect always carries the field and the
    delegating property always exists, reading 0 when the config does not
    declare it.
    """
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    current = site._get_current_defect_name()
    assert 'passivation_level' not in pzt_defects[current]
    assert site.passivation_level == 0

  def test_inactive_site_has_no_applicable_defects(self, pzt_defects, pzt_act_e):
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects, is_active_site=False)
    assert site.applicable_defects == []
    assert site._get_current_defect_name() is None
    assert not hasattr(site, 'sites_generation_layer')

  def test_site_without_defect_config_is_inert(self):
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial')
    assert site.applicable_defects == []
    assert site.defects_config is None
    assert site.Act_E_dict == {}
# =============================================================================
# Migration event generation (interstitial branch)
# =============================================================================
def _migration_grid(origin_idx, origin_site, dest_specs, defect_name):
  """Build the ``{idx: site}`` grid and migration paths for a test.

  The origin site must be part of the grid because ``available_pathways``
  dispatches reaction handlers with ``grid_crystal[idx_origin]``.
  """
  grid = {origin_idx: origin_site}
  paths = {'Plane': [], 'Up': [], 'Down': []}
  for idx, direction, label, kwargs in dest_specs:
    defaults = {'site_type': 'interstitial', 'chemical_specie': 'Empty'}
    defaults.update(kwargs)
    defaults.setdefault('destination_CN', {defect_name: 0})
    grid[idx] = GridSiteStub(**defaults)
    paths[direction].append((idx, label))
  return grid, paths


class TestMigrationEvents:
  def test_migration_events_use_the_real_directional_barriers(
      self, pzt_defects, pzt_act_e):
    """Plane/Up/Down events carry the real ``E_mig_*`` values from the JSON."""
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    act = pzt_act_e['hydrogen_interstitial']
    grid, paths = _migration_grid(
      (0, 0), site,
      [((1, 0), 'Plane', PLANE_EVENT, {}),
       ((2, 0), 'Up', UP_EVENT, {}),
       ((3, 0), 'Down', DOWN_EVENT, {})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert len(site.site_events) == 3
    plane_event, up_event, down_event = site.site_events
    assert plane_event.destination == (1, 0) and plane_event.label == PLANE_EVENT
    assert up_event.destination == (2, 0) and up_event.label == UP_EVENT
    assert down_event.destination == (3, 0) and down_event.label == DOWN_EVENT
    assert plane_event.barrier == pytest.approx(act['E_mig_plane'])
    assert up_event.barrier == pytest.approx(act['E_mig_upward'])
    assert down_event.barrier == pytest.approx(act['E_mig_downward'])

  def test_destination_coordination_adds_the_real_clustering_energy(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    act = pzt_act_e['hydrogen_interstitial']
    cn = 3
    grid, paths = _migration_grid(
      (0, 0), site,
      [((1, 0), 'Plane', PLANE_EVENT,
        {'destination_CN': {'hydrogen_interstitial': cn}})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    expected = act['E_mig_plane'] + act['CN_clustering_energy'][cn]
    assert site.site_events[0].barrier == pytest.approx(expected)

  def test_negative_energy_difference_is_clamped_to_zero(
      self, pzt_defects, pzt_act_e):
    """V_O clustering energies are negative; the barrier must not drop."""
    site = make_site(pzt_defects, pzt_act_e, 'V_O', 'O')
    act = pzt_act_e['oxygen_vacancy']
    assert act['CN_clustering_energy'][2] < 0, (
      "fixture expectation: V_O clustering energy must be attractive")
    grid, paths = _migration_grid(
      (0, 0, 0), site,
      [((1, 0, 0), 'Plane', PLANE_EVENT,
        {'site_type': 'O', 'chemical_specie': 'O',
         'destination_CN': {'oxygen_vacancy': 2}})],
      'oxygen_vacancy')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0, 0), facets_type=None)

    assert site.site_events[0].barrier == pytest.approx(act['E_mig_plane'])

  def test_destination_on_a_foreign_sublattice_is_skipped(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    allowed = pzt_defects['hydrogen_interstitial']['allowed_sublattices']
    foreign = 'sublattice_not_in_config'
    assert foreign not in allowed
    grid, paths = _migration_grid(
      (0, 0), site,
      [((1, 0), 'Plane', PLANE_EVENT, {'site_type': foreign})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_occupied_destination_is_skipped(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    valid = pzt_defects['hydrogen_interstitial']['valid_target_species']
    blocked = 'H'  # a real species, but not a valid target for migration
    assert blocked not in valid
    grid, paths = _migration_grid(
      (0, 0), site,
      [((1, 0), 'Plane', PLANE_EVENT, {'chemical_specie': blocked})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_destination_in_the_support_set_is_skipped(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     supp_by=((1, 0),))
    grid, paths = _migration_grid(
      (0, 0), site,
      [((1, 0), 'Plane', PLANE_EVENT, {})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_site_without_migration_paths_emits_nothing(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    assert site.migration_paths == {'Plane': [], 'Up': [], 'Down': []}

    site.available_pathways({(0, 0): site}, (0, 0), facets_type=None)

    assert site.site_events == []
# =============================================================================
# Event filtering (enabled_events from the real defect config)
# =============================================================================
class TestEventFiltering:
  def test_disabled_migration_emits_no_migration_events(self, pzt_defects, pzt_act_e):
    defects = config_with_events(pzt_defects, 'hydrogen_interstitial', [])
    assert defects['hydrogen_interstitial']['enabled_events'] == []
    site = make_site(defects, pzt_act_e, 'H', 'interstitial')
    grid, paths = _migration_grid(
      (0, 0), site, [((1, 0), 'Plane', PLANE_EVENT, {})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_reaction_only_defect_emits_no_migration_events(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    defects = config_with_events(pzt_defects, 'hydrogen_interstitial',
                                 ['reaction'])
    site = make_site(defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    grid, paths = _migration_grid(
      (0, 0), site, [((1, 0), 'Plane', PLANE_EVENT, {})],
      'hydrogen_interstitial')
    site.migration_paths = paths

    site.available_pathways(grid, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_reduction_requires_the_event_flag(self, ceria_defects, ceria_act_e):
    defects = config_with_events(ceria_defects, 'Ag_interstitial', ['migration'])
    assert 'reduction' not in defects['Ag_interstitial']['enabled_events']
    site = make_site(defects, ceria_act_e, 'Ag', 'interstitial')
    site.ion_charge = ceria_defects['Ag_interstitial']['charge']
    site.CN_redox_energy = site.calculate_CN_contribution_redox_energy()

    site.available_pathways({}, (0, 0), facets_type=None)

    assert site.site_events == []

  def test_reduction_event_uses_the_real_barrier(self, ceria_defects, ceria_act_e):
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial')
    site.ion_charge = ceria_defects['Ag_interstitial']['charge']
    site.CN_redox_energy = site.calculate_CN_contribution_redox_energy()
    act = ceria_act_e['Ag_interstitial']

    site.available_pathways({}, (0, 0), facets_type=None)

    reductions = [e for e in site.site_events if e.label == 'reduction']
    assert len(reductions) == 1
    assert reductions[0].barrier == pytest.approx(
      act['E_reduction'] - site.CN_redox_energy)

  def test_oxidation_event_uses_the_real_barrier(self, ceria_defects, ceria_act_e):
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial')
    site.ion_charge = 0
    site.nearest_neighbors_idx = [(1, 0, 0)]  # surface atom -> can oxidize
    site.CN_redox_energy = site.calculate_CN_contribution_redox_energy()
    act = ceria_act_e['Ag_interstitial']

    site.available_pathways({}, (0, 0), facets_type=None)

    oxidations = [e for e in site.site_events if e.label == 'oxidation']
    assert len(oxidations) == 1
    assert oxidations[0].barrier == pytest.approx(
      act['E_oxidation'] + site.CN_redox_energy)

  def test_charged_species_cannot_oxidize(self, ceria_defects, ceria_act_e):
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial')
    site.ion_charge = ceria_defects['Ag_interstitial']['charge']
    site.nearest_neighbors_idx = [(1, 0, 0)]
    site.CN_redox_energy = site.calculate_CN_contribution_redox_energy()

    site.available_pathways({}, (0, 0), facets_type=None)

    assert [e for e in site.site_events if e.label == 'oxidation'] == []

  def test_neutral_fully_coordinated_bulk_atom_cannot_oxidize(
      self, ceria_defects, ceria_act_e):
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial')
    site.ion_charge = 0
    site.nearest_neighbors_idx = [(1, 0, 0)]
    site.supp_by = ((1, 0, 0),)  # fully coordinated -> not a surface atom
    site.CN_redox_energy = site.calculate_CN_contribution_redox_energy()

    site.available_pathways({}, (0, 0), facets_type=None)

    assert [e for e in site.site_events if e.label == 'oxidation'] == []
# =============================================================================
# Reactions (real reaction config)
# =============================================================================
def _defect_name_by_symbol(defects_config: dict[str, dict], symbol: str) -> str:
  """Resolve a defect name from its symbol without hardcoding config keys."""
  for name, cfg in defects_config.items():
    if cfg['symbol'] == symbol:
      return name
  raise AssertionError(f"no defect with symbol {symbol!r} in the real config")


def _reaction_by_type(reactions_config: dict[str, dict], rxn_type: str) -> dict:
  matches = [r for r in reactions_config.values() if r['type'] == rxn_type]
  assert matches, f"no {rxn_type!r} reaction in the real config"
  return matches[0]


class TestReactionEvents:
  def test_h2_formation_uses_the_real_neighbour_barrier(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    h_name = _defect_name_by_symbol(pzt_defects, 'H')
    reaction = _reaction_by_type(pzt_reactions, 'bimolecular_neighbor')
    neighbour = GridSiteStub(site_type='interstitial', chemical_specie='H')
    site.nearest_neighbors_idx = ['nb']
    grid = {'origin': site, 'nb': neighbour}

    site.available_reactions(grid, 'origin')

    assert len(site.site_events) == 1
    event = site.site_events[0]
    assert event.destination == 'nb' and event.label == reaction['name']
    assert event.barrier == pytest.approx(pzt_act_e[h_name][reaction['name']])

  def test_neighbour_reaction_requires_a_matching_partner(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    reaction = _reaction_by_type(pzt_reactions, 'bimolecular_neighbor')
    # A V_O neighbour satisfies the capture reaction's partner, not H2's.
    neighbour = GridSiteStub(
      site_type='O', chemical_specie='V_O',
      current_defect=_defect_name_by_symbol(pzt_defects, 'V_O'))
    site.nearest_neighbors_idx = ['vo']
    grid = {'origin': site, 'vo': neighbour}

    site.available_reactions(grid, 'origin')

    assert [e for e in site.site_events if e.label == reaction['name']] == []

  def test_removal_event_respects_the_configured_removal_layer(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    h_name = _defect_name_by_symbol(pzt_defects, 'H')
    reaction = next(r for r in pzt_reactions.values()
                    if r.get('sites_removal_layer'))
    assert reaction['sites_removal_layer'] == 'bottom_layer'
    grid = {'origin': site}

    # Default: not at the bottom interface -> no removal.
    site.available_reactions(grid, 'origin')
    assert [e for e in site.site_events if e.label == reaction['name']] == []

    site.site_events = []
    site.set_interface_flags(bottom_z=0.0, top_z=10.0)
    assert site.is_at_bottom_interface is True

    site.available_reactions(grid, 'origin')

    removal = [e for e in site.site_events if e.label == reaction['name']]
    assert len(removal) == 1
    assert removal[0].barrier == pytest.approx(pzt_act_e[h_name][reaction['name']])

  def test_disabled_reactions_are_ignored(self, pzt_defects, pzt_act_e,
                                          pzt_reactions):
    disabled = {
      name: ({**rxn, 'enabled': False} if rxn['type'] == 'bimolecular_neighbor'
             else dict(rxn))
      for name, rxn in pzt_reactions.items()
    }
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=disabled)
    neighbour = GridSiteStub(site_type='interstitial', chemical_specie='H')
    site.nearest_neighbors_idx = ['nb']

    site.available_reactions({'origin': site, 'nb': neighbour}, 'origin')

    neighbour_rxn = _reaction_by_type(pzt_reactions, 'bimolecular_neighbor')
    assert [e for e in site.site_events if e.label == neighbour_rxn['name']] == []
class TestPassivation:
  def test_vacancy_config_declares_the_passivation_budget(self, pzt_defects):
    vo_name = _defect_name_by_symbol(pzt_defects, 'V_O')
    assert pzt_defects[vo_name]['max_passivation_level'] is not None
    assert pzt_defects[vo_name]['max_passivation_level'] > 0

  def test_capture_event_is_keyed_by_the_neighbour_passivation_level(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    vo_name = _defect_name_by_symbol(pzt_defects, 'V_O')
    reaction = _reaction_by_type(pzt_reactions, 'bimolecular_capture')
    barrier_table = pzt_act_e['hydrogen_interstitial'][reaction['name']]
    assert isinstance(barrier_table, dict), (
      "fixture expectation: capture barrier must be passivation-keyed")

    neighbour = GridSiteStub(site_type='O', chemical_specie='V_O',
                             current_defect=vo_name, passivation_level=0)
    site.nearest_neighbors_idx = ['vo']

    site.available_reactions({'origin': site, 'vo': neighbour}, 'origin')

    events = [e for e in site.site_events if e.label == reaction['name']]
    assert len(events) == 1
    assert events[0].destination == 'vo'
    assert events[0].barrier == pytest.approx(barrier_table['0'])

  def test_saturated_vacancy_blocks_further_passivation(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config=pzt_reactions)
    vo_name = _defect_name_by_symbol(pzt_defects, 'V_O')
    reaction = _reaction_by_type(pzt_reactions, 'bimolecular_capture')
    max_pass = pzt_defects[vo_name]['max_passivation_level']

    neighbour = GridSiteStub(site_type='O', chemical_specie='V_O',
                             current_defect=vo_name,
                             passivation_level=max_pass)
    site.nearest_neighbors_idx = ['vo']

    site.available_reactions({'origin': site, 'vo': neighbour}, 'origin')

    assert [e for e in site.site_events if e.label == reaction['name']] == []

  def test_depassivation_is_gated_by_the_real_min_passivation(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'V_O', 'O',
                     reactions_config=pzt_reactions)
    vo_name = _defect_name_by_symbol(pzt_defects, 'V_O')
    reaction = next(r for r in pzt_reactions.values()
                    if r['type'] == 'unimolecular_reaction'
                    and 'min_passivation' in r['reactants'][0])
    min_pass = reaction['reactants'][0]['min_passivation']
    barrier_table = pzt_act_e[vo_name][reaction['name']]

    dest = GridSiteStub(site_type='interstitial', chemical_specie='Empty')
    site.nearest_neighbors_idx = ['dest']
    grid = {'origin': site, 'dest': dest}

    # Below the threshold: blocked.
    site.passivation_level = min_pass - 1
    site.available_reactions(grid, 'origin')
    assert [e for e in site.site_events if e.label == reaction['name']] == []

    # At the threshold: released, using the level-keyed real barrier.
    site.site_events = []
    site.passivation_level = min_pass
    site.available_reactions(grid, 'origin')

    events = [e for e in site.site_events if e.label == reaction['name']]
    assert len(events) == 1
    assert events[0].barrier == pytest.approx(barrier_table[str(min_pass)])

  def test_depassivation_needs_a_valid_destination(
      self, pzt_defects, pzt_act_e, pzt_reactions):
    site = make_site(pzt_defects, pzt_act_e, 'V_O', 'O',
                     reactions_config=pzt_reactions)
    reaction = next(r for r in pzt_reactions.values()
                    if r['type'] == 'unimolecular_reaction'
                    and 'min_passivation' in r['reactants'][0])
    min_pass = reaction['reactants'][0]['min_passivation']
    site.passivation_level = min_pass
    site.nearest_neighbors_idx = []  # no empty interstitial nearby

    site.available_reactions({'origin': site}, 'origin')

    assert [e for e in site.site_events if e.label == reaction['name']] == []
# =============================================================================
# Transition rates (Arrhenius + field corrections)
# =============================================================================
def arrhenius(barrier: float, temperature: float = 300.0) -> float:
  """Independent reference: rate = nu0 * exp(-E_act / kBT)."""
  return NU0 * np.exp(-barrier / (KB * temperature))


class TestTransitionRates:
  def test_rate_follows_the_arrhenius_law_for_the_real_barrier(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    barrier = pzt_act_e['hydrogen_interstitial']['E_mig_plane']
    dest = (1, 0)
    site.site_events = [Event(label=PLANE_EVENT, destination=dest,
                              barrier=barrier)]

    site.transition_rates(T=300)

    assert len(site.site_events) == 1
    event = site.site_events[0]
    assert event.rate == pytest.approx(arrhenius(barrier))
    assert event.destination == dest
    assert event.label == PLANE_EVENT
    assert event.barrier == pytest.approx(barrier)

  def test_rate_increases_with_temperature(self, pzt_defects, pzt_act_e):
    barrier = pzt_act_e['hydrogen_interstitial']['E_mig_plane']

    cold_site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    cold_site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                                   barrier=barrier)]
    cold_site.transition_rates(T=300)

    hot_site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    hot_site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                                  barrier=barrier)]
    hot_site.transition_rates(T=600)

    assert hot_site.site_events[0].rate > cold_site.site_events[0].rate
    assert hot_site.site_events[0].rate == pytest.approx(arrhenius(barrier, 600))
    assert cold_site.site_events[0].rate == pytest.approx(arrhenius(barrier, 300))

  def test_rate_cache_is_temperature_aware(self, pzt_defects, pzt_act_e):
    """The cache key is ``(round(Act_E, 3), round(T, 1))``.

    Reusing the same Site instance across a temperature change must recompute
    the rate instead of replaying the stale cached one.
    """
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    barrier = pzt_act_e['hydrogen_interstitial']['E_mig_plane']

    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=barrier)]
    site.transition_rates(T=300)
    cold = site.site_events[0].rate

    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=barrier)]
    site.transition_rates(T=600)
    hot = site.site_events[0].rate

    assert hot == pytest.approx(arrhenius(barrier, 600))
    assert cold == pytest.approx(arrhenius(barrier, 300))
    assert hot != pytest.approx(cold)
    assert set(site.cache_TR) == {(round(barrier, 3), 300.0),
                                  (round(barrier, 3), 600.0)}

    # Same barrier at the same temperature still shares one cache entry.
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=barrier)]
    site.transition_rates(T=300)
    assert site.site_events[0].rate == pytest.approx(cold)
    assert len(site.cache_TR) == 2

  def test_zero_barrier_gives_the_attempt_frequency(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=0.0)]

    site.transition_rates(T=300)

    assert site.site_events[0].rate == pytest.approx(NU0)

  def test_negative_barrier_is_clamped_but_the_event_keeps_its_energy(
      self, pzt_defects, pzt_act_e):
    """`Act_E = max(Act_E, 0)` only affects the rate, not the stored energy."""
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=-0.5)]

    site.transition_rates(T=300)

    assert site.site_events[0].rate == pytest.approx(NU0)
    assert site.site_events[0].barrier == -0.5

  def test_equal_barriers_share_one_cache_entry(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    barrier = pzt_act_e['hydrogen_interstitial']['E_mig_plane']
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=barrier),
                        Event(label=UP_EVENT, destination=(2, 0),
                              barrier=barrier)]

    site.transition_rates(T=300)

    assert len(site.cache_TR) == 1
    assert site.site_events[0].rate == site.site_events[1].rate == pytest.approx(
      arrhenius(barrier))

  def test_pre_rated_event_rate_is_overwritten_in_place(
      self, pzt_defects, pzt_act_e):
    """Deposition events are pre-seeded with a placeholder rate (Event.rate)."""
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    barrier = pzt_act_e['hydrogen_interstitial']['E_mig_plane']
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=barrier, rate=0.0)]

    site.transition_rates(T=300)

    assert site.site_events[0].rate == pytest.approx(arrhenius(barrier))

  def test_field_lowers_the_forward_migration_barrier(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    act = pzt_act_e['hydrogen_interstitial']
    h_name = _defect_name_by_symbol(pzt_defects, 'H')
    site.ion_charge = pzt_defects[h_name]['charge']
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=act['E_mig_plane'])]
    field = np.array([0.0, 0.0, 1e7])
    migration_pathways = {
      PLANE_EVENT: {'direction': np.array([0.0, 0.0, 1.0]), 'distance': 2.0},
    }

    site.transition_rates(T=300, E_site_field=field,
                          migration_pathways=migration_pathways,
                          clusters=None, atom_to_cluster={})

    shifted = (act['E_mig_plane']
               - site.ion_charge * np.dot(field, [0, 0, 1]) * 1e-10)
    assert shifted > act['E_min_mig'], "fixture expectation: floor not reached"
    assert site.site_events[0].rate == pytest.approx(arrhenius(shifted))

  def test_field_correction_is_floored_at_the_real_E_min_mig(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    act = pzt_act_e['hydrogen_interstitial']
    h_name = _defect_name_by_symbol(pzt_defects, 'H')
    site.ion_charge = pzt_defects[h_name]['charge']
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=act['E_mig_plane'])]
    # Large enough that the raw field shift would drive the barrier negative.
    field = np.array([0.0, 0.0, 1e10])
    migration_pathways = {
      PLANE_EVENT: {'direction': np.array([0.0, 0.0, 1.0]), 'distance': 2.0},
    }

    site.transition_rates(T=300, E_site_field=field,
                          migration_pathways=migration_pathways,
                          clusters=None, atom_to_cluster={})

    assert site.site_events[0].rate == pytest.approx(arrhenius(act['E_min_mig']))

  def test_reduction_barrier_is_shifted_at_the_top_electrode(
      self, ceria_defects, ceria_act_e):
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial',
                     supp_by=('top_layer',))
    act = ceria_act_e['Ag_interstitial']
    site.site_events = [Event(label='reduction', destination=(0,),
                              barrier=act['E_reduction'])]
    field = np.array([0.0, 0.0, 1e7])

    site.transition_rates(T=300, E_site_field=field,
                          migration_pathways={}, clusters=None,
                          atom_to_cluster={})

    field_proj = np.dot(field, [0, 0, 1]) * 1e-10
    expected_barrier = max(act['E_reduction'] - 0.5 * field_proj,
                           act['E_reduction_min'])
    assert site.site_events[0].rate == pytest.approx(arrhenius(expected_barrier))

  def test_subthreshold_field_leaves_the_barrier_untouched(
      self, ceria_defects, ceria_act_e):
    """`relevant_field` requires |E| > 1e6 V/m."""
    site = make_site(ceria_defects, ceria_act_e, 'Ag', 'interstitial',
                     supp_by=('top_layer',))
    act = ceria_act_e['Ag_interstitial']
    site.site_events = [Event(label='reduction', destination=(0,),
                              barrier=act['E_reduction'])]

    site.transition_rates(T=300, E_site_field=np.array([0.0, 0.0, 1e5]),
                          migration_pathways={}, clusters=None,
                          atom_to_cluster={})

    assert site.site_events[0].rate == pytest.approx(arrhenius(act['E_reduction']))
# =============================================================================
# Edge cases: boundaries, support sets, species bookkeeping
# =============================================================================
class TestEdgeCases:
  def test_site_at_the_bottom_boundary_is_flagged(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     position=(1.0, 1.0, 0.0))
    site.set_interface_flags(bottom_z=0.0, top_z=10.0)
    assert site.is_at_bottom_interface is True
    assert site.is_at_top_interface is False

  def test_site_at_the_top_boundary_is_flagged(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     position=(1.0, 1.0, 10.0))
    site.set_interface_flags(bottom_z=0.0, top_z=10.0)
    assert site.is_at_top_interface is True
    assert site.is_at_bottom_interface is False

  def test_bulk_site_is_not_flagged(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     position=(1.0, 1.0, 5.0))
    site.set_interface_flags(bottom_z=0.0, top_z=10.0)
    assert site.is_at_top_interface is False
    assert site.is_at_bottom_interface is False

  def test_supported_by_collects_identical_neighbours(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    h_name = _defect_name_by_symbol(pzt_defects, 'H')
    assert pzt_defects[h_name]['CN_matters'] is True
    assert pzt_defects[h_name]['symbol'] == site.chemical_specie
    n1, n2 = ('n1',), ('n2',)
    site.nearest_neighbors_idx = [n1, n2]
    grid = {n1: GridSiteStub(site_type='interstitial', chemical_specie='H'),
            n2: GridSiteStub(site_type='interstitial', chemical_specie='H')}

    site.supported_by(grid, wulff_facets=None, dir_edge_facets=None,
                      idx_origin=('origin',))

    assert set(site.supp_by) == {n1, n2}
    assert site.energy_site == pytest.approx(
      pzt_act_e[h_name]['CN_clustering_energy'][2])

  def test_supported_by_ignores_neighbours_of_other_species(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    n1 = ('n1',)
    foreign = ('foreign',)
    site.nearest_neighbors_idx = [n1, foreign]
    grid = {n1: GridSiteStub(site_type='interstitial', chemical_specie='H'),
            foreign: GridSiteStub(site_type='O', chemical_specie='V_O')}

    site.supported_by(grid, wulff_facets=None, dir_edge_facets=None,
                      idx_origin=('origin',))

    assert set(site.supp_by) == {n1}

  def test_supported_by_records_the_interface_marker(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     position=(0.0, 0.0, 0.0))
    site.set_interface_flags(bottom_z=0.0, top_z=10.0)
    assert site.is_at_bottom_interface is True
    site.nearest_neighbors_idx = []

    site.supported_by({}, wulff_facets=None, dir_edge_facets=None,
                      idx_origin=('origin',))

    assert 'bottom_layer' in site.supp_by

  def test_introduce_specie_uses_the_configured_charge(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'Empty', 'interstitial')
    h_name = _defect_name_by_symbol(pzt_defects, 'H')

    site.introduce_specie('H')

    assert site.chemical_specie == 'H'
    assert site.ion_charge == pzt_defects[h_name]['charge']

  def test_introduce_specie_accepts_an_explicit_charge(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'Empty', 'interstitial')

    site.introduce_specie('H', ion_charge=0)

    assert site.ion_charge == 0

  def test_remove_specie_clears_charge_and_events(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=0.4)]

    site.remove_specie('Empty')

    assert site.chemical_specie == 'Empty'
    assert site.ion_charge == 0
    assert site.site_events == []

  def test_get_migrating_state_follows_the_real_migrating_attributes(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'V_O', 'O')
    vo_name = _defect_name_by_symbol(pzt_defects, 'V_O')
    attrs = pzt_defects[vo_name]['migrating_attributes']
    assert attrs is not None
    assert 'passivation_level' in attrs, (
      "fixture expectation: V_O must migrate its passivation level")

    state = site.get_migrating_state(pzt_defects)

    assert 'passivation_level' in state
    assert state['passivation_level'] == site.passivation_level
    assert 'ion_charge' not in state  # handled by _introduce_specie_site

  def test_get_migrating_state_is_none_without_a_defect(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'Xe', 'Xe')
    assert site.get_migrating_state(pzt_defects) is None
# =============================================================================
# Latent coupling with the topology pass (documented, not fixed)
# =============================================================================
class TestLatentCoupling:
  def test_constructor_does_not_create_topology_attributes(
      self, pzt_defects, pzt_act_e):
    """``supp_by`` / ``energy_site`` / ``destination_CN`` come from the topology pass."""
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    assert not hasattr(site, 'supp_by')
    assert not hasattr(site, 'energy_site')
    assert not hasattr(site, 'destination_CN')

  def test_transition_rates_defaults_to_thermal_without_field_dependent_key(
      self, pzt_defects, pzt_act_e):
    """A missing ``field_dependent_generation`` key no longer
    raises KeyError - it defaults to False (purely thermal generation)."""
    stripped = {
      name: {k: v for k, v in cfg.items() if k != 'field_dependent_generation'}
      for name, cfg in pzt_defects.items()
    }
    site = make_site(stripped, pzt_act_e, 'H', 'interstitial')
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=0.4)]

    site.transition_rates(T=300)  # must not raise

    assert site.site_events[0].rate == pytest.approx(arrhenius(0.4))

  def test_available_migrations_requires_supp_by(self, pzt_defects, pzt_act_e):
    """Before the topology pass, the migration filter raises AttributeError."""
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)
    dest = GridSiteStub(site_type='interstitial', chemical_specie='Empty',
                        destination_CN={'hydrogen_interstitial': 0})
    site.migration_paths = {'Plane': [((1, 0), PLANE_EVENT)], 'Up': [], 'Down': []}

    with pytest.raises(AttributeError):
      site.available_migrations({(1, 0): dest}, (0, 0), None)

  def test_available_pathways_returns_early_without_a_current_defect(
      self, pzt_defects, pzt_act_e):
    """The None-defect guard prevents the topology coupling from being hit."""
    site = Site(chemical_specie='Xe', position=(0.0, 0.0, 0.0),
                site_type='Xe', Act_E_dict=pzt_act_e,
                defects_config=pzt_defects)

    site.available_pathways({}, (0, 0), facets_type=None)

    assert site.site_events == []
# =============================================================================
# Empty-config boundary conditions (isolated, no real config needed)
# =============================================================================
class TestEmptyConfigBoundaries:
  def test_site_with_an_empty_defect_config_is_inert(self):
    site = Site(chemical_specie='H', position=(0.0, 0.0, 0.0),
                site_type='interstitial', Act_E_dict={}, defects_config={})
    site.supp_by = ()

    site.available_pathways({}, (0, 0), facets_type=None)

    assert site.site_events == []
    assert site.applicable_defects == []
    assert site._get_current_defect_name() is None

  def test_empty_reactions_config_yields_no_reaction_events(
      self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial',
                     reactions_config={})
    neighbour = GridSiteStub(site_type='interstitial', chemical_specie='H')
    site.nearest_neighbors_idx = ['nb']

    site.available_reactions({'origin': site, 'nb': neighbour}, 'origin')

    assert site.site_events == []

  def test_remove_event_type_drops_the_matching_label(self, pzt_defects, pzt_act_e):
    site = make_site(pzt_defects, pzt_act_e, 'H', 'interstitial')
    # Pre-rated events, as transition_rates leaves them.
    site.site_events = [Event(label=PLANE_EVENT, destination=(1, 0),
                              barrier=0.4, rate=1.0),
                        Event(label=UP_EVENT, destination=(2, 0),
                              barrier=0.2, rate=1.0)]

    site.remove_event_type(UP_EVENT)

    assert len(site.site_events) == 1
    assert site.site_events[0].label == PLANE_EVENT