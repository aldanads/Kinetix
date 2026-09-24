"""Behavioral spec for :class:`kinetix.lattice.grain_boundary.GrainBoundary`.

Regression net for the ``simulator.py`` split.  Covers:

* **Distance functions** - ``_distance_to_planar_gb`` and
  ``_distance_to_cylindrical_gb``.
* **Configuration processing** - ``_process_configurations`` injecting the
  derived fields (``distance_function``, ``inner_boundary``,
  ``outer_boundary``, ``mig_cfg``/``gen_cfg``/``rxn_cfg``, linear
  slope/intercept) into the raw config dicts.
* **Boundary detection** - ``get_site_gb_region`` /
  ``is_site_in_grain_boundary``.
* **Event modifications** - ``modify_act_energy_GB`` (linear and
  direction-dependent barrier models) and ``_get_gb_reduction_for_site``.

Config wiring (no hardcoded parameter literals)
-----------------------------------------------
The real shipped GB YAML files are loaded through the production loader
:meth:`kinetix.configs.grain_boundary_config.GrainBoundariesConfig.from_yaml`
(``gb_vertical_planar.yaml`` and ``gb_cylindrical_VCM_HfO2.yaml``), and the
direction-dependent barriers come from the real PZT activation-energy JSON via
``load_activation_energies`` + ``_process_activation_energies``.  Test
positions are chosen relative to the geometry read from those files.  Only
boundary-condition fixtures (empty GB list, zero-width GB) use synthetic dicts.

TODO
----------------------------------------
* A GB entry with a non-``direction_dependent`` barrier model and no
  ``Act_E_diff_GB`` raises ``KeyError`` from inside the *constructor*
  (``_process_configurations``) rather than at use time.
* ``triple_junction_planes`` GBs get ``distance_function = None`` ("NEED TO
  WRITE THE FUNCTION" in the source), so they are silently ignored by
  region detection and ``modify_act_energy_GB``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinetix.configs.config_loader import load_activation_energies
from kinetix.configs.grain_boundary_config import GrainBoundariesConfig
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.initialization import _process_activation_energies
from kinetix.lattice.grain_boundary import GrainBoundary


# Same path-resolution pattern as tests/test_presets.py / tests/test_site.py
PARAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters"


# =============================================================================
# Fixtures: REAL config files via the production loaders
# =============================================================================
# NOTE: the fixtures below still hand every test its own deep copy of the
# file content. (GrainBoundary deep-copies internally too now - fixed bug -
# but isolated inputs keep the tests honest.)
@pytest.fixture()
def planar_gb_dicts() -> list[dict]:
  """The REAL gb_vertical_planar.yaml through the production loader."""
  cfg = GrainBoundariesConfig.from_yaml(
    PARAMS_DIR / "grain_boundaries" / "gb_vertical_planar.yaml")
  return copy.deepcopy(cfg.to_dict())


@pytest.fixture()
def cylindrical_gb_dicts() -> list[dict]:
  """The REAL gb_cylindrical_VCM_HfO2.yaml through the production loader."""
  cfg = GrainBoundariesConfig.from_yaml(
    PARAMS_DIR / "grain_boundaries" / "gb_cylindrical_VCM_HfO2.yaml")
  return copy.deepcopy(cfg.to_dict())


@pytest.fixture(scope="module")
def pzt_preset():
  """The REAL PZT preset (pulls in defects, reactions, GB, settings files)."""
  path = PARAMS_DIR / "presets" / "PZT_ZrPbO3.yaml"
  assert path.exists(), f"preset not found: {path}"
  return SimulationConfig.from_yaml(path)


@pytest.fixture(scope="module")
def pzt_h_act_e(pzt_preset) -> dict:
  """REAL PZT activation energies for H, processed as the lattice does."""
  ae_data = load_activation_energies(
    PARAMS_DIR / "presets" / "PZT_ZrPbO3.yaml", pzt_preset.settings)
  act_e = _process_activation_energies(
    pzt_preset.defects.to_dict(), ae_data, pzt_preset.settings.technology)
  return copy.deepcopy(act_e['hydrogen_interstitial'])




# =============================================================================
# Distance functions
# =============================================================================
class TestDistanceFunctions:
  def test_planar_yz_distance_is_along_x(self):
    """A synthetic yz-oriented GB: distance is |x - position|."""
    gb = {'type': 'vertical_planar', 'orientation': 'yz', 'position': 12.0,
          'width': 4.0, 'Act_E_diff_GB': 1.0, 'event_modifications': {}}
    model = GrainBoundary([30.0, 30.0, 6.0], [gb])
    assert model._distance_to_planar_gb((15.0, 0.0, 0.0), gb) == \
      pytest.approx(3.0)
    assert model._distance_to_planar_gb((9.0, 0.0, 0.0), gb) == \
      pytest.approx(3.0)

  def test_planar_xz_distance_is_along_y(self, planar_gb_dicts):
    """The REAL gb_vertical_planar.yaml is xz-oriented: distance |y - pos|."""
    gb = dict(planar_gb_dicts[0])
    model = GrainBoundary([50.0, 54.0, 6.0], [gb])
    d = model._distance_to_planar_gb((0.0, gb['position'] - 2.0, 0.0), gb)
    assert d == pytest.approx(2.0)
    d = model._distance_to_planar_gb((0.0, gb['position'] + 3.0, 0.0), gb)
    assert d == pytest.approx(3.0)

  def test_unknown_orientation_raises_value_error(self):
    """FIXED (B8): an unknown orientation fails loudly with ValueError
    instead of silently returning None (which caused a TypeError later)."""
    gb = {'type': 'vertical_planar', 'orientation': 'zw', 'position': 5.0,
          'width': 2.0, 'Act_E_diff_GB': 1.0, 'event_modifications': {}}
    model = GrainBoundary([10.0, 10.0, 10.0], [gb])
    with pytest.raises(ValueError, match='Unknown GB orientation'):
      model._distance_to_planar_gb((1.0, 1.0, 1.0), gb)

  def test_cylindrical_distance_is_radial(self, cylindrical_gb_dicts):
    gb = dict(cylindrical_gb_dicts[0])
    cx, cy = gb['center']
    model = GrainBoundary([50.0, 50.0, 6.0], [gb])
    assert model._distance_to_cylindrical_gb(
      (cx + 7.0, cy, 0.0), gb) == pytest.approx(7.0)
    assert model._distance_to_cylindrical_gb(
      (cx + 3.0, cy + 4.0, 0.0), gb) == pytest.approx(5.0)
    # z is irrelevant for the radial distance
    assert model._distance_to_cylindrical_gb(
      (cx + 3.0, cy + 4.0, 99.0), gb) == pytest.approx(5.0)

  def test_on_axis_site_has_zero_distance(self, cylindrical_gb_dicts):
    gb = dict(cylindrical_gb_dicts[0])
    model = GrainBoundary([50.0, 50.0, 6.0], [gb])
    assert model._distance_to_cylindrical_gb(
      (gb['center'][0], gb['center'][1], 1.0), gb) == pytest.approx(0.0)



# =============================================================================
# Configuration processing (_process_configurations)
# =============================================================================
def _first(dicts):
  return dicts[0]


class TestProcessConfigurations:
  def test_planar_gb_gets_derived_fields(self, planar_gb_dicts):
    """The production vertical_planar YAML injects boundaries + callable."""
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]  # the processed deep copy carries injections
    assert callable(gb['distance_function'])
    assert gb['inner_boundary'] == pytest.approx(gb['width'] / 2.0)
    assert gb['outer_boundary'] == pytest.approx(
      gb.get('outer_width', gb['width']) / 2.0)
    assert gb['inner_boundary'] < gb['outer_boundary']
    assert model.vertical_gbs == [gb]
    assert model.max_gb_influence >= gb['outer_boundary']

  def test_cylindrical_gb_gets_derived_fields(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]  # the processed deep copy carries injections
    assert callable(gb['distance_function'])
    assert gb['inner_boundary'] == pytest.approx(gb['radius'])
    assert gb['outer_boundary'] == pytest.approx(gb['outer_radius'])
    assert model.cylindrical_gbs == [gb]
    assert model.max_gb_influence == pytest.approx(gb['outer_radius'])

  def test_event_modifications_are_normalized_to_lists(self, planar_gb_dicts):
    """mig_cfg/gen_cfg/rxn_cfg become lists with *set-based* lookups."""
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    assert isinstance(gb['mig_cfg'], list)
    assert isinstance(gb['gen_cfg'], list)
    assert isinstance(gb['rxn_cfg'], list)
    for entry in gb['mig_cfg']:
      assert isinstance(entry['affected_defects_set'], set)
      assert entry['affected_defects_set'] == set(entry['affected_defects'])
    for entry in gb['rxn_cfg']:
      assert isinstance(entry['affected_reactions_set'], set)

  def test_linear_model_computes_slope_and_intercept(self, planar_gb_dicts):
    """slope = -diff/(outer-inner); at inner -> diff, at outer -> 0."""
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    vo_entry = next(e for e in model.gb_configurations[0]['mig_cfg']
                    if 'oxygen_vacancy' in e['affected_defects'])
    diff = vo_entry['Act_E_diff_GB']
    inner, outer = vo_entry['inner_boundary'], vo_entry['outer_boundary']
    assert vo_entry['linear_slope'] == pytest.approx(-diff / (outer - inner))
    assert vo_entry['linear_intercept'] == pytest.approx(
      diff - vo_entry['linear_slope'] * inner)
    # Sanity: the linear form hits diff at the inner edge and 0 at the outer
    assert vo_entry['linear_slope'] * inner + \
      vo_entry['linear_intercept'] == pytest.approx(diff)
    assert vo_entry['linear_slope'] * outer + \
      vo_entry['linear_intercept'] == pytest.approx(0.0)

  def test_direction_dependent_entry_needs_no_act_e_diff(self, planar_gb_dicts):
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    h_entry = next(e for e in model.gb_configurations[0]['mig_cfg']
                   if 'hydrogen_interstitial' in e['affected_defects'])
    assert h_entry['barrier_model'] == 'direction_dependent'
    assert 'Act_E_diff_GB' not in h_entry

  def test_linear_entry_without_act_e_diff_raises_keyerror(self):
    """FINDING: the error is raised from the constructor, not at use time."""
    bad = {'type': 'vertical_planar', 'orientation': 'yz', 'position': 5.0,
           'width': 2.0,
           'event_modifications': {'migration': {
             'region': 'inner_boundary',
             'affected_defects': ['any_defect']}}}
    with pytest.raises(KeyError, match='Act_E_diff_GB'):
      GrainBoundary([10.0, 10.0, 10.0], [bad])

  def test_triple_junction_gets_no_distance_function(self):
    """FINDING: triple_junction_planes is unimplemented (distance None)."""
    cfg = {'type': 'triple_junction_planes', 'center': [5.0, 5.0],
           'width': 2.0, 'Act_E_diff_GB': 1.0}
    model = GrainBoundary([10.0, 10.0, 10.0], [cfg])
    processed = model.triple_junction_gbs[0]
    assert processed['distance_function'] is None
    assert model.triple_junction_gbs == [model.gb_configurations[0]]
    # The caller's dict is no longer mutated (fixed bug): derived fields
    # live only on GrainBoundary's internal deep copy.
    assert 'distance_function' not in cfg

  def test_unknown_gb_type_is_silently_skipped(self):
    model = GrainBoundary([10.0, 10.0, 10.0],
                          [{'type': 'not_a_real_type'}])
    assert model.vertical_gbs == []
    assert model.cylindrical_gbs == []
    assert model.triple_junction_gbs == []
    assert model.max_gb_influence == 0.0

  def test_empty_gb_list_means_no_gbs(self):
    """An *explicitly empty* list means 'no grain boundaries'"""
    model = GrainBoundary([10.0, 10.0, 10.0], [])
    assert model.gb_configurations == []
    assert model.vertical_gbs == []
    assert model.cylindrical_gbs == []
    assert model.triple_junction_gbs == []
    assert model.max_gb_influence == 0.0
    assert model.get_site_gb_region((5.0, 5.0, 5.0)) == 'bulk'
    assert model.is_site_in_grain_boundary((5.0, 5.0, 5.0)) is False

  def test_no_gbs_when_none_is_passed(self):
    """``None`` still yields the default GB set. An explicitly EMPTY list is
    the supported way to disable GBs entirely (test_empty_gb_list_means_no_gbs)."""
    model = GrainBoundary([10.0, 10.0, 10.0], None)
    assert len(model.gb_configurations) > 0


  def test_zero_width_gb_only_matches_at_the_plane(self):
    cfg = {'type': 'vertical_planar', 'orientation': 'yz', 'position': 5.0,
           'width': 0.0, 'Act_E_diff_GB': 1.0,
           'event_modifications': {'migration': {
             'region': 'inner_boundary',
             'affected_defects': ['d'], 'Act_E_diff_GB': 1.0}}}
    model = GrainBoundary([10.0, 10.0, 10.0], [cfg])
    # Derived fields live on the model's processed deep copy.
    processed = model.gb_configurations[0]
    assert processed['inner_boundary'] == 0.0 and processed['outer_boundary'] == 0.0
    assert model.get_site_gb_region((5.0, 3.0, 3.0)) == 'inner_boundary'
    assert model.get_site_gb_region((5.0 + 1e-9, 3.0, 3.0)) == 'bulk'


# =============================================================================
# Boundary detection
# =============================================================================
class TestBoundaryDetection:
  def test_planar_regions(self, planar_gb_dicts):
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]  # the *processed* copy carries the injected boundaries
    assert model.get_site_gb_region(
      (3.0, gb['position'], 0.0)) == 'inner_boundary'
    d_inner = gb['inner_boundary']
    d_outer = gb['outer_boundary']
    mid = (d_inner + d_outer) / 2.0
    assert model.get_site_gb_region(
      (3.0, gb['position'] + mid, 0.0)) == 'outer_boundary'
    assert model.get_site_gb_region(
      (3.0, gb['position'] + d_outer + 1.0, 0.0)) == 'bulk'
    # the inner edge itself belongs to the inner region
    assert model.get_site_gb_region(
      (3.0, gb['position'] + d_inner, 0.0)) == 'inner_boundary'
    # the outer edge itself belongs to the outer region
    assert model.get_site_gb_region(
      (3.0, gb['position'] + d_outer, 0.0)) == 'outer_boundary'

  def test_cylindrical_regions(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    assert model.get_site_gb_region((cx, cy, 0.0)) == 'inner_boundary'
    mid = (gb['radius'] + gb['outer_radius']) / 2.0
    assert model.get_site_gb_region((cx + mid, cy, 0.0)) == 'outer_boundary'
    assert model.get_site_gb_region(
      (cx + gb['outer_radius'] + 1.0, cy, 0.0)) == 'bulk'

  def test_is_site_in_grain_boundary_wrapper(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    assert model.is_site_in_grain_boundary((cx, cy, 0.0)) is True
    assert model.is_site_in_grain_boundary(
      (cx + gb['outer_radius'] + 5.0, cy, 0.0)) is False


  def test_overlapping_gbs_resolve_to_innermost_region(self):
    """Overlapping GBs resolve to the innermost matching region
    ('inner_boundary' outranks 'outer_boundary'), independent of list order."""
    wide = {'type': 'vertical_planar', 'orientation': 'yz', 'position': 5.0,
            'width': 2.0, 'outer_width': 4.0, 'Act_E_diff_GB': 1.0,
            'event_modifications': {}}
    narrow = {'type': 'vertical_planar', 'orientation': 'xz', 'position': 5.0,
              'width': 2.0, 'Act_E_diff_GB': 1.0, 'event_modifications': {}}
    # Site (6.5, 5.7, 0.0) relative to the two GBs above:
    #   wide  (yz): |x-5.0| = 1.5 -> OUTER band (1.0 < 1.5 <= 2.0)
    #   narrow(xz): |y-5.0| = 0.7 -> INNER core (0.7 <= 1.0)
    site = (6.5, 5.7, 0.0)

    # The innermost match must win whichever GB is listed first. Under the
    # old first-match behavior the [wide, narrow] ordering returned
    # 'outer_boundary'; the fix makes both orderings agree.
    outer_first = GrainBoundary([10.0, 10.0, 10.0], [wide, narrow])
    inner_first = GrainBoundary([10.0, 10.0, 10.0], [narrow, wide])
    assert outer_first.get_site_gb_region(site) == 'inner_boundary'
    assert inner_first.get_site_gb_region(site) == 'inner_boundary'

    # Both GBs matching inner still resolve to inner.
    assert outer_first.get_site_gb_region((5.0, 5.0, 0.0)) == 'inner_boundary'

    # Outside every band -> bulk.
    assert outer_first.get_site_gb_region((0.5, 0.5, 0.0)) == 'bulk'


# =============================================================================
# Event modifications (modify_act_energy_GB and helpers)
# =============================================================================
def make_mod_site(defect_name, position, base_mig=0.5):
  """Minimal site stand-in exposing what modify_act_energy_GB reads."""
  return SimpleNamespace(
    applicable_defects=[defect_name],
    position=position,
    Act_E_dict={defect_name: {'E_mig': {'Plane': base_mig}}},
  )


def mig_pathways(direction, distance):
  """Migration pathway dict in the format built by simulator.py."""
  return {'Plane': {'direction': direction, 'distance': distance}}


class TestEventModifications:
  def test_linear_model_reduces_destination_barrier(self, cylindrical_gb_dicts):
    """VCM cylindrical YAML: linear model, defect-scoped reduction."""
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    entry = gb['mig_cfg'][0]
    defect = entry['affected_defects'][0]
    diff = entry['Act_E_diff_GB']
    inner, outer = gb['inner_boundary'], gb['outer_boundary']

    # Source deep in the bulk, destination half-way down the linear ramp
    # (destination = position - direction * distance, per _calculate_dest_pos)
    mid = (inner + outer) / 2.0
    site = make_mod_site(defect, (cx + outer + 1.0, cy, 0.0))
    model.modify_act_energy_GB(
      site, mig_pathways((-1.0, 0.0, 0.0), outer + 1.0 - mid),
      {defect: {}}, {})
    # Linear model: reduction = diff * (outer - d) / (outer - inner) at d=mid
    expected_reduction = diff * (outer - mid) / (outer - inner)
    assert site.Act_E_dict[defect]['E_mig']['Plane'] == \
      pytest.approx(0.5 - expected_reduction)


  def test_barrier_reaches_the_full_difference_at_the_gb_core(
      self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    entry = gb['mig_cfg'][0]
    defect = entry['affected_defects'][0]
    diff = entry['Act_E_diff_GB']
    outer = gb['outer_radius']

    site = make_mod_site(defect, (cx + outer + 1.0, cy, 0.0))
    model.modify_act_energy_GB(
      site, mig_pathways((-1.0, 0.0, 0.0), outer + 1.0),  # land on the axis
      {defect: {}}, {})
    # destination distance 0 <= inner -> full reduction
    assert site.Act_E_dict[defect]['E_mig']['Plane'] == \
      pytest.approx(0.5 - diff)

  def test_unaffected_defect_keeps_its_barrier(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    far = gb['outer_radius'] + 1.0
    site = make_mod_site('some_other_defect', (cx + far, cy, 0.0))
    model.modify_act_energy_GB(
      site, mig_pathways((-1.0, 0.0, 0.0), far),
      {'some_other_defect': {}}, {})
    assert site.Act_E_dict['some_other_defect']['E_mig']['Plane'] == \
      pytest.approx(0.5)

  def test_bulk_to_bulk_hop_is_untouched(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    far = gb['outer_radius'] + 5.0
    site = make_mod_site('oxygen_interstitial', (cx + far, cy, 0.0))
    model.modify_act_energy_GB(
      site, mig_pathways((0.0, 1.0, 0.0), 1.0),  # tangential, stays in bulk
      {'oxygen_interstitial': {}}, {})
    assert site.Act_E_dict['oxygen_interstitial']['E_mig']['Plane'] == \
      pytest.approx(0.5)

  def test_direction_dependent_entering_barrier_from_real_json(
      self, planar_gb_dicts, pzt_h_act_e):
    """PZT YAML + PZT JSON: outer->inner hop gets the JSON's enter barrier."""
    barriers = pzt_h_act_e['gb_direction_barriers']
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    y0 = gb['position']
    d_outer = gb['outer_boundary']

    site = make_mod_site('hydrogen_interstitial', (3.0, y0 + d_outer, 0.0))
    site.Act_E_dict['hydrogen_interstitial'] = dict(pzt_h_act_e)
    # Source at the outer edge, destination at the GB plane (inner)
    model.modify_act_energy_GB(
      site, mig_pathways((0.0, -1.0, 0.0), d_outer),
      {'hydrogen_interstitial': {}}, {})
    assert site.Act_E_dict['hydrogen_interstitial']['E_mig']['Plane'] == \
      pytest.approx(barriers['outer_boundary_to_inner_boundary'])

  def test_direction_dependent_leaving_barrier_from_real_json(
      self, planar_gb_dicts, pzt_h_act_e):
    barriers = pzt_h_act_e['gb_direction_barriers']
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    y0 = gb['position']

    site = make_mod_site('hydrogen_interstitial', (3.0, y0, 0.0))
    site.Act_E_dict['hydrogen_interstitial'] = dict(pzt_h_act_e)
    # Source on the GB plane, destination at the outer edge
    model.modify_act_energy_GB(
      site, mig_pathways((0.0, 1.0, 0.0), gb['outer_boundary']),
      {'hydrogen_interstitial': {}}, {})
    assert site.Act_E_dict['hydrogen_interstitial']['E_mig']['Plane'] == \
      pytest.approx(barriers['inner_boundary_to_outer_boundary'])

  def test_direction_dependent_missing_json_barriers_raise(
      self, planar_gb_dicts):
    cfgs = [dict(gb) for gb in planar_gb_dicts]
    model = GrainBoundary([50.0, 54.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    site = make_mod_site('hydrogen_interstitial',
                         (3.0, gb['position'] + gb['outer_boundary'], 0.0))
    with pytest.raises(ValueError, match='gb_direction_barriers'):
      model.modify_act_energy_GB(
        site, mig_pathways((0.0, -1.0, 0.0), gb['outer_boundary']),
        {'hydrogen_interstitial': {}}, {})

  def test_reaction_reduction_applied_to_site_energies(
      self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    rxn_entry = gb['rxn_cfg'][0]
    rxn_name = rxn_entry['affected_reactions'][0]
    diff = rxn_entry['Act_E_diff_GB']
    inner, outer = gb['inner_boundary'], gb['outer_boundary']
    base_e = 3.0

    # Site sitting in the outer region: reaction barrier is lowered
    mid = (inner + outer) / 2.0
    site = make_mod_site('oxygen_interstitial', (cx + mid, cy, 0.0))
    site.Act_E_dict['oxygen_interstitial'][rxn_name] = base_e
    model.modify_act_energy_GB(
      site, {}, {'oxygen_interstitial': {}}, {rxn_name: {'name': rxn_name}})
    expected = diff * (outer - mid) / (outer - inner)
    assert site.Act_E_dict['oxygen_interstitial'][rxn_name] == \
      pytest.approx(base_e - expected)

  def test_site_without_applicable_defects_is_untouched(
      self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    site = make_mod_site('oxygen_interstitial', (0.0, 0.0, 0.0))
    site.applicable_defects = []
    model.modify_act_energy_GB(site, {}, {}, {})  # must not raise
    assert site.Act_E_dict['oxygen_interstitial']['E_mig']['Plane'] == \
      pytest.approx(0.5)

  def test_gb_reduction_helper_regions(self, cylindrical_gb_dicts):
    cfgs = [dict(gb) for gb in cylindrical_gb_dicts]
    model = GrainBoundary([50.0, 50.0, 6.0], cfgs)
    gb = model.gb_configurations[0]
    cx, cy = gb['center']
    entry = gb['mig_cfg'][0]
    defect = entry['affected_defects'][0]
    diff = entry['Act_E_diff_GB']

    inner = model._get_gb_reduction_for_site(
      (cx, cy, 0.0), gb, 'migration', defect_name=defect)
    assert inner == pytest.approx(diff)
    outside = model._get_gb_reduction_for_site(
      (cx + gb['outer_radius'] + 1.0, cy, 0.0), gb, 'migration',
      defect_name=defect)
    assert outside == 0.0
    wrong_defect = model._get_gb_reduction_for_site(
      (cx, cy, 0.0), gb, 'migration', defect_name='unaffected_defect')
    assert wrong_defect == 0.0
    wrong_event = model._get_gb_reduction_for_site(
      (cx, cy, 0.0), gb, 'generation', defect_name=defect)
    assert wrong_event == 0.0




