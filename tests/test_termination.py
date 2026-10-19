# tests/test_termination.py
"""Unit + integration tests for kinetix.termination (early stopping).

Conventions: 2-space indent (tests), real `EarlyStoppingConfig` objects for
the registry/config tests, and the committed miniature PZT grid for the
integration tests (same harness `test_pzt_golden_trace.py` uses).
"""

from __future__ import annotations

import numpy as np
import pytest

from kinetix.configs.config_loader import (
  get_parameters_root,
  load_activation_energies,
)
from kinetix.configs.simulation_config import (
  EarlyStoppingConfig,
  SimulationConfig,
)
from kinetix.initialization import _process_activation_energies
from kinetix.termination import (
  AllVOPassivated,
  GBVOPassivated,
  StoppingConditionKey,
  get_stopping_condition,
)
from tests.test_pzt_golden_trace import (
  PRESET_NAME as PZT_PRESET_NAME,
  UPDATE_HINT,
  _build_crystal,
  _fixture_grid_source,
  _install_grid,
  _mini_config,
  _seed_v_o,
)


# =============================================================================
# Lightweight doubles (no lattice build needed)
# =============================================================================

class _FakeDefect:
  def __init__(self, specie, level):
    self.chemical_specie = specie
    self.passivation_level = level


class _FakeSite:
  def __init__(self, specie, level):
    self.defect = _FakeDefect(specie, level)


class _FakeSimulator:
  def __init__(self, entries):
    self.grid_crystal = {
      i: _FakeSite(specie, level) for i, (specie, level) in enumerate(entries)
    }
    # Mirror the production occupancy invariant: every occupied site is
    # listed in active_event_sites (events._track_occupancy_update).
    self.active_event_sites = list(self.grid_crystal)


class _FakeGBModel:
  """GrainBoundary double: position -> region lookup, no geometry needed."""
  def __init__(self, region_by_position, gb_configurations=None):
    # A real GrainBoundary always carries configurations; [] models
    # "GB configured but empty/disabled".
    self.gb_configurations = (
      [{}] if gb_configurations is None else gb_configurations
    )
    self._region_by_position = region_by_position

  def get_site_gb_region(self, position):
    return self._region_by_position.get(tuple(position), "bulk")


class _NoConfigsGBModel:
  """gb_model double exposing the region API but no ``gb_configurations``."""
  def get_site_gb_region(self, position):
    return "inner_boundary"


_AUTO_GB = object()


def _fake_gb_simulator(entries, gb_model=_AUTO_GB):
  """Simulator double whose sites carry positions plus a fake GB model.

  entries: (specie, level, region) triples; site i gets position (i, 0, 0)
  and the default GB model maps that position to ``region``. Pass
  ``gb_model=None`` (or any object) to model missing/no GB configuration.
  """
  entries = list(entries)
  sim = _FakeSimulator([(specie, level) for specie, level, _region in entries])
  region_by_position = {}
  for i, site in sim.grid_crystal.items():
    site.position = (i, 0, 0)
    region_by_position[site.position] = entries[i][2]
  if gb_model is _AUTO_GB:
    gb_model = _FakeGBModel(region_by_position)
  sim.gb_model = gb_model
  return sim


# =============================================================================
# Unit: AllVOPassivated predicate
# =============================================================================

def test_all_vo_passivated_false_when_vacancies_remain():
  sim = _FakeSimulator([("V_O", 0), ("V_O", 1), ("O", 0), ("Empty", 0)])
  assert AllVOPassivated().should_stop(sim) is False


def test_all_vo_passivated_true_when_all_passivated():
  sim = _FakeSimulator([("V_O", 1), ("V_O", 1), ("O", 0), ("H", 0)])
  assert AllVOPassivated().should_stop(sim) is True


def test_all_vo_passivated_threshold_fraction():
  entries = [("V_O", 0), ("V_O", 1), ("V_O", 1), ("V_O", 1)]  # 75% passivated
  assert AllVOPassivated(threshold=0.75).should_stop(_FakeSimulator(entries)) is True
  assert AllVOPassivated(threshold=0.76).should_stop(_FakeSimulator(entries)) is False
  assert AllVOPassivated(threshold=1.0).should_stop(_FakeSimulator(entries)) is False


def test_all_vo_passivated_empty_lattice_does_not_stop():
  # No vacancies at all: nothing has saturated, so the run must continue.
  sim = _FakeSimulator([("O", 0), ("Empty", 0), ("H", 0)])
  assert AllVOPassivated().should_stop(sim) is False


def test_all_vo_passivated_rejects_bad_threshold():
  with pytest.raises(ValueError):
    AllVOPassivated(threshold=0.0)
  with pytest.raises(ValueError):
    AllVOPassivated(threshold=1.5)


def test_all_vo_passivated_describe_and_parameters():
  cond = AllVOPassivated(threshold=0.95)
  assert "all_vo_passivated" in cond.describe()
  assert cond.parameters()["threshold"] == pytest.approx(0.95)


# =============================================================================
# Unit: GBVOPassivated predicate (GB-region-filtered V_O saturation)
# =============================================================================

def test_gb_vo_passivated_counts_only_inner_boundary_by_default():
  # Core V_O passivated; outer + bulk V_O unpassivated.
  entries = [
    ("V_O", 1, "inner_boundary"),
    ("V_O", 0, "outer_boundary"),
    ("V_O", 0, "bulk"),
  ]
  # 1/1 inner V_O passivated -> stop; the outer/bulk vacancies must not
  # dilute the fraction (they are outside the default regions).
  assert GBVOPassivated().should_stop(_fake_gb_simulator(entries)) is True
  # Flip it: only the core V_O is unpassivated -> must not stop, even though
  # every outer/bulk V_O is passivated (they are never counted).
  entries = [
    ("V_O", 0, "inner_boundary"),
    ("V_O", 1, "outer_boundary"),
    ("V_O", 1, "bulk"),
  ]
  assert GBVOPassivated().should_stop(_fake_gb_simulator(entries)) is False


def test_gb_vo_passivated_region_filter_includes_outer_boundary():
  entries = [
    ("V_O", 1, "inner_boundary"),
    ("V_O", 0, "outer_boundary"),
    ("V_O", 0, "bulk"),
  ]
  both = GBVOPassivated(regions=["inner_boundary", "outer_boundary"])
  # Outer now counts: 1/2 passivated -> below the default threshold, whereas
  # the inner-only default (1/1) stops. Bulk is excluded either way.
  assert both.should_stop(_fake_gb_simulator(entries)) is False
  assert GBVOPassivated().should_stop(_fake_gb_simulator(entries)) is True
  # With the filter, saturating inner + outer stops even with bulk left over.
  entries = [
    ("V_O", 1, "inner_boundary"),
    ("V_O", 1, "outer_boundary"),
    ("V_O", 0, "bulk"),
  ]
  assert both.should_stop(_fake_gb_simulator(entries)) is True


def test_gb_vo_passivated_vacuous_when_no_gb_vacancies():
  # V_O exist but none in the requested region -> vacuous count -> False,
  # even though every one of them is passivated.
  entries = [
    ("V_O", 1, "outer_boundary"),
    ("V_O", 1, "bulk"),
    ("O", 1, "inner_boundary"),
  ]
  assert GBVOPassivated().should_stop(_fake_gb_simulator(entries)) is False
  # No vacancies anywhere at all: the GB subset guard fires the same way.
  assert GBVOPassivated().should_stop(_fake_gb_simulator([("O", 0, "bulk")])) is False


def test_gb_vo_passivated_no_gb_model_returns_false():
  entries = [("V_O", 1, "inner_boundary")]  # fully saturated ...
  # ... but no GB configured in any of its shapes -> False, never a crash:
  # gb_model explicitly None.
  sim = _fake_gb_simulator(entries, gb_model=None)
  assert GBVOPassivated().should_stop(sim) is False
  # gb_model attribute absent entirely (plain _FakeSimulator has none).
  assert GBVOPassivated().should_stop(_FakeSimulator([("V_O", 1)])) is False
  # gb_model present but with no configurations.
  sim = _fake_gb_simulator(
    entries, gb_model=_FakeGBModel({}, gb_configurations=[])
  )
  assert GBVOPassivated().should_stop(sim) is False
  # gb_model without a gb_configurations attribute at all.
  sim = _fake_gb_simulator(entries, gb_model=_NoConfigsGBModel())
  assert GBVOPassivated().should_stop(sim) is False


def test_gb_vo_passivated_threshold_fraction():
  entries = [
    ("V_O", 1, "inner_boundary"),
    ("V_O", 0, "inner_boundary"),
    ("V_O", 0, "bulk"),  # never counted
  ]
  # 50% of the GB V_O passivated: threshold 0.5 triggers, 0.51 does not.
  assert GBVOPassivated(threshold=0.5).should_stop(
    _fake_gb_simulator(entries)
  ) is True
  assert GBVOPassivated(threshold=0.51).should_stop(
    _fake_gb_simulator(entries)
  ) is False
  assert GBVOPassivated(threshold=1.0).should_stop(
    _fake_gb_simulator(entries)
  ) is False


def test_gb_vo_passivated_rejects_bad_threshold():
  with pytest.raises(ValueError):
    GBVOPassivated(threshold=0.0)
  with pytest.raises(ValueError):
    GBVOPassivated(threshold=1.5)


def test_gb_vo_passivated_describe_and_parameters():
  cond = GBVOPassivated(threshold=0.95)
  assert "gb_vo_passivated" in cond.describe()
  assert "inner_boundary" in cond.describe()
  params = cond.parameters()
  assert params["threshold"] == pytest.approx(0.95)
  assert params["regions"] == ["inner_boundary"]


# =============================================================================
# Registry + config validation
# =============================================================================

def test_get_stopping_condition_none_and_disabled():
  assert get_stopping_condition(None) is None
  disabled = EarlyStoppingConfig(enabled=False, condition="all_vo_passivated")
  assert get_stopping_condition(disabled) is None


def test_get_stopping_condition_unknown_keyword_lists_options():
  cfg = EarlyStoppingConfig(enabled=True, condition="bogus", parameters={})
  with pytest.raises(ValueError, match="all_vo_passivated"):
    get_stopping_condition(cfg)


def test_get_stopping_condition_enabled_without_keyword():
  cfg = EarlyStoppingConfig(enabled=True, condition=None)
  with pytest.raises(ValueError, match="condition"):
    get_stopping_condition(cfg)


def test_config_from_dict_validates_keyword_at_load():
  with pytest.raises(ValueError, match="all_vo_passivated"):
    EarlyStoppingConfig.from_dict({"enabled": True, "condition": "bogus"})


def test_config_from_dict_carries_extra_keys_as_parameters():
  cfg = EarlyStoppingConfig.from_dict(
    {"enabled": True, "condition": "all_vo_passivated", "threshold": 0.95}
  )
  assert cfg.enabled is True
  assert cfg.condition == "all_vo_passivated"
  assert cfg.parameters == {"threshold": 0.95}
  cond = get_stopping_condition(cfg)
  assert isinstance(cond, AllVOPassivated)
  assert cond.parameters()["threshold"] == pytest.approx(0.95)


def test_get_stopping_condition_gb_vo_passivated_keyword():
  cfg = EarlyStoppingConfig(
    enabled=True,
    condition="gb_vo_passivated",
    parameters={"threshold": 0.8,
                "regions": ["inner_boundary", "outer_boundary"]},
  )
  cond = get_stopping_condition(cfg)
  assert isinstance(cond, GBVOPassivated)
  assert cond.parameters()["threshold"] == pytest.approx(0.8)
  assert cond.parameters()["regions"] == ["inner_boundary", "outer_boundary"]


def test_registry_covers_every_enum_member():
  from kinetix.termination import registry as _registry

  assert set(_registry._REGISTRY) == set(StoppingConditionKey.valid_options())


def test_pzt_preset_enables_early_stopping():
  config = SimulationConfig.from_yaml(
    get_parameters_root() / "presets" / PZT_PRESET_NAME
  )
  es = config.settings.early_stopping
  assert es is not None and es.enabled is True
  assert es.condition == StoppingConditionKey.GB_VO_PASSIVATED.value
  cond = get_stopping_condition(es)
  assert isinstance(cond, GBVOPassivated)
  assert cond.parameters()["threshold"] == pytest.approx(1.0)
  assert cond.parameters()["regions"] == ["inner_boundary"]


def test_presets_without_block_stay_disabled():
  config = SimulationConfig.from_yaml(
    get_parameters_root() / "presets" / "VCM_mock.yaml"
  )
  assert config.settings.early_stopping is None
  assert get_stopping_condition(config.settings.early_stopping) is None


# =============================================================================
# Integration: real miniature PZT grid through the production path
# =============================================================================

@pytest.fixture(scope="module")
def pzt_crystal():
  if _fixture_grid_source() is None:
    pytest.skip(f"miniature PZT grid fixture missing; {UPDATE_HINT}")
  config = _mini_config()
  defects = config.defects.to_dict()
  ae_data = load_activation_energies(
    get_parameters_root() / "presets" / PZT_PRESET_NAME, config.settings
  )
  act_e = _process_activation_energies(
    defects, ae_data, config.settings.technology
  )
  _install_grid()
  crystal = _build_crystal(config, defects, act_e)
  _seed_v_o(crystal)
  return crystal


def _vo_sites(crystal):
  return [
    site for site in crystal.grid_crystal.values()
    if site.defect.chemical_specie == "V_O"
  ]


def test_integration_gb_condition_counts_only_gb_core_vo(pzt_crystal):
  # Runs FIRST in this section (module-scoped fixture is fresh: `_seed_v_o`
  # has just introduced 2 core + 2 bulk V_O) and leaves every V_O
  # unpassivated, so the tests that follow see a clean state.
  cond = GBVOPassivated()
  vo = _vo_sites(pzt_crystal)
  assert vo, "expected seeded V_O on the miniature grid"
  inner = [
    site for site in vo
    if pzt_crystal.gb_model.get_site_gb_region(tuple(site.position))
    == "inner_boundary"
  ]
  assert inner, "expected seeded V_O in the GB core (_seed_v_o picks 2)"
  # Fresh state: nothing passivated anywhere -> nothing stops.
  for site in vo:
    site.defect.passivation_level = 0
  assert cond.should_stop(pzt_crystal) is False
  # Saturate the GB core ONLY: the GB condition stops while bulk V_O are
  # still unpassivated (the global condition must keep running).
  for site in inner:
    site.defect.passivation_level = 1
  assert cond.should_stop(pzt_crystal) is True
  assert AllVOPassivated().should_stop(pzt_crystal) is False
  # Leave every V_O unpassivated for the tests that follow.
  for site in vo:
    site.defect.passivation_level = 0
  assert cond.should_stop(pzt_crystal) is False


def test_integration_vo_saturation_triggers_stop(pzt_crystal):
  cond = AllVOPassivated()
  vo = _vo_sites(pzt_crystal)
  assert vo, "expected seeded V_O on the miniature grid"
  # Completeness pin: every V_O in the full grid is in active_event_sites,
  # so the O(N_active) scan cannot miss a vacancy on production state.
  full_vo = {
    idx for idx, site in pzt_crystal.grid_crystal.items()
    if site.defect.chemical_specie == "V_O"
  }
  listed_vo = {
    idx for idx in pzt_crystal.active_event_sites
    if pzt_crystal.grid_crystal[idx].defect.chemical_specie == "V_O"
  }
  assert listed_vo == full_vo
  assert any(site.defect.passivation_level == 0 for site in vo)
  assert cond.should_stop(pzt_crystal) is False
  # Saturate exactly as the passivation reaction product carries it.
  for site in vo:
    site.defect.passivation_level = 1
  assert cond.should_stop(pzt_crystal) is True


def test_integration_dump_cadence_loop_terminates_early(pzt_crystal):
  cond = AllVOPassivated()
  # Reset to unsaturated (idempotent regardless of test order).
  vo = _vo_sites(pzt_crystal)
  assert vo, "expected seeded V_O on the miniature grid"
  for site in vo:
    site.defect.passivation_level = 0
  assert cond.should_stop(pzt_crystal) is False

  # Short live simulation: real kMC steps, checked at dump cadence.
  rng = np.random.default_rng(42)
  max_snapshots = 25
  stopped_at = None
  for j in range(1, max_snapshots + 1):
    pzt_crystal.last_field_solve_time = pzt_crystal.time
    pzt_crystal.step_kmc(rng)
    if j == 3:
      # Saturation reached mid-run: every V_O passivated.
      for site in vo:
        site.defect.passivation_level = 1
    if cond.should_stop(pzt_crystal):
      stopped_at = j
      break
  assert stopped_at == 3
  assert stopped_at < max_snapshots
