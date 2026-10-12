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


def test_registry_covers_every_enum_member():
  from kinetix.termination import registry as _registry

  assert set(_registry._REGISTRY) == set(StoppingConditionKey.valid_options())


def test_pzt_preset_enables_early_stopping():
  config = SimulationConfig.from_yaml(
    get_parameters_root() / "presets" / PZT_PRESET_NAME
  )
  es = config.settings.early_stopping
  assert es is not None and es.enabled is True
  assert es.condition == StoppingConditionKey.ALL_VO_PASSIVATED.value
  cond = get_stopping_condition(es)
  assert isinstance(cond, AllVOPassivated)


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
