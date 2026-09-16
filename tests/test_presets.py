"""Ensure every shipped preset loads through the typed config pipeline.

Parametrizes over all YAML files in ``data/parameters/presets/`` and loads
each with :meth:`SimulationConfig.from_yaml`. If any preset raises
(``ConfigValidationError`` or any other exception), the test fails with the
preset name and error message visible in the output, so schema drift in a
shipped preset is caught here instead of at simulation time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kinetix.configs.simulation_config import SimulationConfig
from kinetix.configs.electrical_config import VoltageMode

PRESETS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters" / "presets"
PRESET_FILES = sorted(PRESETS_DIR.glob("*.yaml"))


def test_preset_directory_exists_and_is_populated() -> None:
  """Guard against silently passing with zero parametrized cases."""
  assert PRESETS_DIR.is_dir(), f"presets directory not found: {PRESETS_DIR}"
  assert PRESET_FILES, f"no *.yaml presets found in {PRESETS_DIR}"


@pytest.mark.parametrize("preset_path", PRESET_FILES, ids=lambda p: p.name)
def test_preset_loads(preset_path: Path) -> None:
  """Every shipped preset must load into a valid SimulationConfig."""
  try:
    config = SimulationConfig.from_yaml(preset_path)
  except Exception as exc:  # report any failure mode verbatim
    pytest.fail(
      f"Preset '{preset_path.name}' failed to load: {type(exc).__name__}: {exc}"
    )
  assert isinstance(config, SimulationConfig)


def _check_semantics(config: SimulationConfig) -> None:
  """Raise AssertionError with a precise message on the first violation."""
  # --- Basic physical plausibility -------------------------------------
  assert config.experimental.temperature > 0, (
    f"temperature must be > 0, got {config.experimental.temperature}")
  assert config.material.selection.radius_neighbors > 0, (
    f"radius_neighbors must be > 0, got {config.material.selection.radius_neighbors}")
  # --- Structural completeness: heat -----------------------------------
  if config.heat.solve_heat:
    kappa = config.heat.thermal_conductivity
    assert kappa is not None, "solve_heat=true requires thermal_conductivity"
    assert 'kappa_dielectric' in kappa and 'kappa_metal' in kappa, (
      f"thermal_conductivity missing kappa keys: {sorted(kappa)}")
    assert kappa['kappa_metal'] > kappa['kappa_dielectric'], (
      f"kappa_metal ({kappa['kappa_metal']}) must exceed "
      f"kappa_dielectric ({kappa['kappa_dielectric']})")
  # --- Structural completeness: Poisson --------------------------------
  if config.poisson.solve_Poisson:
    cond = config.poisson.conductivity
    assert cond is not None, "solve_Poisson=true requires conductivity"
    assert 'conductive_filament' in cond and 'dielectric' in cond, (
      f"conductivity missing keys: {sorted(cond)}")
  # --- Cross-field consistency: electrical ------------------------------
  electrical = config.electrical
  if electrical is not None and electrical.voltage is not None:
    mode = electrical.voltage.mode
    if mode == VoltageMode.RAMP_CYCLE:
      num_cycles = electrical.voltage.num_cycles
      assert num_cycles is not None and num_cycles > 0, (
        f"RAMP_CYCLE requires num_cycles > 0, got {num_cycles}")
    elif mode == VoltageMode.CONSTANT:
      assert electrical.voltage.constant_voltage is not None, (
        "CONSTANT mode requires constant_voltage to be set")


@pytest.mark.parametrize("preset_path", PRESET_FILES, ids=lambda p: p.name)
def test_preset_semantic_validation(preset_path: Path) -> None:
  """Every shipped preset must pass cross-field semantic validation."""
  try:
    config = SimulationConfig.from_yaml(preset_path)
  except Exception as exc:
    pytest.fail(
      f"Preset '{preset_path.name}' failed to load: {type(exc).__name__}: {exc}"
    )
  try:
    _check_semantics(config)
  except AssertionError as exc:
    pytest.fail(f"Preset '{preset_path.name}' failed semantic validation: {exc}")

