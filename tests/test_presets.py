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
