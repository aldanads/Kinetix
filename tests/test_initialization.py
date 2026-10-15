# tests/test_initialization.py
"""Regression tests for the guarded parameter-sweep overrides in
``kinetix.initialization``.

The HPC PZT batch workflow passes a ``params`` dict (see
``kinetix.cli.get_parameters_from_sim_id``) that overrides the oxygen-vacancy
concentration, the temperature and the hydrogen-generation barrier. The old
unguarded implementation of those overrides crashed the test suite with
``KeyError: 'PZT'`` whenever a non-PZT preset (e.g. the VCM HfO2 mock, whose
activation-energy table is keyed ``'VCM'``) or no sweep dict at all reached
``initialization()`` — most visibly in ``tests/test_percolation.py``.

These tests pin the guard contract:

* Unit level (fast, no lattice build): ``_apply_parameter_sweep_overrides``
  with ``params`` ``None`` / ``{}`` / partial / full, on REAL presets and
  REAL activation-energy files through the production loaders.
* End-to-end: ``initialization()`` itself runs to completion with
  ``params=None`` and ``params={}`` on the VCM mock preset (non-PZT).

The companion scenario — non-PZT preset *with* a full params dict — is
exercised end-to-end by the ``tests/test_percolation.py`` module fixture
(``initialization(0, {...}, "VCM_mock.yaml")``), i.e. the exact call that
used to raise in CI; here it is pinned at unit level
(``test_non_pzt_material_is_never_overridden``).

Run from the repository root::

    pytest tests/test_initialization.py -v
"""
from pathlib import Path

import pytest

from kinetix.configs.config_loader import load_activation_energies
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.initialization import (
    _apply_parameter_sweep_overrides,
    initialization,
)

# Same path-resolution pattern as tests/test_percolation.py / test_site.py
PARAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "parameters"

VCM_PRESET = PARAMS_DIR / "presets" / "VCM_mock.yaml"        # non-PZT (technology VCM)
PZT_PRESET = PARAMS_DIR / "presets" / "PZT_ZrTi_PbO3_2.yaml"  # HPC sweep target

# Full sweep dict, as produced by kinetix.cli.get_parameters_from_sim_id
FULL_PARAMS = {
    "vo_initial_concentration": 2.0e-2,
    "temperature": 373.0,
    "h_generation": 0.50,
}


def _snapshot(config, ae_data):
    """Values the sweep could touch (None when the target does not exist)."""
    vo = config.defects.defects.get("oxygen_vacancy")
    pzt_entries = ae_data.get("PZT")
    e_gen = None
    if (isinstance(pzt_entries, list) and len(pzt_entries) > 1
            and isinstance(pzt_entries[1], dict)
            and isinstance(pzt_entries[1].get("activation_energies"), dict)):
        e_gen = pzt_entries[1]["activation_energies"].get("E_gen_defect")
    return {
        "vo_bulk": vo.initial_concentration_bulk if vo else None,
        "vo_gb": vo.initial_concentration_GB if vo else None,
        "temperature": config.experimental.temperature,
        "e_gen_defect": e_gen,
    }


class TestParameterSweepGuards:
    """``_apply_parameter_sweep_overrides`` guard contract (fast)."""

    def test_none_params_is_a_noop(self):
        """params=None must not crash and must change nothing (PZT preset)."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        ae_data = load_activation_energies(PZT_PRESET, config.settings)
        before = _snapshot(config, ae_data)

        _apply_parameter_sweep_overrides(config, None, ae_data)

        assert _snapshot(config, ae_data) == before

    def test_empty_params_is_a_noop(self):
        """params={} must not crash and must change nothing (PZT preset)."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        ae_data = load_activation_energies(PZT_PRESET, config.settings)
        before = _snapshot(config, ae_data)

        _apply_parameter_sweep_overrides(config, {}, ae_data)

        assert _snapshot(config, ae_data) == before

    def test_non_pzt_material_is_never_overridden(self):
        """The CI regression: VCM preset + full params dict used to raise
        KeyError('PZT') at the old ae_data['PZT'] override line. Now every
        override must be silently skipped and the config left untouched."""
        config = SimulationConfig.from_yaml(VCM_PRESET)
        ae_data = load_activation_energies(VCM_PRESET, config.settings)
        assert "PZT" not in ae_data  # precondition: non-PZT table
        before = _snapshot(config, ae_data)

        _apply_parameter_sweep_overrides(config, dict(FULL_PARAMS), ae_data)

        assert _snapshot(config, ae_data) == before

    def test_full_params_apply_on_pzt(self):
        """The PZT HPC workflow: with the correct params dict all three
        overrides must be applied exactly as the original unguarded lines."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        ae_data = load_activation_energies(PZT_PRESET, config.settings)
        assert ae_data["PZT"][1]["specie"] == "H"  # precondition: entry 1 = H

        _apply_parameter_sweep_overrides(config, dict(FULL_PARAMS), ae_data)

        vo = config.defects.defects["oxygen_vacancy"]
        assert vo.initial_concentration_bulk == FULL_PARAMS["vo_initial_concentration"]
        assert vo.initial_concentration_GB == FULL_PARAMS["vo_initial_concentration"]
        assert config.experimental.temperature == FULL_PARAMS["temperature"]
        assert (ae_data["PZT"][1]["activation_energies"]["E_gen_defect"]
                == FULL_PARAMS["h_generation"])

    def test_partial_params_apply_only_present_keys(self):
        """A params dict with a single key must apply only that override."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        ae_data = load_activation_energies(PZT_PRESET, config.settings)
        before = _snapshot(config, ae_data)

        _apply_parameter_sweep_overrides(config, {"temperature": 350.0}, ae_data)

        after = _snapshot(config, ae_data)
        assert after["temperature"] == 350.0
        assert after["vo_bulk"] == before["vo_bulk"]
        assert after["vo_gb"] == before["vo_gb"]
        assert after["e_gen_defect"] == before["e_gen_defect"]

    def test_missing_oxygen_vacancy_target_is_skipped(self):
        """A preset without an oxygen_vacancy defect must not crash the
        vo override; the other two overrides still apply."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        del config.defects.defects["oxygen_vacancy"]
        ae_data = load_activation_energies(PZT_PRESET, config.settings)

        _apply_parameter_sweep_overrides(config, dict(FULL_PARAMS), ae_data)

        assert config.experimental.temperature == FULL_PARAMS["temperature"]
        assert (ae_data["PZT"][1]["activation_energies"]["E_gen_defect"]
                == FULL_PARAMS["h_generation"])

    def test_malformed_pzt_entry_is_skipped(self):
        """An ae_data['PZT'] list without the expected second entry must not
        crash the h_generation override; the other two overrides still apply."""
        config = SimulationConfig.from_yaml(PZT_PRESET)
        ae_data = load_activation_energies(PZT_PRESET, config.settings)
        ae_data["PZT"] = ae_data["PZT"][:1]  # drop the 'H' entry

        _apply_parameter_sweep_overrides(config, dict(FULL_PARAMS), ae_data)

        vo = config.defects.defects["oxygen_vacancy"]
        assert vo.initial_concentration_bulk == FULL_PARAMS["vo_initial_concentration"]
        assert config.experimental.temperature == FULL_PARAMS["temperature"]


class TestInitializationWithoutSweepParams:
    """End-to-end: ``initialization()`` runs clean without sweep params.

    Uses the non-PZT VCM mock preset (the same lattice as the golden trace,
    loaded from the tracked grid cache), so no Materials Project access and
    no new lattice build is needed.
    """

    @pytest.mark.parametrize("params", [None, {}], ids=["params-none", "params-empty"])
    def test_vcm_initialization_runs_without_sweep_params(self, params):
        simulator, *_ = initialization(0, params, "VCM_mock.yaml")
        assert simulator is not None
        assert len(simulator.grid_crystal) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
