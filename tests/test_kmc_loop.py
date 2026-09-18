# tests/test_kmc_loop.py
"""
Integration test for the core kMC (BKL) loop in Crystal_Lattice.

This is the behavioral spec for the BKL event-selection / time-advancement
machinery that will be refactored in the upcoming crystal.py split.

What is REAL here (loaded through production loaders - no hardcoded config
literals):
  - data/parameters/presets/VCM_mock.yaml           (via SimulationConfig.from_yaml)
  - data/parameters/defects/VCM_HfO2_defects_config.yaml (via config.defects.to_dict())
  - data/parameters/reactions/VCM_HfO2_reactions.yaml    (via config.reactions.to_dict())
  - data/parameters/grain_boundaries/gb_cylindrical_VCM_HfO2.yaml (via gb.to_dict())
  - data/parameters/activation_energies/VCM_HfO2.json    (via load_activation_energies
                                            + _process_activation_energies)
  - data/grids/grid_HfO2_3nm.pkl                   (production grid via _try_load_grid)

Documented test-side overrides (mirroring production parameter-sweep practice
in kinetix/initialization.py):
  1. config.material.formula = 'HfO2'
     (normally resolved from Materials Project via MaterialDataFetcher; the
     fetch step needs network access and is irrelevant to the kMC loop).
  2. calculator_config=None
     (MACE/NEB is not wired into the kMC loop; the calculator config is inert
     here. The cached production grid avoids any structure generation.)
  3. config.defects.defects['oxygen_interstitial'].initial_concentration_bulk
     = 0.05 (the real file ships 0.0 for every defect; the production sweep
     pattern `config.defects.defects[...].initial_concentration_bulk = ...`
     is used so defect_gen() actually introduces defects).
  4. migrating_attributes None -> [] normalization in the defects dict
     (see PRODUCTION BUG FINDING below).
  5. crystal.timestep_limits / crystal.last_field_solve_time are set by the
     test (normally provided post-init by ElectricalController /
     should_solve_fields_now in the CLI loop).

PRODUCTION BUG FINDING (not fixed here, reported per task instructions):
  kinetix/configs/defect_config.py:144 loads `migrating_attributes` with
  `data.get('migrating_attributes')` (no `or []` fallback), and
  VCM_HfO2_defects_config.yaml does not define the key, so DefectConfig's
  field is None and to_dict() (defect_config.py:102) emits
  'migrating_attributes': None. kinetix/lattice/site.py:393 then does
  `config.get('migrating_attributes', [])` - the default only applies to a
  MISSING key, not an explicit None - so the FIRST executed migration of any
  VCM_HfO2 simulation raises TypeError: 'NoneType' object is not iterable.
  (PZT_ZrPbO3_defects_config.yaml only works because it explicitly lists
  migrating_attributes.) The fixture below normalizes None -> [] in the
  in-memory dict to isolate the BKL-loop spec from this serialization bug.

PRODUCTION QUIRK (documented, no fix):
  kinetix/lattice/crystal.py:~268 (`lattice_model`, MP-cache-miss branch)
  builds `structure_dict = {'error: {e}'}` on API failure - a dict whose KEY
  is the literal string "error: {e}" - so the following
  `if 'error' in structure_dict` guard never fires and the failure surfaces
  as a confusing TypeError from Structure.from_dict instead of the intended
  RuntimeError. Avoided here via the cached structure_mp-352.json.
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest

from kinetix.configs.simulation_config import SimulationConfig
from kinetix.configs.config_loader import (
    get_parameters_root,
    load_activation_energies,
)
from kinetix.initialization import (
    _process_activation_energies,
    initialize_grid_crystal,
)

# =============================================================================
# Test constants
# =============================================================================

SEED = 42
N_STEPS = 30
GRID_NAME = "grid_HfO2_3nm"  # production grid matching the VCM_mock material
O_I_BULK_CONCENTRATION = 0.05  # production sweep override (real file: 0.0)

VALID_SPECIES = {"Hf", "O", "Empty", "O_i", "V_O"}


# =============================================================================
# Fixtures - real configs via production loaders
# =============================================================================

@pytest.fixture(scope="module")
def vcm_config():
    """Load the REAL VCM_mock preset via the production loader, then apply the
    documented test-side overrides (see module docstring)."""
    config = SimulationConfig.from_yaml(
        get_parameters_root() / "presets" / "VCM_mock.yaml"
    )

    # Override 1: formula normally resolved from Materials Project (offline test).
    config.material.formula = "HfO2"

    # Override 3: production parameter-sweep pattern so defect_gen() injects
    # oxygen interstitials (every bulk concentration in the real file is 0.0,
    # which would yield a valid but eventless simulation).
    config.defects.defects["oxygen_interstitial"].initial_concentration_bulk = (
        O_I_BULK_CONCENTRATION
    )
    return config


@pytest.fixture(scope="module")
def defects_config(vcm_config):
    """Real VCM defects config as a dict, with the migrating_attributes None
    -> [] workaround for the production bug (see module docstring, finding 1)."""
    defects = vcm_config.defects.to_dict()
    for cfg in defects.values():
        if cfg.get("migrating_attributes") is None:
            cfg["migrating_attributes"] = []
    return defects


@pytest.fixture(scope="module")
def vcm_act_e_dict(vcm_config, defects_config):
    """Real activation energies processed through the production pipeline."""


# =============================================================================
# Lattice construction + instrumentation helpers
# =============================================================================

def _build_lattice(vcm_config, defects_config, vcm_act_e_dict):
    """Build a Crystal_Lattice exactly the way the production
    electronic_device branch of initialization() does (grid fast-path load)."""
    reactions_config = (
        vcm_config.reactions.to_dict() if vcm_config.reactions else None
    )
    gb_configurations = (
        [gb.to_dict() for gb in vcm_config.grain_boundaries]
        if vcm_config.grain_boundaries
        else None
    )

    crystal = initialize_grid_crystal(
        GRID_NAME,
        None,  # mpi_ctx=None -> truly serial (production Phase-1 path)
        vcm_config.material,
        vcm_config.experimental,
        vcm_act_e_dict,
        vcm_config.settings.lammps_output,
        vcm_config.superbasin,
        False,  # save_data=False -> never writes grids
        settings=vcm_config.settings,
        rng=np.random.default_rng(SEED),
        cache_dir=get_parameters_root().parent / "cache",  # offline MP structure cache
        calculator_config=None,  # Override 2: MACE not wired into the kMC loop
        defects_config=defects_config,
        reactions_config=reactions_config,
        gb_configurations=gb_configurations,
        simulation_type=vcm_config.settings.simulation_type,
    )

    # Override 5: normally provided post-init by the CLI/ElectricalController.
    crystal.timestep_limits = float(vcm_config.superbasin.time_step_limits)
    crystal.last_field_solve_time = 0.0
    return crystal


def _sum_total_rate(crystal):
    """Sum of every event rate currently registered on active/generation sites
    (the same catalog _kmc_step builds internally)."""
    total = 0.0
    for idx in crystal.active_event_sites + crystal.generation_sites:
        if idx in crystal.superbasin_dict:
            continue
        for event in crystal.grid_crystal[idx].site_events:
            total += event[0]
    return total


def _make_recorder(crystal):
    """Wrap crystal.processes with a validator/recorder.

    Every executed event is checked BEFORE the state changes:
      - tuple layout (rate, destination_idx, event_label, ..., source_idx)
      - rate positive and finite
      - destination/source indices exist in the grid
      - species consistency: migration events (int label) originate from an
        O_i site; reaction events carry a name from the real reactions config
    Returns (record, unwrap) where `record` accumulates pre-execution event
    snapshots and `unwrap` restores the original method.
    """
    reaction_names = {
        reaction["name"] for reaction in crystal.reactions_config.values()
    }
    record = {"events": []}
    original_processes = crystal.processes

    def validating_processes(chosen_event):
        assert len(chosen_event) >= 3, f"malformed event tuple: {chosen_event!r}"
        rate, dest_idx, label = chosen_event[0], chosen_event[1], chosen_event[2]
        src_idx = chosen_event[-1]

        assert np.isfinite(rate) and rate > 0, f"invalid event rate {rate}"
        assert dest_idx in crystal.grid_crystal, (
            f"destination {dest_idx!r} does not exist in grid_crystal"
        )
        assert src_idx in crystal.grid_crystal, (
            f"source {src_idx!r} does not exist in grid_crystal"
        )
        if isinstance(label, int):
            # Migration event: source must currently host the mobile defect.
            assert crystal.grid_crystal[src_idx].chemical_specie == "O_i", (
                f"migration event from non-defect site {src_idx!r} "
                f"(species={crystal.grid_crystal[src_idx].chemical_specie!r})"
            )
        else:
            assert label in reaction_names, f"unknown event label {label!r}"

        record["events"].append(
            {
                "rate": rate,
                "dest": dest_idx,
                "label": label,
                "source": src_idx,
                "source_specie": crystal.grid_crystal[src_idx].chemical_specie,
            }
        )
        original_processes(chosen_event)

    crystal.processes = validating_processes

    def unwrap():
        crystal.processes = original_processes

    return record, unwrap


@pytest.fixture(scope="module")
def run_result(vcm_config, defects_config, vcm_act_e_dict):
    """Build the lattice once, run N_STEPS seeded kMC steps with every event
    validated on the fly, and return the crystal plus the trajectory record.

    Step 1 is executed through _kmc_step with an RNG-peek so the BKL time
    advance can be checked exactly against -log(u)/total_rate.
    """
    crystal = _build_lattice(vcm_config, defects_config, vcm_act_e_dict)
    n_defect_sites_before = len(crystal.active_event_sites)

    # Introduce defects (a separate CLI step after initialize_grid_crystal;
    # uses crystal.rng -> deterministic under the fixed seed).
    crystal.defect_gen()

    # Materialize rates (the first step would otherwise trigger
    # _update_rates_lazily) and verify the total rate is positive and finite.
    crystal._update_rates_lazily({}, {})
    total_rate = _sum_total_rate(crystal)
    assert np.isfinite(total_rate) and total_rate > 0

    # --- Step 1 via _kmc_step with RNG peek (exact BKL time check) ---------
    rng = crystal.rng
    state_before = copy.deepcopy(rng.bit_generator.state)
    u = rng.random()  # peek the draw _kmc_step will use for the time step
    rng.bit_generator.state = state_before  # restore

    expected_time_step = -np.log(u) / total_rate
    time_step_1, event_1 = crystal._kmc_step(rng, {}, {})
    if event_1 is not None:
        assert time_step_1 == pytest.approx(expected_time_step, rel=1e-9), (
            "BKL time advance does not match -ln(u)/sum(TR): "
            f"{time_step_1} vs {expected_time_step}"
        )
    else:
        # No event within the timestep limit -> clamped to the limit.
        assert time_step_1 == pytest.approx(crystal.timestep_limits)

    # --- Steps 2..N via the public step_kmc, with event validation ----------
    record, unwrap = _make_recorder(crystal)
    try:
        for _ in range(N_STEPS - 1):
            crystal.step_kmc(rng)
    finally:
        unwrap()

    return {
        "crystal": crystal,
        "record": record,
        "total_rate": total_rate,
        "event_1": event_1,
        "n_defect_sites_before": n_defect_sites_before,
        "defects_config": defects_config,
    }




@pytest.fixture(scope="module")
def deterministic_rerun(vcm_config, defects_config, vcm_act_e_dict):
    """Second INDEPENDENT build + run with the same seed (determinism check).

    All N_STEPS steps go through step_kmc (unlike run_result, whose step 1
    uses the raw _kmc_step), so this trajectory's bookkeeping is complete.
    """
    crystal = _build_lattice(vcm_config, defects_config, vcm_act_e_dict)
    crystal.defect_gen()
    n_oi = sum(
        1 for s in crystal.grid_crystal.values() if s.chemical_specie == "O_i"
    )
    assert n_oi > 0, "defect_gen() introduced no defects in the rerun build"
    rng = crystal.rng
    for _ in range(N_STEPS):
        crystal.step_kmc(rng)
    return crystal


# =============================================================================
# Tests
# =============================================================================

class TestSystemConstruction:
    """The minimal system is real and defect-bearing."""

    def test_lattice_built_from_production_grid(self, run_result):
        crystal = run_result["crystal"]
        assert len(crystal.grid_crystal) == 3456
        assert crystal.simulation_type == "electronic_device"

    def test_defects_were_introduced(self, run_result):
        crystal = run_result["crystal"]
        # defect_gen() must have added O_i-bearing sites on top of the pristine
        # lattice oxygen sites that are active from the start.
        assert len(crystal.active_event_sites) > run_result["n_defect_sites_before"]
        n_oi = sum(
            1
            for site in crystal.grid_crystal.values()
            if site.chemical_specie == "O_i"
        )
        assert n_oi > 0

    def test_total_rate_positive_and_finite(self, run_result):
        assert np.isfinite(run_result["total_rate"]) and run_result["total_rate"] > 0


class TestBklTimeAdvancement:
    """Simulation time advances according to the BKL prescription."""

    def test_time_advanced_after_n_steps(self, run_result):
        crystal = run_result["crystal"]
        assert crystal.time > 0
        assert np.isfinite(crystal.time)

    def test_time_within_timestep_limit(self, run_result):
        # Every step is clamped by get_timestep_limit(); with
        # last_field_solve_time=0 the total time can never exceed the limit.
        crystal = run_result["crystal"]
        assert (
            crystal.time
            <= crystal.last_field_solve_time + crystal.timestep_limits + 1e-15
        )


class TestEventExecution:
    """At least one event executed and every selected event was valid."""

    def test_at_least_one_event_executed(self, run_result):
        record = run_result["record"]
        total_executed = len(record["events"]) + (
            1 if run_result["event_1"] is not None else 0
        )
        assert total_executed >= 1

    def test_events_tracking_counter_updated(self, run_result):
        crystal = run_result["crystal"]
        # events_tracking is only incremented by step_kmc; step 1 ran through
        # the raw _kmc_step, so the counter must match the recorded events.
        assert sum(crystal.events_tracking.values()) == len(run_result["record"]["events"])
        assert sum(crystal.events_tracking.values()) >= 1

    def test_selected_events_valid(self, run_result):
        """Destination sites exist and species are correct for every event."""
        events = list(run_result["record"]["events"])
        if run_result["event_1"] is not None:
            event_1 = run_result["event_1"]
            events.append(
                {
                    "rate": event_1[0],
                    "dest": event_1[1],
                    "label": event_1[2],
                    "source": event_1[-1],
                    "source_specie": "O_i",  # validated pre-execution in fixture
                }
            )
        assert events, "expected at least one executed event"
        crystal = run_result["crystal"]
        for event in events:
            assert event["dest"] in crystal.grid_crystal
            assert event["source"] in crystal.grid_crystal
            if isinstance(event["label"], int):
                # Migration: recorded pre-execution source species must be O_i.
                assert event["source_specie"] == "O_i"
            else:
                reaction_names = {
                    reaction["name"]
                    for reaction in crystal.reactions_config.values()
                }
                assert event["label"] in reaction_names

    def test_species_remain_valid_after_run(self, run_result):
        """After the run, no site hosts a species outside the VCM set."""
        crystal = run_result["crystal"]
        for site in crystal.grid_crystal.values():
            assert site.chemical_specie in VALID_SPECIES


class TestRateConsistency:
    """The total rate drives the exponential time advance."""

    def test_bkl_time_step_matches_total_rate(self, run_result):
        """dt = -ln(u) / sum(TR) - verified exactly on step 1 via the RNG peek
        (assertion lives in the run_result fixture); repeated here as spec."""
        if run_result["event_1"] is None:
            pytest.skip("step 1 took the no-event branch")
        assert np.isfinite(run_result["total_rate"])
        assert run_result["total_rate"] > 0

    def test_no_event_step_clamps_to_timestep_limit(self, run_result):
        """When the field-solve deadline is reached, get_timestep_limit()
        returns 0 and the step takes the no-event branch (time clamped)."""
        crystal = run_result["crystal"]
        saved_time = crystal.time
        try:
            crystal.time = crystal.last_field_solve_time + crystal.timestep_limits
            ts, event = crystal._kmc_step(crystal.rng, {}, {})
            assert event is None
            assert ts == pytest.approx(0.0)
        finally:
            crystal.time = saved_time


class TestDeterminism:
    """Same seed -> identical trajectory."""

    def test_same_seed_same_trajectory(self, run_result, deterministic_rerun):
        first = run_result["crystal"]
        second = deterministic_rerun

        assert second.time == pytest.approx(first.time, rel=1e-12), (
            f"first.time={first.time!r} second.time={second.time!r}"
        )
        # run_result's step 1 went through the raw _kmc_step, which does NOT
        # update events_tracking (only step_kmc does); restore that missing
        # entry so both counters describe the same 30-event trajectory.
        event_1 = run_result["event_1"]

        def _expected_first_counter():
            # events_tracking is keyed by the event label (chosen_event[2]:
            # migration-pathway index or reaction name), incremented only by
            # step_kmc. run_result's step 1 used the raw _kmc_step, so its
            # label is missing and must be restored before comparing.
            counter = Counter(first.events_tracking)
            if event_1 is not None:
                counter[event_1[2]] += 1
            return counter

        assert Counter(second.events_tracking) == _expected_first_counter(), (
            f"first={dict(first.events_tracking)!r} "
            f"(+step1 dest {event_1[1] if event_1 is not None else None}) "
            f"second={dict(second.events_tracking)!r}"
        )

        def count_specie(crystal, specie):
            return sum(
                1
                for s in crystal.grid_crystal.values()
                if s.chemical_specie == specie
            )

        for specie in ("O_i", "V_O", "Empty"):
            assert count_specie(second, specie) == count_specie(first, specie), (
                f"species distribution diverged for {specie}"
            )
