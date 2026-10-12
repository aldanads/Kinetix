# tests/test_pzt_physics_validation.py
# -*- coding: utf-8 -*-
"""Physics validation: Boltzmann-weighted event selection in the PZT device.

These tests verify that kMC event selection *respects the relative Boltzmann
probabilities* encoded in the PZT configuration (``PZT_ZrTi_PbO3_2.yaml`` +
``PZT_ZrPbO3.json`` + ``gb_vertical_planar.yaml`` +
``PZT_reactions_H_Ovac_passivation.yaml``). Three single-defect experiments
drive the production kMC loop (``KMCSimulator.step_kmc``) and check the
*executed-event statistics* against the configured barriers:

1. ``test_gb_entry_bias`` - an H interstitial in the GB ``outer_boundary``
   region must migrate *toward* the GB core, never away from it: the
   ``direction_dependent`` GB model gives ``outer->inner`` 0.15 eV vs
   ``inner->outer`` 0.65 eV (``gb_direction_barriers`` in the JSON).
2. ``test_bulk_migration_dominates_passivation`` - an H adjacent to a V_O in
   the bulk under a synthetic downward field must migrate (field-assisted,
   floored at ``E_min_mig``) far more often than passivate the vacancy
   (0.26 eV, ``"H + V_O -> V_OH"``), moving preferentially downward.
3. ``test_gb_residence_and_escape`` - an H inside the GB core (neutral per
   the GB ``charge_state``) must essentially never escape (0.65 eV): it
   random-walks inside the core (0.15 eV), passivates a V_O there
   (0.26 eV), or stays put.

What is REAL here (production loaders end-to-end, no config literals):
the PZT preset, defects/reactions/GB YAMLs, the ``PZT_ZrPbO3.json``
activation energies (``load_activation_energies`` +
``_process_activation_energies``), and the production grid
``data/grids/grid_ZrTi(PbO3)2_10nm.pkl`` (44 574 sites, ~51x51x102 A, planar
GB at y=27 A, core +/-3 A, influence +/-6 A) loaded through
``initialize_grid_crystal``. GB geometry and every barrier are READ from the
loaded configs/GB model at test time - only seed and step counts are literals.

Documented test-side overrides (mirroring production sweep practice, in-memory):
  1. ``config.material.formula = 'ZrTi(PbO3)2'`` - normally MP-resolved;
     offline here (MP structure cache present), irrelevant to event selection.
  2. ``hydrogen_interstitial.enabled_events = ['migration', 'reaction']`` -
     electrode H *generation* is not under test; leaving it on would inject
     stray H at the top electrode and pollute the single-H statistics.
  3. ``calculator_config=None`` - MACE is not wired into the kMC loop.
  4. ``solver_config={'poisson_config': config.poisson}`` - REQUIRED although
     no FEM runs: ``_initialize_migration_pathways`` only builds the
     directional ``E_mig`` table when ``poisson_config.solve_Poisson`` is
     true. The solver object is constructed lazily in cli.py, so no dolfinx
     is touched; with no solver ``_evaluate_fields_for_kmc`` returns zero
     fields (thermal run) and test 2 injects a synthetic uniform field.
  5. ``timestep_limits``/``last_field_solve_time`` set post-init (CLI's job).
     The run helper advances ``last_field_solve_time`` each step to mimic the
     production field-solve schedule; without it kMC time clamps at the
     1e-4 s deadline and later steps become forced no-ops.

SOLVER-FREE field injection (test 2): the coordinator instance's
``_evaluate_fields_for_kmc`` is monkeypatched to return a uniform
``(0, 0, -1e9) V/m`` field for every site - the position-keyed dict shape
``PoissonSolver.evaluate_electric_field_at_points`` produces. 10 MV/cm is a
realistic memristor switching field, large enough to floor *downward* hops at
``E_min_mig`` while *upward* hops pay the full barrier - exactly the
``transition_rates`` term ``Act_E = max(barrier - q*E.d, E_min_mig)``.

COST / MARKING: the module-scoped build takes ~2.5 min (dominated by the
production GB-barrier pass over all 44k sites; no FEniCS involved), so the
tests are marked ``slow`` - run with ``pytest --runslow`` or ``-m slow``.
Individual kMC runs take 2-8 s. The grid pickle is NOT tracked in git
(288 MB); the module SKIPS cleanly when absent (fresh clones / CI).

Reproducibility: seed 42 end-to-end (the kMC rng is re-seeded per test by the
reset protocol; the module crystal is built once and reset between tests).
"""
from __future__ import annotations

import numpy as np
import pytest

from kinetix.configs.config_loader import (
    get_grids_root,
    get_parameters_root,
    load_activation_energies,
)
from kinetix.configs.simulation_config import SimulationConfig
from kinetix.initialization import (
    _process_activation_energies,
    initialize_grid_crystal,
)
from kinetix.lattice.site import Site

# Init-heavy (44k-site grid, ~2.5 min build) but solver-free: skipped by
# default, run with `pytest --runslow` (see module docstring).
pytestmark = pytest.mark.slow

PRESET_NAME = "PZT_ZrTi_PbO3_2.yaml"
GRID_NAME = "grid_ZrTi(PbO3)2_10nm"
SEED = 42
PASSIVATION = "H + V_O -> V_OH"
REMOVAL = "H -> Empty"
DEPASSIVATION = "V_OH -> V_O + H"
H_SYMBOL = "H"
VO_SYMBOL = "V_O"
LATTICE_O = "O"
# Uniform downward field for test 2 (V/m). transition_rates assists a hop
# when q * dot(E, mig_vec) > 0; downward hops have mig_vec_z < 0, so the
# field must point along -z.
DOWN_FIELD = np.array([0.0, 0.0, -1.0e9])

GRID_PATH = get_grids_root() / f"{GRID_NAME}.pkl"

if not GRID_PATH.exists():  # pragma: no cover - depends on local data
    pytest.skip(
        f"{GRID_PATH} is not tracked in git (288 MB, local-only); build it "
        f"with the PZT preset before running these physics-validation tests.",
        allow_module_level=True,
    )


# =============================================================================
# Fixtures - real configs via production loaders; ONE module-scoped lattice
# =============================================================================

@pytest.fixture(scope="module")
def pzt_config() -> SimulationConfig:
    """The REAL PZT preset with the documented in-memory overrides."""
    config = SimulationConfig.from_yaml(
        get_parameters_root() / "presets" / PRESET_NAME
    )
    # Override 1: formula normally resolved from Materials Project (offline).
    config.material.formula = "ZrTi(PbO3)2"
    return config


@pytest.fixture(scope="module")
def pzt_defects(pzt_config) -> dict:
    """Real defects config dict + override 2 (H generation disabled)."""
    defects = pzt_config.defects.to_dict()
    defects["hydrogen_interstitial"]["enabled_events"] = ["migration", "reaction"]
    return defects


@pytest.fixture(scope="module")
def pzt_act_e(pzt_config, pzt_defects) -> dict:
    """Per-defect activation energies from the REAL JSON via the production
    pipeline (same recipe as tests/test_site.py)."""
    ae_data = load_activation_energies(
        get_parameters_root() / "presets" / PRESET_NAME, pzt_config.settings
    )
    return _process_activation_energies(
        pzt_defects, ae_data, pzt_config.settings.technology
    )


@pytest.fixture(scope="module")
def pzt_crystal(pzt_config, pzt_defects, pzt_act_e):
    """Production PZT device lattice (grid fast-path; ~2.5 min - see docstring)."""
    crystal = initialize_grid_crystal(
        GRID_NAME,
        None,  # mpi_ctx=None -> truly serial (production Phase-1 path)
        pzt_config.material,
        pzt_config.experimental,
        pzt_act_e,
        pzt_config.settings.lammps_output,
        pzt_config.superbasin,
        False,  # save_data=False -> never writes grids
        settings=pzt_config.settings,
        rng=np.random.default_rng(SEED),
        cache_dir=get_parameters_root().parent / "cache",  # offline MP cache
        calculator_config=None,  # Override 3: MACE not wired into the kMC loop
        defects_config=pzt_defects,
        reactions_config=pzt_config.reactions.to_dict(),
        gb_configurations=[gb.to_dict() for gb in pzt_config.grain_boundaries],
        solver_config={"poisson_config": pzt_config.poisson},  # Override 4
        simulation_type=pzt_config.settings.simulation_type,
    )
    # Override 5: normally provided post-init by the CLI/ElectricalController.
    crystal.timestep_limits = float(pzt_config.superbasin.time_step_limits)
    crystal.last_field_solve_time = 0.0
    assert len(crystal.grid_crystal) == 44574
    return crystal


@pytest.fixture
def crystal(pzt_crystal):
    """The module lattice RESET to pristine state before every test."""
    _reset_crystal(pzt_crystal)
    yield pzt_crystal


# =============================================================================
# Harness helpers
# =============================================================================

def _reset_crystal(crystal) -> None:
    """Return the shared lattice to its pristine (post-build) state.

    The kMC runs only mutate site Defects (the H hops by object transfer;
    its vacated sites already get fresh empty Defects from the production
    bookkeeping) and the simulator scalars below. Clearing the placed H/V_O
    through the production ``_remove_species_at_site`` API and refreshing the
    topology restores every piece of state the next test depends on
    (active_event_sites, supp_by, destination_CN, energy_site, event lists).
    """
    support, events = set(), set()
    for idx, site in list(crystal.grid_crystal.items()):
        if site.defect.chemical_specie == H_SYMBOL:
            crystal.event_handler._remove_species_at_site(idx, support, events)
        elif site.defect.chemical_specie == VO_SYMBOL:
            crystal.event_handler._remove_species_at_site(idx, support, events)
            # Restore the pristine lattice occupant: an empty Defect carrying
            # 'O' - exactly what a never-occupied O-sublattice site hosts.
            site.remove_specie(LATTICE_O)
    if support or events:
        crystal.event_handler.update_sites_topology(support, events)
    crystal._dirty_sites.clear()
    crystal._fields_changed = True
    crystal.time = 0
    crystal.list_time.clear()
    crystal.events_tracking.clear()
    crystal.superbasin_dict.clear()
    crystal.superbasin_tracker.clear()
    crystal.nothing_happen_count = 0
    crystal.last_field_solve_time = 0.0
    crystal.rng = np.random.default_rng(SEED)
    assert not crystal.active_event_sites, crystal.active_event_sites[:5]
    assert not crystal.generation_sites


def _gb(crystal) -> dict:
    """The pre-processed vertical-planar GB config (position, inner/outer).

    Geometry comes from gb_vertical_planar.yaml through the production
    GrainBoundary model - never hardcoded here.
    """
    return crystal.gb_model.gb_configurations[0]


def _region(crystal, pos) -> str:
    return crystal.gb_model.get_site_gb_region(tuple(pos))


def _distance_to_gb(crystal, pos) -> float:
    """Distance of a position to the GB plane (A), from the loaded config."""
    return crystal.gb_model._distance_to_planar_gb(tuple(pos), _gb(crystal))


def _find_sites(crystal, site_type: str, specie: str, predicate) -> list:
    """Sorted indices of empty sites of a sublattice matching a position rule."""
    return sorted(
        idx
        for idx, site in crystal.grid_crystal.items()
        if site.site_type == site_type
        and site.defect.chemical_specie == specie
        and predicate(site.position)
    )


def _place(crystal, placements: list) -> None:
    """Introduce species through the production EventHandler bookkeeping.

    placements: list of (idx, chemical_specie, ion_charge).
    """
    support, events = set(), set()
    for idx, specie, charge in placements:
        crystal.event_handler._introduce_specie_site(
            idx, support, events, specie, charge
        )
    crystal.event_handler.update_sites_topology(support, events)
    crystal._dirty_sites.update(support | events)
    crystal._fields_changed = True


def _o_neighbors_of(crystal, idx: int, inside_core: bool = False) -> list:
    """Neighbor sites of ``idx`` that can host a V_O (pristine O sublattice)."""
    gb = _gb(crystal)
    out = []
    for n_idx in crystal.grid_crystal[idx].nearest_neighbors_idx:
        site = crystal.grid_crystal[n_idx]
        if site.site_type != LATTICE_O or site.defect.chemical_specie != LATTICE_O:
            continue
        if inside_core and _distance_to_gb(crystal, site.position) > gb["inner_boundary"]:
            continue
        out.append(n_idx)
    return out


def _run_steps(crystal, n_steps: int) -> list:
    """Run ``n_steps`` seeded kMC steps; return the executed-event log.

    Wraps the INSTANCE attribute ``crystal.processes`` (the same observation
    point the golden trace uses) to record ``(rate, dest, label, source)``
    before each event executes. Also advances ``last_field_solve_time`` each
    step to mimic the production field-solve schedule (see module docstring).
    """
    log = []
    original_processes = crystal.processes

    def recording_processes(chosen_event):
        log.append(
            (float(chosen_event[0]), chosen_event[1], chosen_event[2], chosen_event[-1])
        )
        original_processes(chosen_event)

    crystal.processes = recording_processes
    rng = crystal.rng
    try:
        for _ in range(n_steps):
            crystal.last_field_solve_time = crystal.time
            crystal.step_kmc(rng)
    finally:
        crystal.processes = original_processes
    return log


def _track_h(crystal, h_idx: int, log: list) -> dict:
    """Follow the single H through the executed-event log.

    Returns {migrations, passivations, removals, unexpected, final_idx} where
    ``final_idx`` is the site hosting the free H at the end (None if the H was
    passivated or removed).
    """
    stats = {"migrations": 0, "passivations": 0, "removals": 0,
             "unexpected": [], "final_idx": h_idx}
    cur = h_idx
    for _rate, dest, label, src in log:
        if isinstance(label, int) and src == cur:
            stats["migrations"] += 1
            cur = dest
        elif label == PASSIVATION and src == cur:
            stats["passivations"] += 1
            cur = None
        elif label == REMOVAL and src == cur:
            stats["removals"] += 1
            cur = None
        elif label == DEPASSIVATION:
            # H re-emitted next to a V_OH - re-acquire it by scanning.
            stats["migrations"] += 0  # depassivation itself is not a hop
            found = [idx for idx, s in crystal.grid_crystal.items()
                     if s.defect.chemical_specie == H_SYMBOL]
            cur = found[0] if found else None
        else:
            stats["unexpected"].append((label, src, dest))
    stats["final_idx"] = cur
    return stats


def _effective_barrier(rate: float, temperature: float) -> float:
    """Invert the Arrhenius law the engine uses: E = -kT ln(rate / nu0)."""
    return -Site.KB * temperature * np.log(rate / Site.NU0)


# =============================================================================
# Test 1: GB entry bias (direction-dependent barriers)
# =============================================================================

def test_gb_entry_bias(crystal, pzt_act_e):
    """An H in the GB outer region is drawn INTO the core and never leaves.

    Physics: gb_direction_barriers makes entering the GB cheap
    (outer->inner = 0.15 eV) and leaving expensive (inner->outer = 0.65 eV),
    so the executed-hop statistics must be strongly biased toward the core.
    """
    gb = _gb(crystal)
    inner, outer = gb["inner_boundary"], gb["outer_boundary"]
    barriers = pzt_act_e["hydrogen_interstitial"]["gb_direction_barriers"]
    enter_e = barriers["outer_boundary_to_inner_boundary"]
    leave_e = barriers["inner_boundary_to_outer_boundary"]
    assert enter_e < leave_e  # the asymmetry the trajectory will test

    # Place one H (+1, charged outside the core) at mid-device height in the
    # outer_boundary shell: inner < |y - y_GB| <= outer.
    candidates = _find_sites(
        crystal, "interstitial", "Empty",
        lambda p: inner < _distance_to_gb(crystal, p) <= outer and 30.0 < p[2] < 70.0,
    )
    assert candidates, "no empty interstitial sites in the GB outer region"
    h_idx = candidates[len(candidates) // 2]
    _place(crystal, [(h_idx, H_SYMBOL, 1)])
    h_site = crystal.grid_crystal[h_idx]
    assert _region(crystal, h_site.position) == "outer_boundary"

    # --- Registered catalog: inward hops must be cheaper than outward ones ---
    inward, outward = [], []
    for event in h_site.defect.events:
        assert isinstance(event.label, int), event.label  # migrations only
        dest_pos = crystal.grid_crystal[event.destination].position
        d_before = _distance_to_gb(crystal, h_site.position)
        d_after = _distance_to_gb(crystal, dest_pos)
        if d_after < d_before:
            inward.append(event.barrier)
        elif d_after > d_before:
            outward.append(event.barrier)
    assert inward and outward, (f"expected both inward and outward offers, got "
                                f"{len(inward)} inward / {len(outward)} outward")
    print(f"[T1] initial catalog: inward n={len(inward)} min={min(inward):.3f} eV | "
          f"outward n={len(outward)} min={min(outward):.3f} eV "
          f"(config: enter={enter_e}, leave={leave_e})")
    assert min(inward) < min(outward)
    assert min(outward) - min(inward) >= 0.3, (
        "GB direction-dependent barrier asymmetry missing from the catalog"
    )

    # --- Trajectory: 1000 steps of executed hops ---
    n_steps = 1000
    log = _run_steps(crystal, n_steps)
    stats = _track_h(crystal, h_idx, log)
    assert stats["unexpected"] == []
    assert stats["passivations"] == 0 and stats["removals"] == 0

    # Classify the executed H hops.
    cur = h_idx
    hops_in = hops_out = hops_parallel = 0  # while H sits in the outer shell
    entries = exits = 0                     # region-crossing hops, whole run
    for _rate, dest, label, src in log:
        if not (isinstance(label, int) and src == cur):
            continue
        r_before = _region(crystal, crystal.grid_crystal[cur].position)
        r_after = _region(crystal, crystal.grid_crystal[dest].position)
        if r_before == "outer_boundary":
            d_before = _distance_to_gb(crystal, crystal.grid_crystal[cur].position)
            d_after = _distance_to_gb(crystal, crystal.grid_crystal[dest].position)
            if d_after < d_before:
                hops_in += 1
            elif d_after > d_before:
                hops_out += 1
            else:
                hops_parallel += 1
        if r_before != r_after:
            entered = r_after == "inner_boundary" or (
                r_before == "bulk" and r_after == "outer_boundary"
            )
            entries += int(entered)
            exits += int(not entered)
        cur = dest

    directional = hops_in + hops_out
    inward_ratio = hops_in / directional if directional else float("nan")
    print(f"[T1] {n_steps} steps -> migrations={stats['migrations']}; "
          f"outer-phase hops: in={hops_in} out={hops_out} parallel={hops_parallel} "
          f"(inward ratio={inward_ratio:.3f}); region entries={entries} exits={exits}; "
          f"final region={_region(crystal, crystal.grid_crystal[stats['final_idx']].position)}")

    assert stats["migrations"] >= 100, "H barely moved - event selection stalled"
    assert directional >= 1, "no directional outer-phase hops executed"
    assert inward_ratio > 0.5, "GB entry bias reversed: H moves away from the core"
    assert entries >= 1 and exits == 0, "H never entered the GB or leaked out of it"
    final_site = crystal.grid_crystal[stats["final_idx"]]
    assert _region(crystal, final_site.position) == "inner_boundary", (
        "H should end up trapped in the GB core"
    )


# =============================================================================
# Test 2: Bulk migration vs passivation probability (field-assisted)
# =============================================================================

def test_bulk_migration_dominates_passivation(crystal, pzt_act_e, monkeypatch):
    """An H next to a V_O in the bulk migrates (downward) instead of passivating.

    Physics: with the synthetic downward field the fastest downward hops sit
    near the E_min_mig floor while the "H + V_O -> V_OH" capture keeps its
    0.26 eV barrier (the reaction is not field-dependent), so migrations must
    outnumber passivations by orders of magnitude and point downward.
    """
    act_h = pzt_act_e["hydrogen_interstitial"]
    passivation_barrier = act_h[PASSIVATION]["0"]
    e_min_mig = act_h["E_min_mig"]
    e_mig_down = act_h["E_mig_downward"]
    temperature = crystal.temperature
    kT = Site.KB * temperature

    # --- Solver-free uniform field: same dict shape the Poisson solver emits
    e_field_dict = {
        tuple(np.round(site.position, 6)): DOWN_FIELD
        for site in crystal.grid_crystal.values()
    }
    monkeypatch.setattr(
        crystal.solver_coordinator,
        "_evaluate_fields_for_kmc",
        lambda: (e_field_dict, {}),
    )

    # --- Place one V_O (+ a single H next to it) in the BULK, mid-device ---
    bulk = _find_sites(
        crystal, "interstitial", "Empty",
        lambda p: _distance_to_gb(crystal, p) > _gb(crystal)["outer_boundary"] + 3.0
        and 30.0 < p[2] < 60.0,
    )
    assert bulk, "no empty interstitial bulk sites found"
    h_idx = bulk[len(bulk) // 2]
    vo_candidates = _o_neighbors_of(crystal, h_idx)
    assert vo_candidates, "H site has no pristine O neighbor for the V_O"
    vo_idx = vo_candidates[0]
    _place(crystal, [(vo_idx, VO_SYMBOL, 0), (h_idx, H_SYMBOL, 1)])
    h_site = crystal.grid_crystal[h_idx]
    assert _region(crystal, h_site.position) == "bulk"
    assert _region(crystal, crystal.grid_crystal[vo_idx].position) == "bulk"

    # Materialize rates under the injected field (what the first kMC step uses)
    crystal.event_handler._update_rates_lazily(e_field_dict, {})

    def _split_migrations(events):
        down, up, lateral = [], [], []
        for event in events:
            if not isinstance(event.label, int):
                continue
            z = crystal.migration_pathways[event.label]["direction"][2]
            (down if z < -1e-9 else up if z > 1e-9 else lateral).append(event)
        return down, up, lateral

    down, up, lateral = _split_migrations(h_site.defect.events)
    passivation = [e for e in h_site.defect.events if e.label == PASSIVATION]
    assert down and passivation, (
        f"expected downward hops and a passivation offer, got "
        f"{len(down)} down / {len(passivation)} passivation"
    )

    fastest_down = max(down, key=lambda e: e.rate)
    slowest_up = min(up, key=lambda e: e.rate) if up else None
    e_eff_down = _effective_barrier(fastest_down.rate, temperature)
    e_eff_up = (_effective_barrier(slowest_up.rate, temperature)
                if slowest_up is not None else float("inf"))
    ratio_catalog = sum(e.rate for e in down) / passivation[0].rate
    arrhenius_ratio = np.exp((passivation_barrier - e_eff_down) / kT)
    print(f"[T2] catalog @T={temperature:.0f}K: fastest downward E_eff={e_eff_down:.3f} eV "
          f"(E_min_mig={e_min_mig}, bare E_mig_downward={e_mig_down}); "
          f"slowest upward E_eff={e_eff_up:.3f} eV; "
          f"passivation barrier={passivation[0].barrier:.3f} eV "
          f"(config {passivation_barrier}); sum(down)/passivation = "
          f"{ratio_catalog:.1f} (Arrhenius prediction ~{arrhenius_ratio:.0f})")

    # Catalog-level Boltzmann checks: the field assists downward hops below
    # the bare barrier, upward hops stay costlier, and the passivation rate
    # lags the fastest migration channel by a wide margin.
    assert passivation[0].barrier == pytest.approx(passivation_barrier)
    assert e_eff_down < e_mig_down - 0.05, "field did not assist downward hops"
    assert e_eff_down >= e_min_mig - 1e-9, "field correction broke the E_min_mig floor"
    if slowest_up is not None:
        assert e_eff_up > e_eff_down + 0.1, "no up/down field asymmetry in the rates"
    assert ratio_catalog >= 10, "passivation rate too competitive vs migration"

    # --- Trajectory: 5000 seeded steps ---
    n_steps = 5000
    log = _run_steps(crystal, n_steps)
    stats = _track_h(crystal, h_idx, log)
    assert stats["unexpected"] == []

    cur = h_idx
    down_hops = up_hops = 0
    for _rate, dest, label, src in log:
        if isinstance(label, int) and src == cur:
            z_before = crystal.grid_crystal[cur].position[2]
            z_after = crystal.grid_crystal[dest].position[2]
            if z_after < z_before:
                down_hops += 1
            elif z_after > z_before:
                up_hops += 1
            cur = dest

    migrations, passivations = stats["migrations"], stats["passivations"]
    ratio = (migrations / passivations) if passivations else float("inf")
    print(f"[T2] {n_steps} steps -> migrations={migrations} "
          f"(down={down_hops}, up={up_hops}), passivations={passivations}, "
          f"electrode removals={stats['removals']}, ratio={ratio:.1f}")

    assert migrations >= 10 * passivations, (
        "migration does not dominate passivation as the barriers dictate "
        f"({migrations} migrations vs {passivations} passivations)"
    )
    assert migrations > 0, "no migration executed at all"
    assert down_hops > up_hops, "field-driven downward bias missing from the trajectory"
    assert down_hops >= 0.8 * (down_hops + up_hops), (
        "downward preference too weak for the injected field"
    )


# =============================================================================
# Test 3: GB residence and escape probability
# =============================================================================

def test_gb_residence_and_escape(crystal, pzt_act_e):
    """An H inside the GB core stays there: it walks internally or passivates.

    Physics: H is neutral in the core (GB charge_state) and leaving costs the
    inner->outer barrier (0.65 eV) while internal hops cost ~0.15 eV, so over
    10 000 kMC steps no escape may occur and the H must either be free inside
    the core or be captured by the V_O placed there.
    """
    act_h = pzt_act_e["hydrogen_interstitial"]
    barriers = act_h["gb_direction_barriers"]
    core_hop_e = barriers["inner_boundary_to_inner_boundary"]
    leave_e = barriers["inner_boundary_to_outer_boundary"]
    gb = _gb(crystal)

    # --- Place one V_O_0 inside the core next to an interstitial site of the
    # core, then one NEUTRAL H (+0, per the GB charge_state) at that site ---
    inner_sites = _find_sites(
        crystal, "interstitial", "Empty",
        lambda p: _distance_to_gb(crystal, p) <= 0.6 * gb["inner_boundary"]
        and 30.0 < p[2] < 70.0,
    )
    assert inner_sites, "no empty interstitial sites in the GB core"
    placed = False
    for h_idx in inner_sites:
        vo_candidates = _o_neighbors_of(crystal, h_idx, inside_core=True)
        if vo_candidates:
            vo_idx = vo_candidates[0]
            placed = True
            break
    assert placed, "no core interstitial site with a pristine O neighbor in the core"
    _place(crystal, [(vo_idx, VO_SYMBOL, 0), (h_idx, H_SYMBOL, 0)])
    h_site = crystal.grid_crystal[h_idx]
    assert _region(crystal, h_site.position) == "inner_boundary"
    assert _region(crystal, crystal.grid_crystal[vo_idx].position) == "inner_boundary"

    # --- Registered catalog: core hops are cheap, escapes are expensive ---
    core_bars, escape_bars = [], []
    for event in h_site.defect.events:
        if isinstance(event.label, int):
            dest_region = _region(crystal, crystal.grid_crystal[event.destination].position)
            (core_bars if dest_region == "inner_boundary" else escape_bars).append(
                event.barrier
            )
        else:
            assert event.label == PASSIVATION, event.label
            assert event.barrier == pytest.approx(act_h[PASSIVATION]["0"])
    assert core_bars, "H in the core has no internal migration channel"
    print(f"[T3] catalog: core hops n={len(core_bars)} min={min(core_bars):.3f} eV | "
          f"escape hops n={len(escape_bars)} "
          f"min={min(escape_bars) if escape_bars else float('nan'):.3f} eV "
          f"(config: core={core_hop_e}, leave={leave_e})")
    assert min(core_bars) <= core_hop_e + 0.05
    if escape_bars:
        assert min(escape_bars) >= 0.5, "an escape channel is cheaper than configured"
        assert min(escape_bars) - min(core_bars) >= 0.3

    # --- Trajectory: 10 000 seeded steps inside the core ---
    n_steps = 10000
    log = _run_steps(crystal, n_steps)
    stats = _track_h(crystal, h_idx, log)
    assert stats["unexpected"] == []

    escapes = 0
    cur = h_idx
    for _rate, dest, label, src in log:
        if isinstance(label, int) and src == cur:
            if (_region(crystal, crystal.grid_crystal[cur].position) == "inner_boundary"
                    and _region(crystal, crystal.grid_crystal[dest].position)
                    != "inner_boundary"):
                escapes += 1
            cur = dest

    passivated_vos = [
        (idx, site.position)
        for idx, site in crystal.grid_crystal.items()
        if site.defect.chemical_specie == VO_SYMBOL and site.defect.passivation_level >= 1
    ]
    print(f"[T3] {n_steps} steps -> core migrations={stats['migrations']}, "
          f"passivations={stats['passivations']}, escapes={escapes}, "
          f"removals={stats['removals']}, passivated V_O in core="
          f"{any(_region(crystal, p) == 'inner_boundary' for _, p in passivated_vos)}")

    assert stats["migrations"] > 0 or stats["passivations"] > 0, (
        "H neither moved nor passivated inside the core"
    )
    assert escapes == 0, (
        f"H escaped the GB core {escapes} time(s) despite the {leave_e} eV leave barrier"
    )
    assert stats["passivations"] <= 1, "more passivations than the single V_O allows"
    assert stats["removals"] == 0, "H reached the bottom electrode without escaping"

    # Final-state disjunction: free H inside the core, or a passivated V_O there.
    if stats["passivations"] == 0:
        final_site = crystal.grid_crystal[stats["final_idx"]]
        assert final_site.defect.chemical_specie == H_SYMBOL
        assert _region(crystal, final_site.position) == "inner_boundary", (
            "free H left the GB core"
        )
    else:
        assert passivated_vos, "passivation recorded but no passivated V_O found"
        assert all(
            _region(crystal, pos) == "inner_boundary" for _idx, pos in passivated_vos
        ), "the passivated V_O is outside the GB core"
