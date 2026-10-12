# tests/test_pzt_golden_trace.py
# -*- coding: utf-8 -*-
"""Golden-trace regression test for a MINIATURE PZT device.

Purpose
-------
Pin the exact sequence of kMC events (generation, migration, passivation,
GB trapping) executed by the PZT physics stack so future refactors cannot
silently change event selection, ordering, or bookkeeping. This is the PZT
counterpart of ``tests/test_golden_trace.py`` (VCM/HfO2), with the same
contract: the recorded trace must stay byte-identical unless a physics
change is *deliberate* (then regenerate with ``--update-pzt-trace`` and
review the fixture diff).

Why a miniature system
----------------------
The production PZT grid ``grid_ZrTi(PbO3)2_10nm.pkl`` is 288 MB / 44 574
sites and cannot be tracked in Git. This module therefore drives a miniature
device built by the PRODUCTION lattice builder from the REAL preset with one
override:

  * ``material.structure.size = (20, 60, 20)`` A  (production: (50, 50, 100))

The actual supercell pymatgen returns is ``(22.8, 62.6, 22.2)`` A (5 375
sites) - large enough to keep the planar GB at y=27 A with its +/-3 A core,
+/-6 A influence zone and bulk material on both sides. Everything else is
the production PZT configuration, loaded through the production loaders:

  * ``PZT_ZrTi_PbO3_2.yaml``            (SimulationConfig.from_yaml)
  * ``PZT_ZrPbO3.json``                 (load_activation_energies +
                                         _process_activation_energies)
  * ``gb_vertical_planar.yaml``         (config.grain_boundaries)
  * ``PZT_reactions_H_Ovac_passivation.yaml`` (config.reactions)
  * ``PZT_ZrPbO3_defects_config.yaml``  (config.defects.to_dict(), verbatim -
    electrode H *generation* stays enabled: it sustains the event stream)

Initial state (deterministic, no RNG involved)
----------------------------------------------
Four V_O are introduced through the production EventHandler API at
sorted-index landmarks: 2 inside the GB core (|y-27| <= 3 A) and 2 in the
bulk (|y-27| >= 9 A), all mid-height (8 < z < 14 A). Free H atoms enter
through the normal electrode-generation channel (top interface), migrate,
get GB-trapped, and passivate V_O: with seed 42 the first passivation lands
around event ~330 and two core V_O are passivated within the 2 000-event
window (pinned by the fixture, asserted as a sanity check).

Fixture files (tests/fixtures/)
-------------------------------
  * ``grid_PZT_trace.pkl.gz``   - the pristine miniature lattice.
    NOTE: the raw production pickle (protocol default, exactly the
    ``_save_grid_atomic`` payload) is **29.2 MB**, far over the 5 MB budget,
    and ``*.pkl`` is gitignored repo-wide; the committed artifact is
    therefore the gzip of those exact bytes (**2.8 MB**, gzip level 9,
    mtime=0 so regeneration is byte-reproducible). The test decompresses it
    into the gitignored ``data/grids/grid_PZT_trace.pkl`` and loads it
    through the production ``initialize_grid_crystal`` fast path. An
    uncompressed ``grid_PZT_trace.pkl`` dropped next to it (``git add -f``)
    is also honoured if preferred.
  * ``pzt_golden_trace.json``   - ``{"meta": {...}, "events": [...]}`` with
    2000 records ``{"timestep": float (s, kMC time AFTER the event),
    "event_label": int pathway ID | str reaction name,
    "source_site_idx": [ix, iy, iz], "dest_site_idx": [ix, iy, iz] | same
    as source for generation}``. Site indices are the grid's tuple keys,
    serialized as JSON arrays.

Regenerating the fixtures
-------------------------
    pytest tests/test_pzt_golden_trace.py --update-pzt-trace

(Rebuilds the grid from scratch - ~1 min - records 2000 events through the
production load path, overwrites both fixture files, then runs the normal
comparison as a round-trip check. Review ``git diff`` of the JSON before
committing: one changed line == one event whose time/label/sites moved.)

Runtime: ~13 s per run (grid decompress+load ~2 s, pathway/GB init over 5 375
sites ~4 s, the 2000 recorded steps ~7 s). Deterministic end-to-end: seed 42
is re-applied to ``crystal.rng`` immediately before the recorded run, so the
trace does not depend on anything the build consumed earlier.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

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

PRESET_NAME = "PZT_ZrTi_PbO3_2.yaml"
GRID_NAME = "grid_PZT_trace"
MINI_SIZE = (20, 60, 20)  # Angstroms (x, y, z) - GB geometry needs y >= ~40
SEED = 42
N_EVENTS = 2000
MAX_STEPS = 6000  # safety net: fail loudly if the loop stalls before N_EVENTS
TRACE_SCHEMA = 1

PASSIVATION = "H + V_O -> V_OH"
GENERATION = "generation"
V_O = "V_O"
LATTICE_O = "O"

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GRID_PKL = FIXTURES_DIR / f"{GRID_NAME}.pkl"       # optional uncompressed form
GRID_GZ = FIXTURES_DIR / f"{GRID_NAME}.pkl.gz"     # committed artifact
TRACE_JSON = FIXTURES_DIR / "pzt_golden_trace.json"

UPDATE_HINT = "run: pytest tests/test_pzt_golden_trace.py --update-pzt-trace"


# =============================================================================
# Miniature system: real preset + one documented geometry override
# =============================================================================

def _mini_config() -> SimulationConfig:
    """The REAL PZT preset with only the crystal size shrunk to miniature."""
    config = SimulationConfig.from_yaml(
        get_parameters_root() / "presets" / PRESET_NAME
    )
    # Offline (MP resolution needs no network: the structure cache is present).
    config.material.formula = "ZrTi(PbO3)2"
    # THE miniature override. Everything else - defects, reactions, GB,
    # activation energies, temperature, superbasin - stays production.
    config.material.structure.size = MINI_SIZE
    return config


def _build_crystal(config: SimulationConfig, defects: dict, act_e: dict):
    """Load (or first time: build) the miniature lattice via the PRODUCTION path.

    Identical to ``initialize_grid_crystal``'s Phase-4 fast path used by the
    CLI: grid pickle -> live-config binding -> interface flags -> k-d tree ->
    migration pathways -> GB barrier pass. ``save_data=False``: the harness
    owns serialization (gzip) and never writes grids itself.
    """
    crystal = initialize_grid_crystal(
        GRID_NAME,
        None,  # mpi_ctx=None -> truly serial
        config.material,
        config.experimental,
        act_e,
        config.settings.lammps_output,
        config.superbasin,
        False,  # save_data=False
        settings=config.settings,
        rng=np.random.default_rng(SEED),
        cache_dir=get_parameters_root().parent / "cache",  # offline MP cache
        calculator_config=None,
        defects_config=defects,
        reactions_config=config.reactions.to_dict(),
        gb_configurations=[gb.to_dict() for gb in config.grain_boundaries],
        solver_config={"poisson_config": config.poisson},  # builds the E_mig table
        simulation_type=config.settings.simulation_type,
    )
    # Normally provided post-init by the CLI/ElectricalController.
    crystal.timestep_limits = float(config.superbasin.time_step_limits)
    crystal.last_field_solve_time = 0.0
    return crystal


# =============================================================================
# Fixture installation & deterministic defect seeding
# =============================================================================

def _fixture_grid_source() -> Path | None:
    """The committed miniature-grid artifact (uncompressed preferred)."""
    if GRID_PKL.exists():
        return GRID_PKL
    if GRID_GZ.exists():
        return GRID_GZ
    return None


def _install_grid() -> Path:
    """Materialize the committed fixture into ``data/grids`` for the loader.

    ``initialize_grid_crystal`` only reads ``get_grids_root()``; the fixture
    (gzip, 2.8 MB) is decompressed into the gitignored
    ``data/grids/grid_PZT_trace.pkl`` (29 MB, exactly the production
    ``_save_grid_atomic`` payload) and loaded from there through the
    production fast path. Overwritten on every run, so a stale copy in
    ``data/grids`` can never leak into the trace.
    """
    src = _fixture_grid_source()
    assert src is not None, f"missing fixture grid ({UPDATE_HINT})"
    dst = get_grids_root() / f"{GRID_NAME}.pkl"
    if src.suffix == ".gz":
        dst.write_bytes(gzip.decompress(src.read_bytes()))
    else:
        shutil.copyfile(src, dst)
    return dst


def _pick_landmark(crystal, site_type: str, specie: str, predicate) -> tuple:
    """Deterministic site pick: the sorted-index landmarks at 1/3 and 2/3."""
    candidates = sorted(
        idx
        for idx, site in crystal.grid_crystal.items()
        if site.site_type == site_type
        and site.defect.chemical_specie == specie
        and predicate(site.position)
    )
    assert candidates, f"no {site_type}/{specie} site matched the landmark rule"
    return tuple(candidates[int(len(candidates) * frac)] for frac in (0.33, 0.66))


def _seed_v_o(crystal) -> dict:
    """Introduce exactly 4 V_O through the production EventHandler API.

    2 in the GB core (|y - y_GB| <= inner) and 2 in the bulk
    (|y - y_GB| >= outer + 3), all mid-height (8 < z < 14 A) to stay away
    from the electrode interfaces. Selection is by sorted site index - no
    RNG - so the seeded configuration is identical on every run and on
    every machine. Returns the seeded indices for reporting.
    """
    gb = crystal.gb_model.gb_configurations[0]
    gb_pos = gb["position"]

    def _dist(pos):
        return abs(pos[1] - gb_pos)

    core = _pick_landmark(
        crystal, LATTICE_O, LATTICE_O,
        lambda p: _dist(p) <= gb["inner_boundary"] and 8.0 < p[2] < 14.0,
    )
    bulk = _pick_landmark(
        crystal, LATTICE_O, LATTICE_O,
        lambda p: _dist(p) >= gb["outer_boundary"] + 3.0 and 8.0 < p[2] < 14.0,
    )

    support, events = set(), set()
    for idx in core + bulk:
        crystal.event_handler._introduce_specie_site(idx, support, events, V_O, 0)
    crystal.event_handler.update_sites_topology(support, events)
    crystal._dirty_sites.update(support | events)
    crystal._fields_changed = True
    # _native recurses into the tuples: grid keys carry np.int64 elements.
    return {"core": _native(core), "bulk": _native(bulk)}


# =============================================================================
# Trace recording
# =============================================================================

def _native(value):
    """Recursively convert NumPy scalars to Python natives.

    Grid site indices are tuples whose elements may be ``np.int64`` (the
    keys ``get_idx_coords`` produces); JSON cannot serialize those, and the
    text must also be identical across NumPy versions (same rationale as
    ``test_golden_trace.py::_py``).
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return value


def _record_trace(crystal, n_events: int = N_EVENTS) -> list[dict]:
    """Execute seeded kMC steps and record the first ``n_events`` events.

    Hooks the INSTANCE attribute ``crystal.processes`` (the same observation
    point ``test_golden_trace.py`` uses) before each event executes. The
    recorded ``timestep`` is ``crystal.time`` AFTER the step that executed
    the event. ``crystal.rng`` is re-seeded here so the recorded stream is
    independent of anything the lattice build consumed. ``last_field_solve_time``
    is advanced every step to mimic the production field-solve schedule (no
    solvers run here; without this, kMC time would clamp at the 1e-4 s
    deadline and later steps would become forced no-ops).
    """
    trace: list[dict] = []
    original_processes = crystal.processes

    def recording_processes(chosen_event):
        label = _native(chosen_event[2])
        source = _native(chosen_event[-1])
        dest = _native(chosen_event[1])
        trace.append({
            "timestep": None,  # filled after the step completes
            "event_label": label if isinstance(label, str) else int(label),
            "source_site_idx": list(source),
            # Migration/capture: destination site. Generation & unimolecular
            # events register the owning site as destination (production
            # convention) - recorded verbatim.
            "dest_site_idx": list(dest),
        })
        original_processes(chosen_event)

    crystal.processes = recording_processes
    crystal.rng = np.random.default_rng(SEED)
    rng = crystal.rng
    steps = 0
    try:
        while len(trace) < n_events and steps < MAX_STEPS:
            crystal.last_field_solve_time = crystal.time
            crystal.step_kmc(rng)
            steps += 1
            if trace and trace[-1]["timestep"] is None:
                trace[-1]["timestep"] = float(crystal.time)
    finally:
        crystal.processes = original_processes

    assert len(trace) == n_events, (
        f"only {len(trace)} events executed in {steps} steps "
        f"(wanted {n_events}); the event stream dried up"
    )
    return trace


# =============================================================================
# Fixture generation (--update-pzt-trace)
# =============================================================================

def _gzip_pickle(payload) -> bytes:
    """Deterministic gzip: level 9, mtime=0 -> byte-reproducible artefact."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def _generate_fixtures(config, defects, act_e) -> dict:
    """Rebuild the miniature grid from scratch and re-record the 2000 events.

    1. Remove every stale copy of the grid so ``initialize_grid_crystal``
       performs a FRESH production build (~1 min) instead of loading.
    2. Serialize the pristine grid exactly like ``_save_grid_atomic``
       (``{name: grid_crystal}``, default pickle protocol), gzipped.
    3. Re-load through the SAME production path the test uses (the recorded
       trace must come from a loaded lattice, not the just-built one), seed
       the 4 V_O, record, and write the JSON fixture.
    """
    stale = (
        get_grids_root() / f"{GRID_NAME}.pkl",
        GRID_PKL, GRID_GZ, TRACE_JSON,
    )
    for path in stale:
        path.unlink(missing_ok=True)

    print("[update] building miniature lattice (~1 min, production builder)...")
    crystal = _build_crystal(config, defects, act_e)
    raw = pickle.dumps({GRID_NAME: crystal.grid_crystal})  # _save_grid_atomic payload
    GRID_GZ.write_bytes(_gzip_pickle(raw))
    grid_sha256 = hashlib.sha256(raw).hexdigest()
    print(f"[update] wrote {GRID_GZ} "
          f"({GRID_GZ.stat().st_size / 1e6:.2f} MB gzip; raw {len(raw) / 1e6:.1f} MB; "
          f"{len(crystal.grid_crystal)} sites; sha256={grid_sha256[:16]}...)")
    del crystal

    # Reload exactly like the test does, so fixture == round-trip.
    _install_grid()
    crystal = _build_crystal(config, defects, act_e)
    seeded = _seed_v_o(crystal)
    trace = _record_trace(crystal)

    labels = Counter(str(record["event_label"]) for record in trace)
    meta = {
        "schema": TRACE_SCHEMA,
        "scenario": "miniature PZT device (GB trapping / passivation)",
        "preset": PRESET_NAME,
        "miniature_size": list(MINI_SIZE),
        "actual_crystal_size": [float(v) for v in crystal.crystal_size],
        "n_sites": len(crystal.grid_crystal),
        "seed": SEED,
        "n_events": len(trace),
        "grid_sha256": grid_sha256,
        "seeded_v_o": seeded,
        "event_summary": dict(sorted(labels.items())),
    }
    TRACE_JSON.write_text(json.dumps({"meta": meta, "events": trace}, indent=1))
    print(f"[update] wrote {TRACE_JSON} ({TRACE_JSON.stat().st_size / 1e3:.0f} KB); "
          f"labels={meta['event_summary']}")
    print("[update] review `git diff tests/fixtures/pzt_golden_trace.json` "
          "before committing.")
    return meta


# =============================================================================
# Pytest wiring
# =============================================================================

@pytest.fixture(scope="module")
def pzt_mini_inputs():
    """Real PZT configs (production loaders) for the miniature device."""
    config = _mini_config()
    defects = config.defects.to_dict()  # production values verbatim
    ae_data = load_activation_energies(
        get_parameters_root() / "presets" / PRESET_NAME, config.settings
    )
    act_e = _process_activation_energies(
        defects, ae_data, config.settings.technology
    )
    return config, defects, act_e


def test_pzt_golden_trace(pzt_mini_inputs, request):
    """The 2000-event kMC trace of the miniature PZT device must not drift.

    Runs the production kMC loop (seed 42) on the committed miniature grid
    and asserts the executed-event sequence - time, label, source and
    destination site - matches tests/fixtures/pzt_golden_trace.json exactly
    (labels/indices exact; float timestamps via pytest.approx). Any refactor
    that changes which event is selected, in which order, or where it lands,
    fails here instead of silently altering the physics.
    """
    config, defects, act_e = pzt_mini_inputs

    if request.config.getoption("--update-pzt-trace"):
        _generate_fixtures(config, defects, act_e)

    if _fixture_grid_source() is None:
        pytest.skip(
            f"miniature PZT grid fixture missing (expected "
            f"{GRID_GZ.name} in tests/fixtures); {UPDATE_HINT}"
        )
    if not TRACE_JSON.exists():
        pytest.skip(f"trace fixture {TRACE_JSON.name} missing; {UPDATE_HINT}")

    _install_grid()
    crystal = _build_crystal(config, defects, act_e)
    seeded = _seed_v_o(crystal)
    actual = _record_trace(crystal)

    payload = json.loads(TRACE_JSON.read_text())
    expected = payload["events"]
    assert payload["meta"]["schema"] == TRACE_SCHEMA
    assert payload["meta"]["n_events"] == len(expected) == len(actual)

    # --- 1. Exact event sequence: labels and site indices ---
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert got["event_label"] == want["event_label"], (
            f"event #{step}: label {got['event_label']!r} != "
            f"fixture {want['event_label']!r}"
        )
        assert got["source_site_idx"] == want["source_site_idx"], (
            f"event #{step} ({want['event_label']}): source "
            f"{got['source_site_idx']} != fixture {want['source_site_idx']}"
        )
        assert got["dest_site_idx"] == want["dest_site_idx"], (
            f"event #{step} ({want['event_label']}): destination "
            f"{got['dest_site_idx']} != fixture {want['dest_site_idx']}"
        )

    # --- 2. kMC timestamps: approx (physics pinned by the sequence itself) ---
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert got["timestep"] == pytest.approx(
            want["timestep"], rel=1e-9, abs=1e-18
        ), f"event #{step}: time {got['timestep']!r} != fixture {want['timestep']!r}"
    times = [record["timestep"] for record in actual]
    assert all(t2 > t1 for t1, t2 in zip(times, times[1:])), (
        "kMC time must advance strictly between executed events"
    )

    # --- 3. Physics sanity: the behaviours the fixture exists to protect ---
    labels = Counter(str(record["event_label"]) for record in actual)
    n_migrations = sum(
        count for label, count in labels.items() if label.isdigit()
    )
    print(
        f"[pzt-trace] {len(actual)} events, t_final={times[-1]:.3e} s; "
        f"migrations={n_migrations}, generation={labels.get(GENERATION, 0)}, "
        f"passivation={labels.get(PASSIVATION, 0)}; seeded V_O={seeded}"
    )
    assert labels.get(PASSIVATION, 0) >= 1, (
        "no passivation event in the window - the seeded scenario lost its "
        "capture physics (regenerating the fixture cannot fix the physics)"
    )
    assert labels.get(GENERATION, 0) >= 1, "no electrode H generation recorded"
    assert n_migrations >= int(0.9 * len(actual)), (
        "migrations should dominate the event stream"
    )
