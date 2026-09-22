# tests/test_golden_trace.py
"""Refactor-invariance golden trace for the kMC loop (VCM_mock + HfO2 grid).

Purpose
-------
This module is the safety net for the ``Site``/``Defect`` decoupling refactor
(``Site`` has-a ``Defect``; all dynamic defect state - charge,
passivation_level, events - moves off ``Site`` onto ``Defect``).  The refactor
is only allowed to change *where* state lives, never the numbers, so the test
fingerprints the whole execution profile of the kMC loop:

  * preset    : ``data/parameters/presets/VCM_mock.yaml`` (production loader)
  * grid      : ``data/grids/grid_HfO2_3nm.pkl`` (production ``_try_load_grid``)
  * seed      : 42 (``np.random.default_rng(42)``, used by ``defect_gen`` and
                every ``step_kmc`` draw)
  * steps     : 30 ``Crystal_Lattice.step_kmc`` calls

What is recorded (per step, JSON fixture)
-----------------------------------------
  * ``catalog``   : one line per registered event in the exact catalog that
                    ``_kmc_step`` builds, formatted
                    ``rate|barrier|dest|label|origin`` (rate in s^-1,
                    barrier = registered ``E_act`` in eV, dest/origin = grid
                    indices, label = int migration label or event-name string).
                    This is the requested ``(step, rate, dest, label, origin)``
                    record, with the barrier kept as an extra column because it
                    is the physics.
  * ``sum_rate``  : per-step sum of every rate in that catalog (the ``sumTR``
                    driving the BKL time step).
  * ``chosen``    : the event actually executed (same 5-column format; the
                    barrier is resolved from the catalog entry, which also
                    proves the selected event came from the catalog).
  * ``dt``        : elapsed kMC time for the step (``crystal.time`` delta),
                    ``time`` = cumulative ``crystal.time`` after the step.
  * ``n_catalog``, ``n_active_sites``, ``n_generation_sites``,
    ``n_processes_calls`` : occupancy/bookkeeping counters
                    (``n_processes_calls`` is 1 for a normal step, 0 when the
                    time step is clamped to ``timestep_limits`` with no event;
                    a superbasin-driven extra ``processes`` call would raise it
                    - and fail the fixture).

Representation independence
---------------------------
Only two helpers know the internal representation of an event
(``_event_fields`` for a registered ``site_events`` entry and
``_chosen_fields`` for the tuple handed to ``processes``).  Phase 4 of the
refactor replaces the raw ``[rate, dest, label, E_act]`` list with an ``Event``
dataclass: the physics columns in the fixture must NOT change, only those two
helpers.  Likewise this file never touches ``ion_charge`` /
``passivation_level`` directly, so Phases 3, 5 and 6 must leave the trace
bit-identical without any edit here.

Regenerating the fixture
------------------------
Only when the *inputs* legitimately change (YAML parameters, grid file) or the
physics is intentionally changed::

    KINETIX_UPDATE_GOLDEN_TRACE=1 python -m pytest tests/test_golden_trace.py
    python -m tests.test_golden_trace          # same, without pytest

Then inspect ``git diff tests/fixtures/golden_trace_*.json`` (one changed line
== exactly one event whose rate/barrier/destination/label/origin moved) and
commit the fixture together with the change that caused it.  The fixture stores
a provenance block (grid sha256 + semantic hashes of the defect/reaction/
activation-energy inputs + schema version) so a mismatch is reported as
"the inputs changed" instead of "the code changed".
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
import os
import platform
from pathlib import Path

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

# The trace constants are deliberately shared with the kMC-loop behavioural
# test so both always describe the same system (that file builds its lattice
# with the same overrides; this module re-implements the builder because it
# additionally forwards the preset's Poisson config - see _build_lattice).
from tests.test_kmc_loop import (
    GRID_NAME,
    N_STEPS,
    SEED,
)

# =============================================================================
# Constants
# =============================================================================

PRESET_NAME = "VCM_mock.yaml"
FIXTURE_NAME = f"golden_trace_vcm_hfo2_seed{SEED}_n{N_STEPS}.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / FIXTURE_NAME

# Bump when the *fixture schema* changes (new/renamed recorded field).  A bump
# makes the provenance test fail with an explicit "regenerate" message instead
# of a cryptic KeyError.
TRACE_SCHEMA = 1

# Grids shipped with the repository carry their own cache; if the file is
# missing the test skips (creating a grid needs network + Materials Project).
GRID_PATH = get_parameters_root().parent / "grids" / f"{GRID_NAME}.pkl"
EXPECTED_N_SITES = 3456

UPDATE_ENV_VAR = "KINETIX_UPDATE_GOLDEN_TRACE"

# Float comparison tolerance for the trace.  The golden trace protects against
# *behavioural* drift, so it is strict (relative 1e-9) while staying immune to
# platform libm/BLAS differences in the last few ULPs of exp().
RATE_REL_TOL = 1e-9
RATE_ABS_TOL = 1e-12

# =============================================================================
# Cross-sublattice scenario (pins the object-transfer hop semantics)
# =============================================================================
# The shipped presets CANNOT produce a cross-sublattice migration hop:
#   * every mobile defect is interstitial-borne and declares
#     ``valid_target_species=['Empty']``, and
#   * ``Empty``-specie sites exist only on the ``interstitial`` sublattice:
#     every O / Hf / Ce / Zr / Ti / Pb site hosts its lattice specie, and the
#     reactions only ever write 'O' or 'V_O' onto O sites.
# Measured on the main trace: 18,341 offered migration events, all
# interstitial->interstitial, and 0 Empty-specie O sites at any point.
#
# This scenario therefore documents the two test-side overrides that make the
# engine's *own* destination contract reachable in an offline run (the migration
# gate's comment reads "Is the site available (Empty or Vacancy)?"):
#   1. ``oxygen_interstitial.valid_target_species`` gains 'V_O' so an O
#      interstitial may hop onto an oxygen-vacancy site, and
#   2. ``oxygen_vacancy.initial_concentration_bulk = 0.02`` so such sites exist
#      from the start (the shipped YAML seeds none).
# Both live in the in-memory ``defects_config`` only: no YAML edit, no grid
# rebuild, no production change.  Note that sites carry a *pickled copy* of the
# config (the Poisson re-injection at crystal.py:542-566 refreshes only
# Act_E_dict), so the harness re-injects the live dict into every site -
# without that step the overrides are silently ignored.
CROSS_FIXTURE_NAME = "golden_trace_vcm_hfo2_cross_sublattice.json"
CROSS_FIXTURE_PATH = (Path(__file__).resolve().parent / "fixtures"
                      / CROSS_FIXTURE_NAME)

# Bump when the cross fixture's *schema* changes (new/renamed recorded field).
CROSS_SCHEMA = 1

CROSS_SEED = SEED
CROSS_N_STEPS = N_STEPS
CROSS_DEFECT_NAME = "oxygen_interstitial"
CROSS_DEFECT_SYMBOL = "O_i"
CROSS_DESTINATION_SUBLATTICE = "O"
CROSS_V_O_CONCENTRATION = 0.02
CROSS_VALID_TARGET_SPECIES = ("Empty", "V_O")

CROSS_PROVENANCE_KEYS = ("schema", "scenario", "preset", "grid", "n_sites",
                         "seed", "n_steps", "grid_sha256", "inputs_sha256",
                         "overrides", "cross_hops", "cross_hops_sha256")

# =============================================================================
# Fixture encoding: one event == one text line "rate|barrier|dest|label|origin"
# =============================================================================


def _format_event(rate, barrier, dest, label, origin) -> str:
  """Encode one event as a single diff-friendly, exactly-representable line.

  ``repr`` of float round-trips exactly, so no precision is lost by the text
  encoding; ``ast.literal_eval`` restores the native types (grid-index tuples
  stay tuples, int migration labels stay ints, reaction-name labels stay
  strings).
  """
  return "|".join((
      repr(float(rate)),
      repr(None if barrier is None else float(barrier)),
      repr(dest),
      repr(label),
      repr(origin),
  ))


def _parse_event(text: str) -> tuple:
  """Inverse of :func:`_format_event` -> (rate, barrier, dest, label, origin)."""
  rate, barrier, dest, label, origin = text.split("|")
  return (float(rate),
          None if barrier == "None" else float(barrier),
          ast.literal_eval(dest),
          ast.literal_eval(label),
          ast.literal_eval(origin))


def _event_fields(event):
  """(rate, barrier, dest, label) of a *registered* ``site_events`` entry.

  SINGLE POINT OF CONTACT for the internal event representation.  Phase 4 of
  the refactor replaced the raw ``[rate, dest, label, E_act]`` list with the
  ``Event`` dataclass, so the four physics columns now come from named fields;
  the fixture content is unaffected.
  """
  return (float(event.rate), float(event.barrier), event.destination,
          event.label)


def _chosen_fields(chosen_event):
  """(rate, dest, label, origin) of the tuple handed to ``processes``.

  SINGLE POINT OF CONTACT for the executed-event tuple (the kMC catalog tuple
  built by ``_kmc_step``; it does not carry the barrier, which the recorder
  resolves from the catalog).
  """
  return (float(chosen_event[0]), chosen_event[1], chosen_event[2],
          chosen_event[-1])


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _sha256_object(obj) -> str:
  payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# =============================================================================
# Lattice construction (production recipe) and trace capture
# =============================================================================


def _build_lattice(vcm_config, defects_config, vcm_act_e_dict):
  """Build the lattice exactly like the production ``electronic_device`` path.

  Deliberately mirrors ``tests/test_kmc_loop.py::_build_lattice`` (same grid
  fast-path load, same documented overrides) with ONE addition: the preset's
  Poisson configuration is forwarded so that

    * ``Crystal_Lattice.poisson_config`` is set, which makes
      ``_initialize_migration_pathways`` (crystal.py:542-566) re-inject the
      LIVE activation energies into every site and build the directional
      ``E_mig`` table, and
    * ``processes`` refreshes every mobile site's pathways each step.

  That is what the real CLI does (`initialize_grid_crystal`, initialization.py:
  416-423 passes ``solver_config`` including ``poisson_config``).  It also
  makes the trace depend on the live ``VCM_HfO2.json`` parameters instead of
  the activation energies baked into the pickled grid.  The Poisson solver
  itself is constructed lazily in ``cli.py``, so no dolfinx/mesh is needed
  here and ``_evaluate_fields_for_kmc`` returns empty field dicts.
  """
  reactions_config = (
      vcm_config.reactions.to_dict() if vcm_config.reactions else None
  )
  gb_configurations = (
      [gb.to_dict() for gb in vcm_config.grain_boundaries]
      if vcm_config.grain_boundaries else None
  )

  crystal = initialize_grid_crystal(
      GRID_NAME,
      None,  # mpi_ctx=None -> truly serial (production Phase-1 path)
      vcm_config.material,
      vcm_config.experimental,
      vcm_act_e_dict,
      vcm_config.settings.lammps_output,
      vcm_config.superbasin,
      False,  # save_data=False -> never writes grids/meshes
      settings=vcm_config.settings,
      rng=np.random.default_rng(SEED),
      cache_dir=get_parameters_root().parent / "cache",  # offline MP cache
      calculator_config=None,  # MACE is not wired into the kMC loop
      defects_config=defects_config,
      reactions_config=reactions_config,
      gb_configurations=gb_configurations,
      simulation_type=vcm_config.settings.simulation_type,
      solver_config={"poisson_config": vcm_config.poisson},
  )

  # Normally provided post-init by the CLI / ElectricalController.
  crystal.timestep_limits = float(vcm_config.superbasin.time_step_limits)
  crystal.last_field_solve_time = 0.0
  return crystal


def _catalog_snapshot(crystal):
  """The catalog ``_kmc_step`` builds, as ``(rate, barrier, dest, label, origin)``.

  Mirrors crystal.py:2718-2728: iterate ``active_event_sites +
  generation_sites``, using the superbasin's absorbing-state list for sites
  owned by a superbasin.  Snapshotting at ``processes`` entry means the rates
  were already refreshed by ``_update_rates_lazily`` and no state has mutated
  yet, so this is exactly the table the balanced tree was built from.
  """
  records = []
  for idx in crystal.active_event_sites + crystal.generation_sites:
    if idx in crystal.superbasin_dict:
      for item in crystal.superbasin_dict[idx].site_events_absorbing:
        records.append((float(item[0]), float(item[3]), item[1], item[2], idx))
      continue
    for event in crystal.grid_crystal[idx].defect.events:
      rate, barrier, dest, label = _event_fields(event)
      records.append((rate, barrier, dest, label, idx))
  return records


def _resolve_chosen_line(catalog, chosen):
  """Format the executed event with its barrier taken from the catalog entry.

  The kMC catalog tuple carries only ``(rate, dest, label, origin)``, so the
  barrier is recovered by matching those four values against the catalog.  The
  match being found is itself an invariant ("the selected event came from the
  catalog"); if it is not found, the line is emitted with ``None`` and the
  fixture comparison fails loudly.
  """
  rate, dest, label, origin = chosen
  for record in catalog:
    if (record[0] == rate and record[2] == dest and record[3] == label
        and record[4] == origin):
      return _format_event(*record)
  return _format_event(rate, None, dest, label, origin)



def _run_trace(crystal):
  """Run ``N_STEPS`` seeded steps and return the per-step records (in memory).

  ``crystal.processes`` is wrapped only to observe the catalog at the exact
  moment a kMC step executes it; the wrapper calls straight through to the
  original method, so the physics is untouched and the trace is a pure
  observation of the loop.
  """
  crystal.defect_gen()
  crystal._update_rates_lazily({}, {})  # materialize rates before step 1
  rng = crystal.rng
  calls = []
  original_processes = crystal.processes

  def recording_processes(chosen_event):
    calls.append((_catalog_snapshot(crystal), _chosen_fields(chosen_event)))
    original_processes(chosen_event)

  crystal.processes = recording_processes
  steps = []
  try:
    for step in range(1, N_STEPS + 1):
      n_calls_before = len(calls)
      time_before = crystal.time
      crystal.step_kmc(rng)
      dt = crystal.time - time_before
      n_processes_calls = len(calls) - n_calls_before
      if n_processes_calls:
        catalog, chosen = calls[n_calls_before]
        chosen_line = _resolve_chosen_line(catalog, chosen)
      else:
        # No event inside timestep_limits: the state is untouched, so taking
        # the catalog after the step is still the table that was used.
        catalog, chosen_line = _catalog_snapshot(crystal), None
      steps.append({
          "step": step,
          "time": float(crystal.time),
          "dt": float(dt),
          "sum_rate": float(sum(record[0] for record in catalog)),
          "n_catalog": len(catalog),
          "n_active_sites": len(crystal.active_event_sites),
          "n_generation_sites": len(crystal.generation_sites),
          "n_processes_calls": n_processes_calls,
          "chosen": chosen_line,
          "catalog": catalog,
      })
  finally:
    crystal.processes = original_processes
  return steps


def _run_cross_trace(crystal):
  """``CROSS_N_STEPS`` seeded steps plus every cross-sublattice hop seen.

  Same observation protocol as :func:`_run_trace` (the wrapper calls straight
  through to ``processes``, so the physics is untouched), with one addition:
  for every *migration* whose source and destination sublattice differ, the
  post-hop state of both sites is recorded.  Those records ARE the contract:

    * ``dst_defect`` must equal ``src_defect`` - the Defect object travelled,
      so the O-sublattice destination hosts the source's configuration
      (``oxygen_interstitial``) instead of re-deriving one from its own
      sublattice (which is what the pre-Phase-5 attribute flow did), and
    * ``src_is_empty`` must be True - the source is left with a fresh empty
      Defect (``clear_defect``), not with stale state.
  """
  crystal.defect_gen()
  crystal._update_rates_lazily({}, {})  # materialize rates before step 1
  rng = crystal.rng
  calls = []
  hops = []
  state = {"step": 0}
  original_processes = crystal.processes

  def recording_processes(chosen_event):
    catalog = _catalog_snapshot(crystal)
    chosen_line = _resolve_chosen_line(catalog, _chosen_fields(chosen_event))
    origin_idx, dest_idx = chosen_event[-1], chosen_event[1]
    origin_site = crystal.grid_crystal[origin_idx]
    dest_site = crystal.grid_crystal[dest_idx]
    src_sublattice = origin_site.site_type
    dst_sublattice = dest_site.site_type
    src_defect = origin_site.defect.config.name
    calls.append((catalog, _chosen_fields(chosen_event)))
    original_processes(chosen_event)
    if (isinstance(chosen_event[2], int)          # migration label
        and src_sublattice != dst_sublattice):
      rate, barrier, dest, label, origin = _parse_event(chosen_line)
      hops.append({
          "step": state["step"],
          "origin": list(origin),
          "dest": list(dest),
          "label": label,
          "rate": rate,
          "barrier": barrier,
          "src_sublattice": src_sublattice,
          "dst_sublattice": dst_sublattice,
          "src_defect": src_defect,
          "dst_defect": dest_site.defect.config.name,
          "dst_defect_sublattice": dest_site.defect.sublattice,
          "src_is_empty": origin_site.defect.is_empty,
      })

  crystal.processes = recording_processes
  steps = []
  try:
    for step in range(1, CROSS_N_STEPS + 1):
      state["step"] = step
      n_calls_before = len(calls)
      time_before = crystal.time
      crystal.step_kmc(rng)
      dt = crystal.time - time_before
      n_processes_calls = len(calls) - n_calls_before
      if n_processes_calls:
        catalog, chosen = calls[n_calls_before]
        chosen_line = _resolve_chosen_line(catalog, chosen)
      else:
        # No event inside timestep_limits: the state is untouched, so taking
        # the catalog after the step is still the table that was used.
        catalog, chosen_line = _catalog_snapshot(crystal), None
      steps.append({
          "step": step,
          "time": float(crystal.time),
          "dt": float(dt),
          "sum_rate": float(sum(record[0] for record in catalog)),
          "n_catalog": len(catalog),
          "n_active_sites": len(crystal.active_event_sites),
          "n_generation_sites": len(crystal.generation_sites),
          "n_processes_calls": n_processes_calls,
          "chosen": chosen_line,
          "catalog": catalog,
      })
  finally:
    crystal.processes = original_processes
  return steps, hops


# =============================================================================
# Fixture encoding (lossless, diff-minimal)
# =============================================================================


def _catalog_digest(lines) -> str:
  """Order-sensitive digest of one step's catalog lines."""
  return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _line_delta(old_lines, new_lines) -> list:
  """Positional, order-sensitive delta between two catalog line lists."""
  ops = []
  matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
  for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == "equal":
      continue
    ops.append({"op": tag, "i1": i1, "old": old_lines[i1:i2],
                "new": new_lines[j1:j2]})
  return ops


def _apply_line_delta(old_lines, delta) -> list:
  """Inverse of :func:`_line_delta` (opcodes are emitted in i1 order)."""
  lines = list(old_lines)
  shift = 0
  for op in delta:
    start = op["i1"] + shift
    n_old, n_new = len(op["old"]), len(op["new"])
    if op["op"] == "delete":
      del lines[start:start + n_old]
    elif op["op"] == "insert":
      lines[start:start] = op["new"]
    elif op["op"] == "replace":
      lines[start:start + n_old] = op["new"]
    else:
      raise AssertionError(f"unknown catalog delta op: {op['op']!r}")
    shift += n_new - n_old
  return lines



def _step_catalog_lines(step) -> list:
  """Catalog lines of a step, whether it came from memory or from the fixture."""
  catalog = step["catalog"]
  if catalog and isinstance(catalog[0], str):
    return list(catalog)
  return [_format_event(*record) for record in catalog]


def _encode_steps(steps) -> list:
  """JSON-ready steps: step 1 stores the full catalog, later steps store deltas."""
  encoded = []
  previous = None
  for entry in steps:
    lines = _step_catalog_lines(entry)
    block = {key: value for key, value in entry.items() if key != "catalog"}
    block["catalog_digest"] = _catalog_digest(lines)
    if previous is None:
      block["catalog"] = lines
    else:
      block["catalog_delta"] = _line_delta(previous, lines)
    previous = lines
    encoded.append(block)
  return encoded


def _decode_catalogs(encoded_steps) -> list:
  """Rebuild every step's catalog lines, verifying delta == recorded digest."""
  catalogs = []
  previous = None
  for index, block in enumerate(encoded_steps, start=1):
    if "catalog" in block:
      lines = list(block["catalog"])
    elif previous is not None:
      lines = _apply_line_delta(previous, block["catalog_delta"])
    else:
      raise AssertionError(
          f"step {index}: catalog_delta without a preceding catalog")
    digest = _catalog_digest(lines)
    assert digest == block["catalog_digest"], (
        f"step {index}: fixture integrity check failed - the recorded delta "
        f"does not reproduce catalog_digest ({digest} != "
        f"{block['catalog_digest']}). Regenerate the fixture with "
        f"{UPDATE_ENV_VAR}=1 instead of hand-editing it.")
    catalogs.append(lines)
    previous = lines
  return catalogs



# =============================================================================
# Provenance and fixture I/O
# =============================================================================

PROVENANCE_KEYS = ("schema", "preset", "grid", "n_sites", "seed", "n_steps",
                   "grid_sha256", "inputs_sha256")


def _build_meta(defects_config, vcm_act_e_dict, crystal) -> dict:
  """Provenance block: exactly what the trace was recorded against."""
  inputs = {
      "defects_config": defects_config,
      "reactions_config": crystal.reactions_config,
      "activation_energies": vcm_act_e_dict,
      "mode": crystal.mode,
      "technology": crystal.technology,
      "affected_site": crystal.affected_site,
      "sites_generation_layer": crystal.sites_generation_layer,
      "temperature": crystal.temperature,
      "simulation_type": crystal.simulation_type,
  }
  return {
      "schema": TRACE_SCHEMA,
      "generator": "tests/test_golden_trace.py",
      "preset": PRESET_NAME,
      "grid": GRID_NAME,
      "n_sites": len(crystal.grid_crystal),
      "seed": SEED,
      "n_steps": N_STEPS,
      "grid_sha256": _sha256_file(GRID_PATH) if GRID_PATH.exists() else None,
      "inputs_sha256": _sha256_object(inputs),
      "rate_rel_tol": RATE_REL_TOL,
      "rate_abs_tol": RATE_ABS_TOL,
      "python": platform.python_version(),
      "numpy": np.__version__,
      "site_config_source": (
          "loaded-grid path: each Site carries the defects_config/Act_E_dict "
          "pickled with data/grids/<grid>.pkl, and the live YAML/JSON inputs "
          "fingerprinted by inputs_sha256 are re-injected because the preset "
          "enables Poisson (crystal.py:542-566)"),
  }


def _build_cross_meta(defects_config, vcm_act_e_dict, crystal, hops) -> dict:
  """Provenance block of the cross-sublattice scenario.

  Records the scenario's own inputs - including the two test-side overrides, so
  that changing them fails the provenance test instead of silently re-blessing
  a different run - and fingerprints the recorded hops.
  """
  overrides = {
      f"{CROSS_DEFECT_NAME}.valid_target_species":
          list(CROSS_VALID_TARGET_SPECIES),
      "oxygen_vacancy.initial_concentration_bulk": CROSS_V_O_CONCENTRATION,
      "site.defects_config": (
          "live defects_config re-injected into every Site (the pickled grid "
          "copy would silently ignore the overrides above)"),
  }
  inputs = {
      "defects_config": defects_config,
      "reactions_config": crystal.reactions_config,
      "activation_energies": vcm_act_e_dict,
      "mode": crystal.mode,
      "technology": crystal.technology,
      "affected_site": crystal.affected_site,
      "sites_generation_layer": crystal.sites_generation_layer,
      "temperature": crystal.temperature,
      "simulation_type": crystal.simulation_type,
  }
  return {
      "schema": CROSS_SCHEMA,
      "scenario": "cross_sublattice",
      "generator": "tests/test_golden_trace.py",
      "preset": PRESET_NAME,
      "grid": GRID_NAME,
      "n_sites": len(crystal.grid_crystal),
      "seed": CROSS_SEED,
      "n_steps": CROSS_N_STEPS,
      "grid_sha256": _sha256_file(GRID_PATH) if GRID_PATH.exists() else None,
      "inputs_sha256": _sha256_object(inputs),
      "rate_rel_tol": RATE_REL_TOL,
      "rate_abs_tol": RATE_ABS_TOL,
      "python": platform.python_version(),
      "numpy": np.__version__,
      "overrides": overrides,
      "cross_hops": len(hops),
      "cross_hops_sha256": _sha256_object(hops),
  }


def _write_fixture(trace) -> None:
  """Atomic write, mirroring kinetix/initialization.py:_save_grid_atomic."""
  FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
  temp_path = FIXTURE_PATH.with_suffix(".json.tmp")
  with open(temp_path, "w") as handle:
    json.dump(trace, handle, indent=1)
    handle.write("\n")
  os.replace(str(temp_path), str(FIXTURE_PATH))


def _read_fixture() -> dict:
  with open(FIXTURE_PATH) as handle:
    return json.load(handle)


def _write_cross_fixture(trace) -> None:
  """Atomic write of the cross fixture (same pattern as _write_fixture)."""
  CROSS_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
  temp_path = CROSS_FIXTURE_PATH.with_suffix(".json.tmp")
  with open(temp_path, "w") as handle:
    json.dump(trace, handle, indent=1)
    handle.write("\n")
  os.replace(str(temp_path), str(CROSS_FIXTURE_PATH))


def _read_cross_fixture() -> dict:
  with open(CROSS_FIXTURE_PATH) as handle:
    return json.load(handle)


def _load_or_update_cross_fixture(actual_trace) -> dict:
  """Recorded cross trace, or (re)written when regeneration was asked for."""
  if os.environ.get(UPDATE_ENV_VAR):
    _write_cross_fixture(actual_trace)
    pytest.skip(f"{UPDATE_ENV_VAR} set: rewrote {CROSS_FIXTURE_PATH}")
  if not CROSS_FIXTURE_PATH.exists():
    pytest.fail(
        f"cross-sublattice golden trace fixture is missing: "
        f"{CROSS_FIXTURE_PATH}\n"
        f"Generate it with:  {UPDATE_ENV_VAR}=1 python -m pytest "
        f"tests/test_golden_trace.py::test_golden_trace_cross_sublattice")
  return _read_cross_fixture()


# =============================================================================
# Trace comparison
# =============================================================================


def _values_equal(expected, actual) -> bool:
  """Tolerant, type-aware comparison (floats vs tuples vs labels/strings)."""
  if isinstance(expected, tuple) and isinstance(actual, tuple):
    return (len(expected) == len(actual) and
            all(_values_equal(exp, act) for exp, act in zip(expected, actual)))
  if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
    return math.isclose(float(expected), float(actual),
                        rel_tol=RATE_REL_TOL, abs_tol=RATE_ABS_TOL)
  if expected is None or actual is None:
    return expected is actual
  return expected == actual


def _normalize_block(block) -> dict:
  """Scalars of one step plus the parsed executed event, for comparison."""
  return {
      "step": int(block["step"]),
      "time": float(block["time"]),
      "dt": float(block["dt"]),
      "sum_rate": float(block["sum_rate"]),
      "n_catalog": int(block["n_catalog"]),
      "n_active_sites": int(block["n_active_sites"]),
      "n_generation_sites": int(block["n_generation_sites"]),
      "n_processes_calls": int(block["n_processes_calls"]),
      "chosen": None if block["chosen"] is None else _parse_event(block["chosen"]),
  }


SCALAR_KEYS = ("time", "dt", "sum_rate")
COUNTER_KEYS = ("n_catalog", "n_active_sites", "n_generation_sites",
                "n_processes_calls")


def _diff_steps(expected_blocks, actual_blocks, max_diffs=25) -> list:
  """Per-step scalar/kernel differences between two encoded traces."""
  expected = [_normalize_block(block) for block in expected_blocks]
  actual = [_normalize_block(block) for block in actual_blocks]
  diffs = []
  if len(expected) != len(actual):
    diffs.append(f"number of steps: fixture {len(expected)} != actual "
                 f"{len(actual)}")
  for exp, act in zip(expected, actual):
    if exp["step"] != act["step"]:
      diffs.append(f"step index mismatch: {exp['step']} != {act['step']}")
    for key in SCALAR_KEYS:
      if not _values_equal(exp[key], act[key]):
        diffs.append(f"step {exp['step']}: {key} {exp[key]!r} != {act[key]!r}")
    for key in COUNTER_KEYS:
      if exp[key] != act[key]:
        diffs.append(f"step {exp['step']}: {key} {exp[key]} != {act[key]}")
    if not _values_equal(exp["chosen"], act["chosen"]):
      diffs.append(f"step {exp['step']}: executed event differs\n"
                   f"    fixture: {exp['chosen']}\n    actual : {act['chosen']}")
  if len(diffs) > max_diffs:
    diffs = diffs[:max_diffs] + [f"... ({len(diffs) - max_diffs} more)"]
  return diffs


def _diff_catalog(expected_lines, actual_lines, step, max_diffs=10,
                  context=1) -> list:
  """Differences between one step's catalog lines (order-sensitive)."""
  if expected_lines == actual_lines:
    return []
  diffs = []
  if len(expected_lines) != len(actual_lines):
    diffs.append(f"step {step}: catalog has {len(actual_lines)} events, "
                 f"fixture has {len(expected_lines)}")
  for index, (exp_line, act_line) in enumerate(zip(expected_lines,
                                                   actual_lines)):
    if _values_equal(_parse_event(exp_line), _parse_event(act_line)):
      continue
    diffs.append(f"step {step}: catalog[{index}] {exp_line} != {act_line}")
    if len(diffs) >= max_diffs:
      break
  report = list(difflib.unified_diff(expected_lines, actual_lines,
                                     fromfile=f"fixture/step{step}",
                                     tofile=f"actual/step{step}", n=context))
  if report:
    diffs.append(f"--- unified diff of step {step} (first 40 lines) ---")
    diffs.extend(report[:40])
  return diffs


def _load_or_update_fixture(actual_trace) -> dict:
  """Return the recorded trace, or (re)write it when regeneration was asked for."""
  if os.environ.get(UPDATE_ENV_VAR):
    _write_fixture(actual_trace)
    pytest.skip(f"{UPDATE_ENV_VAR} set: rewrote {FIXTURE_PATH}")
  if not FIXTURE_PATH.exists():
    pytest.fail(
        f"golden trace fixture is missing: {FIXTURE_PATH}\n"
        f"Generate it with:  {UPDATE_ENV_VAR}=1 python -m pytest "
        f"tests/test_golden_trace.py")
  return _read_fixture()




# =============================================================================
# Fixtures
# =============================================================================

# Production parameter-sweep pattern (the shipped YAML uses 0.0 for every
# defect, which would give a valid but eventless run).  Same value as
# tests/test_kmc_loop.py.
BULK_CONCENTRATION = 0.05


def _load_vcm_config() -> SimulationConfig:
  """REAL VCM_mock preset plus the documented offline overrides.

  Mirrors the fixtures of ``tests/test_kmc_loop.py``:
    1. ``material.formula`` would normally come from Materials Project.
    2. ``oxygen_interstitial.initial_concentration_bulk`` is raised so
       ``defect_gen()`` actually introduces defects.
  """
  config = SimulationConfig.from_yaml(
      get_parameters_root() / "presets" / PRESET_NAME)
  config.material.formula = "HfO2"
  config.defects.defects["oxygen_interstitial"].initial_concentration_bulk = (
      BULK_CONCENTRATION)
  return config


def _load_act_e(config: SimulationConfig, defects_config: dict) -> dict:
  """Activation energies through the production pipeline (real JSON file)."""
  return _process_activation_energies(
      defects_config,
      load_activation_energies(get_parameters_root() / "presets" / PRESET_NAME,
                               config.settings),
      config.settings.technology)


@pytest.fixture(scope="module")
def grid_path():
  """The cached production grid, or skip (creating one needs network)."""
  if not GRID_PATH.exists():
    pytest.skip(f"production grid not found: {GRID_PATH} - run a VCM_mock "
                "simulation once so the grid cache exists")
  return GRID_PATH


@pytest.fixture(scope="module")
def vcm_config():
  return _load_vcm_config()


@pytest.fixture(scope="module")
def defects_config(vcm_config):
  return vcm_config.defects.to_dict()


@pytest.fixture(scope="module")
def vcm_act_e_dict(vcm_config, defects_config):
  """LIVE activation energies through the production pipeline.

  NOTE: ``tests/test_kmc_loop.py::vcm_act_e_dict`` has an empty body (returns
  ``None``), so that test runs on the activation energies pickled inside the
  grid.  Here the live JSON parameters drive the run instead, which is what the
  production CLI does - and which is the binding Phase 6 of the refactor
  rewrites.
  """
  return _load_act_e(vcm_config, defects_config)


@pytest.fixture(scope="module")
def actual_trace(grid_path, vcm_config, defects_config, vcm_act_e_dict):
  """One deterministic 30-step run of the kMC loop (JSON-ready steps)."""
  crystal = _build_lattice(vcm_config, defects_config, vcm_act_e_dict)
  assert len(crystal.grid_crystal) == EXPECTED_N_SITES, (
      f"expected the cached {EXPECTED_N_SITES}-site grid, got "
      f"{len(crystal.grid_crystal)} - was the grid rebuilt from scratch?")
  steps = _run_trace(crystal)
  return {"meta": _build_meta(defects_config, vcm_act_e_dict, crystal),
          "steps": _encode_steps(steps)}


# -----------------------------------------------------------------------------
# Cross-sublattice scenario fixtures (see the module-level comment for why the
# shipped presets cannot produce one and what the two overrides change).
# -----------------------------------------------------------------------------

def _load_cross_config() -> SimulationConfig:
  """VCM_mock preset plus the cross-sublattice scenario's overrides."""
  config = _load_vcm_config()
  config.defects.defects["oxygen_vacancy"].initial_concentration_bulk = (
      CROSS_V_O_CONCENTRATION)
  return config


def _cross_defects_config(config: SimulationConfig) -> dict:
  """Live defects_config where O_i may also hop onto oxygen-vacancy sites."""
  defects = config.defects.to_dict()
  defects[CROSS_DEFECT_NAME]["valid_target_species"] = (
      list(CROSS_VALID_TARGET_SPECIES))
  return defects


def _build_cross_lattice(vcm_config, defects_config, vcm_act_e_dict):
  """Build the cross scenario lattice and re-inject the live defects_config.

  Every Site is unpickled with the defects_config stored inside the grid file,
  and the production re-injection refreshes ``Act_E_dict`` only
  (crystal.py:542-566).  Without this step the scenario's overrides would never
  reach the migration gate - ``Site.available_migrations`` reads
  ``self.defects_config`` - and no cross-sublattice event would be offered.
  """
  crystal = _build_lattice(vcm_config, defects_config, vcm_act_e_dict)
  for site in crystal.grid_crystal.values():
    site.defects_config = defects_config
  return crystal


@pytest.fixture(scope="module")
def cross_config():
  return _load_cross_config()


@pytest.fixture(scope="module")
def cross_defects_config(cross_config):
  return _cross_defects_config(cross_config)


@pytest.fixture(scope="module")
def cross_act_e_dict(cross_config, cross_defects_config):
  return _load_act_e(cross_config, cross_defects_config)


@pytest.fixture(scope="module")
def cross_actual_trace(grid_path, cross_config, cross_defects_config,
                       cross_act_e_dict):
  """One deterministic run whose catalogs contain cross-sublattice hops."""
  crystal = _build_cross_lattice(cross_config, cross_defects_config,
                                 cross_act_e_dict)
  assert len(crystal.grid_crystal) == EXPECTED_N_SITES, (
      f"expected the cached {EXPECTED_N_SITES}-site grid, got "
      f"{len(crystal.grid_crystal)} - was the grid rebuilt from scratch?")
  steps, hops = _run_cross_trace(crystal)
  return {
      "meta": _build_cross_meta(cross_defects_config, cross_act_e_dict,
                                crystal, hops),
      "steps": _encode_steps(steps),
      "cross_hops": hops,
  }



# =============================================================================
# Tests
# =============================================================================

REGENERATE_HINT = (
    "If the change is intentional and only moves state (not physics), the\n"
    "trace must NOT change - fix the refactor.  If the physics or the input\n"
    "parameters changed intentionally, regenerate and review the diff:\n"
    f"    {UPDATE_ENV_VAR}=1 python -m pytest tests/test_golden_trace.py\n"
    f"    git diff {FIXTURE_PATH.relative_to(FIXTURE_PATH.parent.parent.parent)}"
)


def test_fixture_provenance(actual_trace):
  """The fixture must have been recorded on exactly these inputs."""
  fixture = _load_or_update_fixture(actual_trace)
  expected_meta = fixture["meta"]
  actual_meta = actual_trace["meta"]
  mismatches = [
      f"  {key}: fixture={expected_meta.get(key)!r}  "
      f"actual={actual_meta.get(key)!r}"
      for key in PROVENANCE_KEYS
      if expected_meta.get(key) != actual_meta.get(key)
  ]
  assert not mismatches, (
      "the golden trace was recorded against different inputs "
      "(grid file, defect/reaction/activation-energy parameters or schema).\n"
      + "\n".join(mismatches) + "\n\n" + REGENERATE_HINT)


def test_golden_trace_matches_fixture(actual_trace):
  """The kMC execution profile must be identical to the recorded one.

  Compares, for every step: catalog size and every
  ``(rate, barrier, dest, label, origin)`` record in order, the sum of rates,
  the executed event, the elapsed and cumulative kMC time and the occupancy
  counters.  This must hold unchanged across every phase of the Site/Defect
  refactor.
  """
  fixture = _load_or_update_fixture(actual_trace)

  # Cheap first pass: the digest is order-sensitive, so a mismatch here means
  # "something in this step moved" before the expensive positional comparison.
  digest_diffs = [
      f"step {expected['step']}: catalog digest {expected['catalog_digest']} "
      f"!= {actual['catalog_digest']}"
      for expected, actual in zip(fixture["steps"], actual_trace["steps"])
      if expected["catalog_digest"] != actual["catalog_digest"]
  ]

  expected_catalogs = _decode_catalogs(fixture["steps"])
  actual_catalogs = _decode_catalogs(actual_trace["steps"])
  diffs = list(digest_diffs)
  diffs.extend(_diff_steps(fixture["steps"], actual_trace["steps"]))
  for step, (expected_lines, actual_lines) in enumerate(
      zip(expected_catalogs, actual_catalogs), start=1):
    if len(diffs) >= 60:
      break
    diffs.extend(_diff_catalog(expected_lines, actual_lines, step, max_diffs=5))

  assert not diffs, (
      "the kMC execution profile drifted from the golden trace "
      f"({len(diffs)} difference(s) reported).\n\n"
      + "\n".join(diffs[:60]) + "\n\n" + REGENERATE_HINT)


CROSS_REGENERATE_HINT = (
    "This fixture is the contract for object-transfer hops: after a\n"
    "cross-sublattice migration the destination must host the SOURCE's\n"
    "DefectConfig and the source must be left with a fresh empty Defect.\n"
    "If this fails after a refactor, the refactor is wrong - regenerate only\n"
    "when the scenario's inputs changed on purpose:\n"
    f"    {UPDATE_ENV_VAR}=1 python -m pytest tests/test_golden_trace.py::"
    f"test_golden_trace_cross_sublattice\n"
    f"    git diff "
    f"{CROSS_FIXTURE_PATH.relative_to(CROSS_FIXTURE_PATH.parent.parent.parent)}"
)


def _diff_cross_hops(expected_hops, actual_hops, max_diffs=10) -> list:
  """Field-level differences between two cross-hop record lists."""
  diffs = []
  if len(expected_hops) != len(actual_hops):
    diffs.append(f"cross-sublattice hops: fixture {len(expected_hops)} != "
                 f"actual {len(actual_hops)}")
  for index, (expected, actual) in enumerate(zip(expected_hops, actual_hops)):
    if set(expected) != set(actual):
      diffs.append(f"hop {index}: fields differ: fixture {sorted(expected)} "
                   f"!= actual {sorted(actual)}")
      continue
    for key in sorted(expected):
      if not _values_equal(expected[key], actual[key]):
        diffs.append(f"hop {index} (step {expected.get('step')}): {key} "
                     f"{expected[key]!r} != {actual[key]!r}")
    if len(diffs) >= max_diffs:
      break
  return diffs


def test_golden_trace_cross_sublattice(cross_actual_trace):
  """A hop between sublattices must move the Defect object, config included.

  The shipped presets cannot offer a cross-sublattice migration at all (see the
  module-level comment), so this scenario adds the two documented test-side
  overrides that make the engine's "Empty or Vacancy" destination contract
  reachable.  The trace is compared exactly like the main one - provenance,
  per-step digests, catalog lines, executed event - plus the recorded hop
  records, which pin the semantics object transfer changed:

    * the O-sublattice destination hosts the SOURCE's config
      (``oxygen_interstitial``, whose own site_type is ``interstitial``) rather
      than a config re-derived from its own sublattice, and
    * the source is left with a fresh empty Defect.
  """
  fixture = _load_or_update_cross_fixture(cross_actual_trace)

  # --- provenance ----------------------------------------------------------
  expected_meta = fixture["meta"]
  actual_meta = cross_actual_trace["meta"]
  mismatches = [
      f"  {key}: fixture={expected_meta.get(key)!r}  "
      f"actual={actual_meta.get(key)!r}"
      for key in CROSS_PROVENANCE_KEYS
      if expected_meta.get(key) != actual_meta.get(key)
  ]
  assert not mismatches, (
      "the cross-sublattice trace was recorded against different inputs.\n"
      + "\n".join(mismatches) + "\n\n" + CROSS_REGENERATE_HINT)

  # --- per-step trace, same machinery as the main trace --------------------
  digest_diffs = [
      f"step {expected['step']}: catalog digest {expected['catalog_digest']} "
      f"!= {actual['catalog_digest']}"
      for expected, actual in zip(fixture["steps"],
                                  cross_actual_trace["steps"])
      if expected["catalog_digest"] != actual["catalog_digest"]
  ]
  expected_catalogs = _decode_catalogs(fixture["steps"])
  actual_catalogs = _decode_catalogs(cross_actual_trace["steps"])
  diffs = list(digest_diffs)
  diffs.extend(_diff_steps(fixture["steps"], cross_actual_trace["steps"]))
  for step, (expected_lines, actual_lines) in enumerate(
      zip(expected_catalogs, actual_catalogs), start=1):
    if len(diffs) >= 60:
      break
    diffs.extend(_diff_catalog(expected_lines, actual_lines, step, max_diffs=5))
  assert not diffs, (
      "the cross-sublattice trace drifted from its golden fixture "
      f"({len(diffs)} difference(s) reported).\n\n"
      + "\n".join(diffs[:60]) + "\n\n" + CROSS_REGENERATE_HINT)

  # --- the recorded hops ---------------------------------------------------
  expected_hops = fixture["cross_hops"]
  actual_hops = cross_actual_trace["cross_hops"]
  assert len(expected_hops) >= 1, (
      "the fixture records no cross-sublattice hop - the scenario no longer "
      "exercises what it exists for.\n\n" + CROSS_REGENERATE_HINT)
  hop_diffs = _diff_cross_hops(expected_hops, actual_hops)
  assert not hop_diffs, (
      "the cross-sublattice hops drifted from the golden fixture "
      f"({len(hop_diffs)} difference(s) reported).\n\n"
      + "\n".join(hop_diffs[:10]) + "\n\n" + CROSS_REGENERATE_HINT)

  # --- semantic contract, independent of the fixture -----------------------
  for hop in actual_hops:
    where = f"step {hop['step']}: {hop['origin']} -> {hop['dest']}"
    assert hop["src_sublattice"] == "interstitial", where
    assert hop["dst_sublattice"] == CROSS_DESTINATION_SUBLATTICE, where
    assert hop["src_defect"] == CROSS_DEFECT_NAME, where
    assert hop["dst_defect"] == hop["src_defect"], (
        f"{where}: the destination hosts {hop['dst_defect']!r} instead of the "
        f"migrating {hop['src_defect']!r} - the hop did not carry the Defect "
        f"object (pre-Phase-5 semantics re-derived the destination config "
        f"from its own sublattice)")
    assert hop["dst_defect_sublattice"] == "interstitial", (
        f"{where}: expected the carried config to stay an interstitial-borne "
        f"one, got site_type {hop['dst_defect_sublattice']!r}")
    assert hop["src_is_empty"] is True, (
        f"{where}: the source was not left with a fresh empty Defect")


def test_trace_internal_invariants(actual_trace):
  """The captured trace must satisfy the kMC loop's own bookkeeping rules.

  These invariants hold regardless of the refactor phase and catch recorder /
  encoding regressions that a fixture comparison alone could mask (e.g. an
  executed event that was fabricated outside the catalog).  Cross-build
  determinism ("same seed -> same trajectory") is covered by
  ``tests/test_kmc_loop.py::TestDeterminism`` - deliberately not repeated here
  because every lattice build costs ~7 s.
  """
  blocks = actual_trace["steps"]
  catalogs = _decode_catalogs(blocks)

  previous_time = 0.0
  for block, lines in zip(blocks, catalogs):
    step = int(block["step"])
    time_after = float(block["time"])
    dt = float(block["dt"])

    assert time_after == pytest.approx(previous_time + dt, rel=1e-9, abs=1e-15), (
        f"step {step}: kMC time is not cumulative "
        f"({previous_time} + {dt} != {time_after})")
    previous_time = time_after

    assert int(block["n_catalog"]) == len(lines), (
        f"step {step}: n_catalog {block['n_catalog']} != {len(lines)} catalog lines")
    if lines:
      assert float(block["sum_rate"]) > 0.0, (
          f"step {step}: catalog is non-empty but the total rate is not positive")

    assert int(block["n_processes_calls"]) in (0, 1), (
        f"step {step}: {block['n_processes_calls']} processes() calls in one "
        "step - an extra (superbasin/virtual-move) execution would be a "
        "behaviour change and must be reviewed, not silently absorbed")

    chosen = block["chosen"]
    if chosen is None:
      assert int(block["n_processes_calls"]) == 0, (
          f"step {step}: an event executed but no event was recorded")
      continue
    parsed = _parse_event(chosen)
    assert parsed[1] is not None, (
        f"step {step}: the executed event {chosen!r} was not found in the "
        "catalog it was drawn from")
    assert int(block["n_processes_calls"]) == 1
    assert chosen in lines, (
        f"step {step}: executed event {chosen!r} is absent from the recorded "
        "catalog it was drawn from")



def test_comparator_detects_drift(actual_trace):
  """The golden machinery itself must reject every kind of drift it claims to.

  A golden fixture is only as good as its comparator, so this test perturbs the
  freshly captured trace in memory and asserts each perturbation is caught
  (and that an unperturbed copy produces no differences).
  """
  baseline = actual_trace["steps"]
  baseline_catalogs = _decode_catalogs(baseline)
  lines = baseline_catalogs[0]

  # No false positives on an untouched trace.
  assert _diff_steps(baseline, baseline) == []
  assert _diff_catalog(lines, lines, 1) == []

  # --- catalog entry drift: rate, destination, ordering, size ---------------
  record = list(_parse_event(lines[0]))
  rate_drifted = [_format_event(record[0] * (1.0 + 1e-6), *record[1:])] + lines[1:]
  assert _diff_catalog(lines, rate_drifted, 1), "rate drift was not detected"

  alternative = next(_parse_event(line) for line in lines
                     if _parse_event(line)[2] != record[2])
  dest_record = list(record)
  dest_record[2] = alternative[2]
  dest_drifted = [_format_event(*dest_record)] + lines[1:]
  assert _diff_catalog(lines, dest_drifted, 1), "destination drift was not detected"
  assert _catalog_digest(dest_drifted) != _catalog_digest(lines), (
      "catalog digest is not sensitive to the destination")

  swap_index = next(index for index, line in enumerate(lines) if line != lines[0])
  swapped = list(lines)
  swapped[0], swapped[swap_index] = swapped[swap_index], swapped[0]
  assert _catalog_digest(swapped) != _catalog_digest(lines), (
      "catalog digest is not order sensitive")
  assert _diff_catalog(lines, swapped, 1), "event reordering was not detected"

  assert _diff_catalog(lines, lines[:10], 1), "catalog truncation was not detected"

  # --- per-step scalar / executed-event / counter drift ---------------------
  blocks = [dict(block) for block in baseline]

  sum_rate_drifted = [dict(block) for block in blocks]
  sum_rate_drifted[4]["sum_rate"] = blocks[4]["sum_rate"] * (1.0 + 1e-6)
  assert _diff_steps(baseline, sum_rate_drifted), "sum_rate drift was not detected"

  time_drifted = [dict(block) for block in blocks]
  time_drifted[6]["time"] = blocks[6]["time"] * (1.0 + 1e-6)
  assert _diff_steps(baseline, time_drifted), "kMC time drift was not detected"

  dt_drifted = [dict(block) for block in blocks]
  dt_drifted[8]["dt"] = blocks[8]["dt"] * (1.0 + 1e-6)
  assert _diff_steps(baseline, dt_drifted), "per-step dt drift was not detected"

  event_drifted = [dict(block) for block in blocks]
  replacement = next(line for line in lines if line != blocks[9]["chosen"])
  event_drifted[9]["chosen"] = replacement
  assert _diff_steps(baseline, event_drifted), (
      "executed-event drift was not detected")

  counter_drifted = [dict(block) for block in blocks]
  counter_drifted[2]["n_active_sites"] = blocks[2]["n_active_sites"] + 1
  assert _diff_steps(baseline, counter_drifted), (
      "occupancy counter drift was not detected")

  call_drifted = [dict(block) for block in blocks]
  call_drifted[3]["n_processes_calls"] = blocks[3]["n_processes_calls"] + 1
  assert _diff_steps(baseline, call_drifted), (
      "superbasin/virtual-move drift was not detected")


# =============================================================================
# Fixture generator
# =============================================================================


def generate_fixture() -> dict:
  """Build the lattice, run the trace and (over)write the fixture."""
  assert GRID_PATH.exists(), (
      f"cannot record a golden trace without the cached production grid: "
      f"{GRID_PATH}")
  config = _load_vcm_config()
  defects_config = config.defects.to_dict()
  act_e_dict = _load_act_e(config, defects_config)
  crystal = _build_lattice(config, defects_config, act_e_dict)
  steps = _run_trace(crystal)
  trace = {"meta": _build_meta(defects_config, act_e_dict, crystal),
           "steps": _encode_steps(steps)}
  _write_fixture(trace)
  return trace


if __name__ == "__main__":
  trace = generate_fixture()
  blocks = trace["steps"]
  print(f"golden trace written: {FIXTURE_PATH} "
        f"({FIXTURE_PATH.stat().st_size} bytes)")
  print(f"  preset={trace['meta']['preset']} grid={trace['meta']['grid']} "
        f"sites={trace['meta']['n_sites']} seed={trace['meta']['seed']} "
        f"steps={len(blocks)}")
  print(f"  inputs_sha256={trace['meta']['inputs_sha256']}")
  for block in blocks[:3] + blocks[-1:]:
    print(f"  step {block['step']:>2}: n_catalog={block['n_catalog']:>5} "
          f"sum_rate={block['sum_rate']:.9e} dt={block['dt']:.6e} "
          f"time={block['time']:.6e} chosen={block['chosen']}")

