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
``passivation_level`` / ``migrating_attributes`` directly, so Phases 3 and 5
must leave the trace bit-identical without any edit here.

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
    for event in crystal.grid_crystal[idx].site_events:
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

