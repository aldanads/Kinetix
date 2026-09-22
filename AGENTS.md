# Kinetix — Agent Context

> Read this file at the start of every session. It is maintained by the agents
> working on the Site/Defect decoupling refactor; update it when architecture,
> phases, or bugs change (see *Standing Tasks* at the bottom).

## Project Overview
- **kMC simulator for resistive switching** (ReRAM/memristors): defect migration,
  filament formation/dissolution in dielectrics.
- Supports **VCM** (valence-change) and **ECM** (electrochemical metallization)
  mechanisms; also legacy deposition/annealing paths (NOT test-covered).
- Multiphysics coupling: Poisson (electrostatics) + heat equation (Joule heating)
  + stochastic defect kinetics, solved with DOLFINx/FEniCS on FEM meshes (gmsh).
- Event selection: rejection-free **BKL algorithm on a balanced binary tree**.
- CLI entry: `kinetix/cli.py` drives `System_state.step_kmc()`
  (`kinetix/lattice/crystal.py`). Poisson/heat solvers are Linux-only.
- Author: Samuel Aldana Delgado (Tyndall Institute), MIT license.

## Environment
- **Python 3.12.8**, conda env `Kinetix` (requires-python >=3.10).
- **IMPORTANT**: the *base* conda env lacks pymatgen — always use the Kinetix env:
  - `conda activate Kinetix`
  - Full interpreter path: `/local/anaconda3/envs/Kinetix/bin/python`
- Key deps: numpy (`>=1.24,<2` — must stay <2 for pymatgen), scipy, pandas,
  pymatgen 2025.10.7, mpi4py, fenics-dolfinx 0.8.0, gmsh, petsc4py, pytest 9.0.2.
- Optional `mace` extra (pyproject): ase, mace-torch, torch, huggingface_hub —
  MACE NEB barrier calculator; models fetched from the HF Hub.
- pytest config lives in `pyproject.toml` (`[tool.pytest.ini_options]`: the
  `slow`/`solver`/`mace` markers); `tests/` is a package (`tests/__init__.py`),
  so plain `python -m pytest tests/` from the repo root works without sys.path
  hacks.
- `git` binary must be available (provenance capture in `_get_git_provenance`).

## Architecture (Key Modules)
| Module | Role |
|---|---|
| `kinetix/lattice/crystal.py` (~4.5k lines) | `Crystal_Lattice`/`System_state` — kMC loop, event execution (`processes` :2905, handlers :3004+), field solving, GB charge states |
| `kinetix/lattice/site.py` (~1.3k lines) | `Site` — lattice site with topology + **Defect composition**; event generation (`available_pathways` :610, `available_reactions` :818) |
| `kinetix/lattice/defect.py` (190 lines) | `Event` (:40, `catalog_tuple` :69), `Defect` (:80), `EMPTY_DEFECT_CONFIG` (:157), `make_empty_defect` (:175) |
| `kinetix/lattice/cluster.py` | Cluster/filament analysis (`_find_clusters`) |
| `kinetix/lattice/island.py` | Island — deposition morphology |
| `kinetix/lattice/grain_boundary.py` | GB geometry + barrier modifications (planar/cylindrical/triple-junction) |
| `kinetix/utils/superbasin.py` | Superbasin acceleration (note quirk at :319 — bug B2) |
| `kinetix/utils/balanced_tree.py` | Balanced binary tree for BKL event selection |
| `kinetix/utils/state_loader.py` | LAMMPS dump parsing, state restore |
| `kinetix/configs/` | Typed config dataclasses: `simulation_config`, `defect_config`, `reaction_config`, `electrical_config`, `grain_boundary_config`, `material_config`, `mesh_config`, `solver_config`, `calculator_config` + `config_loader` |
| `kinetix/initialization.py` | Lattice construction, grid loading/caching, config wiring |
| `kinetix/solvers/` | Poisson + heat FEM solvers (DOLFINx) |
| `kinetix/calculators/mace_neb.py` | MACE NEB barrier calculator (optional, GPU; `max_passivation_level` validated here) |
| `kinetix/logging_config.py` | Root logger `propagate=False` (affects caplog — see Testing) |
| `kinetix/cli.py` / `kinetix/__main__.py` | Simulation workflow driver |

## Site/Defect Refactor Status (Epic: decouple defect state from Site)
**Target:** Site *has-a* Defect composition model; the Defect carries all dynamic
state; the flat attribute-by-attribute migration machinery is gone.

| Phase | Description | Status |
|---|---|---|
| 0 | Golden trace test (physics-preservation contract) | ✅ `6ba9dcf` |
| 1 | Introduce `Event`/`Defect` dataclasses (unwired) | ✅ `dfbf17d` |
| 2 | `Site.defect` always present + `Site.idx` | ✅ `4a6cb46` |
| 3 | Move attributes to Defect (delegating properties) | ✅ `dc8859f` |
| 4 | Hot-path migration to `list[Event]` (`site_events` is `list[Event]`) | ✅ `f806250` |
| 5 | Remove `migrating_attributes` runtime use → object-transfer hops | ✅ `a224b49` |
| 6 | Delete dict flow + legacy bridges (delegating properties, YAML keys) | ⏭️ Next |

**Phase 5 essentials (current state):**
- Migration hop = `dst.install_defect(src.defect)` + `src.clear_defect()`
  (crystal.py `_handle_migration_event` :3004). The Defect object transfers by
  reference; the source gets a fresh empty Defect.
- `_install_defect_site` (:3337) / `_remove_species_at_site` (:3386) do the
  dirty-site bookkeeping; `extra_state`/`attributes_to_reset` parameters are gone.
- GB charge override is written through to `defect.charge` **before** the hop.
- `Site.get_migrating_state` is deleted; `DefectConfig.migrating_attributes`
  (YAML key) still exists for Phase 6 cleanup.
- `Site.introduce_specie` (:563): re-introducing the **same species keeps the
  Defect object** and only refreshes charge (passivation is physics-carrying);
  a species swap installs a fresh Defect (no state inheritance).

**Key design decisions:**
- Every Site always hosts exactly one Defect (empty sites get an empty Defect, never None).
- `site_type` (sublattice) belongs to Site, not Defect; `EMPTY_DEFECT_CONFIG.site_type = "Empty"`.
- Config-time-immutable: `enabled_events`, `max_passivation_level`, barriers. Runtime-dynamic (on Defect): `chemical_specie`, `charge`, `passivation_level`, `events`.
- No DefectRegistry — lookup is plain `dict[str, DefectConfig]` (legacy dict-of-dicts still flows through until Phase 6).
- No shared/singleton Defect instances across sites (per-site mutable state).
- Sublattice-based config resolution (`_get_current_defect_name` :301) preserved until Phase 6 (compat finding C1).
- `Event.catalog_tuple()` (:69) format is depended on by the balanced tree + superbasin.
- Pickle compat: `Site.__setstate__` (:227) migrates legacy flat-attribute pickles onto a Defect and normalizes events to `Event`.

## Golden Trace Contract
- Files: `tests/test_golden_trace.py` (5 tests) +
  `tests/fixtures/golden_trace_vcm_hfo2_seed42_n30.json` (main trace) and
  `tests/fixtures/golden_trace_vcm_hfo2_cross_sublattice.json` (Finding A).
- Main scenario: `VCM_mock.yaml` preset, `grid_HfO2_3nm.pkl`, seed 42, 30 kMC steps.
- Records per step: time, dt, sum_rate, event-catalog digests, executed events.
- **MUST remain unchanged across all refactor phases** — the only end-to-end
  physics-preservation check.
- Regenerate **only** when physics or input parameters intentionally change:
  `KINETIX_UPDATE_GOLDEN_TRACE=1 python -m pytest tests/test_golden_trace.py`,
  then review `git diff` of the fixture.
- Cross-sublattice scenario (`test_golden_trace_cross_sublattice`): same preset /
  grid / seed plus two **test-side-only** overrides — O_i
  `valid_target_species += 'V_O'` and V_O `initial_concentration_bulk = 0.02` —
  and the live `defects_config` re-injected into every Site (sites otherwise run
  on the copy pickled in the grid; crystal.py:542-566 re-injects `Act_E_dict`
  only). Shipped presets CANNOT produce such a hop: `Empty`-specie sites exist
  only on the `interstitial` sublattice (measured on the main trace: 18 341
  offered migration events, 100 % interstitial→interstitial, 0 Empty-specie O
  sites). The scenario captures **4 hops** (steps 1/5/18/21) and pins that the
  destination hosts the SOURCE's config (`oxygen_interstitial`, whose site_type
  stays `interstitial`) and that the source is left empty.
  Regenerate with
  `KINETIX_UPDATE_GOLDEN_TRACE=1 python -m pytest "tests/test_golden_trace.py::test_golden_trace_cross_sublattice"`.

## Testing
- Full suite: `pytest tests/ -q` → **366 passed, 1 skipped** (~23 min); the 39
  `solver`-marked tests are ~22 min of that (see *Test Execution* below).
- Golden trace alone: `pytest tests/test_golden_trace.py -v` → 5 tests (~25 s).
- Notable files: `test_site.py` (64), `test_migration_pathways.py` (37),
  `test_cluster_island.py` (43), `test_balanced_tree.py` (29),
  `test_state_loader.py` (31), `test_kmc_loop.py` (12),
  `test_gb_charge_and_state_transfer.py` (23), `test_superbasin.py` (17),
  `test_golden_trace.py` (5).
- `test_mace_adapter.py` (marked `mace`) self-skips at module level without the
  `mace` extra — collects nothing; its `slow`-marked pathway sweeps
  (`--runslow`) also live there.
- Conventions: tests use **real parameter files via the production loaders**
  (`load_activation_energies`, preset loader) — avoid hardcoded literals.
- **caplog quirk**: `pytest.caplog` does NOT capture `kinetix.*` records
  (root logger `propagate=False`, see `kinetix/logging_config.py`). Use the
  `kinetix_log_collector` fixture in `tests/conftest.py` instead.
- `--runslow` flag (conftest) enables `@pytest.mark.slow` tests (real MACE
  CI-NEB pathway sweeps; many hours). Skipped by default.

## Test Execution

**Default (fast feedback, ~40 s):**

```bash
pytest tests/ -q -m "not solver and not mace"
```
Runs: 326 tests + 1 module-level skip (39 solver tests deselected).
**Use this by default — do NOT run the full suite for quick feedback.**

**Full suite (slow, ~23 min):**

```bash
pytest tests/ -q
```
Runs: all 365 tests (the 39 `solver` tests take ~22 min; measured 21m43s).

**Specific categories:**

```bash
# Only solver tests (when changing Poisson/heat/FEM code)
pytest tests/ -m solver -v

# Only MACE tests (when changing the MACE adapter)
pytest tests/ -m mace -v

# Everything except slow tests
pytest tests/ -m "not slow" -q

# Golden trace only (physics preservation check)
pytest tests/test_golden_trace.py -v
```

**When to run what:**

| Changed code | Run these tests |
|---|---|
| Site/Defect/Event classes | `pytest tests/ -q -m "not solver and not mace"` |
| kMC loop (`crystal.py`) | `pytest tests/test_golden_trace.py tests/test_kmc_loop.py -v` |
| Poisson/heat solvers | `pytest tests/ -m solver -v` |
| MACE adapter | `pytest tests/ -m mace -v` |
| Config loading | `pytest tests/test_config_loader.py tests/test_presets.py -v` |
| Any refactor phase | Golden trace + `pytest tests/ -q -m "not solver and not mace"` |

Markers are registered in `pyproject.toml` (`[tool.pytest.ini_options]`);
`solver` (test_poisson_solver.py, test_heat_solver.py, test_FEMSolver.py) and
`mace` (test_mace_adapter.py) are module-level `pytestmark` declarations.

Measured selections (Kinetix env):

| Selection | Tests | Wall time |
|---|---|---|
| `-m "not solver and not mace"` | 326 (+1 module skip) | ~40 s |
| `-m solver` | 39 | ~22 min |
| `-m mace` (no `mace` extra installed) | 0 (module skip) | ~5 s |
| full suite (`pytest tests/ -q`) | 365 (+1 module skip) | ~23 min |

## Known Bugs (from the 7-part decoupling investigation; report not stored in repo)
### Fixed
- H1 `remove_event_type()` missing parentheses ✅
- H2 `event[3]` accessed before isinstance check ✅ (structurally gone in Phase 4)
- H5 `electrode_scavenging` bool form ✅
- B11 `_find_clusters` 3-arg TypeError ✅
- M1 `passivation_level` conditionally absent ✅ (Phase 3: always on Defect)
- Phase-5 regression class (session of `a224b49`): deleted-method dangling
  callers, dropped Poisson refresh / dirty-site bookkeeping, discarded GB
  charge override, passivation reset on re-introduction — all fixed pre-commit.

### Open (tracked for later phases)
- **B2**: Superbasin label convention `num_event - 2` (`superbasin.py:319`) — Phase 6.
- **B3**: Two producers of the 5-element list event shape — verify remnants in Phase 6.
- **B6**: Superbasin absorbing moves bypass `catalog_tuple` — Phase 6.
- **C1**: Defect identity derived from sublattice, not occupant — composition model fixes this; Phase 6 completes.
- **H7**: Superbasin virtual moves not state-preserving — post-refactor.
- **M4**: Energy caches keyed by `supp_by` only — Phase 4+.
- **M7**: Config references pickled per site — Phase 6 optimization.
- **Finding A (coverage closed, behaviour intended)**: cross-sublattice hops now
  keep the source's DefectConfig (object transfer) where legacy re-derived it
  from the destination sublattice. Pinned by
  `test_golden_trace_cross_sublattice` + its fixture (4 hops); note the *main*
  trace still contains 0 such hops and that the scenario needs the test-side
  overrides documented in *Golden Trace Contract* — the shipped presets cannot
  produce a cross-sublattice hop at all.
- **Live-vs-pickled `defects_config` (open)**: Sites run on the `defects_config`
  pickled inside the grid; the Poisson re-injection (crystal.py:542-566)
  refreshes `Act_E_dict` only. A defects-YAML edit therefore does NOT change an
  existing grid's runtime behaviour (the main trace's `inputs_sha256` still fails
  provenance, so it is caught there). Rebuild the grid or re-inject explicitly.
- **`destination_CN` gap (open, latent)**: `Site.available_migrations` reads
  `dest_site.destination_CN[current_defect]` (site.py:785) for every destination
  that passes its "Empty or Vacancy" gate, but `destination_CN` is populated by
  `supported_by` (site.py:404-427) only for sites that were Empty when it ran. An
  occupied destination whose occupant has `CN_matters=True` (PZT
  `oxygen_vacancy`) would raise AttributeError/KeyError. Latent today: no shipped
  config declares an occupied target species for a mobile defect.

## Coding Conventions
- **4-space indent in production**, **2-space indent in tests** — match the file you edit.
- `from __future__ import annotations` at the top of new files.
- `@dataclass(slots=True)` for hot-path objects (`Event`, `Defect`).
- Absolute imports: `from kinetix.lattice.site import Site`.
- Type hints on new code; docstrings with Args/Returns for public methods.
- Config dataclasses use `from_dict(name, data)` classmethods with YAML defaults.
- Parameter files drive everything: resolve values from configs in tests, never hardcode.

## Data Files
- `data/parameters/presets/` — simulation presets (YAML; e.g. `VCM_mock.yaml`, `PZT_ZrPbO3.yaml`)
- `data/parameters/defects/` — defect configs (YAML; legacy `migrating_attributes` keys live until Phase 6)
- `data/parameters/reactions/` — reaction configs (YAML)
- `data/parameters/electrical/`, `data/parameters/grain_boundaries/` — field / GB configs
- `data/parameters/activation_energies/` — barrier data (JSON)
- `data/grids/` — cached lattice grids (pickle) + `.lock` files
- `data/cache/` — Materials Project structure cache (`mp-*.json`)
- `data/mesh/`, `data/experimental/` — FEM meshes, experimental data

## Important Warnings
- **DO NOT** modify the golden trace fixture without regenerating it — and never
  regenerate to "make a refactor pass". If the trace changes, STOP and report the diff.
- **DO NOT** change physics (barriers, rates, field corrections, GB rules) during refactor phases.
- **DO NOT** add attributes to `@dataclass(slots=True)` classes casually — slots are fixed at class creation.
- **DO NOT** touch the kMC loop / event handlers in `crystal.py` without running `pytest tests/test_golden_trace.py`.
- **DO NOT** delete `_remove_species_at_site` / `_install_defect_site` bookkeeping — dangling callers crash the kMC loop (this bit a previous session).
- **DO NOT** reset `passivation_level` on species re-introduction — it keys
  activation energies (`Act_E[str(level)]`) and gates capture/depassivation; resetting it yields invalid barrier keys (`KeyError`).
- **DO NOT** remove delegating properties on `Site` (`chemical_specie`, `ion_charge`, `passivation_level`, `site_events`) until Phase 6.
- **DO NOT** change `Event.catalog_tuple()` format (balanced tree + superbasin depend on it).
- **DO NOT** share `Defect` instances between sites (per-site mutable state).
- **DO NOT** delete `DefectConfig.migrating_attributes` or the YAML keys yet (Phase 6).
- **DO NOT** use `caplog` for `kinetix.*` loggers (`propagate=False`); use `kinetix_log_collector`.
- **DO NOT** use the base conda python (no pymatgen); use the `Kinetix` env.

## Standing Tasks (Every Refactor Task)
When completing any refactor task, the agent MUST:
1. **Run relevant tests** — use the "When to run what" table in *Test Execution*
   above (default: fast feedback via `-m "not solver and not mace"`, NOT the
   full 22-minute suite).
2. **Update tests** — behavior changed → update affected tests; new code → add tests.
3. **Update README** — only if user-facing behavior changed (config schema, CLI, presets).
4. **Update this file (AGENTS.md)** — phase completed, architecture changed, bugs fixed/found.
5. **Report architectural findings** — categorize by severity (H/B/M per the
   investigation convention); do NOT fix unless blocking; STOP and report if a
   code path contradicts the plan.

## Quick Start
```bash
conda activate Kinetix                          # base env lacks pymatgen
pytest tests/test_golden_trace.py -v            # physics contracts (~25 s)
pytest tests/ -q                                # full suite (~23 min)
pytest tests/test_site.py -v                    # single file
pytest tests/ -q --runslow                      # + MACE pathway sweeps (hours)
KINETIX_UPDATE_GOLDEN_TRACE=1 pytest tests/test_golden_trace.py   # regen both fixtures (review diff!)
```
