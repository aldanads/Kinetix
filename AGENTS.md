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
- CLI entry: `kinetix/cli.py` drives `simulator.step_kmc()`
  (`kinetix/lattice/simulator.py`). Poisson/heat solvers are Linux-only.
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
| `kinetix/lattice/simulator.py` (~1.9k lines) | `KMCSimulator`/`simulator` — rate evaluation, superbasin driving; delegates events/solvers/metadata/lattice-construction/kMC loop out |
| `kinetix/lattice/site.py` (~1.3k lines) | `Site` — lattice site with topology + **Defect composition**; event generation (`available_pathways` :610, `available_reactions` :818) |
| `kinetix/lattice/lattice_builder.py` (~1.7k lines) | LatticeBuilder — lattice construction/init (36 methods): MP structure + cache, grid assembly, migration pathways, k-d tree/neighbours, interstitial generation, Wulff/coords, cluster tracking |
| `kinetix/lattice/events.py` | EventHandler — kMC event dispatch/handlers, dirty-site bookkeeping, affected-state refresh |
| `kinetix/lattice/kmc_loop.py` | KMCLoop — BKL kMC step orchestration (step/catalog/BKL + superbasin search & activation policy) |
| `kinetix/lattice/defect.py` (190 lines) | `Event` (:40, `catalog_tuple` :69), `Defect` (:80), `EMPTY_DEFECT_CONFIG` (:157), `make_empty_defect` (:175) |
| `kinetix/lattice/cluster.py` | Cluster/filament analysis (`_find_clusters`) |
| `kinetix/lattice/island.py` | Island — deposition morphology |
| `kinetix/lattice/grain_boundary.py` | GB geometry + barrier modifications (planar/cylindrical/triple-junction) |
| `kinetix/utils/superbasin.py` | Superbasin acceleration (note quirk at :319 — bug B2) |
| `kinetix/utils/balanced_tree.py` | Balanced binary tree for BKL event selection |
| `kinetix/utils/state_loader.py` | LAMMPS dump parsing, state restore |
| `kinetix/utils/metadata.py` | MetadataWriter — metadata output (JSON now, H5MD/NOMAD future) |
| `kinetix/configs/` | Typed config dataclasses: `simulation_config`, `defect_config`, `reaction_config`, `electrical_config`, `grain_boundary_config`, `material_config`, `mesh_config`, `solver_config`, `calculator_config` + `config_loader` |
| `kinetix/initialization.py` | Lattice construction, grid loading/caching, config wiring |
| `kinetix/solvers/` | Poisson + heat FEM solvers (DOLFINx) |
| `kinetix/solvers/coordinator.py` | SolverCoordinator — orchestrates Poisson/Heat field solving |
| `kinetix/calculators/mace_neb.py` | MACE NEB barrier calculator (optional, GPU; `max_passivation_level` validated here) |
| `kinetix/logging_config.py` | Root logger `propagate=False` (affects caplog — see Testing) |
| `kinetix/cli.py` / `kinetix/__main__.py` | Simulation workflow driver |

## simulator.py Split Progress (epic: break up the 4.5k-line god class)
Extraction order: metadata → solvers → events → kMC loop → lattice construction.
Each phase moves code verbatim into a class that holds **no simulation state**
(everything goes through `self.simulator`) and leaves thin one-line delegates on
`KMCSimulator`, so `cli.py`, `superbasin.py`, `state_loader.py` and the tests
are unchanged.

| Phase | Module | Status |
|---|---|---|
| 1 | `kinetix/utils/metadata.py` — `MetadataWriter` (`write_json`; H5MD/NOMAD stubs) | ✅ `6b040e3` |
| 2 | `kinetix/solvers/coordinator.py` — `SolverCoordinator` | ✅ `f15d6aa` |
| 3 | `kinetix/lattice/events.py` — `EventHandler` (17 methods) | ✅ Phase 3 |
| 4 | `kinetix/lattice/kmc_loop.py` — `KMCLoop` (9 methods) | ✅ Phase 4 |
| 5 | `kinetix/lattice/lattice_builder.py` — `LatticeBuilder` (36 methods) | ✅ Phase 5 |
| Rename | `crystal.py` → `simulator.py`, `Crystal_Lattice` → `KMCSimulator`, `System_state` → `simulator` | ✅ pure rename |

**Rename essentials (pure rename — no logic, no physics):**
- `Crystal_Lattice` → **`KMCSimulator`**; `kinetix/lattice/crystal.py` →
  **`kinetix/lattice/simulator.py`** (`git mv`, so history follows the file).
- Handle name `System_state` → **`simulator`** in `cli.py`,
  `initialization.py`, `utils/analysis.py`, `utils/extract_data.py`,
  `utils/superbasin.py`, `lattice/island.py`, `lattice/site.py`,
  `lattice/cluster.py` and the tests (`simulator, *_ = initialization(...)`).
- Collaborators (`EventHandler`, `KMCLoop`, `SolverCoordinator`,
  `LatticeBuilder`) now hold **`self.simulator`** (was `self.system`):
  `def __init__(self, simulator: KMCSimulator) -> None`. Tests assert the
  attribute (`builder.simulator is system`) and the source text
  (`"self.simulator.rank"`, `"self.simulator.mpi_ctx.bcast"`); the MPI guards
  and the structure/time bcasts are unchanged — only the receiver name changed.
- `kinetix/__init__.py` exports `KMCSimulator` (`__all__` no longer carries the
  stale name). The logger name followed the file, so child loggers are now
  `kinetix.lattice.simulator` (comment updated in `logging_config.py`; the
  `propagate=False` / caplog quirk is unchanged).
- **Legacy alias** — the only place the old identifier survives:
  `Crystal_Lattice = KMCSimulator` at the bottom of `simulator.py`, so
  `from kinetix.lattice.simulator import Crystal_Lattice` keeps working. Pinned
  by `tests/test_simulator_rename.py` (`test_alias_is_the_last_statement_of_simulator`,
  `test_legacy_alias_is_the_same_class`, `test_no_stale_identifiers_in_package_or_tests`).
- **Deliberately NOT renamed** (the Step-4 list): `grid_crystal` (site dict),
  `crystal_grid` (builder entry point), `crystal_size`, `plot_crystal`,
  `measurements_crystal`, pymatgen `crystal_system`/`structure` contexts,
  `MetadataWriter.write_json(crystal: KMCSimulator)`'s parameter name, the
  lowercase `system_state` parameter of `state_loader.load_state_from_dump`, and
  the test doubles `MockSystemState` / `FakeSystemState`.
- **Pickle safety (verified byte-wise)**: grids are `{filename: {idx: Site}}`
  and reference ONLY `kinetix.lattice.site` (checked on all five
  `data/grids/*.pkl`), so the rename cannot break them — loading
  `grid_HfO2_3nm.pkl` through the production path still yields 3456 sites in
  `simulator.grid_crystal`. Pinned by
  `test_grid_pickles_reference_only_the_site_class`.
  NOT covered: the `variables.pkl` *result* artefacts written by
  `save_variables` (`cli.py`) store the old **module path**, so they no longer
  unpickle (pinned as a known limitation by
  `test_legacy_result_pickle_module_path_no_longer_resolves`); their dict key
  is now `'simulator'`, and the readers in `analysis.py` / `extract_data.py`
  follow. No such artefact exists in the repo.
- **Hazard**: a naive `lattice.crystal` → `lattice.simulator` substitution
  corrupts `lattice.crystal_size` — it hit 15 lines in
  `tests/test_migration_pathways.py` during this rename (reverted). Always
  `git diff` a mechanical rename.
- Verification: golden trace **5/5** with the fixtures byte-identical, fast
  suite **407 passed / 1 skipped**, new `tests/test_simulator_rename.py` **8/8**.

**Phase 3 essentials:**
- Moved: `processes`, `_handle_{migration,generation,redox,reaction}_event`,
  `_should_scavenge`, `_defect_by_name`, `_find_empty_neighbor`,
  `_is_at_top_electrode`, `_get_mobile_sites`, `_get_gb_charge_state`,
  `_install_defect_site`, `_introduce_specie_site`, `_track_occupancy_update`,
  `_remove_species_at_site`, `update_sites_topology`, `_update_rates_lazily`.
  Bodies are byte-identical (verified against `git show HEAD:…`), only
  re-indented to the 4-space convention with `self.` → `self.simulator.`.
- **Delegates kept** (external callers): `processes` (`superbasin.py:134/138/351`),
  `_update_rates_lazily` (`_kmc_step`, golden trace, `test_kmc_loop.py`),
  `update_sites_topology` (`defect_gen`, `state_loader.py:180`),
  `_introduce_specie_site` (deposition paths, `state_loader.py:166`),
  `_get_mobile_sites` (init :210).
- `_is_active_site` **stays on `KMCSimulator`** (lattice construction and
  `_resolve_defect_config` use it); the handler calls
  `self.simulator._is_active_site(...)`.
- `_kmc_step` calls `self.processes(...)` **through the delegate on purpose**:
  the golden trace wraps the *instance* attribute to observe the event catalog
  (`tests/test_golden_trace.py:344`). Never call `self.event_handler.processes`
  from the loop.
- Lazy `event_handler` property (same pattern as `solver_coordinator`), so
  `KMCSimulator.__new__` buildouts and legacy pickles resolve it.

**Phase 4 essentials:**
- Moved (9 methods): `step_kmc`, `_kmc_step`, `_search_superbasin`,
  `update_superbasin` + the superbasin activation policy
  (`should_activate_superbasin`, `is_filament_percolating`,
  `_check_event_based_superbasin`, `_check_time_based_superbasin`,
  `_slow_timesteps`). Bodies byte-identical vs `git show HEAD:…`
  (modulo `self.simulator.`), verified two ways: body-diff AND a routing check
  (no bare `self.<system-attr>` may survive — a *missing* rewrite is invisible
  to the body-diff alone; this exact silent no-op bug was caught by the golden
  trace during the phase).
- **Delegates kept** for **all 9** names (one-liners under the "KMC logic"
  banner): `step_kmc` (`cli.py:449`, golden trace, tests), `_kmc_step`
  (`test_kmc_loop.py`, `test_event_handler.py`, `test_kmc_loop_class.py`),
  the rest kept for API stability. `_evaluate_fields_for_kmc` remains the
  SolverCoordinator delegate and is what `step_kmc`/`_kmc_step` call.
- `track_time` / `add_time` **stay on `KMCSimulator`**
  (`initialization.py:351/355`, `cli.py` call them); the loop reaches
  `track_time` through `self.simulator.track_time(...)`.
- Lazy `kmc_loop` property with a **local import** (same pattern as
  `solver_coordinator`).
- **MPI premise correction (found in Phase 4)**: `step_kmc` is NOT MPI-free —
  it keeps the pre-existing `rank == 0` guard and the
  `mpi_ctx.bcast(payload, root=0)` of `system.time`, moved verbatim to
  `kmc_loop.py` (serial runs pass `mpi_ctx=None`, so the bcast is skipped).
  No MPI logic was added/removed, and the loop holds no rank/mpi state.
- Routing rule now lives in `kmc_loop.py`: `_kmc_step` MUST call
  `self.simulator.processes(...)` — the *system* delegate — never
  `event_handler.processes` (the golden trace wraps the instance attribute).
  Pinned by `test_kmc_loop_class.py::test_kmc_loop_reaches_processes_via_instance_delegate`.
- Tests: `tests/test_kmc_loop_class.py` (13) incl. a 5-step integration that
  reproduces the first 5 golden-fixture steps through the delegate.

**Phase 5 essentials:**
- Moved (36 methods, `simulator.py` 3410 → **1935 lines**, −1475): structure/MP
  model (`_load_mp_cache`, `_save_mp_cache`, `lattice_model`,
  `_is_inside_supercell`, `_apply_miller_orientation`, `_get_rotation_matrix`,
  `_create_supercell`, `_compute_basis_vectors`); migration pathways
  (`_initialize_migration_pathways`, `_validate_migration_network`); neighbour
  search (`_build_kdtree`, `_get_neighbors_for_site`, `_generate_periodic_images`,
  `_check_percolation_at_radius`, `find_optimal_radius`, `diagnose_steep_down`,
  `diagnose_interstitial_presence`); grid assembly (`crystal_grid`);
  site-init/interstitials (`_efficient_act_e_copy`,
  `_get_applicable_defects_for_site`, `_compute_interface_flags`,
  `_generate_interstitial_sites`, `_find_interstitials_voronoi`,
  `_refine_interstitial_positions`, `_cluster_and_average`,
  `_validate_interstitial_positions`, `create_ovito_xyz_file`,
  `_handle_missing_neighbors`); neighbour analysis
  (`_parallel_neighbors_analysis`, `_sequencial_neighbors_analysis`,
  `get_num_cores`, `_process_batch_sites_worker`); `get_idx_coords`,
  `Wulff_Shape`, `create_edges`, `_initialize_cluster_tracking`.
- **Delegates kept for all 36 names** (one-liners under the "Lattice construction"
  banner after `__init__`), so `initialization.py`, `cli.py`, `metadata.py`,
  `state_loader.py` and the tests are unchanged. Pinned:
  `test_only_crystal_touches_lattice_builder` (no module outside `simulator.py`
  references `lattice_builder`) and `test_delegate_signatures_match_builder`
  (parameter names + defaults identical).
- Lazy `lattice_builder` property with a **local import** (same pattern as
  `solver_coordinator`/`kmc_loop`).
- **Collaborators that stay on `KMCSimulator`** and are reached through
  `self.simulator`: `_is_active_site` (Phase 3 rule — construction calls it) and
  `_minimum_image_vector` (runtime callers in the MACE NEB calculator/active
  learning). Neither is duplicated on the builder.
- **MPI premise (verbatim, unchanged)**: construction is NOT rank-free —
  `_save_mp_cache`/`lattice_model` guard with `rank == 0`, `lattice_model`
  broadcasts the fetched structure, and `_initialize_migration_pathways` /
  `_generate_interstitial_sites` carry their own rank guards; all read as
  `self.simulator.rank` / `self.simulator.mpi_ctx`. No collective added/removed, no
  rank state on the builder.
- Extraction mechanics: generator `/tmp/gen_lattice_builder.py` extracts **by
  method name** from `git show HEAD:…` (not line numbers), re-indents to the
  4-space convention and routes `self.X` → `self.simulator.X` with pure string ops.
  Verified **two ways** (Phase 4 recipe): byte-identical body-diff (modulo
  routing, reversed) **plus** a routing invariant (no un-routed `self.X` may
  survive — a missing rewrite is invisible to the body-diff alone). 3 bare-`self`
  argument fixes: `self._kdtree`, `self.gb_configurations`, `kx=self`. Output is
  reproducible (re-running the generator yields identical bytes).
- Tests: `tests/test_lattice_builder_class.py` (20) — delegate/thinness/signature
  contract, statelessness, no runtime `crystal` import, MPI routing, behavioural
  checks through the delegates (`_get_rotation_matrix`, `_compute_basis_vectors`,
  `get_idx_coords` cache, `_efficient_act_e_copy` isolation,
  `_get_applicable_defects_for_site`, `_process_batch_sites_worker` call contract,
  `_validate_migration_network` read-only, `get_num_cores`) and an integration
  test on the golden lattice (3456 sites, k-d tree, neighbours, interface flags,
  Phase-6 live `defects_config` binding for every site).
- **Pre-existing findings (Phase 5, pinned not fixed)**:
  `_validate_migration_network()`'s default `radius=None` crashes in
  `_generate_periodic_images` (`None / float`) — the only in-package caller
  passes a real radius (`_initialize_migration_pathways`);
  `_process_batch_sites_worker` forwards SIX positional args to
  `Site.neighbors_analysis` (which declares five) and
  `_parallel_neighbors_analysis` has no in-package callers.

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
| 6 | Final cleanup: delegating properties, `migrating_attributes`, latent bugs | ✅ Phase 6 |

**Phase 6 essentials (epic complete):**
- The four delegating properties (`chemical_specie`, `ion_charge`,
  `passivation_level`, `site_events`) are **gone**: all state is read/written
  as `site.defect.chemical_specie` / `.charge` / `.passivation_level` /
  `.events`. A missed read raises AttributeError, but a missed *write* would
  silently create a shadow attribute — grep these names after touching them.
- `DefectConfig.migrating_attributes`, its YAML keys and every runtime
  reference are deleted. Both fixtures' `meta.inputs_sha256` was regenerated
  for that schema change with **steps proven byte-identical** (main `1a3b3bb3…`,
  cross `e2bf330b…`, 4 hops unchanged) — nothing else in them changed.
- Migration hop = `dst.install_defect(src.defect)` + `src.clear_defect()`
  (`_handle_migration_event` :3014); GB charge is written through to
  `defect.charge` **before** the hop; `_install_defect_site` (:3347) /
  `_remove_species_at_site` (:3396) do the dirty-site bookkeeping.
- `destination_CN` is guarded (`available_migrations` :777) with an actionable
  error, and loaded grids bind the **live** `defects_config` onto every site
  (`crystal_grid`, now `lattice_builder.py:837` after the Phase-5 move) — the
  pickled copy can no longer go stale.
- `introduce_specie` (:549): same species keeps the Defect (charge refresh
  only, passivation untouched); a species swap installs a fresh Defect.

**Key design decisions:**
- Every Site always hosts exactly one Defect (empty sites get an empty Defect, never None).
- `site_type` (sublattice) belongs to Site, not Defect; `EMPTY_DEFECT_CONFIG.site_type = "Empty"`.
- Config-time-immutable: `enabled_events`, `max_passivation_level`, barriers. Runtime-dynamic (on Defect): `chemical_specie`, `charge`, `passivation_level`, `events`.
- No DefectRegistry — lookup is plain `dict[str, DefectConfig]`; sites share
  ONE live reference to it (Phase 6 binding at `lattice_builder.py:837`).
- No shared/singleton Defect instances across sites (per-site mutable state).
- Config resolution: **occupied** sites read `defect.name`; **empty** sites
  still use the sublattice registry lookup (`_get_current_defect_name` :272),
  kept deliberately — an empty Defect carries no candidate information, and
  `destination_CN`/GB barrier tables depend on it (C1 resolved & documented).
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
- Phase 6 regenerated **only** `meta.inputs_sha256` in both fixtures (the
  config schema lost `migrating_attributes`); the `steps` blocks and all 4
  cross-hops were re-hashed byte-identical before/after (proof above).
- Cross-sublattice scenario (`test_golden_trace_cross_sublattice`): same preset /
  grid / seed plus two **test-side-only** overrides — O_i
  `valid_target_species += 'V_O'` and V_O `initial_concentration_bulk = 0.02` —
  and the live `defects_config` re-injected into every Site (sites otherwise run
  on the copy pickled in the grid; lattice_builder.py:412-414 re-injects `Act_E_dict`
  only). Shipped presets CANNOT produce such a hop: `Empty`-specie sites exist
  only on the `interstitial` sublattice (measured on the main trace: 18 341
  offered migration events, 100 % interstitial→interstitial, 0 Empty-specie O
  sites). The scenario captures **4 hops** (steps 1/5/18/21) and pins that the
  destination hosts the SOURCE's config (`oxygen_interstitial`, whose site_type
  stays `interstitial`) and that the source is left empty.
  Regenerate with
  `KINETIX_UPDATE_GOLDEN_TRACE=1 python -m pytest "tests/test_golden_trace.py::test_golden_trace_cross_sublattice"`.

## Testing
- Full suite: `pytest tests/ -q` → **446 passed, 1 skipped** (~23 min); the 39
  `solver`-marked tests are ~22 min of that (see *Test Execution* below).
- Golden trace alone: `pytest tests/test_golden_trace.py -v` → 5 tests (~25 s).
- Notable files: `test_site.py` (64), `test_cluster_island.py` (43),
  `test_migration_pathways.py` (37), `test_state_loader.py` (31),
  `test_balanced_tree.py` (29), `test_gb_charge_and_state_transfer.py` (22),
  `test_lattice_builder_class.py` (20), `test_event_handler.py` (17),
  `test_superbasin.py` (17), `test_solver_coordinator.py` (14),
  `test_kmc_loop_class.py` (13), `test_kmc_loop.py` (12),
  `test_metadata_writer.py` (9), `test_simulator_rename.py` (8),
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

**Default (fast feedback, ~86 s):**

```bash
pytest tests/ -q -m "not solver and not mace"
```
Runs: 407 tests + 1 module-level skip (39 solver tests deselected).
**Use this by default — do NOT run the full suite for quick feedback.**

**Full suite (slow, ~23 min):**

```bash
pytest tests/ -q
```
Runs: all 446 tests (the 39 `solver` tests take ~22 min; measured 21m43s).

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
| kMC loop (`simulator.py`) | `pytest tests/test_golden_trace.py tests/test_kmc_loop.py -v` |
| Lattice construction (`lattice_builder.py`) | `pytest tests/test_lattice_builder_class.py -v` (golden lattice, ~12 s) |
| Poisson/heat solvers | `pytest tests/ -m solver -v` |
| MACE adapter | `pytest tests/ -m mace -v` |
| Config loading | `pytest tests/test_config_loader.py tests/test_presets.py -v` |
| Renames / public API (`KMCSimulator`, `simulator.py`) | `pytest tests/test_simulator_rename.py -v` (8, ~5 s) |
| Any refactor phase | Golden trace + `pytest tests/ -q -m "not solver and not mace"` |

Markers are registered in `pyproject.toml` (`[tool.pytest.ini_options]`);
`solver` (test_poisson_solver.py, test_heat_solver.py, test_FEMSolver.py) and
`mace` (test_mace_adapter.py) are module-level `pytestmark` declarations.

Measured selections (Kinetix env):

| Selection | Tests | Wall time |
|---|---|---|
| `-m "not solver and not mace"` | 407 (+1 module skip) | ~86 s |
| `-m solver` | 39 | ~22 min |
| `-m mace` (no `mace` extra installed) | 0 (module skip) | ~5 s |
| full suite (`pytest tests/ -q`) | 446 (+1 module skip) | ~23 min |
| `test_lattice_builder_class.py` alone | 20 | ~12 s |
| `test_simulator_rename.py` alone | 8 | ~5 s |

## Known Bugs (from the 7-part decoupling investigation; report not stored in repo)
### Fixed
- H1 `remove_event_type()` missing parentheses ✅
- H2 `event[3]` accessed before isinstance check ✅ (structurally gone in Phase 4)
- H5 `electrode_scavenging` bool form ✅
- B11 `_find_clusters` 3-arg TypeError ✅
- M1 `passivation_level` conditionally absent ✅ (Phase 3: always on Defect)
- `destination_CN` gap (latent KeyError/AttributeError on occupied
  destinations) ✅ guarded with an actionable error (site.py:777, Phase 6)
- Live-vs-pickled `defects_config` staleness ✅ live registry bound onto every
  loaded site (now `lattice_builder.py:837` after the Phase-5 move; Phase 6)
- C1 / M7 ✅ occupant sites read `defect.name`, empty-site lookup documented
  (site.py:249), one shared live config reference
- Phase-5 regression class (Site/Defect epic, session of `a224b49`): deleted-method
  dangling callers, dropped Poisson refresh / dirty-site bookkeeping, discarded
  GB charge override, passivation reset on re-introduction — all fixed pre-commit.
- H: `kinetix/__init__.py` unconditional FEM import → fixed with try/except
  (Phase 2 of the simulator.py split)
- B: Missing `return` in `_evaluate_fields_for_kmc` → fixed
  (`kinetix/solvers/coordinator.py`)
- M: Unbound `clusters` in `prepare_clusters_for_bcs` → fixed
  (`kinetix/solvers/coordinator.py`)

### Open (post-epic debt — NOT addressed by Phase 6)
- **B2**: Superbasin label convention `num_event - 2` (`superbasin.py:319`).
- **B3**: Two producers of the 5-element list event shape — verify remnants.
- **B6**: Superbasin absorbing moves bypass `catalog_tuple`.
- **H7**: Superbasin virtual moves not state-preserving.
- **M4**: Energy caches keyed by `supp_by` only.
- **`site.py:465` bare name**: `detect_edges(..., chemical_specie)` references
  an undefined local (deposition path, not test-covered) — pre-existing, Phase 6
  did not touch it.
- **`_is_at_top_electrode` dead + wrong flag** (found in Phase 3, moved verbatim
  to `events.py:346`): the method returns
  `grid_crystal[site_idx].is_at_bottom_interface` although its name says *top*,
  and it has **no callers anywhere** in the package. Behaviour pinned by
  `tests/test_event_handler.py::test_is_at_top_electrode_reads_interface_flag` so
  a future caller cannot silently inherit the wrong flag.
- **`step_kmc` MPI premise (found in Phase 4)**: the phase brief assumed the
  kMC loop runs MPI-free, but `step_kmc` carries a pre-existing `rank == 0`
  guard plus `mpi_ctx.bcast(payload, root=0)` of `system.time` — moved verbatim
  to `kmc_loop.py`. Serial runs never hit the bcast (`mpi_ctx=None`,
  `initialization.py`). Not a bug; documented so a future "MPI-free loop"
  refactor does not silently drop the guard or the time broadcast.
- **`_validate_migration_network(radius=None)` crashes (found in Phase 5, moved
  verbatim to `lattice_builder.py:420`)**: the default `None` reaches
  `_generate_periodic_images` (`radius / np.linalg.norm(...)` → `TypeError`).
  The only in-package caller passes a real radius
  (`_initialize_migration_pathways`), so no shipped path hits it. Pinned by
  `test_lattice_builder_class.py::test_validate_migration_network_is_read_only`
  (called with a real radius AND asserting the default still raises) so a fix
  is deliberate.
- **`_parallel_neighbors_analysis` dead + arity mismatch (found in Phase 5,
  moved verbatim to `lattice_builder.py:1430`)**: the method has no in-package
  callers, and its worker `_process_batch_sites_worker` forwards SIX positional
  arguments to `Site.neighbors_analysis`, which declares five
  (`site.py:328`). Reproduces at HEAD; pinned by
  `test_process_batch_sites_worker_call_contract` (stub site, documents the
  historical call shape incl. `neighbors_positions`).
- **`site.py:93-94` interface flags**: `is_at_bottom_interface` /
  `is_at_top_interface` are set by `neighbors_analysis` only for the outermost
  layers; `Site.calculate_site_energy` and the removal-layer rules read them.
- **Finding A (closed, behaviour intended)**: cross-sublattice hops keep the
  source's DefectConfig (object transfer), pinned by
  `test_golden_trace_cross_sublattice` (4 hops); scenario needs its documented
  overrides — shipped presets offer 0 such hops, and the *main* trace has none.

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
- `data/parameters/defects/` — defect configs (YAML)
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
- **DO NOT** touch the kMC loop / event handlers in `simulator.py` without running `pytest tests/test_golden_trace.py`.
- **DO NOT** delete `_remove_species_at_site` / `_install_defect_site` bookkeeping — dangling callers crash the kMC loop (this bit a previous session). They now live on `EventHandler` (`events.py`) and are reached through the handler.
- **DO NOT** call `self.event_handler.processes(...)` from `_kmc_step` — the kMC
  loop must go through the `KMCSimulator.processes` **delegate**, because the
  golden trace wraps the *instance* attribute (`tests/test_golden_trace.py:344`)
  to observe the event catalog; bypassing it silently voids the physics contract.
- **DO NOT** add simulation state to `EventHandler` / `SolverCoordinator` /
  `KMCLoop` / `LatticeBuilder` — each one reads and writes the simulator through
  `self.simulator` (pickles, MPI rank ownership and the golden trace depend on the
  state staying on `KMCSimulator`).
- **DO NOT** rename the collaborators' `simulator` attribute back to `system` —
  tests pin both the attribute (`builder.simulator is system`) and the source
  text (`"self.simulator.rank"` / `"self.simulator.mpi_ctx.bcast"`), and
  `kinetix/__init__.py` exports `KMCSimulator`.
- **DO NOT** delete the `Crystal_Lattice = KMCSimulator` alias at the bottom of
  `simulator.py` without deciding what happens to pre-rename `variables.pkl`
  artefacts (the alias and the pickle limitation are pinned by
  `tests/test_simulator_rename.py`).
- **DO NOT** call the lattice builder from outside `simulator.py` — external code
  uses the `KMCSimulator` delegates (the 36 one-liners are the API; pinned by
  `test_only_crystal_touches_lattice_builder`). Changing a delegate signature
  without the builder's drifts the defaults for `initialization.py`/`cli.py`
  (pinned by `test_delegate_signatures_match_builder`).
- **DO NOT** reset `passivation_level` on species re-introduction — it keys
  activation energies (`Act_E[str(level)]`) and gates capture/depassivation; resetting it yields invalid barrier keys (`KeyError`).
- **DO NOT** re-add flat state accessors to `Site` — Phase 6 deleted the four
  delegating properties; use `site.defect.chemical_specie` / `.charge` /
  `.passivation_level` / `.events` everywhere.
- **DO NOT** change `Event.catalog_tuple()` format (balanced tree + superbasin depend on it).
- **DO NOT** share `Defect` instances between sites (per-site mutable state).
- **DO NOT** delete `_get_current_defect_name` / `applicable_defects` — the
  empty-site registry lookup feeds `destination_CN` and the GB barrier tables;
  removing them drifts both golden traces (verified Phase 6).
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
