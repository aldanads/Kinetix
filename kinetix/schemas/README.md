# Kinetix configuration data contracts (JSON Schema)

This directory holds **machine-readable data contracts** for the Kinetix
multi-scale pipeline: one [JSON Schema](https://json-schema.org/) (Draft
2020-12) per configuration dataclass, generated automatically from
`kinetix/configs/`.

They exist so that the YAML that drives a simulation can be checked, completed
and documented **without importing Python** — by editors, CI, workflow managers
or external partners that need to know what a valid Kinetix configuration looks like.

## These files are generated — do not hand-edit them

```bash
python kinetix/schemas/generate_schemas.py                    # (re)generate
python kinetix/schemas/generate_schemas.py --check            # validate only, write nothing
python kinetix/schemas/generate_schemas.py --validate-presets  # + check the shipped YAML
```

`generate_schemas.py` introspects the dataclasses, which remain the single
source of truth; no dataclass was modified to produce these files. The generator
uses the **standard library only** (`pydantic` would work too, but it is not a
declared dependency of Kinetix and a documentation tool must not add one). The
optional validation steps need `jsonschema`; generation never does.

Re-run the generator after any change to `kinetix/configs/` — the schemas are
artefacts, the dataclasses are the contract.

## The 22 schemas

| File | Class | Describes |
|---|---|---|
| `simulation_config.schema.json` | `SimulationConfig` | the master configuration object (18 properties, 23 `$defs`) |
| `simulation_settings.schema.json` | `SimulationSettings` | the `settings:` block of a preset |
| `experimental_conditions.schema.json` | `ExperimentalConditions` | the `experimental:` block (temperature, pressure, sticking coefficient) |
| `material_config.schema.json` | `MaterialConfig` | material properties (MP id, permittivity, chemenv) |
| `material_selection.schema.json` | `MaterialSelection` | material name / MP id / neighbour radius |
| `crystal_structure.schema.json` | `CrystalStructure` | lattice size, Miller indices, layer selectors |
| `defect_config.schema.json` | `DefectConfig` | one defect species (22 fields) |
| `defects_config.schema.json` | `DefectsConfig` | the `defects:` mapping of a defect file |
| `reaction_config.schema.json` | `ReactionConfig` | one reaction |
| `reaction_species.schema.json` | `ReactionSpecies` | a reactant/product entry |
| `reactions_config.schema.json` | `ReactionsConfig` | the `reactions:` mapping of a reaction file |
| `grain_boundary_config.schema.json` | `GrainBoundaryConfig` | one grain-boundary descriptor (geometry, barrier rule) |
| `grain_boundaries_config.schema.json` | `GrainBoundariesConfig` | the `grain_boundaries:` list of a GB file |
| `electrical_config.schema.json` | `ElectricalConfig` | voltage protocol, series resistance, current model |
| `voltage_config.schema.json` | `VoltageConfig` | the `voltage:` block (mode, ramp, cycles) |
| `current_config.schema.json` | `CurrentConfig` | the `current:` block (Schottky parameters) |
| `mesh_config.schema.json` | `MeshConfig` | gmsh mesh sizing and refinement |
| `poisson_solver_config.schema.json` | `PoissonSolverConfig` | Poisson solver settings and conductivity |
| `heat_solver_config.schema.json` | `HeatSolverConfig` | heat solver settings and thermal properties |
| `superbasin_config.schema.json` | `SuperbasinConfig` | superbasin acceleration parameters |
| `calculator_config.schema.json` | `CalculatorConfig` | MACE-NEB calculator (model, cluster radii, cache) |
| `interstitial_refinement_config.schema.json` | `InterstitialRefinementConfig` | interstitial refinement switch and threshold |

## How types are mapped

| Python | JSON Schema |
|---|---|
| `str` / `int` / `float` / `bool` | `string` / `integer` / `number` / `boolean` |
| `X \| None` | `anyOf: [X, null]` |
| `list[X]` | `array` with `items` |
| `tuple[int, int, int]` | `array` with `prefixItems` + fixed `minItems`/`maxItems` |
| `dict[str, X]` | `object` with `additionalProperties` |
| `Enum` | `enum` of **member names** + `x-enum-values` |
| nested dataclass | `$ref` into `$defs` |
| `Path` | `string` (YAML carries a string) |
| `Any` | unconstrained |

* **Enums by name.** `VoltageMode` / `CurrentModel` are `auto()` enums, and the
  loader selects members with `VoltageMode[mode_str]`, so the *names*
  (`CONSTANT`, `SCHOTTKY`, …) are the file contract; the integers are kept in
  `x-enum-values`.
* **`required`** lists the fields with no default. **`default`** is recorded
  whenever the value is representable in JSON.
* **`description`** comes from the class' Google-style `Attributes:` docstring,
  falling back to a trailing `#` comment on the field line.
* **`x-runtime-injected: true`** marks fields the runtime fills in rather than
  reading from YAML (`rng`, `mpi_ctx`, `base_path`).
* **`additionalProperties: false`** applies to the *typed objects*. The
  production loaders ignore unknown YAML keys, so whole files are validated
  section-wise (see below) rather than as one strict object.

## Validating the shipped configuration files

A preset file is **not** an instance of `SimulationConfig`: its `metadata:`,
`crystal:` and `components:` keys form an envelope that
`SimulationConfig.from_yaml` folds into the typed object (metadata → name /
description / author, crystal → material config, components → the per-family
YAML files). That is why `--validate-presets` checks each typed section against
its own schema, and then each component file against the schema of the config it
feeds.

## Known findings (produced by validating the shipped data with these schemas)

1. **YAML floats written without a signed exponent load as strings.**
   `conductive_filament: 1e5` (and `6.3e2`, `6.3e6`, `series_resistance: 1.0e7`)
   are parsed as `str` by PyYAML, because YAML 1.1 requires `1.0e+5`. The
   dataclass declares `float`, and the value only works because the two
   consumers coerce it (`poisson.py:481`, `cluster.py:172`) — a new consumer
   that forgets `float()` would hand a string to PETSc. Preferred fix in the
   data files: write `1.0e+5`.
2. **Defect names come from the mapping key.** In
   `defects: {oxygen_interstitial: {...}}` the loader injects `name` from the
   key, but `DefectConfig.name` has no default and is therefore `required` in
   the schema, so validating a defects file reports a false positive.
3. **`reactions/*.yaml` carry a `metadata:` block** that `ReactionsConfig` does
   not model; the loader ignores it.
4. **`electrical_ZERO_HOLD.yaml` defines `initial_voltage`**, which
   `ElectricalConfig` does not model; the loader ignores it.
5. **`CalculatorConfig.type` is not dispatched at runtime** — the schema
   documents the declared intent (`"mace_neb"` / `"tabulated"`), but the kMC
   rate path always uses the tabulated `Act_E_dict` (see `AGENTS.md`, Known
   Bugs → Open).

## Not covered

* `data/parameters/activation_energies/*.json` — free-form, per-mechanism
  dictionaries with no backing dataclass.
* Output artefacts: LAMMPS dumps and `metadata.json` (results, not configuration).
* The preset envelope itself (`metadata` / `crystal` / `components`): no
  dataclass describes it, because it exists only inside `from_yaml`.
