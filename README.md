# Kinetix
A fully open-source kinetic Monte Carlo (kMC) simulator for materials deposition, annealing, and memristive device modeling

> [!WARNING]
> **⚠️ Active Development**
> This tool is currently in beta.
> Feel free to contact me directly.
> Best suited for those comfortable with Python, FEniCS/DOLFINx and MPI environments.

## Aim
Kinetix aims to bridge materials science and device physics by providing a transparent, modular, and accessible platform for multiscale simulation of emerging electronic devices, ideal for research in neuromorphic computing, memristors, and thin-film processing.

## Capabilities
Kinetix is a Python-based, open-source simulation framework (**MIT License**) that enables atomic-scale modeling of:

* **Material Deposition:** Nucleation and growth of thin films of fcc metals (e.g., Ni, Pd, Ag, Pt, Au, Cu) from a kinetic gas-phase source, including substrate–metal interaction effects on growth morphology.
* **Vacuum Annealing:** Thermal evolution, defect passivation, and surface morphology relaxation of deposited films.
* **Resistive Switching:** Formation and dissolution of conductive filaments in valence-change (VCM) and electrochemical (ECM) memristors, with electrode scavenging of mobile ions.
* **Grain Boundary (GB) Engineering:** Spatially resolved modification of activation energies at grain boundaries — vertical planar, cylindrical, and triple-junction geometries — including **direction-dependent migration barriers** that distinguish entry into, exit from, and within-GB hops.
* **Multiphysics Coupling:** Real-time feedback between electrostatics (Poisson), Joule heating (heat equation), and stochastic defect kinetics, with field-assisted migration barriers for charged species.
* **Superbasin Acceleration:** Local-superbasin event grouping to escape rare-event bottlenecks in deposition, annealing, and device simulations.

---

## 🔄 Simulation Workflow
Each simulation is driven by [`run_simulation.py`](run_simulation.py), which calls `System_state.step_kmc()` ([`kinetix/lattice/crystal.py`](kinetix/lattice/crystal.py)) in the following loop:

1. **Lattice Generation:** Constructs the 3D atomic grid (lattice, interfaces, and grain boundaries) with `pymatgen`, generates interstitial sites using a Voronoi method, and builds the finite-element mesh with `gmsh`.
2. **FEM Solvers:** Computes the electric potential (Poisson) and, when enabled, the temperature field (heat equation) for the current defect configuration and electrode potentials using DOLFINx/FEniCS.
3. **Energy Landscape Update:** Calculates site-resolved activation energies for all possible defect events (migration, generation, recombination), applying GB barrier modifications (linear or direction-dependent) and electric-field corrections for charged species, and lazily recomputes only the affected rates.
4. **kMC Step:** Selects an event with the rejection-free BKL (Bortz-Kalos-Lebowitz) algorithm implemented on a balanced binary tree of transition rates, advances the simulation time, and updates the lattice.
5. **Loop:** Repeats steps 2–4 — re-solving the fields at every voltage update for device simulations — until the deposition/annealing target or the applied voltage protocol is completed.

> [!NOTE]
> The Poisson/heat solvers only run on **Linux** (guarded by `platform.system() == 'Linux'` in `run_simulation.py`). On other platforms there is no FEM field feedback, while pure deposition/annealing simulations continue to run.

---

## Governing Equations

Kinetix couples three layers of physics in every device simulation: the electrostatic potential, the temperature field, and the stochastic defect kinetics.

### Electrostatics (Poisson equation)
The electric potential $V(\mathbf{x})$ is obtained from the ionic charge density $\rho(\mathbf{x})$:

$$-\nabla \cdot \left(\varepsilon_0 \varepsilon_r(\mathbf{x}) \nabla V\right) = \rho(\mathbf{x})$$

Charges are spread onto the FEM mesh as Gaussians (`epsilon_gaussian_charge`). When a conductive filament bridges the electrodes, the solver instead enforces current continuity, $-\nabla \cdot (\sigma \nabla V) = 0$, using the configurable filament/dielectric conductivities.

### Joule heating (heat equation)
The steady-state temperature field $T(\mathbf{x})$ satisfies

$$-\nabla \cdot \left(\kappa(\mathbf{x}) \nabla T\right) = Q(\mathbf{x})$$

with Joule-heating source $Q$ and space-varying thermal conductivity (dielectric vs. metal). Thermal inertia is treated as a capacitor relaxation, $T(t+\Delta t) = T_\mathrm{ss} + \big(T(t) - T_\mathrm{ss}\big)\exp(-\Delta t/\tau)$ with time constant $\tau = \rho c_p L^2 / \kappa$.

### Defect kinetics (Arrhenius rate law)
Each elementary event $i$ is assigned a first-order transition rate

$$k_i = \nu_0 \exp\left(-\frac{E_{\mathrm{act},i}}{k_\mathrm{B} T}\right)$$

with attempt frequency $\nu_0 = 7\times10^{12}$ s⁻¹ (bond vibration) and activation energies $E_{\mathrm{act},i}$ read from `data/parameters/activation_energies/`. Charged-species barriers may be reduced by the local electric field ($E_\mathrm{act} - q \mathbf{E} \cdot \mathbf{d}$). The kMC clock advances by $\Delta t = -\ln U / \sum_i k_i$ with $U \sim \mathrm{Uniform}(0,1)$, and events are selected from a balanced binary tree.

---

## ⚙️ Configuration & Usage Guide
Kinetix is driven by modular YAML and JSON configuration files under `data/parameters/`. This separation allows you to swap materials, defect chemistries, and device geometries without modifying the Python core. Every simulation is assembled by a *preset* file in `data/parameters/presets/`.

### 📁 Directory structure
```text
data/parameters/
├── defects/               # Mobile species, charges, and enabled events (YAML)
├── reactions/             # Redox, recombination, and passivation reactions (YAML)
├── electrical/            # Voltage protocols and current models (YAML)
├── grain_boundaries/      # GB geometry and barrier modifications (YAML)
├── activation_energies/   # Base DFT/experimental activation energies (JSON)
└── presets/               # Master files assembling a complete simulation (YAML)
```

| Subdirectory | Purpose | Example files |
|---|---|---|
| `defects/` | Species, charge states, sublattices, enabled events, scavenging | `VCM_HfO2_defects_config.yaml`, `PZT_ZrPbO3_defects_config.yaml`, `ECM_CeO2_Ag_defects_config.yaml` |
| `reactions/` | Redox, recombination, and passivation reactions | `VCM_HfO2_reactions.yaml`, `PZT_reactions_H_Ovac_passivation.yaml` |
| `electrical/` | Voltage protocols (`CONSTANT`, `RAMP_CYCLE`, `ZERO_HOLD`) and current models (Schottky, ohmic) | `electrical_RAMP_CYCLE.yaml`, `electrical_CONSTANT_STEP.yaml`, `electrical_ZERO_HOLD.yaml` |
| `grain_boundaries/` | Planar/cylindrical GB geometry and barrier models | `gb_cylindrical_VCM_HfO2.yaml`, `gb_vertical_planar.yaml`, `gb_cylindrical_PZT_ZrPbO3.yaml` |
| `activation_energies/` | Base activation energies per species (JSON) | `VCM_HfO2.json`, `PZT_ZrPbO3.json`, `activation_energies_deposition.json` |
| `presets/` | Master YAML files referencing components, materials, mesh, and solvers | `PZT_ZrTi_PbO3_2.yaml`, `VCM_HfO2_cylindrical_gb.yaml`, `PZT_ZrTi_PbO3_2_annealing.yaml` |

### Example preset
The following is a real snippet from [`data/parameters/presets/VCM_HfO2_cylindrical_gb.yaml`](data/parameters/presets/VCM_HfO2_cylindrical_gb.yaml):

```yaml
# Master file referencing all component configurations
material:
  name: "HfO2"
  mp_id: "mp-550893"                # Materials Project ID
  radius_neighbors: 3.9
  epsilon_r: 23.0

crystal:
  size: [50, 50, 50]                # Angstroms (x, y, z)
  miller_indices: [0, 0, 1]         # Surface orientation

components:
  defects: "defects/VCM_HfO2_defects_config.yaml"
  reactions: "reactions/VCM_HfO2_reactions.yaml"
  grain_boundaries: "grain_boundaries/gb_cylindrical_VCM_HfO2.yaml"
  electrical: "electrical/electrical_RAMP_CYCLE.yaml"

settings:
  simulation_type: "electronic_device"
  technology: "VCM"
  activation_energies: "activation_energies/VCM_HfO2.json"

superbasin:
  enabled_superbasin: true
  n_search_superbasin: 50
  time_step_limits: 1.0e-4          # s
  E_min: 0.5                        # eV

poisson:
  solve_Poisson: true
  screening_factor: 0.01
  conductivity:
    conductive_filament: 1e5        # S/m
    dielectric: 1.0e-1              # S/m

heat:
  solve_heat: true
  kappa_dielectric: 1.1             # W/m-K
  kappa_metal: 23                   # W/m-K
  use_thermal_inertia: true
```

---

## Dependencies
Kinetix is built entirely on free and open-source software. The full pinned environment is provided in [`environment.yml`](environment.yml) (conda). Key packages:

| Package | Version (pinned) | Role |
|---|---|---|
| [pymatgen](https://pymatgen.org) | 2025.10.7 | Crystal structures and properties from the [Materials Project](https://next-gen.materialsproject.org/) |
| [DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCS Project) | 0.8.0 | Finite-element Poisson and heat solvers |
| [gmsh](https://gmsh.info/) | 4.13.1 | Automated 3D mesh generation |
| [SciPy](https://scipy.org/) | 1.15.1 | Spatial queries (cKDTree), physical constants |
| [NumPy](https://numpy.org/) | 1.26.4 | Grid and vectorized lattice operations |
| [mpi4py](https://mpi4py.readthedocs.io/) + MPICH | 4.0.1 / 4.2.3 | MPI parallelization |
| pandas / Matplotlib / Seaborn / PyVista | 2.2.2 / 3.10.0 / 0.13.2 / 0.44.2 | Data analysis, plotting, 3D visualization |
| pytest | 9.0.2 | Automated test suite |
| Python | 3.12.8 | — |

Additional analysis/visualization packages (scikit-image, imageio, tifffile, mp-pyrho, `pymatgen-analysis-defects`) are installed through the `pip:` section of the same file.

> [!NOTE]
> **License**: Kinetix is released under the **MIT License** — free to use, modify, and distribute, with attribution. See [LICENSE](LICENSE) for full terms.

---

## 🛠️ Installation
Kinetix relies on a heavy scientific stack (DOLFINx, gmsh, MPI, pymatgen) that is easiest to install through the provided conda environment file, [`environment.yml`](environment.yml).

> [!NOTE]
> `environment.yml` was generated on Linux (e.g., for HPC use) and pins some Linux-specific builds (`gcc_linux-64`, `binutils_linux-64`, etc.). DOLFINx and the MPI-based solvers are best supported on Linux/macOS.

1. Clone the repository:
   ```bash
   git clone https://github.com/aldanads/Kinetix.git
   cd Kinetix
   ```
2. Create the conda environment from `environment.yml` (this installs the full FEniCS/DOLFINx + MPI stack and can take several minutes):
   ```bash
   conda env create -f environment.yml
   ```
   [Mamba](https://mamba.readthedocs.io/) also works as a faster drop-in replacement:
   ```bash
   mamba env create -f environment.yml
   ```
3. Activate the environment. Its name (`Kinetix`) is set by the `name:` field at the top of `environment.yml`:
   ```bash
   conda activate Kinetix
   ```
4. (Optional) Verify the key dependencies import correctly:
   ```bash
   python -c "import dolfinx, gmsh, pymatgen, mpi4py; print('Environment OK')"
   ```

## 🔑 Materials Project API Key
Kinetix fetches crystal structures and material properties (density, dielectric constants, etc.) directly from the [Materials Project](https://next-gen.materialsproject.org/) via `pymatgen`. This requires a free personal API key.

1. Log in (or register) at the Materials Project and copy your API key from your [account dashboard](https://next-gen.materialsproject.org/api).
2. Copy the provided template to create your own local `config.json` in the project root:
   ```bash
   cp config.template.json config.json
   ```
3. Open `config.json` and replace the placeholder with your key:
   ```json
   {
       "api_key": "YOUR_API_KEY_HERE"
   }
   ```

> [!IMPORTANT]
> `config.json` is listed in [`.gitignore`](.gitignore) and must **never** be committed — it holds your personal, secret API key. Only the placeholder file, `config.template.json`, is tracked in the repository.

Kinetix loads this file automatically at runtime (`kinetix/configs/config_loader.py` → `get_api_key()`), so no environment variables or extra command-line flags are needed once `config.json` exists in the project root.

---

## ▶️ Running a Simulation
Simulations are launched from the project root with [`run_simulation.py`](run_simulation.py), which drives the kMC loop for the material/defect/reaction/electrical setup described in a preset (see `data/parameters/presets/`). To see all available options:

```bash
python run_simulation.py --help
```

### Command-line interface
| Argument | Type | Description |
|---|---|---|
| `sim_id` (positional, optional) | `int` | Simulation ID used to index the parameter-sweep array in `get_parameters_from_sim_id()` (default: `0`) |
| `--config`, `-c` | `str` | Preset YAML file name or path (default: the PZT device preset, `PZT_ZrTi_PbO3_2.yaml`) |
| `--profile` | flag | Wrap the run in `cProfile`; prints the top 15 functions by cumulative time and saves `kmc_profile.prof` |
| `--allow-multi-rank-profile` | flag | Override the single-core enforcement for `--profile` (results are distorted by MPI synchronization) |
| `--dry-run` | flag | Print the resolved arguments and exit before initializing meshes or the kMC lattice |

> [!NOTE]
> Always pass `--config` explicitly: the argparse default currently expands to a name that does not match an on-disk preset, while `main()` itself defaults to `PZT_ZrTi_PbO3_2.yaml`. With `--config`, both code paths resolve to the same file.

### Basic run (single process)
```bash
conda activate Kinetix
python run_simulation.py 0 --config PZT_ZrTi_PbO3_2.yaml
```

### Parameter sweeps and preset selection
`sim_id` indexes the sweep array defined in `get_parameters_from_sim_id()` (initial vacancy concentration × temperature × hydrogen-generation rate; edit that function to customize the mapping). Use `--config` to select a different preset:

```bash
python run_simulation.py 4 --config VCM_HfO2_cylindrical_gb.yaml
```

### Running in parallel (MPI)
The DOLFINx-based Poisson/heat solvers run across MPI ranks, while the kMC lattice is evolved on rank 0 and synchronized through broadcasts (`kinetix/utils/mpi_context.py`). To run on multiple ranks:

```bash
mpiexec -n 8 python run_simulation.py 4 --config PZT_ZrTi_PbO3_2.yaml
```

### Profiling a run
Use the `--profile` flag to wrap the execution in `cProfile`. This prints the top 15 functions by cumulative time and saves the full profile to `kmc_profile.prof`.

> [!WARNING]
> **Single-core enforcement:** `cProfile` is a single-process profiler. If `--profile` is used with more than one MPI rank, only rank 0 is instrumented, and the profiler's overhead creates artificial MPI wait times that distort the bottleneck analysis. Kinetix therefore **automatically aborts** multi-rank profiling runs (see `_enforce_single_rank_profiling()` in `run_simulation.py`).

Profile on a single core:

```bash
python run_simulation.py 0 --config PZT_ZrTi_PbO3_2.yaml --profile
```

If you explicitly want to profile a multi-rank run despite the distortion, use the override flag:

```bash
mpiexec -n 4 python run_simulation.py 0 --profile --allow-multi-rank-profile
```

> [!NOTE]
> In the `--profile` code path, `main()` is invoked with default arguments, so the run always uses the default PZT preset even when `--config` is supplied. Use a single-core profiled run to benchmark a specific preset.

### Dry run (validation)
To verify your command-line arguments without initializing the heavy FEM meshes or the kMC lattice:

```bash
python run_simulation.py 4 --config PZT_ZrTi_PbO3_2.yaml --dry-run
```

### Running on an HPC cluster (PBS)
[`scripts/hpc/pbs_template.sh`](scripts/hpc/pbs_template.sh) is an example PBS submission script with job-array support (`#PBS -t 0-10`); `PBS_ARRAYID` becomes `sim_id`, and the `CONFIG_FILE`, `PYTHON_EXEC`, `MPI_EXEC`, and `#PBS` resource directives are edited at the top of the script before submission:

```bash
qsub pbs_template.sh
```

Cluster-specific notes (FEniCSx JIT compilation, GCC ≥ 9.0, shared `.dolfin_cache`) are documented in [`scripts/hpc/README.md`](scripts/hpc/README.md), and a single-core profiling job script is provided at [`scripts/hpc/profile_job.sh`](scripts/hpc/profile_job.sh).

Simulation outputs (crystal snapshots, saved state, IV curves) are written under the `program/` and `output/` subdirectories of `Sim_<id>` folders created in the preset's `output_path`.

---

## 🧪 Testing
The test suite lives in [`tests/`](tests/) and covers the FEM Poisson and heat solvers, the shared FEM solver base class, grain-boundary barrier/charge modifications, and PBC-aware migration pathways:

```text
tests/
├── conftest.py                          # Adds the project root to sys.path
├── test_FEMSolver.py                    # FEM solver base class
├── test_poisson_solver.py               # Poisson solve, charge spreading, boundary conditions
├── test_heat_solver.py                  # Steady-state heat + thermal relaxation
├── test_gb_charge_and_state_transfer.py # GB barriers and defect state transfer
└── test_migration_pathways.py           # PBC neighbor finding and pathway keys
```

Run the suite from the project root (inside the `Kinetix` environment):

```bash
conda activate Kinetix
python -m pytest tests/
```

The tests use lightweight mocks and do not require an MPI cluster or GPU hardware.

---

## 📁 Project Structure
```text
Kinetix/
├── run_simulation.py          # CLI entry point: argparse + kMC driver loop
├── environment.yml            # Conda environment (pinned builds)
├── config.template.json       # Template for the Materials Project API key
├── LICENSE                    # MIT license
├── kinetix/                   # Core Python package
│   ├── initialization.py      # Builds crystal, configs, solvers, output paths
│   ├── material_fetcher.py    # Materials Project structure/property retrieval
│   ├── configs/               # Typed YAML loaders (simulation, defects, electrical, ...)
│   ├── lattice/               # Crystal_Lattice, Site, GrainBoundary, island, cluster
│   ├── solvers/               # FEM solvers: Poisson, heat, electrical (IV)
│   ├── calculators/           # Pluggable activation-energy providers (MACE-NEB)
│   └── utils/                 # mpi_context, balanced_tree, superbasin, analysis
├── data/
│   ├── parameters/            # All user-facing YAML/JSON parameter files
│   ├── grids/                 # Saved crystal grids (.pkl, generated locally)
│   ├── mesh/                  # gmsh meshes (.msh, generated locally)
│   ├── cache/                 # Materials Project cache (git-ignored)
│   └── experimental/          # Experimental I–V data for comparison
├── scripts/
│   └── hpc/                   # PBS templates and HPC deployment notes
└── tests/                     # pytest suite
```

---

## 📚 How to Cite

If you use Kinetix in your research or adapt part of the code, please cite the following:

### Core Framework (Published Versions)
The core kMC framework has been validated and used in the following publications:

**Aldana, Samuel**, and Michael Nolan. "Control of Growth Morphology of Deposited fcc Metals through Tuning Substrate–Metal Interactions." ACS Applied Materials & Interfaces (2025).
* **DOI:** [10.1021/acsami.5c18081](https://doi.org/10.1021/acsami.5c18081)
* **Code:** [github.com/aldanads/control-of-growth-morphology...](https://github.com/aldanads/Control-of-growth-morphology-of-deposited-fcc-metals-through-tuning-substrate-metal-interactions)
* **Zenodo:** [10.5281/zenodo.18898755](https://doi.org/10.5281/zenodo.18898755)

**Aldana, Samuel**, Cara-Lena Nies, and Michael Nolan. "Control of Cu morphology on TaN barrier and combined Ru-TaN barrier/liner substrates for nanoscale interconnects from atomistic kinetic Monte Carlo simulations." Nanoscale 17, no. 19 (2025): 12450-12464.
* **DOI:** [10.1039/D4NR04505J](https://doi.org/10.1039/D4NR04505J)
* **Code:** [github.com/aldanads/control-of-growth-morphology...](https://github.com/aldanads/Control-of-growth-morphology-of-deposited-fcc-metals-through-tuning-substrate-metal-interactions)
* **Zenodo:** [10.5281/zenodo.19151596](https://doi.org/10.5281/zenodo.19151596)
