# Kinetix
A fully open-source kinetic Monte Carlo (kMC) simulator for materials deposition, annealing, and memristive device modeling

> [!WARNING]
> **⚠️ Active Development**
> This tool is currently in beta. The README and installation workflows are not ready yet. 
> Feel free to contact me directly. 
> Best suited for those comfortable with Python, FEniCS/DOLFINx and MPI environments.

## Aim
Kinetix aims to bridge materials science and device physics by providing a transparent, modular, and accessible platform for multiscale simulation of emerging electronic devices, ideal for research in neuromorphic computing, memristors, and thin-film processing.

## Capabilities
Kinetix is a Python-based, open-source simulation framework (**MIT License**) that enables atomic-scale modeling of:
* **Material Deposition:** Nucleation and growth of thin films (e.g., fcc metals),
* **Vacuum Annealing:** Thermal evolution, defect passivation and morphology relaxation of deposited films.
* **Resistive Switching:** Formation and dissolution of conductive filaments in redox- and valence change memristors (VCM and ECM).
* **Grain Boundary (GB) Engineering:** Spatially resolved modifications of activation energies at grain boundaries
* **Multiphysics Coupling:** Real-time feedback between electrostatics, Joule heating and stochastic defect kinetics.

---

## 🔄 Simulation Workflow
1. **Lattice Generation:** Constructs the 3D atomic grid (including grain boundaries and interfaces) using `pymatgen` and the mesh for the FEM solvers using `gmsh`.
2. **FEM Solvers:** Computes the electric field (Poisson) and temperature distribution (Heat) based on the current defect configuration and applied boundary conditions.
3. **Energy Landscape Update:** Calculates local activation energies for all possible defect events (migration, generation, recombination).
4. **kMC Step:** Selects an event using the BKL (Bortz-Kalos-Lebowitz) or rejection-free algorithm, advances the simulation time, and updates the lattice.
5. **Loop:** Repeats steps 2–4 to capture the dynamic evolution of the device.

---

## ⚙️ Configuration & Usage Guide
Kinetix is driven by a set of modular YAML and JSON configuration files located in the `data/parameters/` directory. This separation allows to swap materials, defect chemistries, and device geometries without modifying the core Python code.

### 📁 Directory Structure
```text
data/parameters/
├── defects/                 # Defines mobile species (e.g., H+, V_O)
├── reactions/               # Defines redox, recombination, and passivation reactions
├── electrical/              # Electrode boundary conditions and voltage sweeps
├── grain_boundaries/        # GB geometry and localized barrier modifications
└── activation_energies/     # Base DFT/experimental energy barriers (JSON)
```
---

## Dependencies
Built entirely on free and open-source software, Kinetix integrates seamlessly with:
* [pymatgen](https://pymatgen.org) – to fetch crystal structures from the [Materials Project](https://next-gen.materialsproject.org/),
* [gmsh](https://gmsh.info/) – for automated 3D mesh generation,
* [DOLFINx](https://github.com/FEniCS/dolfinx) (part of the [FEniCS Project](https://fenicsproject.org/)) – To solve the Poisson and Heat equations via the Finite Element Method.
* [SciPy](https://scipy.org/) – For efficient spatial queries (cKDTree) and neighbor-finding algorithms with periodic boundary conditions.
* MPI + OpenMP – for hybrid parallelization (MPI via DOLFINx, OpenMP for lattice operations).

> [!NOTE]
> **License**: Kinetix is released under the **MIT License** — free to use, modify, and distribute, with attribution. See [LICENSE](LICENSE) for full terms.

---
## 🛠️ Installation

Kinetix relies on a heavy scientific stack (DOLFINx, gmsh, MPI, pymatgen) that is easiest to install through the provided conda environment file, [`environment.yml`](environment.yml)

> [!NOTE]
> `environment.yml` was generated on Linux (e.g., for HPC use) and pins some Linux-specific builds (`gcc_linux-64`, `binutils_linux-64`, etc.). DOLFINx and the MPI-based solvers are best supported on Linux/macOS.

1. Close the repository:
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

Kinetix loads this file automatically at runtime (see `kinetix/configs/config_loader.py` and `get_api_key()`), so no environment variables or extra command-line flags are needed once `config.json` exists in the project root.

***

## ▶️ Running a Simulation
Simulations are launched from the project root with [`run_simulation.py`](run_simulation.py), which drives the kMC loop for the material/defect/reaction/electrical setup described in a YAML preset (see `data/parameters/presets/`, e.g. `PZT_ZrTi_PbO3_2.yaml`).

To see all available options:
```bash
python run_simulation.py --help
```

### Basic run (single process) 
```bash
conda activate Kinetix
python run_simulation.py
```

Running without arguments defaults to `sim_id = 0` and uses the default preset defined in the script (e.g., `PZT_ZrTi_PbO3_2.yaml`)

### Specifying a configuration and parameter set
Use the `--config` (or `-c`) flag to select a specific material/device preset from `data/parameters/presets/`, and pass a `sim_id` integer to index into your parameter sweep array:

```bash
python run_simulation.py 4 --config VCM_HfO2_cylindrical_gb.yaml
```

_Edit `get_parameters_from_sim_id()` in `run_simulation.py` to customize the parameter sweep mappings._

### Running in parallel (MPI)
The DOLFINx-based Poisson/heat solvers support MPI, while lattice operations use OpenMP. To run on multiple ranks:
```bash
mpiexec -n 8 python run_simulation.py 4 --config PZT_ZrTi_PbO3_2.yaml
```

> [!NOTE]
> The coupled electrostatics/heat solvers only run on Linux (guarded by `platform.system() == 'Linux'` in the code). On other platforms, deposition/annealing simulations without the field solvers will still run.

### Profiling a run
Use the `--profile` flag to wrap the execution in `cProfile`. This prints the top 15 functions by cumulative time and saves the full profile to `kmc_profile.prof`.

> [!WARNING]
> **Single-core enforcement:** `cProfile` is a single-process profiler. If launched with multiple MPI ranks, only rank 0 is instrumented, and the profiler's overhead creates artificial MPI wait times that distort the bottleneck analysis. Therefore, Kinetix **automatically aborts** if `--profile` is used with `>1` MPI ranks.
To profile, run with a single core (as shown above). If you explicitly want to profile a multi-rank run despite the distortion, use the override flag:
```bash
mpiexec -n 4 python run_simulation.py 0 --profile --allow-multi-rank-profile
```

### Dry run (validation)
To verify your command-line arguments and configuration paths without initializing the heavy FEM meshes or kMC lattice:
```bash
python run_simulation.py 4 --config PZT_ZrTi(PbO3)2.yaml --dry-run
```

### Running on an HPC cluster (PBS)
[`pbs_template.sh`](pbs_template.sh) is an example PBS submission script. When submitting jobs, you can easily swap the `CONFIG_FILE` variable at the top of the script to run different material presets across your cluster nodes:
```bash
qsub pbs_template.sh
```

Edit the `CONFIG_FILE`, `SIM_ID`, and `#PBS` resource directives (`nodes`, `ppn`) at the top of the script to match your job and cluster paths.
Simulation outputs (crystal snapshots, saved state, IV curves) are written under the program/ and output/ directories.

***

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
