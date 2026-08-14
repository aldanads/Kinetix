# Kinetix
A fully open-source kinetic Monte Carlo (kMC) simulator for materials deposition, annealing, and memristive device modeling

> [!WARNING]
> **⚠️ Active Development**
> This tool is currently in beta. The README and installation workflows are not ready yet. 
> Feel free to contact me directly. 
> Best suited for those comfortable with Python, FEniCS/DOLFINx and MPI environments.

## Aim
Kinetix aims to bridge materials science and device physics by providing a transparent, modular, and accessible platform for multiscale simulation of emerging electronic devices, ideal for research in neuromorphic computing, memristors, and thin-film processing.

## Capacities
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
