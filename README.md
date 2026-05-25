# Kinetix
A fully open-source kinetic Monte Carlo (kMC) simulator for materials deposition, annealing, and memristive device modeling

> [!WARNING]
> **⚠️ Active Development**
> This tool is currently in beta. The README and installation workflows are not ready yet. 
> Feel free to contact me directly. 
> Best suited for those comfortable with Python/FEniCS environments.

## Capacities
Kinetix is a Python-based, open-source simulation framework (**MIT License**) that enables atomic-scale modeling of:
* Material deposition (e.g., fcc metals),
* Vacuum annealing of deposited films, and
* Resistive switching in redox- and filamentary-type memristors.

## Dependencies
Built entirely on free and open-source software, Kinetix integrates seamlessly with:
* [pymatgen](https://pymatgen.org) – to fetch crystal structures from the [Materials Project](https://next-gen.materialsproject.org/),
* [gmsh](https://gmsh.info/) – for automated 3D mesh generation,
* [DOLFINx](https://github.com/FEniCS/dolfinx) (part of the [FEniCS Project](https://fenicsproject.org/)) – to solve the Poisson equation for electrostatics,
* MPI + OpenMP – for hybrid parallelization (MPI via DOLFINx, OpenMP for lattice operations).

> [!NOTE]
> **License**: Kinetix is released under the **MIT License** — free to use, modify, and distribute, with attribution. See [LICENSE](LICENSE) for full terms.

## Aim
Kinetix aims to bridge materials science and device physics by providing a transparent, modular, and accessible platform for multiscale simulation of emerging electronic devices, ideal for research in neuromorphic computing, memristors, and thin-film processing.

## 📚 How to Cite

If you use Kinetix in your research or adapt part of the code, please cite the following:

### Core Framework (Published Versions)
The core kMC framework has been validated and used in the following publications:
**Aldana, Samuel**, and Michael Nolan. "Control of Growth Morphology of Deposited fcc Metals through Tuning Substrate–Metal Interactions." ACS Applied Materials & Interfaces (2025).
**DOI:** [10.1021/acsami.5c18081](https://doi.org/10.1021/acsami.5c18081)
**Code:** [github.com/aldanads/control-of-growth-morphology...](https://github.com/aldanads/Control-of-growth-morphology-of-deposited-fcc-metals-through-tuning-substrate-metal-interactions)
**Zenodo:** [10.5281/zenodo.18898755](https://doi.org/10.5281/zenodo.18898755)

**Aldana, Samuel**, Cara-Lena Nies, and Michael Nolan. "Control of Cu morphology on TaN barrier and combined Ru-TaN barrier/liner substrates for nanoscale interconnects from atomistic kinetic Monte Carlo simulations." Nanoscale 17, no. 19 (2025): 12450-12464.
**DOI:** [10.1039/D4NR04505J](https://doi.org/10.1039/D4NR04505J)
**Code:** [github.com/aldanads/control-of-growth-morphology...](https://github.com/aldanads/Control-of-growth-morphology-of-deposited-fcc-metals-through-tuning-substrate-metal-interactions)
**Zenodo:** [10.5281/zenodo.19151596](https://doi.org/10.5281/zenodo.19151596)
