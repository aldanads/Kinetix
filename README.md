# Kinetix
A fully open-source kinetic Monte Carlo (kMC) simulator for materials deposition, annealing, and memristive device modeling

## Capacities
Kinetix is a Python-based, license-free simulation framework that enables atomic-scale modeling of:
* Material deposition (e.g., fcc metals),
* Vacuum annealing of deposited films, and
* Resistive switching in redox- and filamentary-type memristors.

## Dependencies
Built entirely on free and open-source software, Kinetix integrates seamlessly with:
* [pymatgen](https://pymatgen.org) – to fetch crystal structures from the [Materials Project](https://next-gen.materialsproject.org/),
* [gmsh](https://gmsh.info/) – for automated 3D mesh generation,
* [DOLFINx](https://github.com/FEniCS/dolfinx) (part of the [FEniCS Project](https://fenicsproject.org/)) – to solve the Poisson equation for electrostatics,
* MPI + OpenMP – for hybrid parallelization (MPI via DOLFINx, OpenMP for lattice operations).

## Aim
Kinetix aims to bridge materials science and device physics by providing a transparent, modular, and accessible platform for multiscale simulation of emerging electronic devices, ideal for research in neuromorphic computing, memristors, and thin-film processing.
