\# HPC Deployment Guide



\## Prerequisites



1\. \*\*Modern GCC (>= 9.0)\*\*: FEniCSx JIT compilation requires C++17 support

2\. \*\*MPI matching your conda environment\*\*: Check with `conda list | grep mpi`

3\. \*\*Shared filesystem\*\*: The `.dolfin\_cache` folder must be on a shared filesystem



\## Quick Start



1\. Copy the template (`pbs\_template.pbs`)

2\. Edit the `CUSTOMIZE THESE PATHS` section

3\. Submit the job



