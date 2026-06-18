#!/bin/bash
# =============================================================================
# Kinetix HPC PBS Template
# =============================================================================
# This is a template for running Kinetix on PBS HPC clusters.
# You MUST customize the paths and module names for your specific cluster.
#
# Key requirements:
# 1. GCC >= 9.0 (FEniCSx JIT compilation requires modern C++ support)
# 2. MPI implementation matching your conda environment (MPICH or OpenMPI)
# 3. Shared filesystem for JIT cache (.dolfin_cache folder)
# =============================================================================

#PBS -l nodes=1:ppn=4
#PBS -N Kinetix_simulation
#PBS -q default
#PBS -j oe
#PBS -t 1-10                    # Job array for parameter sweeps


# 1. Navigate to submission directory
cd $PBS_O_WORKDIR

# =============================================================================
# CUSTOMIZE THESE PATHS FOR YOUR CLUSTER
# =============================================================================
# Path to your conda environment's Python
PYTHON_EXEC="/path/to/your/conda/envs/Kinetix/bin/python"
MPI_EXEC="/path/to/your/conda/envs/Kinetix/bin/mpiexec"

# Path to modern GCC (REQUIRED: must be >= 9.0)
# Option A: Use module system (if available in non-interactive shells)
# module load gcc/12.3

# Option B: Hardcode the path (more reliable)
GCC_BIN_DIR="/path/to/gcc-12/bin"
GCC_LIB_DIR="/path/to/gcc-12/lib64"
export PATH="$GCC_BIN_DIR:$PATH"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

# =============================================================================
# FEniCSx JIT Configuration (DO NOT CHANGE unless you know what you're doing)
# =============================================================================

# Shared cache directory prevents MPI race conditions during JIT compilation
export XDG_CACHE_HOME="$PBS_O_WORKDIR/.dolfin_cache"
mkdir -p $XDG_CACHE_HOME

# Set compilers explicitly
export CC="$GCC_BIN_DIR/gcc"
export CXX="$GCC_BIN_DIR/g++"

# Timeout for JIT compilation (increase if you have complex forms)
export DOLFINX_JIT_TIMEOUT=300

# =============================================================================
# Job Configuration
# =============================================================================

SIM_ID=${PBS_ARRAYID:-1}

# Core counting (works with both PBS_NCPUS and PBS_NODEFILE)
if [ -n "$PBS_NCPUS" ]; then
    CORES=$PBS_NCPUS
else
    CORES=$(wc -l < "$PBS_NODEFILE")
fi

echo "=================================================="
echo "Kinetix Simulation ID: $SIM_ID"
echo "Cores: $CORES"
echo "GCC: $(gcc --version | head -n 1)"
echo "Python: $($PYTHON_EXEC --version)"
echo "Working Dir: $PBS_O_WORKDIR"
echo "JIT Cache: $XDG_CACHE_HOME"
echo "Time: $(date)"
echo "=================================================="

# Run simulation
$MPI_EXEC -n $CORES $PYTHON_EXEC run_simulation.py $SIM_ID

echo "=================================================="
echo "Completed at $(date)"
echo "=================================================="