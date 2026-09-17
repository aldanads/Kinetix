#!/bin/bash
# =============================================================================
# Kinetix HPC Slurm Template
# =============================================================================
# This is a template for running Kinetix on Slurm HPC clusters.
#
# Key requirements:
# 1. GCC >= 9.0 (FEniCSx JIT compilation requires modern C++ support)
# 2. MPI implementation matching your conda environment (MPICH or OpenMPI)
# 3. Shared filesystem for JIT cache (.dolfin_cache folder)
# =============================================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=Kinetix_PZT
#SBATCH --partition=cpuq-short             # Customize: check available partitions with 'sinfo'
#SBATCH --output=Kinetix_%A_%a.out      # %A = Job ID, %a = Array Task ID (joins stdout/stderr)
#SBATCH --array=0-5                    # Job array for parameter sweeps

# 1. Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# =============================================================================
# CUSTOMIZE THESE PATHS FOR YOUR CLUSTER
# =============================================================================

# Path to modern GCC (REQUIRED: must be >= 9.0)
# Option A: Use module system
 module load gcc/13.1.0

# Initialize conda for batch scripts
source ~/.bashrc
conda activate Kinetix

# Use Python and MPI from the conda environment
# (This ensures mpi4py/dolfinx match the MPI runtime)
PYTHON_EXEC="python"
MPI_EXEC="mpiexec"

# Option B: Hardcode the path (more reliable)
#GCC_BIN_DIR="/path/to/gcc-12/bin"
#GCC_LIB_DIR="/path/to/gcc-12/lib64"
#export PATH="$GCC_BIN_DIR:$PATH"
#export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

# =============================================================================
# FEniCSx JIT Configuration
# =============================================================================

# Shared cache directory prevents MPI race conditions during JIT compilation
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.dolfin_cache"
mkdir -p $XDG_CACHE_HOME

# Set compilers explicitly to the loaded module
export CC="gcc"
export CXX="g++"

# Timeout for JIT compilation (increase if you have complex forms)
export DOLFINX_JIT_TIMEOUT=300

# =============================================================================
# Job Configuration
# =============================================================================

SIM_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_FILE="PZT_ZrTi_PbO3_2.yaml"

# Core counting (Slurm calculates this automatically from --ntasks-per-node)
CORES=$SLURM_NTASKS

echo "=================================================="
echo "Kinetix Simulation ID: $SIM_ID"
echo "Cores: $CORES"
echo "Partition: cpuq-short"
echo "GCC: $(gcc --version | head -n 1)"
echo "Python: $($PYTHON_EXEC --version)"
echo "Working Dir: $SLURM_SUBMIT_DIR"
echo "JIT Cache: $XDG_CACHE_HOME"
echo "Time: $(date)"
echo "=================================================="

# Run simulation
$MPI_EXEC -n $CORES $PYTHON_EXEC run_simulation.py $SIM_ID --config "$CONFIG_FILE"

echo "=================================================="
echo "Completed at $(date)"
echo "=================================================="
