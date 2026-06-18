#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -N kmc_c_8
#PBS -q default
#PBS -j oe

SIM_ID=4

# 1. Navigate to submission directory
cd $PBS_O_WORKDIR

# ==============================================================================
# The path to GCC 12.3
# ==============================================================================
GCC_BIN_DIR="/local/gcc-12.3.0/bin"
GCC_LIB_DIR="/local/gcc-12.3.0/lib64"

# Prepend to PATH so "which gcc" finds the right one
export PATH="$GCC_BIN_DIR:$PATH"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

# Verify the correct gcc is loaded
echo "GCC path: $(which gcc)"
echo "GCC version: $(gcc --version | head -n 1)"


# This prevents MPI ranks from writing to local /tmp folders and clashing
export XDG_CACHE_DIR="$PBS_O_WORKDIR/.dolfin_cache"
export XDG_CACHE_HOME="$PBS_O_WORKDIR/.dolfin_cache"
mkdir -p $XDG_CACHE_HOME

export CC="$GCC_BIN_DIR/gcc"
export CXX="$GCC_BIN_DIR/g++"

echo "CC is set to: $CC"
echo "CXX is set to: $CXX"


# Disable parallel JIT compilation to prevent race conditions
export DOLFINX_JIT_TIMEOUT=300

# 2. Absolute path to the conda Python environment
PYTHON_EXEC="/sfihome/samuel.delgado/anaconda3/envs/Kinetix/bin/python"
MPI_EXEC="/sfihome/samuel.delgado/anaconda3/envs/Kinetix/bin/mpiexec"

# 3. Capture the number of cores requested by the scheduler
# 3. Bulletproof core counting: Use PBS_NCPUS if available, otherwise count lines in the nodefile
if [ -n "$PBS_NCPUS" ]; then
    CORES=$PBS_NCPUS
else
    CORES=$(wc -l < "$PBS_NODEFILE")
fi

echo "=================================================="
echo "Requested Cores (ncpus): $CORES"
echo "Working Directory: $PBS_O_WORKDIR"
echo "Python Version: $($PYTHON_EXEC --version)"
echo "MPI Version: $(mpirun --version | head -n 1)"
echo "Time: $(date)"
echo "=================================================="

$MPI_EXEC -n $CORES $PYTHON_EXEC run_simulation.py $SIM_ID

echo "=================================================="
echo "Simulation completed at $(date)"
echo "=================================================="