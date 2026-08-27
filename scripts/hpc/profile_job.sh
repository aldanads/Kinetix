#!/bin/bash
#PBS -l nodes=1:ppn=1          # ? Critical: 1 core only
#PBS -N kmc_profile
#PBS -q default
#PBS -j oe

SIM_ID=0
CONFIG_FILE="PZT_ZrTi_PbO3_2.yaml"

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
# This prevents MPI ranks from writing to local /tmp folders and clashing
export XDG_CACHE_DIR="$PBS_O_WORKDIR/.dolfin_cache"
export XDG_CACHE_HOME="$PBS_O_WORKDIR/.dolfin_cache"
mkdir -p $XDG_CACHE_HOME
export CC="$GCC_BIN_DIR/gcc"
export CXX="$GCC_BIN_DIR/g++"
# Disable parallel JIT compilation to prevent race conditions
export DOLFINX_JIT_TIMEOUT=300

# 2. Absolute path to the conda Python environment
PYTHON_EXEC="/sfihome/samuel.delgado/anaconda3/envs/Kinetix/bin/python"

# 3. Set cores to 1
CORES=1

echo "=================================================="
echo "PROFILING JOB"
echo "Simulation ID: $SIM_ID"
echo "Config: $CONFIG_FILE"
echo "Requested Cores: $CORES"
echo "Working Directory: $PBS_O_WORKDIR"
echo "Python Version: $($PYTHON_EXEC --version)"
echo "Time: $(date)"
echo "=================================================="

# Run with profiling flag (no MPI!)
$PYTHON_EXEC run_simulation.py "$SIM_ID" --profile --config "$CONFIG_FILE"

echo "=================================================="
echo "Profiling completed at $(date)"
echo "Profile file: $PBS_O_WORKDIR/kmc_profile.prof"
echo "=================================================="
