"""Kinetix: kinetic Monte Carlo simulator for materials deposition, annealing,
and memristive device modeling."""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Public API re-exports. These make the primary simulation objects importable
# directly from the top-level package, e.g. `from kinetix import KMCSimulator`.
# ---------------------------------------------------------------------------
from kinetix.lattice.simulator import KMCSimulator
from kinetix.lattice.site import Site
from kinetix.lattice.cluster import Cluster
from kinetix.lattice.grain_boundary import GrainBoundary
from kinetix.lattice.island import Island
from kinetix.configs.simulation_config import SimulationConfig
try:
    from kinetix.solvers.poisson import PoissonSolver
    from kinetix.solvers.heat import HeatSolver
except ImportError:
    pass  # dolfinx not available on this system
from kinetix.solvers.electrical import ElectricalController
from kinetix.utils.superbasin import Superbasin
from kinetix.utils.mpi_context import MPIContext

__all__ = [
    "KMCSimulator",
    "Site",
    "Cluster",
    "GrainBoundary",
    "Island",
    "SimulationConfig",
    "PoissonSolver",
    "HeatSolver",
    "ElectricalController",
    "Superbasin",
    "MPIContext",
    "__version__",
]
# Without the FEM stack the solver names above are absent; keep `import *` valid.
__all__ = [name for name in __all__ if name in globals()]
