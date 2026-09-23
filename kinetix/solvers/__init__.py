from .coordinator import SolverCoordinator

try:
    from .poisson import PoissonSolver
    from .heat import HeatSolver
except ImportError:
    pass  # dolfinx not available on this system
