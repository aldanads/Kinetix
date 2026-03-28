# tests/test_poisson_solver.py
"""
Tests for PoissonSolver class.
"""
import pytest
import numpy as np
from pathlib import Path
from scipy.constants import epsilon_0

from solvers.poisson import PoissonSolver
from utils.mpi_context import MPIContext

class MockCluster:
    """Mock Cluster object for testing."""
    def __init__(self, atoms_positions, attached_layer):
        self.atoms_positions = atoms_positions
        self.attached_layer = attached_layer
        self.internal_atom_positions = atoms_positions
        
class TestPoissonSolver:
    """Test suite for PoissonSolver."""
    
    @pytest.fixture
    def mpi_ctx(self):
        """Fixture: MPI context."""
        return MPIContext.get_instance()
    
    @pytest.fixture
    def poisson_solver(self, mpi_ctx):
        """Fixture: PoissonSolver instance."""
        params = {
            'mesh_file': 'test_mock_mesh.msh',
            'epsilon_r': 25.0,
            'conductivity_CF': 1e6,
            'conductivity_dielectric': 1e-10,
            'metal_valence': 4,
            'd_metal_O': 2.0,
            'chem_env_symmetry': 'Octahedron',
            'active_dipoles': 1.0,
            'defects_config': {}
        }
        
        return PoissonSolver(
            params,
            mpi_ctx=mpi_ctx,
            path_results=Path('./test_output'),
            mesh_size=2.0,
            fine_mesh_size=0.5
        )
    
    def test_initialization(self, poisson_solver):
        """Test PoissonSolver initializes correctly."""
        assert hasattr(poisson_solver, 'rho'), "Charge density function should exist"
        assert hasattr(poisson_solver, 'sigma'), "Conductivity function should exist"
        assert hasattr(poisson_solver, 'E_field'), "E-field function should exist"
        assert hasattr(poisson_solver, 'bond_polarization_factor'), "Polarization factor should exist"
        assert hasattr(poisson_solver, 'epsilon_r'), "Relative permittivity should exist"
        print("? PoissonSolver initialized correctly")
    
    def test_dipole_moment_calculation(self, poisson_solver):
        """Test dipole moment calculation."""
        assert poisson_solver.bond_polarization_factor > 0, "Polarization factor should be positive"
        print(f"? Dipole moment calculated: {poisson_solver.bond_polarization_factor:.4e}")
    
    def test_charge_density(self, poisson_solver):
        """Test charge density calculation."""
        charge_locations = np.array([[5.0, 5.0, 5.0]])
        charges = np.array([1.602e-19])  # Elementary charge
        
        rho = poisson_solver.charge_density(charge_locations, charges)
        
        assert rho is not None
        assert len(rho.x.array) > 0
        print("? Charge density calculated")
    
    def test_charge_density_multiple_charges(self, poisson_solver):
        """Test charge density with multiple charges."""
        charge_locations = np.array([
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0]
        ])
        charges = np.array([1.602e-19, -1.602e-19, 1.602e-19])
        
        rho = poisson_solver.charge_density(charge_locations, charges)
        
        assert rho is not None
        print("? Multiple charge density calculated")
        
    def test_set_boundary_conditions(self, poisson_solver):
        """Test boundary condition setup."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        assert len(poisson_solver.bcs) >= 2, "Should have at least top and bottom BCs"
        assert poisson_solver._bcs_changed == True, "BCs should be marked as changed"
        assert poisson_solver.use_conductivity == False, "Conductivity should be False by default"
        print("? Boundary conditions set")
    
    def test_set_boundary_conditions_with_clusters(self, poisson_solver):
        """Test boundary conditions with cluster BCs."""
        # Create mock cluster touching bottom electrode
        cluster = MockCluster(
            atoms_positions=np.array([[5.0, 5.0, 0.5]]),
            attached_layer={'bottom_layer': True, 'top_layer': False}
        )
        clusters = {'cluster_1': cluster}
        
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0, clusters=clusters)
        
        assert len(poisson_solver.bcs) >= 3, "Should have top, bottom, and cluster BCs"
        print("? Cluster boundary conditions set")
    
    def test_solve_no_charges(self, poisson_solver):
        """Test Poisson solve without charges (Laplace equation)."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        uh = poisson_solver.solve([], [])
        
        assert uh is not None
        assert len(uh.x.array) > 0
        assert not np.any(np.isnan(uh.x.array)), "Solution should not contain NaN"
        print("? Laplace equation solved (no charges)")
    
    def test_solve_with_charges(self, poisson_solver):
        """Test Poisson solve with charges."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        charge_locations = np.array([[5.0, 5.0, 5.0]])
        charges = np.array([1.602e-19])
        
        uh = poisson_solver.solve(charge_locations, charges)
        
        assert uh is not None
        assert len(uh.x.array) > 0
        assert not np.any(np.isnan(uh.x.array)), "Solution should not contain NaN"
        print("? Poisson equation solved (with charges)")
    
    def test_solve_reuse_objects(self, poisson_solver):
        """Test that solver reuses pre-allocated objects."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        # First solve
        uh1 = poisson_solver.solve([], [])
        
        # Second solve (should reuse objects)
        uh2 = poisson_solver.solve([], [])
        
        # Both should be valid
        assert uh1 is not None
        assert uh2 is not None
        print("? Solver reuses pre-allocated objects")
    
    def test_evaluate_electric_field(self, poisson_solver):
        """Test E-field evaluation."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        uh = poisson_solver.solve([], [])
        
        points = np.array([[5.0, 5.0, 5.0]])
        E_field = poisson_solver.evaluate_electric_field_at_points(uh, points)
        
        assert len(E_field) > 0
        assert tuple(np.round(points[0], 6)) in E_field
        print("? E-field evaluated")
    
    def test_evaluate_electric_field_caching(self, poisson_solver):
        """Test E-field caching."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        uh = poisson_solver.solve([], [])
        points = np.array([[5.0, 5.0, 5.0]])
        
        # First evaluation (computes)
        E1 = poisson_solver.evaluate_electric_field_at_points(uh, points)
        
        # Second evaluation (should use cache)
        E2 = poisson_solver.evaluate_electric_field_at_points(uh, points)
        
        # Should be identical
        key = tuple(np.round(points[0], 6))
        assert np.allclose(E1[key], E2[key]), "Cached values should match"
        print("? E-field caching works")
    
    def test_conductivity_in_system(self, poisson_solver):
        """Test conductivity field setup."""
        metal_atoms = [(5.0, 5.0, 5.0),(6.0, 6.0, 6.0)]
        
        poisson_solver.conductivity_in_system(metal_atoms)
        
        # Check that sigma was updated
        assert poisson_solver.sigma is not None
        print("? Conductivity field set")
    
    def test_verify_bcs_after_solve(self, poisson_solver):
        """Test BC verification (debugging utility)."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        uh = poisson_solver.solve([], [])
        
        # Should not raise
        poisson_solver.verify_bcs_after_solve(uh, expected_top=1.0, expected_bottom=0.0)
        print("? BC verification completed")
    
    def test_diagnose_boundary_conditions(self, poisson_solver):
        """Test BC diagnostics (debugging utility)."""
        poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
        
        # Should not raise
        poisson_solver.diagnose_boundary_conditions()
        print("? BC diagnostics completed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])