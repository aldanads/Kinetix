"""
Tests for FEMSolverBase class.
"""
import pytest
import numpy as np
from pathlib import Path
from dolfinx import fem

from solvers.FEMSolverBase import FEMSolverBase
from utils.mpi_context import MPIContext



class TestFEMSolverBase:
    """Test suite for FEMSolverBase."""
    
    def test_mpi_context_available(self):
        """Test that MPI context can be created."""
        mpi_ctx = MPIContext.get_instance()
        
        assert mpi_ctx is not None
        assert hasattr(mpi_ctx, 'rank')
        assert hasattr(mpi_ctx, 'comm')
        print(f"? MPI context created (rank={mpi_ctx.rank})")
    
    def test_base_initialization(self):
        """Test that FEMSolverBase can be initialized."""
        mpi_ctx = MPIContext.get_instance()
        
        params = {
            'mesh_file': 'test_mock_mesh.msh',
            'defects_config': {}
        }
        
        
        # This will fail if mesh doesn't exist, which is OK for now
        try:
            solver = FEMSolverBase(
                params,
                mpi_ctx=mpi_ctx,
                path_results=Path('./test_output')
            )
            print("? FEMSolverBase initialized successfully")
        except FileNotFoundError as e:
            # Expected if mesh file doesn't exist yet
            print(f"??  Mesh file not found (expected for first test): {e}")
            pytest.skip("Mesh file not available")
    
    def test_abstract_methods_raise_error(self):
        """Test that abstract methods raise NotImplementedError."""
        mpi_ctx = MPIContext.get_instance()
        
        params = {
            'mesh_file': 'test_mock_mesh.msh',
            'defects_config': {}
        }
        
        try:
            solver = FEMSolverBase(
                params,
                mpi_ctx=mpi_ctx,
                path_results=Path('./test_output')
            )
            
            # These should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                solver.set_boundary_conditions()
            
            with pytest.raises(NotImplementedError):
                solver.solve()
            
            print("? Abstract methods raise NotImplementedError correctly")
            
        except FileNotFoundError:
            pytest.skip("Mesh file not available")
            
    def test_evaluate_at_points(self):
        """Test point evaluation with base class method."""
        mpi_ctx = MPIContext.get_instance()
        params = {'mesh_file': 'test_mock_mesh.msh', 'defects_config': {}}
        
        solver = FEMSolverBase(
            params,
            mpi_ctx=mpi_ctx,
            path_results=Path('./test_output')
        )
        
        # Create test function: f(x,y,z) = x + y + z
        f = fem.Function(solver.V)
        f.interpolate(lambda x: x[0] + x[1] + x[2])
        
        # Test points inside domain
        points = np.array([[5.0, 5.0, 5.0]])
        values = solver.evaluate_at_points(f, points)
        
        assert len(values) == 1
        expected = 5.0 + 5.0 + 5.0  # 15.0
        assert np.isclose(values[0], expected, rtol=1e-5), f"Expected {expected}, got {values[0]}"
        print(f"? Point evaluation works (expected={expected}, got={values[0]:.6f})")
        
    def test_evaluate_at_points_empty(self):
        """Test point evaluation with empty input."""
        mpi_ctx = MPIContext.get_instance()
        params = {'mesh_file': 'test_mock_mesh.msh', 'defects_config': {}}
        
        solver = FEMSolverBase(
            params,
            mpi_ctx=mpi_ctx,
            path_results=Path('./test_output')
        )
        
        f = fem.Function(solver.V)
        
        # Empty input should return empty array
        result = solver.evaluate_at_points(f, [])
        assert len(result) == 0
        print("? Empty input handled correctly")
        
    def test_save_solution(self):
        """Test solution saving."""
        mpi_ctx = MPIContext.get_instance()
        params = {'mesh_file': 'test_mock_mesh.msh', 'defects_config': {}}
        
        solver = FEMSolverBase(
            params,
            mpi_ctx=mpi_ctx,
            path_results=Path('./test_output')
        )
        
        # Setup output
        solver._setup_time_series_output()
        
        # Create test function
        f = fem.Function(solver.V)
        f.interpolate(lambda x: x[0])
        
        # Save
        files = solver.save_solution(
            f,
            filename=solver.output_base,
            time_value=1e-9,
            timestep=1,
            save_csv=True
        )
        
        assert len(files) >= 1
        assert any(f.endswith('.vtu') or f.endswith('.bp') for f in files)
        print(f"? Save solution works: {files}")
        
    def test_evaluate_at_points_scalar_and_vector(self):
      """Test both scalar and vector functions work correctly."""

      mpi_ctx = MPIContext.get_instance()
      params = {'mesh_file': 'test_mock_mesh.msh', 'defects_config': {}}
        
      solver = FEMSolverBase(
        params,
        mpi_ctx=mpi_ctx,
        path_results=Path('./test_output')
      )
      
      # Test 1: Scalar function
      f_scalar = fem.Function(solver.V)
      f_scalar.interpolate(lambda x: x[0] + x[1] + x[2])
      
      points = np.array([[5.0, 5.0, 5.0]])
      values_scalar = solver.evaluate_at_points(f_scalar, points)
      
      # ? Should be 1D array for scalar
      assert values_scalar.shape == (1,), f"Scalar should be (1,), got {values_scalar.shape}"
      print(f"? Scalar function: shape={values_scalar.shape}, value={values_scalar[0]:.2f}")
      
      # Test 2: Vector function
      f_vector = fem.Function(solver.V_vec)
      f_vector.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
      
      values_vector = solver.evaluate_at_points(f_vector, points)
      
      # ? Should be 2D array (1, 3) for vector
      assert values_vector.shape == (1, 3), f"Vector should be (1,3), got {values_vector.shape}"
      print(f"? Vector function: shape={values_vector.shape}, value={values_vector[0]}")
      
      # Test 3: Single point (tests np.atleast_2d)
      single_point = np.array([[5.0, 5.0, 5.0]])
      values_single = solver.evaluate_at_points(f_vector, single_point)
      
      assert values_single.shape == (1, 3), f"Single point should be (1,3), got {values_single.shape}"
      print(f"? Single point: shape={values_single.shape}")

# This allows running the file directly: python test_base.py
if __name__ == '__main__':
    pytest.main([__file__, '-v'])