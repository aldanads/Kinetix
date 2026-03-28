import pytest
import numpy as np
from pathlib import Path

# === Constants ===
NUMERICAL_TOL = 1e-10  # FEM numerical tolerance for assertions
PHYSICS_TOL = 0.01 # 1% tolerance for physics validation

from solvers.heat import HeatSolver
from solvers.poisson import PoissonSolver
from utils.mpi_context import MPIContext


@pytest.fixture
def mpi_ctx():
  """ Fixture: MPI context """
  return MPIContext.get_instance()
    
@pytest.fixture
def heat_solver(mpi_ctx):
  """ Fixture: HeatSolver instance """
  params ={
    'mesh_file': 'test_mock_mesh.msh',
    'thermal_conductivity': 10.0, # W/m/K
    'density': 5000.0, # kg/m^3
    'specific_heat': 500.0, # J/kg/K
    'ambient_temperature': 300.0, # K
    'use_thermal_inertia': True,
    'tau_thermal': 1e-12, # 1 ps
    'defects_config': {}
  }
    
  grid_crystal = None
    
  return HeatSolver(
    params,
    grid_crystal=grid_crystal,
    mpi_ctx=mpi_ctx,
    path_results=Path('./test_output'),
    mesh_size=2.0,
    fine_mesh_size=0.5
  )
    
@pytest.fixture
def poisson_solver(mpi_ctx):
  """ Fixture: HeatSolver instance """
  params = {
    'mesh_file': 'test_mock_mesh.msh',
    'epsilon_r': 25.0,
    'metal_valence': 1,
    'd_metal_O': 2.4,
    'chem_env_symmetry': 'Tetrahedron',
    'active_dipoles': 1,
    'conductivity_CF': 1e6,
    'conductivity_dielectric': 1e-12,
    'defects_config': {}  
  }
    
  return PoissonSolver(
    params,
    grid_crystal=None,
    mpi_ctx=mpi_ctx,
    path_results=Path('./test_output')
  )

class TestHeatSolver:
  """ Tests for HeatSolver """  
  # ============================================================================
  # TEST CLASS 1: Basic HeatSolver Functionality
  # ============================================================================
    
  def test_initialization(self, heat_solver):
    """Test HeatSolver initializes correctly."""
    assert hasattr(heat_solver, 'kappa'), "Thermal conductivity field should exist"
    assert hasattr(heat_solver, 'rho_cp'), "Heat capacity field should exist"
    assert hasattr(heat_solver, 'Q'), "Heat source field should exist"
    assert hasattr(heat_solver, 'T_current'), "Temperature field should exist"
    assert hasattr(heat_solver, 'tau_thermal'), "Thermal relaxation time should exist"
    assert hasattr(heat_solver, 'T_steady_cache'), "Steady-state cache should exist"
    print("=============================")
    print("HeatSolver initialized correctly")
  
  def test_thermal_time_constant(self, heat_solver):
    """Test thermal time constant is set correctly """
    assert heat_solver.tau_thermal > 0, "tau_thermal should be positive"
    
    print("=============================")
    print(f"Thermal time constant: tau = {heat_solver.tau_thermal*1e12:.2f} ps")
    
  def test_set_boundary_conditions(self, heat_solver):
    """Test boundary condition setup."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    
    assert len(heat_solver.bcs) >= 2, "Should have at least top and bottom BCs"
    assert heat_solver._bcs_changed == True, "BCs should be marked as changed"
    
    print("=============================")
    print("Boundary conditions set")
    
  def test_reset_temperature(self, heat_solver):
    """Test temperature reset."""
    # Set to non-ambient
    heat_solver.T_current.x.array[:] = 350.0
        
    # Reset
    heat_solver.reset_temperature(300.0)
        
    assert np.allclose(heat_solver.T_current.x.array, 300.0), "Reset failed"
    assert heat_solver.T_steady_cache is None, "Cache should be cleared"
    
    print("=============================")
    print("Temperature reset works")
    
  def test_get_thermal_time_constant(self, heat_solver):
    """Test thermal time constant getter."""
    tau = heat_solver.get_thermal_time_constant()
        
    assert tau == heat_solver.tau_thermal, "Should return tau_thermal"
    assert tau > 0, "tau should be positive"
    
    print("=============================")
    print(f"Thermal time constant getter: t = {tau*1e12:.2f} ps")
    
  def test_get_maximum_temperature(self, heat_solver):
    """Test maximum temperature calculation."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    
    heat_solver.update_temperature(dt=0.0, poisson_solver=None, recompute_steady=True)
        
    T_max = heat_solver.get_maximum_temperature()
    
    assert np.isclose(T_max, 350.0), f"T_max ({T_max:.6f} K) is not close to maximum BC (350.0 K) "
    
    print("=============================")
    print(f"Maximum temperature: {T_max:.2f} K")
    
  def test_get_average_temperature(self, heat_solver):
    """Test average temperature calculation."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    heat_solver.update_temperature(dt=0.0, poisson_solver=None, recompute_steady=True)
        
    T_avg = heat_solver.get_average_temperature()
  
    assert 300.0 - NUMERICAL_TOL <= T_avg <= 350.0 + NUMERICAL_TOL, f"T_avg out of bounds: {T_avg}"
    
    print("=============================")
    print(f"Average temperature: {T_avg:.2f} K")
    
  # ============================================================================
  # TEST CLASS 2: Steady-State Physics Validation
  # ============================================================================
    
class TestSteadyStatePhysics:
  """Tests for steady-state heat equation physics validation."""
        
  def test_solve_steady_no_heating(self,heat_solver):
    """Test steady-state solve without Joule heating.
        
    Validates that temperature profile is linear (analytical solution for Q=0).
    """
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    
    T_sol = heat_solver._solve(poisson_solver=None)
    
    assert T_sol is not None
    assert len(T_sol.x.array) > 0
    assert not np.any(np.isnan(T_sol.x.array)), "Solution should not contain NaN"
    
    # Check 1: Min/max within BC bounds
    T_min = np.min(T_sol.x.array)
    T_max = np.max(T_sol.x.array)
    assert 300.0 - NUMERICAL_TOL <= T_min <= T_max <= 350.0 + NUMERICAL_TOL, f"Temperature out of bounds: [{T_min}, {T_max}]"
    
    # Check 2: Linear temperature profile (physics validation)
    # Get DOF coordinates
    dof_coords = heat_solver.V.tabulate_dof_coordinates()
    T_values = T_sol.x.array
    
    # Analytical solution: T(z) = T_bottom + (T_top - T_bottom) * (z - z_min) / (z_max - z_min)
    z_min = heat_solver.z_min
    z_max = heat_solver.z_max
    T_bottom = 300.0
    T_top = 350.0
    
    T_analytical = T_bottom + (T_top - T_bottom) * (dof_coords[:, 2] - z_min) / (z_max - z_min)
    
    # Compare numerical vs. analytical (allow 1% error for FEM discretization)
    error = np.abs(T_values - T_analytical)
    max_error = np.max(error)
    relative_error = max_error / (T_top - T_bottom)
    
    # FEM with linear elements should be very close to analytical for this simple case
    assert relative_error < 0.01, \
        f"Temperature profile deviates from linear: max error = {max_error:.4f} K ({relative_error*100:.2f}%)"
    
    
    print("=============================")
    # === Print temperature profile at representative z-positions ===
    self._print_temperature_profile(heat_solver, T_sol, n_bins=10)
    print("Steady-state solve (no heating):") 
    print(f"T in [{T_min:.2f}, {T_max:.2f}] K")
    print(f"  Max deviation from linear: {max_error:.4f} K ({relative_error*100:.2f}%)")
    print(f"  Temperature profile is linear (physics validated)")
    
  def test_uniform_temperature_equal_bcs(self, heat_solver):
    """Test that T is uniform when T_top = T_bottom."""
    heat_solver.set_boundary_conditions(top_value=320.0, bottom_value=320.0)
    heat_solver.update_temperature(dt=0.0, poisson_solver=None, recompute_steady=True)
        
    # All values should be ~320 K (uniform)
    T_std = np.std(heat_solver.T_current.x.array)
    assert T_std < 0.1, f"Temperature should be uniform: std = {T_std:.4f} K"
  
  def test_temperature_profile_by_z(self, heat_solver):
    """Test temperature profile at representative z-positions."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    T_sol = heat_solver._solve(poisson_solver=None)
        
    # Get binned profile (for debugging/visualization)
    profile_data = self._get_temperature_profile(heat_solver, T_sol, n_bins=10)
        
    # Verify monotonic increase (temperature should increase from bottom to top)
    T_avg = profile_data['T_avg']
    valid_bins = ~np.isnan(T_avg)
        
    if np.sum(valid_bins) > 1:
      T_valid = T_avg[valid_bins]
      assert np.all(np.diff(T_valid) >= -PHYSICS_TOL * 50), \
          "Temperature should increase monotonically from bottom to top"
    
  def _get_temperature_profile(self, heat_solver, T_sol, n_bins=10):
    """Helper: Get temperature profile at representative z-positions."""
    dof_coords = heat_solver.V.tabulate_dof_coordinates()
    T_values = T_sol.x.array
        
    z_min = heat_solver.z_min
    z_max = heat_solver.z_max
    z_range = z_max - z_min
        
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
    bin_avg_T = []
    bin_std_T = []
    bin_count = []
        
    for i in range(n_bins):
      mask = (dof_coords[:, 2] >= bin_edges[i]) & (dof_coords[:, 2] < bin_edges[i+1])
            
      if np.any(mask):
        T_in_bin = T_values[mask]
        bin_avg_T.append(np.mean(T_in_bin))
        bin_std_T.append(np.std(T_in_bin))
        bin_count.append(np.sum(mask))
      else:
        bin_avg_T.append(np.nan)
        bin_std_T.append(np.nan)
        bin_count.append(0)
        
    return {
      'z': bin_centers,
      'z_norm': (bin_centers - z_min) / z_range,
      'T_avg': np.array(bin_avg_T),
      'T_std': np.array(bin_std_T),
      'count': np.array(bin_count)
    }
    
  # ============================================================================
  # TEST CLASS 3: Thermal Relaxation Physics
  # ============================================================================

class TestThermalRelaxation:
  """Tests for thermal relaxation (capacitor model) physics."""
    
  def test_update_temperature_with_relaxation(self,heat_solver):
    """Test temperature update with thermal relaxation."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    
    # Reset to ambient
    heat_solver.reset_temperature (300.0)
    
    # Compute steady-state
    T_steady = heat_solver._solve(poisson_solver=None)
    heat_solver.T_steady_cache = T_steady
    
    # Get DOF coordinates to identify interior nodes
    dof_coords = heat_solver.V.tabulate_dof_coordinates()
    interior_mask = (
      (dof_coords[:, 2] > heat_solver.z_min + 1e-6) & 
      (dof_coords[:, 2] < heat_solver.z_max - 1e-6)
    )
    
    if not np.any(interior_mask):
      pytest.skip("No interior nodes for this mesh")
        
    # Test 1: Small dt (should barely relax)
    dt_small = 1e-13
    alpha = np.exp(-dt_small / heat_solver.tau_thermal)
    print(f"\n DEBUG: dt={dt_small*1e12:.1f} ps, alpha={alpha:.6f}")
    T1 = heat_solver.update_temperature(dt=dt_small, poisson_solver=None, recompute_steady=True)
    
    diff1 = np.linalg.norm(T1.x.array[interior_mask] - T_steady.x.array[interior_mask])
    
    # Test 2: Large dt (should approach steady-state)
    # Update with large dt (should approach steady-state)
    heat_solver.reset_temperature(300.0)  # Reset for fair comparison
    dt_large = 1e-12 # 1 ns >> tau
    alpha = np.exp(-dt_large / heat_solver.tau_thermal)
    print(f"\n DEBUG: dt={dt_large*1e12:.1f} ps, alpha={alpha:.6f}")
    T2 = heat_solver.update_temperature(dt=dt_large, poisson_solver=None, recompute_steady=False)
    
    diff2 = np.linalg.norm(T2.x.array[interior_mask] - T_steady.x.array[interior_mask])
    
    
    # T2 should be MUCH closer to steady-state than T1
    assert diff2 < diff1 * 0.5, \
            f"Large dt should be much closer: diff1={diff1:.6f}, diff2={diff2:.6f}"
    
    print("=============================")
    print(f"Thermal relaxation works")

  def test_update_temperature_instant_equilibration(self, heat_solver):
    """Test instant equilibration (use_thermal_inertia=False)."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
    heat_solver.use_thermal_inertia = False
    
    heat_solver.reset_temperature (300.0)
    
    # Should instantly reach steady-state
    T = heat_solver.update_temperature(dt=1e-12, poisson_solver=None, recompute_steady = True)
    
    # Should match steady-state exactly
    diff = np.linalg.norm(T.x.array - heat_solver.T_steady_cache.x.array)
    assert diff < 1e-10, f"Should reach steady-state instantly: diff={diff}"
    
    print("=============================")
    print("Instant equilibration works")
    
  def test_exponential_relaxation_time_constant(self, heat_solver):
    """Test that relaxation follows exponential decay with correct time constant."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=300.0)
        
    # Compute steady-state
    T_steady = heat_solver._solve(poisson_solver=None)
    heat_solver.T_steady_cache = T_steady
        
    # Reset to ambient
    heat_solver.reset_temperature(300.0)
        
    # Test with dt = t (should relax to ~63% of steady-state)
    dt = heat_solver.tau_thermal
    T = heat_solver.update_temperature(dt=dt, poisson_solver=None, recompute_steady=False)
        
    # Check interior points (not boundary nodes)
    dof_coords = heat_solver.V.tabulate_dof_coordinates()
    interior_mask = (
      (dof_coords[:, 2] > heat_solver.z_min + 1e-6) & 
      (dof_coords[:, 2] < heat_solver.z_max - 1e-6)
    )
        
    if np.any(interior_mask):
      T_interior = T.x.array[interior_mask]
      T_steady_interior = T_steady.x.array[interior_mask]
      T_initial = 300.0
            
      # Fraction of relaxation achieved
      fraction = (T_interior - T_initial) / (T_steady_interior - T_initial + 1e-10)
      expected_fraction = 1 - np.exp(-1)  # ~0.632
            
      avg_fraction = np.mean(fraction)
            
      # Should be close to 63% (allow 20% error for FEM discretization)
      assert 0.4 < avg_fraction < 0.8, \
        f"Relaxation fraction {avg_fraction:.2f} not close to expected {expected_fraction:.2f}"
    
  # ============================================================================
  # TEST CLASS 4: Analytical Solutions Validation
  # ============================================================================

class TestAnalyticalSolutions:
  """Tests comparing numerical solutions to analytical benchmarks."""
    
  def test_linear_profile_1d(self, heat_solver):
    """Test 1D linear temperature profile (no internal heat generation)."""
    heat_solver.set_boundary_conditions(top_value=400.0, bottom_value=300.0)
    T_sol = heat_solver._solve(poisson_solver=None)
        
    dof_coords = heat_solver.V.tabulate_dof_coordinates()
    T_values = T_sol.x.array
        
    # Analytical: T(z) = T_bottom + (T_top - T_bottom) * (z - z_min) / L
    T_analytical = 300.0 + 100.0 * (dof_coords[:, 2] - heat_solver.z_min) / \
                       (heat_solver.z_max - heat_solver.z_min)
                         
    error = np.abs(T_values - T_analytical)
    max_error = np.max(error)
    relative_error = max_error / 100.0
        
    assert relative_error < PHYSICS_TOL, \
            f"1D linear profile error: {relative_error*100:.2f}%"
    
  def test_uniform_temperature(self, heat_solver):
    """Test uniform temperature (equal BCs, no heating)."""
    heat_solver.set_boundary_conditions(top_value=350.0, bottom_value=350.0)
    T_sol = heat_solver._solve(poisson_solver=None)
        
    # All values should be 350 K
    T_std = np.std(T_sol.x.array)
    T_mean = np.mean(T_sol.x.array)
        
    assert T_std < 0.1, f"Temperature should be uniform: std = {T_std:.4f} K"
    assert np.isclose(T_mean, 350.0, atol=NUMERICAL_TOL), \
            f"Mean temperature should be 350 K: {T_mean:.6f} K"
                       
  # ============================================================================
  # TEST CLASS 5: Electro-Thermal Coupling
  # ============================================================================

class TestElectroThermalCoupling:
  """Tests for coupling between Poisson and Heat solvers."""
  
  def test_set_joule_heating(self, poisson_solver, heat_solver):
    """Test setting Joule heating from Poisson solution."""
    # Solve Poisson with conductivity
    poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
    poisson_solver.conductivity_in_system([])
    uh = poisson_solver.solve([], [])
    poisson_solver._project_electric_field(uh)
        
    # Set Joule heating
    heat_solver.set_joule_heating(poisson_solver)
        
    # Check that heat source is non-zero where conductivity is high
    assert np.any(heat_solver.Q.x.array > 0), "Joule heating not set correctly"
    
  def test_solve_with_joule_heating(self, poisson_solver, heat_solver):
    """Test solve with automatic Joule heating."""
    # Solve Poisson
    poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
    poisson_solver.conductivity_in_system([])
    uh = poisson_solver.solve([], [])
    poisson_solver._project_electric_field(uh)
        
    # Set thermal BCs
    heat_solver.set_boundary_conditions(top_value=300.0, bottom_value=300.0)
        
    # Solve heat (automatically computes Joule heating)
    T_sol = heat_solver._solve(poisson_solver)
        
    # Check results
    assert T_sol is not None
    assert not np.any(np.isnan(T_sol.x.array))
        
    # Sanity check: T should never be below ambient
    T_min = np.min(T_sol.x.array)
    T_max = np.max(T_sol.x.array)
    
    assert T_min >= 300.0 - NUMERICAL_TOL, f"Temperature below ambient: {T_min}"
    assert T_max >= 300.0, f"Temperature should increase with heating: T_max = {T_max:.2f} K"
    
    print(f"_solve() with Joule heating: T ? [{T_min:.2f}, {T_max:.2f}] K")
    
  def test_update_temperature_with_poisson(self, poisson_solver, heat_solver):
    """Test update_temperature() with Poisson solver (full workflow)."""
    # Solve Poisson
    poisson_solver.set_boundary_conditions(top_value=1.0, bottom_value=0.0)
    poisson_solver.conductivity_in_system([])
    uh = poisson_solver.solve([], [])
    poisson_solver._project_electric_field(uh)
        
    # Set thermal BCs
    heat_solver.set_boundary_conditions(top_value=300.0, bottom_value=300.0)
        
    # Update temperature (automatically computes Joule heating + steady-state)
    T_current = heat_solver.update_temperature(
      dt=1e-12,
      poisson_solver=poisson_solver,
      recompute_steady=True
    )
        
    # Check that temperature increased
    T_max = heat_solver.get_maximum_temperature()
    T_avg = heat_solver.get_average_temperature()
    
    assert T_max > 300.0, f"Joule heating should increase T: T_max = {T_max:.2f} K"
    assert T_avg > 300.0, f"Average T should increase: T_avg = {T_avg:.2f} K"
    
    print(f"update_temperature() workflow: T_max = {T_max:.2f} K, T_avg = {T_avg:.2f} K")
    