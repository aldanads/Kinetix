# src/solvers/heat.py
"""
Heat equation solver for thermal calculations in kMC simulations.

Implements steady-state heat conduction with thermal relaxation (capacitor model):
    - Steady-state: -grad(grad(T)) = Q
    - Thermal relaxation: T(t+dt) = T_steady + (T(t) - T_steady)*exp(-dt/t)
"""

import numpy as np
import ufl
from pathlib import Path

from dolfinx import fem, mesh, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import petsc4py.PETSc as PETSc
from mpi4py import MPI
from kinetix.solvers.base import FEMSolverBase


class HeatSolver(FEMSolverBase):
  """
  Heat equation solver for temperature field with thermal relaxation.
    
  Solves steady-state: -grad(grad(T)) = Q
    
  Thermal relaxation (capacitor model):
    T(t+dt) = T_steady + (T(t) - T_steady)*exp(-dt/tau)
    
  where tau = rho*c_p*L^2/kappa is the thermal relaxation time constant.
  """
    
  def __init__(
    self,
    heat_parameters,
    grid_crystal=None,
    mpi_ctx=None,
    **kwargs
  ):
    """
    Initialize Heat solver.
        
    Parameters:
    -----------
    heat_parameters : dict
      Heat-specific parameters:
      - mesh_file : str
      - thermal_conductivity : float (W/m/K)
      - density : float (kg/m^3)
      - specific_heat : float (J/kg/K)
      - ambient_temperature : float (K)
      - use_thermal_inertia : bool (default: True)
      - tau_thermal : float (s) - relaxation time constant
      - compute_tau_from_properties : bool (default: False)
      - characteristic_length : float (m) - for tau calculation
    grid_crystal : optional
      Crystal structure for mesh generation
    mpi_ctx : MPIContext, optional
      MPI context (uses singleton if None)
    **kwargs : dict
      Additional parameters:
      - mesh_size : float (angstroms)
      - path_results : str
    """
    # === Extract heat-specific parameters ===
    self.kappa_default = heat_parameters.get('thermal_conductivity').get('kappa_dielectric',2.0)  # W/m/K
    self.kappa_metal = heat_parameters.get('thermal_conductivity').get('kappa_metal') # W/m/K
    self.rho_default = heat_parameters.get('density', 9700.0)  # kg/m³
    self.cp_default = heat_parameters.get('specific_heat', 450.0)  # J/kg/K
    self.T_ambient = heat_parameters.get('ambient_temperature', 300.0)  # K
        
    # Thermal relaxation (capacitor model)
    self.use_thermal_inertia = heat_parameters.get('use_thermal_inertia', True)
        
    # Option A: Specify relaxation time directly
    self.tau_thermal = heat_parameters.get('tau_thermal', None)  # 1 ps default
        
    # Option B: Compute from material properties and geometry
    if self.tau_thermal is None:
      L = heat_parameters.get('characteristic_length', 50e-10)  # m (default: 50 Å)
      self.tau_thermal = (self.rho_default * self.cp_default * L**2) / self.kappa_default
      if mpi_ctx is None or mpi_ctx.rank == 0:
        print(f"Computed thermal relaxation time: t = {self.tau_thermal:.2e} s = {self.tau_thermal*1e12:.2f} ps")
        
        
    # === Initialize base class ===
    super().__init__(
      heat_parameters,
      grid_crystal=grid_crystal,
      mpi_ctx=mpi_ctx,
      **kwargs
    )
        
    # === Heat-specific initialization ===
    self._setup_thermal_spaces()
    self._setup_thermal_forms()
    
    self._setup_time_series_output(
      output_folder='Temperature_results',
      base_filename='Temperature'
    )
        
    # === Initialize temperature field ===
    self.T_current = fem.Function(self.V, dtype=np.float64)
    self.T_current.x.array[:] = self.T_ambient
    self.T_current.name = "Temperature"
        
    # Cache for steady-state solution
    self.T_steady_cache = None
        
    # === Track state ===
    self._bcs_changed = True
    self.heat_source = None
    
  def _setup_thermal_spaces(self):
    """Setup thermal-specific function spaces and functions."""
    # DG0 space for material properties and heat sources
    if not hasattr(self, 'W'):
      self.W = fem.functionspace(self.domain, ("DG", 0))
        
    # Pre-allocate thermal conductivity field
    self.kappa = fem.Function(self.W, dtype=np.float64)
    self.kappa.x.array[:] = self.kappa_default
        
    # Pre-allocate heat source field
    self.Q = fem.Function(self.W, dtype=np.float64)
    self.Q.x.array[:] = 0.0
        
    # Temperature solution reuses base class self.uh
    self.uh.name = "Temperature"
    
  def _setup_thermal_forms(self):
    """Setup bilinear and linear forms for steady-state heat equation."""
    angstrom_to_m = 1e-10
    angstrom_to_m3 = 1e-30
        
    # Bilinear form: 
    self.a_form = fem.form(
      ufl.inner(self.kappa * ufl.grad(self.u_trial), ufl.grad(self.v_test))
      * angstrom_to_m * ufl.dx
    )
        
    # Linear form: int(Q·v dx)
    self.L_form = fem.form(self.Q * self.v_test * angstrom_to_m3 * ufl.dx)
        
    # Setup matrix in base class
    self._setup_matrix(self.a_form)
    
    
  # ======================================================================
  # Material Properties
  # ======================================================================
  def update_thermal_properties(self, poisson_solver):
    """
    Update thermal conductivity field based on cluster positions.
    
    Metal clusters -> high thermal conductivity (heat flows easily)
    Oxide regions -> low thermal conductivity (thermal insulator)
    
    Parameters:
    -----------
    clusters : dict, optional
        Cluster objects with atoms_positions attribute.
        If None, entire domain gets oxide thermal conductivity.
    """
    # Oxide thermal conductivity everywhere
    self.kappa.x.array[:] = self.kappa_default
    
    if hasattr(poisson_solver, 'metal_cells') and len(poisson_solver.metal_cells) > 0:
      self.kappa.x.array[poisson_solver.metal_cells] = self.kappa_metal 
    
    if hasattr(poisson_solver, 'interface_cells_top') and len(poisson_solver.interface_cells_top) > 0:
      self.kappa.x.array[poisson_solver.interface_cells_top] = self.kappa_metal   
      
    if hasattr(poisson_solver, 'intercace_cells_bottom') and len(poisson_solver.intercace_cells_bottom) > 0:
      self.kappa.x.array[poisson_solver.intercace_cells_bottom] = self.kappa_metal    
      
    self._bcs_changed = True
      
  def set_joule_heating(self, poisson_solver):
    """
    Set Joule heating source: Q = sigma|E|^2.
        
    Parameters:
    -----------
    poisson_solver : PoissonSolver
      Poisson solver with conductivity field
    """
    # Get conductivity from Poisson solver
    sigma = poisson_solver.sigma
    
    # === Ensure E-field is computed  (should already be computed in Poisson solver)
    if not hasattr(poisson_solver, 'E_field') or poisson_solver.E_field is None:
      poisson_solver._project_electric_field(poisson_solver.uh) # (V/angstroms)
    
    # === Compute E-field at cell midpoints from potential gradient ===
    # Evaluate |E|^2 at midpoints 
    # Note: self.cell_midpoints has shape (N_cells, 3)
    E_values = self.evaluate_at_points(poisson_solver.E_field, self.cell_midpoints)
    
    E_values = E_values * 1e10
    
    E_squared = np.sum(E_values ** 2, axis=1)
      
    # Joule heating: Q = s|E|^2 [W/m^3]
    Q_joule = sigma.x.array * E_squared
    
    # Set heat source (only local cells)
    local_size = self.W.dofmap.index_map.size_local
    with self.Q.x.petsc_vec.localForm() as local_Q:
      local_Q.array[:] = Q_joule[:local_size]   
    
  # ======================================================================
  # Boundary Conditions
  # ======================================================================
  
  def set_boundary_conditions(self, top_value=0.0, bottom_value=0.0):
    """
    Set boundary conditions for heat equation.
    
    Parameters:
    -----------
    top_value : float
      Temperature at top electrode (K)
    bottom_value : float
      Temperature at bottom electrode (K)
    """
    all_boundary_conditions = []
    
    # === Dirichlet BCs (fixed temperature) ===
    
    # === Top and bottom electrode BCs ===
    def top_boundary(x):
      return np.isclose(x[2], self.z_max, atol=1e-8)
    
    def bottom_boundary(x):
      return np.isclose(x[2], self.z_min, atol=1e-8)
      
    # Locate boundary facets
    boundary_facets_top = mesh.locate_entities_boundary(
      self.domain, self.fdim, top_boundary
    )
    boundary_facets_bottom = mesh.locate_entities_boundary(
      self.domain, self.fdim, bottom_boundary
    )
    
    u_top = fem.Function(self.V)
    u_top.interpolate(lambda x: np.full_like(x[0], top_value))
      
    u_bottom = fem.Function(self.V)
    u_bottom.interpolate(lambda x: np.full_like(x[0], bottom_value))        
    
    # Obtain the degree of freedom (DOFs): the nodes
    boundary_dofs_top = fem.locate_dofs_topological(
      self.V, self.fdim, boundary_facets_top
    )
    boundary_dofs_bottom = fem.locate_dofs_topological(
      self.V, self.fdim, boundary_facets_bottom
    )

    # Apply Dirichlet boundary conditions
    bc_top = fem.dirichletbc(u_top, boundary_dofs_top)
    bc_bottom = fem.dirichletbc(u_bottom, boundary_dofs_bottom)
    
    all_boundary_conditions.extend([bc_top, bc_bottom]) # Add electrode BCs
    
    self.bcs = all_boundary_conditions
    self._bcs_changed = True

  # ======================================================================
  # Solve Methods
  # ======================================================================    
  
  def _solve(self, poisson_solver=None):
    """
    Solve steady-state heat equation: -grad(grad(T)) = Q
    
    If poisson_solver is provided, automatically computes Joule heating.
    If not, uses existing Q field (e.g., for testing or external heat sources).
    
    Parameters:
    -----------
    poisson_solver : PoissonSolver, optional
        Poisson solver with conductivity field and cached E-field.
        If provided, computes Q = sigma|E|^2 automatically.    
    
    Returns:
    --------
    T_sol : Function
      Temperature solution
    """
    # === Compute Joule heating if Poisson solver provided ===
    if poisson_solver is not None:
      # 1. Update thermal conductivity (reuses Poisson's cell indices)
      self.update_thermal_properties(poisson_solver)
      # 2. Compute Joule heating
      self.set_joule_heating(poisson_solver)
    
    # === Reasemble matrix if BCs changed ===
    if self._bcs_changed:
      self.A.zeroEntries()
      assemble_matrix(self.A, self.a_form, bcs=self.bcs)
      self.A.assemble()
      self.ksp.setOperators(self.A)
      self._bcs_changed = False
      
    # === Assemble RHS ===
    with self.b.localForm() as loc_b:
      loc_b.set(0)
    assemble_vector(self.b, self.L_form)
    
    fem.apply_lifting(self.b, [self.a_form], [self.bcs])
    self.b.ghostUpdate(
      addv=PETSc.InsertMode.ADD_VALUES,
      mode=PETSc.ScatterMode.REVERSE
    )
    fem.set_bc(self.b, self.bcs)
    
    # === Solve ===
    self.uh.x.array[:] = self.T_ambient # Initial guess
    fem.set_bc(self.uh.x.array, self.bcs)
    self.uh.x.scatter_forward()
    self.ksp.setInitialGuessNonzero(False) 
    
    self.ksp.solve(self.b, self.uh.x.petsc_vec)
    self.uh.x.scatter_forward()
    
    return self.uh
  
  def update_temperature(self, dt,poisson_solver=None, recompute_steady=True):
    """
    Update temperature using exponential relaxation toward steady-state.
          
    Thermal "capacitor" model:
      dT/dt = (T_steady - T) / t
        T(t+dt) = T_steady + (T(t) - T_steady) * exp(-dt/tau)
          
    Parameters:
    -----------
    dt : float
      Time step [s] (kMC timestep)
    recompute_steady : bool, optional
      Whether to recompute steady-state solution (default: True)
          
    Returns:
    --------
    T_current : Function
      Updated temperature field
    """
    # === Recompute steady state if needed ===
    if recompute_steady or self.T_steady_cache is None:
      self.T_steady_cache = self._solve(poisson_solver)
      self.T_steady_cache.name = "Temperature_steadyState"
    
    T_steady = self.T_steady_cache
    
    # === Exponential relaxation (capacitor model) ===
    if self.use_thermal_inertia and dt > 0:
      #T_new = T_steady + (T_old - T_steady) * exp(-dt/tau)
      alpha = np.exp(-dt / self.tau_thermal)
      self.T_current.x.array[:] = (
        T_steady.x.array + 
        alpha * (self.T_current.x.array - T_steady.x.array)
      )
    else:
      # Instant equilibration (steady-state only)
      self.T_current.x.array[:] = T_steady.x.array[:]
      
    # Ensure BCs are satisfied (Dirichlet nodes should match exactly)
    if self.bcs:
      fem.set_bc(self.T_current.x.array, self.bcs)
      self.T_current.x.scatter_forward()
      
    return self.T_current
    
  # ======================================================================
  # Utility Methods
  # ======================================================================  
  def get_thermal_time_constant(self):
    """Return thermal relaxation time constant."""
    return self.tau_thermal
    
  def get_maximum_temperature(self):
    """Get maximum temperature in domain (across all MPI ranks)."""
    local_max = np.max(self.T_current.x.array)
    global_max = self.mpi_ctx.allreduce(local_max, op=MPI.MAX)
    return global_max
    
  def get_minimum_temperature(self):
    """Get maximum temperature in domain (across all MPI ranks)."""
    local_min = np.min(self.T_current.x.array)
    global_min = self.mpi_ctx.allreduce(local_min, op=MPI.MIN)
    return global_min
    
  def get_average_temperature(self):
    """Get volume-averaged temperature."""
    
    # === Step 1: Integrate T over local domain (this MPI rank) ===
    local_T_int = fem.assemble_scalar(fem.form(self.T_current * ufl.dx))
    
    # === Step 2: Integrate 1 over local domain (= local volume) ===
    dx = ufl.Measure("dx", domain=self.domain)
    local_V_int = fem.assemble_scalar(fem.form(1.0 * dx))
    
    # === Step 3: Sum across all MPI ranks ===
    global_T_int = self.mpi_ctx.allreduce(local_T_int, op=MPI.SUM)
    global_V_int = self.mpi_ctx.allreduce(local_V_int, op=MPI.SUM)
    
    # === Step 4: Compute volume-average ===
    return global_T_int / global_V_int if global_V_int > 0 else self.T_ambient
    
  def reset_temperature(self, T_value=None):
    """
    Reset temperature field to uniform value.
        
    Parameters:
    -----------
    T_value : float, optional
      Temperature to set (uses ambient if None)
    """
    if T_value is None:
      T_value = self.T_ambient
      
    self.T_current.x.array[:] = T_value
    self.T_steady_cache = None
    
  def evaluate_temperature_at_points(self, points):
    """
    Evaluate temperature field T at given points.
    
    Implements caching to avoid recomputing for same points.
    
    Parameters:
    -----------
    points : array-like
        Points [N, 3] where to evaluate temperature (angstroms)
    
    Returns:
    --------
    T_values : dict
        Dictionary {(x,y,z): T} in Kelvin
    """
    # Initialize cache on first call
    if not hasattr(self, '_T_field_cache'):
      self._T_field_cache = {}
      
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
      points_array = points_array.reshape(1, -1)
      
    # === Separate cached and new points ===
    cached_points = []
    new_points = []
    
    for point in points_array:
      point_key = tuple(np.round(point, 6))
      if point_key in self._T_field_cache:
        cached_points.append(point_key)
      else:
        new_points.append(point)
        
    # === Initialize result dictionary ===
    T_values_global = {}
    
    # === Add cached value ===
    for point_key in cached_points:
      T_values_global[point_key] = self._T_field_cache[point_key]
      
    # === Compute new values ===
    if new_points:
      new_points_array = np.array(new_points, dtype=np.float64)
      
      # Evaluate T_current at new points using base class method
      T_values_array = self.evaluate_at_points(
        self.T_current,
        new_points_array,
      )
      
      # === MPI reduction: sum across ranks ===
      T_values_array = self.mpi_ctx.allreduce(
        T_values_array,
        op=MPI.SUM
      )
      
      for i, point in enumerate(new_points):
        point_key = tuple(np.round(point, 6))
        T_values_global[point_key] = T_values_array[i]
        self._T_field_cache[point_key] = T_values_array[i]
        
    return T_values_global    
        
    
  def save_temperature(self, time_value, timestep):
    """
    Save temperature field (uses internal self.T_current).
    
    Parameters:
    -----------
    time_value : float
        Physical time for metadata
    timestep : int
        Timestep index for filename numbering
    
    Returns:
    --------
    saved_files : list
        List of paths to saved files
    """
    return self.save_solution(
      self.T_current,
      filename=self.output_base,
      time_value=time_value,
      timestep=timestep,
      save_csv=self.save_csv
    )
