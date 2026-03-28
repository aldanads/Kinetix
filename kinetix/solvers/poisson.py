# /solvers/poisson.py
"""
Poisson solver for electrostatic calculations in kMC simulations.
"""
import sys
import numpy as np
import ufl
from scipy.constants import epsilon_0, e
from pathlib import Path

from dolfinx import fem, mesh, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import petsc4py.PETSc as PETSc

from kinetix.solvers.base import FEMSolverBase



class PoissonSolver(FEMSolverBase):
  """
  Poisson equation solver for electric potential.
    
  Solves: -grad*(e grad(f)) = rho/e0
    
  With optional conductivity for bridging clusters:
  -grad*(s grad(f)) = 0
  """
  
  def __init__(
    self,
    poisson_parameters,
    grid_crystal=None,
    mpi_ctx=None,
    **kwargs
  ):
    """
    Initialize Poisson solver.
          
    Parameters:
    -----------
    poisson_parameters : dict
      Poisson-specific parameters:
      - mesh_file : str
      - epsilon_r : float (relative permittivity)
      - conductivity_CF : float (S/m)
      - conductivity_dielectric : float (S/m)
      - metal_valence : float
      - d_metal_O : float (angstrom)
      - chem_env_symmetry : str
      - active_dipoles : float
      - defects_config : dict
    grid_crystal : optional
      Crystal structure for mesh generation
    mpi_ctx : MPIContext, optional
      MPI context (uses singleton if None)
    **kwargs : dict
      Additional parameters:
      - epsilon_gaussian_charge : float (angstrom)
      - activate_mesh_refinement : bool
      - fine_mesh_size : float (angstrom)
      - refinement_radius : float (angstrom)
      - path_results : str
    """  
    # === Extract Poisson-specific paramters ===
    self.epsilon_gc = kwargs.pop('epsilon_gaussian_charge', 0.8) # angstroms
    self.active_mesh_refinement = kwargs.pop('activate_mesh_refinement', True)
    self.fine_mesh_size = kwargs.pop('fine_mesh_size', 0.2) # angstroms
    self.refinement_radius = kwargs.pop('refinement_radius', 1.2) # angstroms
    self.defects_config = poisson_parameters.get('defects_config', {})
    self.epsilon_r = poisson_parameters.get('epsilon_r', 25.0)
    self.poisson_parameters = poisson_parameters
    
    # === Initialize base class ===
    super().__init__(
      poisson_parameters,
      grid_crystal=grid_crystal,
      mpi_ctx=mpi_ctx,
      **kwargs
    )
  
    # === Poisson-specific initialization ===
    self._setup_poisson_specific_spaces()
    self._setup_poisson_forms()
    self._calculate_dipole_moment()
    
    # === Track state ===
    self.use_conductivity = False
    self.metal_atoms = None
    self._bcs_changed = True
    self._field_cache = {}
    self.new_field_cache = True
  
  def _setup_poisson_specific_spaces(self):
    """Setup Poisson-specific function spaces and functions."""
    # DG0 space for charge density and conductivity
    if not hasattr(self,'W'):
      self.W = fem.functionspace(self.domain, ("DG", 0))
      
    # Pre-allocate charge density function
    self.rho = fem.Function(self.W, dtype=np.float64)
    
    # Pre-allocate conductivity function
    self.sigma = fem.Function(self.W, dtype=np.float64)
    
    # Pre-allocate electric field function (vector)
    self.E_field = fem.Function(self.V_vec)
    
    self.uh.name = "ElectricPotential"
    
  def _setup_poisson_forms(self):
    """Setup bilinear form for Poisson equation."""
    angstrom_to_m = 1e-10
    
    # Standard Poisson form: ?grad(u)*grad(v) dx
    self.a_form = fem.form(
      ufl.inner(ufl.grad(self.u_trial), ufl.grad(self.v_test))
      * angstrom_to_m * ufl.dx
    )
    
    # Setup matrix in base class
    self._setup_matrix(self.a_form)
    
  def _calculate_dipole_moment(self):
    """
    Padovani, A., Larcher, L., Pirrotta, O., Vandelli, L., & Bersuker, G. (2015). 
    Microscopic modeling of HfO x RRAM operations: From forming to switching. 
    IEEE Transactions on electron devices, 62(6), 1998-2006.
      
    McPherson, J. W., & Mogul, H. C. (1998). Underlying physics of the thermochemical 
    E model in describing low-field time-dependent dielectric       
    breakdown in SiO 2 thin films. Journal of Applied Physics, 84(3), 1513-1523.
           
    ************* Formula for dielectric moment: *******************
    McPherson, J., J. Y. Kim, A. Shanware, and H. Mogul. "Thermochemical description 
    of dielectric breakdown in high dielectric constant              
    materials." Applied Physics Letters 82, no. 13 (2003): 2121-2123.
    """
    # Dipole moment: Units (e*angstroms); 1D = Debye	? 0.2081943 e*angstrom (large dipole moment is around 11D)
    # Dipole moment: Units (enm)
    
    # Dipole moment geometry factors
    L = {
      'Tetrahedron': 1/3,
      'Octahedron': 1, 
      'Trigonal': np.sqrt(2/3), 
      'Cube': np.sqrt(1/3), 
      'Disheptahedral': np.sqrt(2/3), 
      'Cuboctahedral': np.sqrt(1/3)
    }
    
    metal_valence = self.poisson_parameters['metal_valence'] # Metal valence
    d_metal_O = self.poisson_parameters['d_metal_O'] #Units: angstrom
    chem_env_symmetry = self.poisson_parameters['chem_env_symmetry'] # Symmetry in the local environment (molecule)
    active_dipoles = self.poisson_parameters['active_dipoles']
    
    dipole_moment = (
      active_dipoles *
      (metal_valence / 2) *
      d_metal_O *
      L.get(chem_env_symmetry, 1.0) 
    )
    
    self.bond_polarization_factor = ((2 + self.epsilon_r) / 3) * dipole_moment
    
  # ======================================================================
  # Boundary Conditions
  # ======================================================================
  
  def set_boundary_conditions(self, top_value=0.0, bottom_value=0.0, clusters=None):
    """
    Set Dirichlet boundary conditions on electrodes and clusters.
        
    Parameters:
    -----------
    top_value : float
      Potential at top electrode (V)
    bottom_value : float
      Potential at bottom electrode (V)
    clusters : dict, optional
      Cluster objects for virtual electrode BCs
    """
    all_boundary_conditions = []
    
    # === Top and bottom electrode BCs ===
    def top_boundary(x):
      return np.isclose(x[2], self.z_max, atol=1e-8)
    
    def bottom_boundary(x):
      return np.isclose(x[2], self.z_min, atol=1e-8)
      
    # Find boundaries in domain where top_boundary returns True
    # - domain: finite element mesh
    # - facet dimension: typically tdim-1
    # - top_boundary: condition for selecting the facets
    # + return boundary_facets_top: NumPy array with indices of facets that satisfy the condition
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
        
    # === Cluster BCs (virtual electrodes) ===
    self.use_conductivity = False
    
    if clusters is not None:
      for cluster in clusters.values():
        touches_bottom = cluster.attached_layer.get('bottom_layer', False)
        touches_top = cluster.attached_layer.get('top_layer', False)
        
        if touches_bottom and touches_top:
          # Bridging cluster - use conductivity formulation
          self.use_conductivity = True
          self.metal_atoms = cluster.atoms_positions
        elif touches_bottom:
          # Connected to bottom electrode
          cluster_bcs = self._create_cluster_boundary_conditions(
            cluster.internal_atom_positions, bottom_value
          )
          all_boundary_conditions.extend(cluster_bcs)
        elif touches_top:
          # Connected to top electrode       
          cluster_bcs = self._create_cluster_boundary_conditions(
            cluster.internal_atom_positions, top_value
          ) 
          all_boundary_conditions.extend(cluster_bcs)
        
    self.bcs = all_boundary_conditions
    self._bcs_changed = True
        
  def _create_cluster_boundary_conditions(self,cluster_particle_positions, cluster_potential):
    """
    Create boundary conditions for clusters.
        
    Parameters:
    -----------
    cluster_particle_positions : array-like
      Coordinates of particles in cluster contact
    cluster_potential : float
      Potential value at cluster contact
        
    Returns:
    --------
    list : DirichletBC objects
    """  
    
    if cluster_particle_positions is None or len(cluster_particle_positions) == 0:
      return []
    
    cluster_boundary_conditions = []
    contact_radius = 2.0 # angstroms
    
    cluster_particle_positions = np.array(cluster_particle_positions, dtype=np.float64)
    
    # Find DOFs near cluster
    dofs_near_cluster = self._find_dofs_near_particles_vectorized(
      cluster_particle_positions, contact_radius
    )   
    
    if len(dofs_near_cluster) > 0:
      u_cluster = fem.Function(self.V)
      u_cluster.interpolate(lambda x: np.full_like(x[0], cluster_potential))
      bc_cluster = fem.dirichletbc(u_cluster, dofs_near_cluster)
      cluster_boundary_conditions.append(bc_cluster)
      
    return cluster_boundary_conditions 
    
  def _find_dofs_near_particles_vectorized(self, particle_positions, contact_radius):
    """
    Find DOFs within contact radius of particles (vectorized).
        
    Parameters:
    -----------
    particle_positions : np.ndarray
      Particle positions [N, 3]
    contact_radius : float
      Search radius in angstrom
        
    Returns:
    --------
    nearby_dofs : np.ndarray
      DOF indices within contact radius
    """
    # Vectorized distance calculation: (n_dofs, n_particles)
    dx = self.dof_coords[:, np.newaxis, 0] - particle_positions[np.newaxis, :, 0]
    dy = self.dof_coords[:, np.newaxis, 1] - particle_positions[np.newaxis, :, 1]  
    dz = self.dof_coords[:, np.newaxis, 2] - particle_positions[np.newaxis, :, 2]
      
    # Apply minimum image convention (periodic in x, y)
    dx = dx - self.Lx * np.round(dx / self.Lx)
    dy = dy - self.Ly * np.round(dy / self.Ly)
    # z is NOT periodic
    
    # Calculate all distances
    distances = np.sqrt(dx**2 + dy**2 + dz**2) # Shape: (n_dofs, n_particles)
        
    # Find DOFs within contact radius
    nearby_dofs = np.where(np.any(distances <= contact_radius, axis=1))[0]
      
    return nearby_dofs.astype(np.int32)
    
  # ======================================================================
  # Physics: Charge Density & Conductivity
  # ======================================================================
  
  def charge_density(self, charge_locations, charges, tolerance=3):
    """
    Create DG0 Function representing charge density with Gaussian smearing.
        
    Parameters:
    -----------
    charge_locations : np.ndarray
      Charge positions [N, 3] in angstrom
    charges : np.ndarray
      Charge values [N] in Coulombs
    tolerance : float, optional
      Tolerance in % for charge conservation (default: 3)
        
    Returns:
    --------
    rho : Function
      Charge density in DG0 space [C/m3]
    """
    # Reuse pre-allocated arrays
    cell_values = np.zeros(self.num_cells, dtype=np.float64)
    
    for x0, q, in zip(charge_locations, charges):
      # Vectorized Gaussian computation for all cells
      r_sq = np.sum((self.cell_midpoints - x0) ** 2, axis = 1)
      
      normalization = (2 * np.pi * self.epsilon_gc**2) ** (self.tdim / 2)
      gauss_values = (q / normalization) * np.exp(-r_sq / (2 * self.epsilon_gc ** 2))
        
      cell_values += gauss_values
      
      
    # Set values (only local cells)
    with self.rho.x.petsc_vec.localForm() as local_rho:
      local_rho.setArray(cell_values[:self.W.dofmap.index_map.size_local])
      
    # Validate charge conservation
    local_charge = fem.assemble_scalar(fem.form(self.rho * ufl.dx)) # Local total charge for this process
    total_charge = self.mpi_ctx.allreduce(local_charge, op=self.mpi_ctx.MPI.SUM)       
    expected_charge = sum(charges)
          
    charge_error = (
      100 * abs((total_charge - expected_charge) / expected_charge) 
      if abs(expected_charge) > 0 else 0.0
    )
        
    if charge_error > tolerance:
      if self.rank == 0:
        error_msg = (
          f"\nCHARGE CONSERVATION ERROR:\n"
          f"- Total charge: {total_charge:.4e} C\n"
          f"- Expected:     {expected_charge:.4e} C\n"
          f"- Error:        {charge_error:.2f}% of expected charge\n\n"
          f"SOLUTIONS:\n"
          f"1. Increase epsilon (standard deviation) to control how widely the charge is spread with the Gaussian distribution (current: {self.epsilon_gc:.2f}):\n"
          f"   - Larger epsilon: charge spreads out over more cells (reduces singularities but may lose accuracy)\n"
          f"   - Smaller epsilon: charge is more localized (may lead to numerical issues if the mesh is too coarse)\n"
          f"2. Use finer mesh resolution:\n"
          f"   - Smaller cells better resolve point charges (current: {self.mesh_size:.2f})\n"
        )     
        print(error_msg)
        sys.stdout.flush()
    
      # Synchronize to ensure rank 0 finishes printing, then abort all processes
      self.comm.Barrier()
      self.comm.Abort(1)        
          
    return self.rho
    
    
  def conductivity_in_system(self, metal_atoms):
    """
    Set conductivity field for bridging clusters.
        
    Parameters:
    -----------
    metal_atoms : list
      Positions of metal atoms in bridging cluster
    """
    sigma_metal = self.poisson_parameters.get('conductivity_CF', 1e6) # S/m
    sigma_dielectric = self.poisson_parameters.get('conductivity_dielectric', 1e-10) # S/m
    
    # Initialize to dielectric value
    self.sigma.x.array[:] = sigma_dielectric
    
    if not metal_atoms:
      return
      
    points_array = np.asarray(metal_atoms, dtype=np.float64)
    if points_array.ndim == 1:
      points_array = points_array.reshape(1, -1)
    
    # Find cells containing metal atoms
    cell_candidates = geometry.compute_collisions_points(self.bb_tree, points_array)
    colliding_cells = geometry.compute_colliding_cells(
      self.domain, cell_candidates, points_array
    )
          
    for i in range(len(points_array)):
      if len(colliding_cells.links(i)) > 0:
        cell_id = colliding_cells.links(i)[0]
        if cell_id < len(self.sigma.x.array):
          self.sigma.x.array[cell_id] = sigma_metal
          
  # ======================================================================
  # Solve
  # ======================================================================
  
  def solve(self, charge_locations, charges, charge_err_tol=3):
    """
    Solve Poisson equation.
          
    Parameters:
    -----------
    charge_locations : np.ndarray
      Charge positions [N, 3] in angstrom
    charges : np.ndarray
      Charge values [N] in Coulombs
    charge_err_tol : float, optional
      Tolerance for charge conservation (%)
          
    Returns:
    --------
    uh : Function
      Electric potential solution
    """
    angstrom_to_m = 1e-10
    
    # === Choose formulation ===
    if self.use_conductivity:
      # Conductivity formulation: -?*(s?f) = 0
      self.conductivity_in_system(self.metal_atoms)
      
      a_form = fem.form(
        ufl.inner(self.sigma * ufl.grad(self.u_trial), ufl.grad(self.v_test))
        * angstrom_to_m * ufl.dx
      )
      
      with self.b.localForm() as loc_b:
        loc_b.set(0)
            
      assemble_L = False
    else:
      # Standard Poisson: -e?(?f) = ?
      a_form = self.a_form
      
      # Create charge density
      rho = self.charge_density(charge_locations, charges, charge_err_tol)
      """
      Unit analysis:
      rho = C/m^3
      epsilon_0 = F/m = C/(m*V)
      dx = m^3
      
      L = V * m
      """
      L = (rho / (epsilon_0 * self.epsilon_r)) * self.v_test * ufl.dx
      L_form = fem.form(L)
      assemble_L = True



        
    # === Reassemble matrix if BCs changed ===
    if self._bcs_changed or self.use_conductivity:
      # Zero out matrix and reassemble with new BCs
      self.A.zeroEntries()
      assemble_matrix(self.A, a_form, bcs=self.bcs)
      self.A.assemble()
      self.ksp.setOperators(self.A)
      self._bcs_changed = False # Reset flag
        
    # === Assemble RHS (only for standard Poisson) ===       
    if assemble_L:
      #
      # assemble_vector() it supposed to zero internally so "with self.b.localForm() as loc_b" would be unnecesary, but I haven't tested
      #
      with self.b.localForm() as loc_b:
        loc_b.set(0)
      assemble_vector(self.b,L_form)
    
    # === Apply BCs to RHS ===
    fem.apply_lifting(self.b, [a_form], [self.bcs])
    self.b.ghostUpdate(
      addv=PETSc.InsertMode.ADD_VALUES,
      mode=PETSc.ScatterMode.REVERSE
    )
    fem.set_bc(self.b, self.bcs)


    #---------------
    # Initial guess: Not significant improvement for linear problems. It might be relevant if we have a with voltage-dependent permitivity 
    # I can check layer the convergence
    # num_its = self.ksp.getIterationNumber()
    # --------------
    """
    # Reuse solution function self.uh:
    if self.previous_solution is not None:
      self.uh.x.array[:] = self.previous_solution.x.array[:] # Set initial guess
      self.ksp.setInitialGuessNonzero(True)
    else:
      self.uh.x.array[:] = 0.0 # Zero initial guest on first solve
      self.ksp.setInitialGuessNonzero(False)
    """
    
    # === Solve ===
    self.uh.x.array[:] = 0.0
    fem.set_bc(self.uh.x.array, self.bcs) 
    self.uh.x.scatter_forward()
    self.ksp.setInitialGuessNonzero(False)
        
    self.ksp.solve(self.b, self.uh.x.petsc_vec)
    self.uh.x.scatter_forward()

        
    # DIAGNOSTIC: Check if solution respects BCs
    #self.verify_bcs_after_solve(self.uh, expected_top=1.0, expected_bottom=0.0)
    """
    #Store solution for next iteration
    if self.previous_solution is None:
      self.previous_solution = fem.Function(self.V)

    self.previous_solution.x.array[:] = self.uh.x.array[:]
    """
        
    self.new_field_cache = True
        
    return self.uh
    
  # ======================================================================
  # Electric Field Evaluation
  # ======================================================================    
    
  def evaluate_electric_field_at_points(self, uh, points):
  
    """
    Evaluate electric field E = -?f at given points.
    
    Uses base class evaluate_at_points()
    Implements caching to avoid recomputing E-field for same points.
    
    Parameters:
    -----------
    uh : Function
        Electric potential solution
    points : array-like
        Points [N, 3] where to evaluate E-field (angstroms)
    
    Returns:
    --------
    E_values : dict
        Dictionary {(x,y,z): (Ex, Ey, Ez)} in V/m
    """
    # Initialize cache on firts call
    if self.new_field_cache:
      self._field_cache = {}
      self.new_field_cache = False
    
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
      points_array = points_array.reshape(1, -1)
        
    # === Separate cached and new points ===
    cached_points = []
    new_points = []  
      
    for point in points_array:
      point_key = tuple(np.round(point,6))
      if point_key in self._field_cache:
        cached_points.append(point_key)
      else:
        new_points.append(point)
        
    # === Initialize result dictionary ===       
    E_values_global = {}
      
    # === Add cached values ===
    for point_key in cached_points:
        E_values_global[point_key] = self._field_cache[point_key]
    
    # === Compute new values ===
    if new_points:
      new_points_array = np.array(new_points, dtype=np.float64)
      
      # Project E = -?f to vector function space  
      self._project_electric_field(uh)
        
      # Evaluate E_field at new points using base class method
      E_values_array = self.evaluate_at_points(
        self.E_field,
        new_points_array,
      )
      
      # Apply unit conversion and polarization factor
      E_values_array = E_values_array * 1e10 * self.bond_polarization_factor  # V/m
      
      # === MPI reduction: sum across ranks ===
      # Different ranks may evaluate different points, so we need to combine results
      E_values_array = self.mpi_ctx.allreduce(
        E_values_array,
        op=self.mpi_ctx.MPI.SUM
      )
      
      # Store in dictionary format with caching 
      for i, point in enumerate(new_points):
        point_key = tuple(np.round(point, 6))
        E_values_global[point_key] = E_values_array[i]
        self._field_cache[point_key] = E_values_array[i]
          
    return E_values_global
      
  def _project_electric_field(self, uh):
    """
    Project E = -?f to vector function space.
    
    Call this once before evaluating E-field at multiple points.
    The result is stored in self.E_field.
    
    Parameters:
    -----------
    uh : Function
        Electric potential solution
    """
      
    # Compute gradient expression: E = -?f
    E_expr = -ufl.grad(uh)
    
    # Create expression for interpolation
    expr = fem.Expression(
        E_expr,
        self.V_vec.element.interpolation_points()
    )
    
    # Interpolate to vector function space
    self.E_field.interpolate(expr)
    
  
  # ======================================================================
  # Debugging Utilities
  # ======================================================================
  
  def diagnose_boundary_conditions(self):
    """
    BC diagnosis with proper parallel handling.
    All ranks compute local stats, MPI reduces to global, rank 0 prints.
    """
    # === Header (rank 0 only) ===
    if self.rank == 0:
        print("\n" + "="*70)
        print("BOUNDARY CONDITION DIAGNOSTICS")
        print("="*70)
    
    # === Check domain boundaries (all ranks compute, rank 0 prints) ===
    coords = self.domain.geometry.x
    local_z_min = coords[:, 2].min()
    local_z_max = coords[:, 2].max()
    
    # Reduce to global min/max across all ranks
    global_z_min = self.mpi_ctx.allreduce(local_z_min, op=self.mpi_ctx.MPI.MIN)
    global_z_max = self.mpi_ctx.allreduce(local_z_max, op=self.mpi_ctx.MPI.MAX)
    
    if self.rank == 0:
        print(f"\nDomain Z-range: [{global_z_min:.6f}, {global_z_max:.6f}] angstrom")
        print(f"Stored Z-range: [{self.z_min:.6f}, {self.z_max:.6f}] angstrom")
        
        if abs(global_z_min - self.z_min) > 1e-6 or abs(global_z_max - self.z_max) > 1e-6:
            print("??  WARNING: Stored z_min/z_max don't match actual domain!")
    
    # === Check each BC (all ranks compute, rank 0 prints global stats) ===
    if self.rank == 0:
        print(f"\nTotal boundary conditions: {len(self.bcs)}")
    
    for i, bc in enumerate(self.bcs):
        V_bc = bc.function_space
        bc_value = bc.g
        
        # Get BC values on this rank
        if hasattr(bc_value, 'x'):
            bc_values = bc_value.x.array
            local_min = np.min(bc_values) if len(bc_values) > 0 else 1e10
            local_max = np.max(bc_values) if len(bc_values) > 0 else -1e10
            local_count = len(bc_values)
        else:
            # Constant BC value
            local_min = local_max = float(bc_value)
            local_count = 1
        
        # === Reduce to global statistics ===
        global_min = self.mpi_ctx.allreduce(local_min, op=self.mpi_ctx.MPI.MIN)
        global_max = self.mpi_ctx.allreduce(local_max, op=self.mpi_ctx.MPI.MAX)
        global_count = self.mpi_ctx.allreduce(local_count, op=self.mpi_ctx.MPI.SUM)
        
        # === Rank 0 prints consolidated results ===
        if self.rank == 0:
            print(f"\n  BC {i}:")
            print(f"    Applied Value range: [{global_min:.6e}, {global_max:.6e}]")
            print(f"    Total DOFs (all ranks): {global_count}")
            print(f"    Function Space Dim (this rank): {V_bc.dofmap.index_map.size_local}")
    
    if self.rank == 0:
        print("="*70 + "\n")

    
  def verify_bcs_after_solve(self,uh, expected_top=1.0, expected_bottom=0.0):
    """
    Verify that BCs are satisfied in the solution.
      
    Checks top/bottom boundaries AND interior domain for physics violations.
    All ranks compute local stats, MPI reduces to global, rank 0 prints.
      
    Parameters:
    -----------
    uh : Function
      Electric potential solution
    expected_top : float
      Expected potential at top boundary (V)
    expected_bottom : float
      Expected potential at bottom boundary (V)
    """
    if self.rank == 0:
      print("\n" + "="*70)
      print("VERIFYING BOUNDARY CONDITIONS IN SOLUTION")
      print("="*70)
        
    # === Get DOF coordinates and solution values (all ranks) ===
    dof_coords = self.V.tabulate_dof_coordinates()
    solution_values = uh.x.array
      
    # === Find boundary DOFs (all ranks) ===
    top_mask = np.isclose(dof_coords[:, 2], self.z_max, atol=1e-6)
    top_values = solution_values[top_mask]
       
    bottom_mask = np.isclose(dof_coords[:, 2], self.z_min, atol=1e-6)
    bottom_values = solution_values[bottom_mask]
      
    # === Local statistics (all ranks) ===  
    if len(top_values) > 0:
      top_min_local = np.min(top_values)
      top_max_local = np.max(top_values)
      top_mean_local = np.mean(top_values)
    else:
      top_min_local = top_max_local = top_mean_local = 0.0
    
    if len(bottom_values) > 0:
      bottom_min_local = np.min(bottom_values)
      bottom_max_local = np.max(bottom_values)
      bottom_mean_local = np.mean(bottom_values)
    else:
      bottom_min_local = bottom_max_local = bottom_mean_local = 0.0
        
        
    # === Global statistics via MPI reduction ===
    top_min = self.mpi_ctx.allreduce(
      top_min_local if len(top_values) > 0 else 1e10,
      op=self.mpi_ctx.MPI.MIN
    )
    top_max = self.mpi_ctx.allreduce(
      top_max_local if len(top_values) > 0 else -1e10, 
      op=self.mpi_ctx.MPI.MAX
    )
    bottom_min = self.mpi_ctx.allreduce(
      bottom_min_local if len(bottom_values) > 0 else 1e10,
      op=self.mpi_ctx.MPI.MIN
    )
    bottom_max = self.mpi_ctx.allreduce(
      bottom_max_local if len(bottom_values) > 0 else -1e10,
      op=self.mpi_ctx.MPI.MAX
    )
    
    # === Print boundary results (rank 0 only) ===
    if self.rank == 0:
      print(f"\nTop boundary (z = {self.z_max:.6f}):")
      print(f"  Expected value: {expected_top:.6f} V")
      print(f"  Actual range: [{top_min:.6f}, {top_max:.6f}] V")
        
      top_error = max(abs(top_min - expected_top), abs(top_max - expected_top))
      if top_error > 1e-6:
        print(f"  ? ERROR: Top BC violated by {top_error:.2e} V")
      else:
        print(f"  ? Top BC satisfied")
        
      print(f"\nBottom boundary (z = {self.z_min:.6f}):")
      print(f"  Expected value: {expected_bottom:.6f} V")
      print(f"  Actual range: [{bottom_min:.6f}, {bottom_max:.6f}] V")
        
      bottom_error = max(abs(bottom_min - expected_bottom), abs(bottom_max - expected_bottom))
      if bottom_error > 1e-6:
        print(f"  ? ERROR: Bottom BC violated by {bottom_error:.2e} V")
      else:
        print(f"  ? Bottom BC satisfied")
            
    # === Check interior domain for physics violations (all ranks compute, rank 0 prints) ===
    interior_mask = ~(top_mask | bottom_mask)
    if np.any(interior_mask):
      interior_values = solution_values[interior_mask]
      interior_min_local = np.min(interior_values)
      interior_max_local = np.max(interior_values)
            
      # Reduce to global interior statistics
      interior_min = self.mpi_ctx.allreduce(interior_min_local, op=self.mpi_ctx.MPI.MIN)
      interior_max = self.mpi_ctx.allreduce(interior_max_local, op=self.mpi_ctx.MPI.MAX)
            
      if self.rank == 0:
        print(f"\nInterior domain:")
        print(f"  Value range: [{interior_min:.6f}, {interior_max:.6f}] V")
            
        if interior_min < expected_bottom - 1e-6:
          print(f"  ? ERROR: Interior values below bottom BC!")
        if interior_max > expected_top + 1e-6:
          print(f"  ? ERROR: Interior values above top BC!")
          print(f"     This suggests the Laplace equation is violated")
        
    # === Footer (rank 0 only) ===
    if self.rank == 0:
        print("="*70 + "\n")
        
        
        


    