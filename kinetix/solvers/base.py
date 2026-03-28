"""
Modular FEM solvers for kMC electro-thermal simulations.
Base class + specialized solvers
"""

from mpi4py import MPI
from pathlib import Path
import numpy as np
import ufl

from dolfinx import mesh, fem, geometry, io
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from dolfinx.io import gmshio
import petsc4py.PETSc as PETSc
import gmsh

from kinetix.utils.mpi_context import MPIContext
from kinetix.configs.config_loader import get_mesh_root

class FEMSolverBase:
  """ 
    Shared infrastructure for DOLFINx-based solver       
    Base class
    
    Handles:
    - MPI context (via MPIContext singleton)
    - Mesh loading and management
    - Function space setup
    - Linear algebra pre-allocation
    - KSP solver configuration
    - Domain geometry precomputation
    
    Subclasses must implement:
    - set_boundary_conditions()
    - solve()
    """
  def __init__(
    self, 
    solver_parameters, 
    grid_crystal=None,
    mpi_ctx=None,
    **kwargs):
    
    """
    Initialize FEM solver base.
        
    Parameters:
    -----------
    solver_parameters : dict
      Solver-specific parameters (mesh_file, physical params, etc.)
    grid_crystal : optional
      Crystal structure for mesh generation (if mesh needs to be created)
    mpi_ctx : MPIContext, optional
      MPI context manager (uses singleton if None)
    **kwargs : dict
      Additional parameters:
        - gmsh_model_rank : int (default: 0)
        - gdim : int (default: 3)
        - mesh_size : float (default: 0.8 angstrom)
        - path_results : str (default: "")
    """
    # === MPI handling ===
    self.mpi_ctx = mpi_ctx
    self.comm = self.mpi_ctx.comm
    self.rank = self.mpi_ctx.rank
    self.use_mpi = self.mpi_ctx.available
    
    # === Solver parameters ===
    self.solver_parameters = solver_parameters
    self.path_results_folder = kwargs.get("path_results", "")
    
    # === Mesh parameters ===
    self.gmsh_model_rank = kwargs.get("gmsh_model_rank", 0)
    self.gdim = kwargs.get("gdim", 3)
    self.mesh_size = kwargs.get("mesh_size", 0.8)
    self.padding = kwargs.get("bounding_box_padding",5.0)
    self.epsilon_gc = kwargs.get("epsilon_gaussian_charge",0.8) #(angstrom)
    self.active_mesh_refinement = kwargs.get("activate_mesh_refinement",True)
    self.fine_mesh_size = kwargs.get("fine_mesh_size",0.2) #(angstrom)
    self.refinement_radius = kwargs.get("refinement_radius",1.2) #(angstrom)
    self.defects_config = solver_parameters["defects_config"]
    
    # === Mesh handling ===
    mesh_file = solver_parameters['mesh_file']
    self.mesh_folder = get_mesh_root()
    self.mesh_file = self.mesh_folder / mesh_file
    
    # === Initialize mesh and FEM ===
    self._ensure_mesh(grid_crystal)
    self._load_and_setup_fem()
    self._setup_function_spaces()
    self._setup_linear_algebra()
    self._precompute_domain_geometry()
    self._setup_time_series_output(output_folder="Electric_potential_results")
    
  # ====================================
  # Mesh management
  # ====================================
  def _ensure_mesh(self, grid_crystal):
      """ Ensure mesh file exists; generate if missing (rank 0 only), then sync """
      
      if self.rank == 0:
        self.mesh_folder.mkdir(parents=True, exist_ok=True)
        if not self.mesh_file.exists():
          print(f'Rank {self.rank}: Starting mesh generation')
          self._generate_mesh(grid_crystal)
          print(f'Rank {self.rank}: Mesh generation completed')
        else:
          print(f'Rank {self.rank}: Using existing mesh: {self.mesh_file}')
      else:
        print(f'Rank {self.rank}: Waiting for mesh...')
        
      if self.use_mpi:
        self.comm.Barrier()
        
      print(f'Rank {self.rank}: Mesh ready on all ranks.')        
        
  def _generate_mesh(self, grid_crystal):
          """
          Generate a mesh using GMSH
           - grid_crystal: crystal structure
           - padding: increase margins of the simulation domain (respect to structure)
           - mesh_size: control the coarsen of the mesh
          """
        
          points = self._get_poisson_relevant_sites(grid_crystal)
          # Calculate minimum atom separation for validation
          min_separation = self._calculate_min_atomic_separation(points)
    
          # Validate mesh parameters
          self._validate_mesh_parameters(min_separation)
          
          gmsh.initialize()
          gmsh.model.add(self.mesh_file.name)
          
          # Add lattice points to the model using OCC - OpenCASCADE (OCC) geometry module
          point_tags = []
          for point in points:
              tag = gmsh.model.occ.addPoint(point[0], point[1], point[2])
              point_tags.append(tag)
              
          # Create bounding box -> The smallest rectangular (in 2D) or cuboidal (in 3D) volume that fully encloses a given set of objects.
          # Defined by:
          # Minimum coordinates (min_coords) The smallest values
          # Maximum coordinates (max_coords) The largest values
          min_coords = np.min(points, axis=0)
          max_coords = np.max(points, axis=0)
          
          # Only add padding in x, y
          min_coords[0] -= self.padding
          min_coords[1] -= self.padding
          max_coords[0] += self.padding
          max_coords[1] += self.padding
          min_coords[2] -= self.epsilon_gc * 3
          max_coords[2] += self.epsilon_gc * 3
          
          box_tag = gmsh.model.occ.addBox(
              min_coords[0], min_coords[1], min_coords[2], # Start (x, y, z)
              max_coords[0] - min_coords[0], # Width (x-dimension)
              max_coords[1] - min_coords[1], # Height (y-dimension)
              max_coords[2] - min_coords[2] # Depth (z-dimension)
              )
          
          # Synchronizes the OpenCASCADE (OCC) geometry kernel with gmsh model.
          # Necessary after modifying the geometry (adding points, surfaces, volumes) to recognizes the changes
          gmsh.model.occ.synchronize()
          
          # Embed lattice points (point_tags) into the volume (box_tag)
          # gmsh.model.mesh.embed(dim, tags, target_dim, target_tag)
              # dim = 0, The dimension of the entities being embedded (0 = points).
              # tags = point_tags, A list of point tags (the points to embed).
              # target_dim = 3, The dimension of the target entity (3 = volume).
              # target_tag = box_tag, The tag of the target volume (where points are embedded).
          gmsh.model.mesh.embed(0,point_tags,self.gdim,box_tag)
          
          # Add physical group for the domain (3D elements)
          domain_tag = gmsh.model.addPhysicalGroup(self.gdim,[box_tag])
          gmsh.model.setPhysicalName(self.gdim,domain_tag,"domain")
          
          # Add physical group for the boundary (2D elements)
          # Extract boundary entities (surfaces) of the 3D box -> boundary of 3D volume box_tag
          boundary_tags = gmsh.model.getBoundary([(self.gdim,box_tag)], oriented = False)
          # Define physical group -> Surface (dim=2) -> Extract the tags of the surfaces
          boundary_tag = gmsh.model.addPhysicalGroup(self.gdim-1, [tag for dim, tag, in boundary_tags])
          gmsh.model.setPhysicalName(self.gdim-1,boundary_tag,"boundary")
          
          
          if self.active_mesh_refinement:
            print('Starting refinement')
            self._add_adaptative_refinement(points)
          else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax",self.mesh_size) # Coarsen globally
            
          # Generate a coarse mesh
          gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Frontal-Delaunay for efficiency
          gmsh.option.setNumber("Mesh.Optimize", 1)
          gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
          gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
          
          # Additional mesh quality options
          gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
          gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
          gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
          
          gmsh.model.mesh.generate(self.gdim)
          
          # Verify mesh quality
          self._verify_mesh_quality(points)
        
          gmsh.write(str(self.mesh_file))
          gmsh.finalize()
    
    
  def _load_mesh(self):
        """
        Load the mesh from the gmsh file.
        """
        # All processes load the mesh
        domain, cell_markers, facet_markers = gmshio.read_from_msh(
          str(self.mesh_file), 
          self.comm, 
          self.gmsh_model_rank, 
          gdim=self.gdim
        )
        
        return domain, cell_markers, facet_markers
        
  def _calculate_min_atomic_separation(self, points):
      """
      Calculate minimum separation between atomic sites
      """
      from scipy.spatial.distance import pdist
      
      if len(points) < 2: 
        return float('inf')
        
      distances = pdist(points)
      return np.min(distances)
      
  def _validate_mesh_parameters(self,min_separation):
      """
      Validate mesh parameters for numerical stability
      """
      
      print('\n=== Parameter Validation ===')
      
      issues = []
      warnings = []
      
      # 1) Fine mesh should resolve Gaussian charge distribution
      # Need at least 3-4 elements across the Gaussian width
      recommended_fine_mesh = self.epsilon_gc / 4.0
      if self.fine_mesh_size > recommended_fine_mesh:
        warnings.append(f'Fine mesh size ({self.fine_mesh_size:.3f} angstroms) is larger than recommended({recommended_fine_mesh:.3f} angstroms)')
        warnings.append(f'  -> Should be < epsilon_gc/4 = {self.epsilon_gc}/4 = {recommended_fine_mesh:.3f} angstroms)')
        
      # 2) Refinement radius should cover Gaussian charge
      # Should be at least 3*epsilon_gc for 99.7% of Gaussian
      recommended_radius = 3 * self.epsilon_gc
      if self.refinement_radius < recommended_radius:
        warnings.append(f'Refinement radius ({self.refinement_radius:.3f} angstroms) is smaller than recommended ({recommended_radius:.3f} angstroms)')
        warnings.append(f'  -> Should be >= 3*epsilon_gc = 3*{self.epsilon_gc} = {recommended_radius:.3f} angstroms')
        
      # 3) Fine mesh should be much smaller than coarse mesh
      mesh_ratio = self.fine_mesh_size / self.mesh_size
      if mesh_ratio > 0.5:
        warnings.append(f'Fine/coarse mesh ratio ({mesh_ratio:.2f}) is too large')
        warnings.append(f'  -> Should be < 0.5 for effective refinement')
        
      # 4) Mesh should resolve atomic separations
      if self.fine_mesh_size > min_separation / 3:
        issues.append(f'Fine mesh ({self.fine_mesh_size:.3f} angstroms) is too coarse for atomic separation ({min_separation:.3f} angstroms)')
        issues.append(f'  -> Should be < min_separation/3 = {min_separation/3:.3f} angstroms')
        
      # 5) Check for overlapping Gaussian charges
      if min_separation < 2 * self.epsilon_gc:
        warnings.append(f'Atomic separation ({min_separation:.3f} angstroms) < 2*epsilon_gc ({2*self.epsilon_gc:.3f} angstroms)')
        warnings.append(f'  -> Gaussian charges may significantly overlap')
        
        
        # Print results
      if issues:
        print('CRITICAL ISSUES:')
        for issue in issues:
            print(f'  ? {issue}')
    
      if warnings:
        print('WARNINGS:')
        for warning in warnings:
            print(f'  ??  {warning}')
    
      if not issues and not warnings:
        print('? All parameters look good')
    
      # Provide recommendations
      print('\nRECOMMENDED PARAMETERS:')
      print(f'  fine_mesh_size: {min(self.epsilon_gc/4, min_separation/5):.3f} angstroms')
      print(f'  mesh_size: {min(self.epsilon_gc/4, min_separation/5) * 3:.3f} angstroms')
      print(f'  refinement_radius: {max(3*self.epsilon_gc, min_separation*1.5):.3f} angstroms')
      print(f'  epsilon_gc: {min_separation/3:.3f} angstroms (if adjustable)')
      print('=' * 40)
      
    
  def _verify_mesh_quality(self,site_positions):
      """
      Comprehensive mesh quality verification
      """
      print('\n=== Mesh Quality Analysis ===')
      
      
      # Get mesh statistics
      node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
      node_coords = node_coords.reshape(-1,3)
      elements = gmsh.model.mesh.getElements()
      total_elements = sum(len(elem_tags) for elem_tags in elements[1])
        
      print(f'Total nodes: {len(node_tags)}')
      print(f'Total elements: {total_elements}')
        
      if self.active_mesh_refinement:
        self._analyze_refinement_quality(node_coords,site_positions)
          
      
  def _analyze_refinement_quality(self,node_coords, site_positions):
      """
      Analyze mesh refinement near atomic sites
      """
      print('\n--- Refinement Quality Check ---')
      # Check first few atoms
      atoms_to_check = min(10, len(site_positions))
      
      
      for i in range(atoms_to_check):
        atom_pos = site_positions[i]
        
        # Find nodes near this atom
        distances = np.linalg.norm(node_coords - atom_pos,axis=1)
        # Nodes within refinement radius
        nearby_mask = distances <= self.refinement_radius
        nearby_nodes = node_coords[nearby_mask]
        nearby_distances = distances[nearby_mask]
        
        # Nodes close to charge (within Gaussian width)
        very_close_mask = distances <= self.epsilon_gc
        very_close_count = np.sum(very_close_mask)
        """
        if very_close_count < 5:
          print(f"\nAtom {i} at ({atom_pos[0]:.2f}, {atom_pos[1]:.2f}, {atom_pos[2]:.2f}):")
          print(f'  Nodes within {self.epsilon_gc:.2f} angstroms (Gaussian width): {very_close_count}')
        
        """
        
        print(f'\nAtom {i} at ({atom_pos[0]:.2f}, {atom_pos[1]:.2f}, {atom_pos[2]:.2f}):')
        print(f'  Nodes within {self.refinement_radius:.2f} angstroms: {len(nearby_nodes)}')
        print(f'  Nodes within {self.epsilon_gc:.2f} angstroms (Gaussian width): {very_close_count}')
        
        if len(nearby_nodes) > 1:
          
          # Estimate local mesh size
            sorted_distances = np.sort(nearby_distances[nearby_distances > 1e-10])
            if len(sorted_distances) > 1:
                min_spacing = sorted_distances[0]
                
                print(f'  Closest node distance: {min_spacing:.4f} angstroms')
                print(f"  Mean distance to nodes: {np.mean(sorted_distances[:60]):.6e} angstroms")

                # Check for duplicates
                unique_dists = np.unique(np.round(sorted_distances[:100], decimals=10))
                print(f"  Unique distances (first 100): {len(unique_dists)} / {len(sorted_distances[:100])}")
                # Check if mesh is adequate
                elements_in_gaussian = self.epsilon_gc / min_spacing
                print(f'  Elements across Gaussian width: {elements_in_gaussian:.1f}')
                
                if elements_in_gaussian < 3:
                    print(f'  ??  WARNING: Insufficient resolution (need =3-4 elements)')
                elif elements_in_gaussian >= 4:
                    print(f'  ? Good resolution')
                else:
                    print(f'  ~ Marginal resolution')
                    
  def _get_poisson_relevant_sites(self, grid_crystal):
      """
      Filter sites relevant for Poisson equation.
      Select sites from allowed_sublattices
      """
      relevant_coords = []
      
      for site in grid_crystal.values():
      # Check if this site type is relevant for any defect
        is_relevant = False
      
        for defect_name, cfg in self.defects_config.items():
          allowed_sublattices = cfg.get("allowed_sublattices", [])
          if site.site_type in allowed_sublattices:
            is_relevant = True
            break
        
        if is_relevant:
          relevant_coords.append(site.position)
        
      return np.array(relevant_coords)

  def _add_adaptative_refinement(self,site_positions):
      """
      Add distance-based mesh refinement near particles
      Use Ball fields    
      """
      
      try:
        existing_fields = gmsh.model.mesh.field.list()
        for field_id in existing_fields:
          gmsh.model.mesh.field.remove(field_id)
          
      except:
        pass
      
      
      ball_fields = []
      
      total_sites = len(site_positions)
      
      
      for i, pos in enumerate(site_positions):
        # Create distance field to this particle
        ball_field = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(ball_field, 'VIn', self.fine_mesh_size)
        gmsh.model.mesh.field.setNumber(ball_field, 'VOut', self.mesh_size)
        gmsh.model.mesh.field.setNumber(ball_field, 'XCenter', pos[0])
        gmsh.model.mesh.field.setNumber(ball_field, 'YCenter', pos[1])
        gmsh.model.mesh.field.setNumber(ball_field, 'ZCenter', pos[2])
        gmsh.model.mesh.field.setNumber(ball_field, 'Radius', self.refinement_radius)
        

        # Optional: Add thickness for smooth transition
        #gmsh.model.mesh.field.setNumber(ball_field, 'Thickness', self.refinement_radius * 0.2)
            
        ball_fields.append(ball_field)
        

        # Show the progress
        if total_sites < 20 or i % max(1, total_sites // 10) == 0 or i == total_sites - 1:
          progress = (i + 1) / total_sites * 100
          print(f'Refinement progress: {progress:.1f} % ({i+1}/{total_sites})')
        
      
      if ball_fields:  
        if len(ball_fields) > 1:  
          # Combine all threshold fields using minimum (finest mesh)
          min_field = gmsh.model.mesh.field.add('Min')
          gmsh.model.mesh.field.setNumbers(min_field,'FieldsList', ball_fields)
          gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
          print(f'Combined {len(ball_fields)} ball fields using Min field (ID: {min_field})')

        else:
          gmsh.model.mesh.field.setAsBackgroundMesh(ball_fields[0])
          

        
        # Optional but helpful: set global min size as safety net
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.fine_mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)
        
        print(f"Applied {len(ball_fields)} ball refinement fields")
            
      else:
        # Fallback to global mesh size if no refinement fields
        #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)
        print("No refinement fields applied, using global mesh size")
        
        
  def _load_and_setup_fem(self):
      """Load mesh and create function spaces."""
      # Load the mesh
      self.domain, self.cell_markers, self.facet_markers = self._load_mesh()
      
      # Get mesh dimensions
      self.tdim = self.domain.topology.dim # Get the dimension of the mesh (3D in this case)
      self.fdim = self.tdim - 1 # The dimension of the facets (faces) is one less than the mesh dimension
      
      # Create facet-to-cell connectivity (required to determine boundary facets)
      self.domain.topology.create_connectivity(self.fdim,self.tdim) # Create connectivity between facets and cells
        
      # Example: V ('Lagrange',1) are functions defined in domain, continuous (because Lagrange elements enforce continuity) and degree 1
      self.V = functionspace(self.domain, ('Lagrange',1))
      # Create vector function space for electric field evaluation
      self.V_vec = functionspace(
        self.domain, 
        ("Lagrange",1, (self.domain.topology.dim,))
      )
        
  # ======================================================================
  # Function Spaces & Reusable Objects
  # ======================================================================
  def _setup_function_spaces(self):
    """Setup function spaces and precompute reusable objects."""
    # DG0 space for coefficients (charge density, conductivity, etc.)
    # Discontinuous Galerkin (DG) space of order 0 -> no continuity between elements
    # DG(0) useful for defining constant fields, like properties, density
    self.W = fem.functionspace(self.domain, ("DG",0))
        
    # Pre-compute cell midpoints (for DG0 interpolation)
    # Local cells (cells owned by this process in parallel computation)
    # Ghost cells (cells shared between processes in parallel computing)
    num_cells = (
      self.domain.topology.index_map(self.tdim).size_local + 
      self.domain.topology.index_map(self.tdim).num_ghosts
    )
    self.num_cells = num_cells
    # Computes the midpoint of each cell in the domain
    # Since DG(0) functions (like rho) are piecewise constant per element, we typically evaluate them at the midpoint of each element
    self.cell_midpoints = mesh.compute_midpoints(
      self.domain, 
      self.tdim, 
      np.arange(num_cells, dtype=np.int32)
    )
         
    # Pre-create trial and test functions
    self.u_trial = ufl.TrialFunction(self.V)
    self.v_test = ufl.TestFunction(self.V)
   
  # ======================================================================
  # Linear Algebra & Solver
  # ======================================================================    
  def _setup_linear_algebra(self):
      """Pre-allocate linear algebra objects and configure KSP solver."""
      # Pre-allocate matrix (will be reassembled when BCs change)
      # Note: a_form must be defined by subclass before calling this
      # For now, create placeholder - subclass will update
      
      # Pre-create PETSc vectors
      self.b = fem.petsc.create_vector(
        fem.form(self.v_test * ufl.dx)
      )
      
      # Pre-create solution function
      self.uh = fem.Function(self.V)
              
      # Pre-create KSP solver (reuse across solves)
      # Configure KSP solver with initial guess
      self.ksp = PETSc.KSP().create(self.domain.comm)
      self.ksp.setType(PETSc.KSP.Type.CG)
      self.ksp.getPC().setType(PETSc.PC.Type.HYPRE)
      self.ksp.getPC().setHYPREType("boomeramg")
      self.ksp.setTolerances(rtol=1e-8,atol=1e-10,max_it=1000)
      
      # Track if BCs have changed (triggers matrix reassembly)  
      self._bcs_changed = True
      self.bcs = []

      # Pre-compute bounding box tree for point evaluation
      self.bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim) 
      
      # Cache for previous solution (optional, for initial guess)        
      self.previous_solution = None
      
  def _setup_matrix(self, a_form):
    """
    Setup matrix from bilinear form.
    Call this from subclass after defining a_form.
    """
    self.a_form = a_form
    self.A = fem.petsc.create_matrix(self.a_form)
    self.ksp.setOperators(self.A)
    
  # ======================================================================
  # Domain Geometry
  # ======================================================================
      
  def _precompute_domain_geometry(self):
    """ Pre-compute domain bounding box and dimensions """
    coords = self.domain.geometry.x
    local_x_min, local_x_max = coords[:,0].min(), coords[:,0].max()
    local_y_min, local_y_max = coords[:,1].min(), coords[:,1].max()
    local_z_min, local_z_max = coords[:,2].min(), coords[:,2].max()
    
    # Global min/max across all ranks
    x_min = self.mpi_ctx.allreduce(local_x_min, op=self.mpi_ctx.MPI.MIN)
    x_max = self.mpi_ctx.allreduce(local_x_max, op=self.mpi_ctx.MPI.MAX)
    y_min = self.mpi_ctx.allreduce(local_y_min, op=self.mpi_ctx.MPI.MIN)
    y_max = self.mpi_ctx.allreduce(local_y_max, op=self.mpi_ctx.MPI.MAX)
    self.z_min = self.mpi_ctx.allreduce(local_z_min, op=self.mpi_ctx.MPI.MIN)
    self.z_max = self.mpi_ctx.allreduce(local_z_max, op=self.mpi_ctx.MPI.MAX)
    
    self.Lx = x_max - x_min
    self.Ly = y_max - y_min
    
    # Pre-compute DOF coordinates
    self.dof_coords = self.V.tabulate_dof_coordinates()
    
  # ======================================================================
  # Utility Methods
  # ======================================================================
  
  def evaluate_at_points(self, function, points, fallback_value=None):
    """
    Evaluate a DOLFINx function at arbitrary points.
    
    Parameters:
    -----------
    function : dolfinx.fem.Function
        FEM function to evaluate (scalar or vector)
    points : array-like
        Points [N, 3] where to evaluate (in mesh coordinates, angstroms)
    fallback_value : float or array-like, optional
        Value to return for points outside the domain.
        If None, uses function's value at cell 0 (may be inaccurate).
    
    Returns:
    --------
    values : np.ndarray
        Function values at points [N, ...] where ... depends on function rank.
        Returns empty array with shape (0, ...) if input is empty.
    """
    
    points_array = np.asarray(points, dtype=np.float64)
    
    # Handle empty input
    if len(points_array) == 0:
      if function.function_space.value_shape:
        return np.empty((0,) + function.function_space.value_shape, dtype=function.x.array.dtype)
      else:
        return np.empty(0, dtype=function.x.array.dtype)
        
    # === Step 1: Find candidate cells that might contain each point ===
    cell_candidates = geometry.compute_collisions_points(self.bb_tree, points_array)
    
    # === Step 2: Resolve to actual cells that might contain each point ===
    colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_array)
      
    # === Step 3: Build cells array with fallback for points outside domain ===
    valid_cells = []
    valid_indices = []
    
    for i in range(len(points_array)):
      cell_links = colliding_cells.links(i)
      if len(cell_links) > 0:
        valid_cells.append(cell_links[0]) # Use first matching cell
        valid_indices.append(i)
        
    # === Step 4: Evaluate function ===
    # Note: function.eval expects(points, cells) where cells[i] corresponds to points[i]
    # Initialize result array for all points
    
    # Determine shape from function's value_shape
    if function.function_space.value_shape:
      output_shape = (len(points_array),) + function.function_space.value_shape
    else:
      output_shape = (len(points_array),)
      
    values = np.zeros(output_shape, dtype=np.float64)
    
    if len(valid_cells) > 0:
      valid_points = points_array[valid_indices]
      valid_values = function.eval(valid_points,valid_cells)
      
      valid_values = np.atleast_2d(valid_values)
      
      for j, global_idx in enumerate(valid_indices):
        val = valid_values[j]
        if val.ndim > 0 and val.size == 1 and not function.function_space.value_shape:
          values[global_idx] = val.item()
        else:
          values[global_idx] = val
      
    return values    
    
  def _setup_time_series_output(self,output_folder="Electric_potential_results"):
    """
    Initialize time series output handling.
    Call this once at solver initialization.
    """
    results_folder = Path(self.path_results_folder) / output_folder
    results_folder.mkdir(exist_ok=True, parents=True)
      
    # Base filename for time series
    self.output_base = results_folder / "E_potential"
    
    # Track timesteps for metadata  
    self.timestep_info = []
    
    # Output format preference (configurable)
    self.output_format = self.solver_parameters.get('output_format', 'vtu')
    self.save_csv = self.solver_parameters.get('save_csv', False)
    
  
  def save_solution(self, function, filename, time_value=0.0, timestep=None,
                    output_format=None, save_csv=False, csv_filename=None):
                    
    """
    Save a DOLFINx function to file(s) for visualization and analysis.
    
    Supports both VTK (.vtu) and VTX (.bp) formats, with optional CSV export.
    
    Parameters:
    -----------
    function : dolfinx.fem.Function
        Function to save
    filename : str or Path
        Base filename (without extension)
    time_value : float, optional
        Physical time value for metadata (default: 0.0)
    timestep : int, optional
        Timestep index for auto-numbering (default: None)
    output_format : str, optional
        'vtu' (legacy VTK) or 'bp' (modern VTX). Uses instance default if None.
    save_csv : bool, optional
        If True, also save coordinates + values to CSV (default: False)
    csv_filename : str or Path, optional
        CSV filename. Uses default if None.
    
    Returns:
    --------
    saved_files : list
        List of paths to saved files
    """
    
    saved_files = []
    if output_format is None:
      output_format = getattr(self,'output_format', 'bp')
    
    # === Ensure output directory exist ===
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # === Add timestep numbering if provided ===
    if timestep is not None:
      base_name = f"{filepath}_{timestep:04d}"
    else:
      base_name = str(filepath)
      
    # === Save in requested format ===
    if output_format == 'vtu':
      # Legacy VTK format (universal ParaView support)
      vtu_file = f"{base_name}.vtu"
      with io.VTKFile(self.domain.comm, vtu_file, "w") as vtk:
        vtk.write_function(function)
      saved_files.append(vtu_file)
      
    elif output_format == 'bp':
      # Modern VTX format (efficient parallel I/O)
      bp_file = f"{base_name}.bp"
      with io.VTXWriter(self.domain.comm, bp_file, function) as writer:
        writer.write(time_value)
      saved_files.append(bp_file)
    else:
      raise ValueError(f"Unknown output_format: {output_format}. Use 'vtu' or 'bp'.")
      
    # === Optional CSV export for quick analysis ===
    if save_csv:
      csv_file = Path(csv_filename or f"{base_name}.csv")
      csv_file.parent.mkdir(parents=True, exist_ok=True)
      
      # Get mesh coordinates and function values
      mesh_coords = self.domain.geometry.x
      func_values = function.x.array
      
      # Handle vector/tensor functions
      if function.function_space.value_shape:
        # Vector/tensor: flatten values for CSV
        values_reshaped = func_values.reshape(len(func_values), -1)
        data = np.column_stack([mesh_coords, values_reshaped])
        header = "x,y,z," + ",".join([f"value_{i}" for i in range(values_reshaped.shape[1])])
      else:
        # Scalar function
        data = np.column_stack([mesh_coords, func_values])
        header = "x,y,z,value"
        
      np.savetxt(csv_file,data, delimiter=',', header=header, comments="")
      saved_files.append(str(csv_file))
      
    # === Track timestep metadata
    if timestep is not None:
      self.timestep_info.append({
        'timestep':timestep,
        'time': time_value,
        'files': saved_files
      })
      
    return saved_files              
  
  # ======================================================================
  # Abstract Methods (Must be implemented by subclasses)
  # ======================================================================
    
  def set_boundary_conditions(self, *args, **kwargs):
    """
    Set boundary conditions for the problem.
        
    Must be implemented by subclass.
    """
    raise NotImplementedError(
      "Subclasses must implement set_boundary_conditions()"
    )
  
  def solve(self, *args, **kwargs):
    """
    Solve the PDE.
        
    Must be implemented by subclass.
    """
    raise NotImplementedError(
      "Subclasses must implement solve()"
    ) 
    
    
        