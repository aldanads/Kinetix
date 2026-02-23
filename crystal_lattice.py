# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:19:06 2024

@author: samuel.delgado
"""
import matplotlib.pyplot as plt
from Site import Site,Island,Cluster,GrainBoundary
from scipy import constants
import numpy as np
import math
from matplotlib import cm
import time
import copy


# Pymatgen for creating crystal structure and connect with Crystallography Open Database or Material Project
# from pymatgen.ext.cod import COD
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.defects.generators import ChargeInterstitialGenerator
from pymatgen.core import Structure, PeriodicSite


import json
import os
from pathlib import Path
import platform


# Rotation of a vector - Is copper growing in [111] direction?
# The basis vector is in [001]
# https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python


class Crystal_Lattice():
    
    def __init__(
      self,
      crystal_features,
      experimental_conditions,
      Act_E_dict,
      lammps_file,
      superbasin_parameters,
      **kwargs
    ):
        
        # --- Crystal features ---
        self.id_material = crystal_features['id_material_Material_Project']
        self.crystal_size = crystal_features['crystal_size']
        self.miller_indices = crystal_features['miller_indices']
        api_key = crystal_features['api_key']
        use_parallel = crystal_features['use_parallel']
        self.facets_type = crystal_features['facets_type']
        self.affected_site = crystal_features['affected_site']
        self.mode = crystal_features['mode']
        self.radius_neighbors = crystal_features['radius_neighbors']
        self.sites_generation_layer = crystal_features['sites_generation_layer']
        self.gb_configurations = crystal_features['gb_configurations']
        self.defects_config = crystal_features.get('defects_config',{})
        self.reactions_config = crystal_features.get('reactions_config',{})
             
        # --- Experimental conditions ---
        self.sticking_coefficient = experimental_conditions['sticking_coeff']
        self.partial_pressure = experimental_conditions['partial_pressure']
        self.temperature = experimental_conditions['T']
        self.experiment = experimental_conditions['experiment']
        
        # --- Activation energies (structured by defect name) ---
        self.Act_E_dict = Act_E_dict
        
         # --- Superbasin ---
        self.n_search_superbasin = superbasin_parameters['n_search_superbasin']
        self.time_step_limits = superbasin_parameters['time_step_limits']
        self.E_min = superbasin_parameters['E_min']
        self.energy_step = superbasin_parameters['energy_step']
        self.superbasin_dict = {}
        self.superbasin_tracker = []
        self.nothing_happen_count = 0
        self.time_based_superbasin = superbasin_parameters['time_based_superbasin']
        self.allow_specie_removal = True # We need this variable to desactivate specie removal during superbasin creation
        
        # --- Poisson solver ---s
        self.poissonSolver_parameters = kwargs.get('poissonSolver_parameters', None)

        # The device is globally neutral. Charged ions + electrons/dielectric reduced
        # Example: Ag/CeO2 (Ce4+) --> Ag+ + CeO2 (Ce3+)
        # Debye length:
        # There is a distance from a charged particle beyond which its electrostatic influence 
        # is effectively blocked due to the redistribution of neighboring charges
        if self.poissonSolver_parameters:
          self.screening_factor = self.poissonSolver_parameters['screening_factor']
          self.ion_charge = self.poissonSolver_parameters['ion_charge']
          self.conductivity = self.poissonSolver_parameters['conductivity_CF']

        self.time = 0
        self.list_time = []
        
        # --- Grid generation ---n
        self.lattice_model(api_key, self.mode, self.affected_site, self.miller_indices)
        grid_crystal = kwargs.get('grid_crystal', None)
        self.crystal_grid(grid_crystal,self.radius_neighbors,self.mode,self.affected_site,api_key,use_parallel)

        self.sites_occupied = [] # Sites occupy be a chemical specie
        self.adsorption_sites = [] # Sites availables for deposition or migration
        
        #Transition rate for adsortion of chemical species
        if self.experiment != 'ECM memristor':
            self.transition_rate_adsorption(experimental_conditions)
            # self.E_min_lim_superbasin = self.Act_E_gen * 0.9 # Don't create superbasin that include the deposition process
            self.E_min_lim_superbasin = 0.25 # Don't create superbasin that include the deposition process
            # Wulff shape and edge types for this kind of material
            self.Wulff_Shape(api_key)
            self.create_edges(self.facets_type)
            
        else:
            self.wulff_facets = None
            self.dir_edge_facets = None
            self._initialize_cluster_tracking()
            
            kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
            nu0=7E12;  # nu0 (s^-1) bond vibration frequency
            T = 300


        
        # Obtain all the positions in the grid that are supported by the
        # substrate or other deposited chemical species
        support_update_sites = set(self._get_mobile_sites(self.grid_crystal.keys()))
        event_update_sites = set()
        
        self.update_sites(support_update_sites,event_update_sites)

        self.lammps_file = lammps_file
        
        
    def lattice_model(self, api_key, mode, affected_site=None, miller_indices=(0, 0, 1)):
        """
        Generate a PRISTINE lattice model. Defects are added later in crystal_grid.
        """
        with MPRester(api_key) as mpr:
          structure = mpr.get_structure_by_material_id(self.id_material)
        
        # Always use pristine structure -> Defects insertion in grid_crystal  
        sga = SpacegroupAnalyzer(structure)
        structure_conv = sga.get_conventional_standard_structure()
        
        self.miller_indices = miller_indices
        structure_oriented = self._apply_miller_orientation(structure_conv, miller_indices)
        
        self.structure_basic = structure_oriented
        self.lattice_constants = tuple(np.array(structure_oriented.lattice.abc) / 10) # nm
        
        # Set chemical specie notation
        if mode == 'vacancy' and affected_site:
          self.chemical_specie = f"V_{affected_site}"
        elif mode == 'interstitial':
          # Use first interstitial symbol for naming
          interstitials = [
            cfg["symbol"] for cfg in self.defects_config.values()
            if cfg["site_type"] == "interstitial"
          ]
          self.chemical_specie = interstitials[0] if interstitials else "interstitial"
        else:
          self.chemical_specie = structure_oriented.composition.reduced_formula
          
        # Create full supercell
        self.structure = self._create_supercell(structure_oriented)
        self.crystal_size = self.structure.lattice.abc
        
        self._compute_basis_vectors()
        

    def _apply_miller_orientation(self, structure, miller_indices):
        """
        Orient structure so that the specified Miller direction aligns with the z-axis.
        
        This method:
        1. Converts Miller indices to Cartesian coordinates
        2. Creates a rotation matrix to align this direction with z-axis
        3. Chooses an appropriate in-plane orientation for x and y
        4. Applies the rotation to the structure
        
        Parameters:
            structure: Input pymatgen Structure
            miller_indices (tuple): (h, k, l) Miller indices
            
        Returns:
            Structure: Rotated structure with [hkl] along z-axis
        """

        h, k, l = miller_indices
    
        # Handle special case: (0,0,0) or default to no rotation
        if h == 0 and k == 0 and l == 0:
            return structure.copy()

        # Convert Miller indices to Cartesian direction in the original lattice
        # [hkl] in fractional coordinates -> Cartesian
        miller_direction = structure.lattice.get_cartesian_coords([h, k, l])
        miller_direction = miller_direction / np.linalg.norm(miller_direction)

        # Target: align miller_direction with z-axis [0, 0, 1]
        z_axis = np.array([0, 0, 1])

        # Create rotation matrix using Rodrigues' rotation formula
        # We need to rotate miller_direction onto z_axis
        rotation_matrix = self._get_rotation_matrix(miller_direction, z_axis)

        # Apply rotation to structure
        symm_op = SymmOp.from_rotation_and_translation(rotation_matrix, [0, 0, 0])
        structure_rotated = structure.copy()
        structure_rotated.apply_operation(symm_op)
    
        return structure_rotated
     
        
    def _get_rotation_matrix(self, vec1, vec2):
        """
        Calculate rotation matrix that rotates vec1 to align with vec2.
        
        Uses Rodrigues' rotation formula. Handles the special case where
        vectors are parallel or anti-parallel.
        
        Parameters:
            vec1 (array): Starting vector (will be rotated)
            vec2 (array): Target vector (destination)
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """

        # Normalize vectors
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)

        # Check if vectors are already aligned
        if np.allclose(v1, v2):
            return np.eye(3)

        # Check if vectors are opposite (anti-parallel)
        if np.allclose(v1, -v2):
            # Rotate 180° around any perpendicular axis
            # Find a perpendicular vector
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            perp = np.cross(v1, perp)
            perp = perp / np.linalg.norm(perp)
            # 180° rotation around perp
            return 2 * np.outer(perp, perp) - np.eye(3)
        
        # Rodrigues' rotation formula
        # Rotation axis (perpendicular to both vectors)
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)

        # Rotation angle
        cos_angle = np.dot(v1, v2)
        sin_angle = np.linalg.norm(np.cross(v1, v2))

        # Rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R


    def _create_supercell(self, unit_cell):
        """
        Create a full supercell of the pristine lattice.
        
        Parameters:
            unit_cell (Structure): Oriented unit cell from Materials Project
            
        Returns:
            Structure: Supercell matching target dimensions in self.crystal_size
        """
        lattice_params = np.array(unit_cell.lattice.abc)
        target_dims = np.array(self.crystal_size)
        repetitions = np.ceil(target_dims / lattice_params).astype(int)
        repetitions = np.maximum(repetitions, 1)
        
        scaling_matrix = np.diag(repetitions)
        return unit_cell * scaling_matrix
        
        
    def _compute_basis_vectors(self):
        """
        Compute basis vectors for KMC grid based on minimum fractional coordinate spacing.
        
        Sets self.basis_vectors to lattice vectors scaled by minimum atomic spacing.
        This creates a grid where integer multiples correspond to atomic positions.
        """
        # Find minimum non-zero fractional coordinate spacing
        frac_coords = np.array([site.frac_coords for site in self.structure_basic])
        
        # Get unique sorted fractional coordinates for each direction
        min_spacing = []
    
        for dim in range(3):
            coords = np.sort(np.unique(np.round(frac_coords[:, dim], decimals=10)))
            coords = coords[coords > 1e-10]  # Remove zeros
            
            if len(coords) > 1:
                # Minimum spacing between adjacent positions
                spacings = np.diff(coords)
                min_spacing.append(np.min(spacings[spacings > 1e-10]))
            elif len(coords) == 1:
                # Single position means spacing is the coordinate itself
                min_spacing.append(coords[0])
            else:
                # Fallback: use 1.0 (full lattice vector)
                min_spacing.append(1.0)
        
        # Use minimum spacing across all dimensions for uniform scaling
        min_non_zero_element = min(min_spacing)
        
        # Scale lattice vectors by minimum spacing
        # This gives basis vectors where integer steps land on atomic sites
        self.basis_vectors = np.array(self.structure_basic.lattice.matrix) * min_non_zero_element
                 
            
    def _initialize_migration_pathways(self, radius_neighbors):
      """Initialize migration pathways from the COMPLETE grid_crystal."""
      self.event_labels = {}
      self.migration_pathways = {}
      i = 0
      
      
      # Brute-force neighbor search (O(N2)), but only done during initialization
      for site_idx in self.grid_crystal.keys():
        site_pos = self.grid_crystal[site_idx].position
        
        # Get neighbors using k-d tree
        neighbor_site_indices = self._get_neighbors_for_site(site_idx, radius_neighbors)
        
        # Process each neighbor
        for neighbor_idx in neighbor_site_indices:
          
          if neighbor_idx == site_idx:
            continue
            
          neighbor_pos = self.grid_crystal[neighbor_idx].position
          dist = np.linalg.norm(np.array(site_pos) - np.array(neighbor_pos))
          # Create migration key
          key = tuple(
            np.array(self.get_idx_coords(neighbor_pos,self.basis_vectors)) -
            np.array(self.get_idx_coords(site_pos,self.basis_vectors))
          )
          
          if key not in self.event_labels:
            self.event_labels[key] = i
            vector = np.array(neighbor_pos) - np.array(site_pos)
            self.migration_pathways[i] = {
              'direction': vector / np.linalg.norm(vector),
              'distance': dist
            }
            i += 1
            
      self.num_event = len(self.event_labels) + 2
      
      # Electric field-dependent barriers
      if (self.poissonSolver_parameters and
          self.poissonSolver_parameters.get('solve_Poisson', False)):
          
          for name in self.defects_config.keys():
            Act_E_mig = {}
            for key, migration_vector in self.migration_pathways.items():
              z_component = migration_vector['direction'][2]
              if np.isclose(z_component, 0.0, atol=1e-9):
                Act_E_mig[key] = self.Act_E_dict[name].get('E_mig_plane')
              elif z_component > 0:
                Act_E_mig[key] = self.Act_E_dict[name].get('E_mig_upward')
              else:
                Act_E_mig[key] = self.Act_E_dict[name].get('E_mig_downward')
            self.Act_E_dict[name]['E_mig'] = Act_E_mig  
            
          for site in self.grid_crystal.values():
                site.site_events = []
                site.Act_E_dict = copy.deepcopy(self.Act_E_dict)
            
    
    def _build_kdtree(self):
      """Build and store k-d tree for reuse."""
      from scipy.spatial import cKDTree
      positions = np.array([site.position for site in self.grid_crystal.values()])
      self._kdtree_positions = positions
      self._kdtree_indices = list(self.grid_crystal.keys())
      self._kdtree = cKDTree(positions)
      
    def _get_neighbors_for_site(self,site_idx,radius):
      """Get neighbors for a specific site using stored k-d tree."""
      site_pos = self.grid_crystal[site_idx].position
      neighbor_array_indices = self._kdtree.query_ball_point(site_pos,radius)
      
      #Convert to site indices
      neighbor_site_indices = [self._kdtree_indices[i] for i in neighbor_array_indices]
      return neighbor_site_indices      
      
            
    def crystal_grid(self,grid_crystal,radius_neighbors,mode,affected_site,api_key,use_parallel=None):
        
        self.coord_cache = {}
        
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.rank
            use_mpi = True
        except (ImportError, AttributeError):
            # MPI not available or not initialized
            rank = 0
            use_mpi = False
 
        if grid_crystal == None:
            
            if rank == 0:
            
                # We obtain integer idx
                print(f'Initializing grid_crystal with {len(self.structure)} host sites')
                total_start_time = time.perf_counter()
                
                # --- STEP 1: Build host lattice with REAL chemical species ---
                start_time = time.perf_counter()
                self.grid_crystal = {}
                for site in self.structure:
                  idx = self.get_idx_coords(site.coords, self.basis_vectors)
                  self.grid_crystal[idx] = Site(
                    chemical_specie=site.specie.symbol,
                    position=tuple(site.coords),
                    site_type=site.specie.symbol,
                    Act_E_dict=copy.deepcopy(self.Act_E_dict), # Its own copy
                    defects_config = self.defects_config,
                    reactions_config = self.reactions_config
                  )
                step1_time = time.perf_counter() - start_time
                print(f"Step 1 (Build host lattice): {step1_time:.4f} seconds")
                  
                # --- STEP 2: Handle boundary sites (if needed) ---
                start_time = time.perf_counter()
                self._handle_missing_neighbors(radius_neighbors, affected_site)
                step2_time = time.perf_counter() - start_time
                print(f"Step 2 (Boundary sites): {step2_time:.4f} seconds")     
                
                # --- STEP 3: Add interstitial/hollow sites ---
                start_time = time.perf_counter()
                interstitial_count = 0
                if mode == "interstitial":
                  interstitial_sites = self._generate_interstitial_sites(api_key)
                  for pos in interstitial_sites:
                    idx = self.get_idx_coords(pos, self.basis_vectors)
                    #print(f'Is {idx} in grid_crystal already? {idx in self.grid_crystal}')
                    if idx not in self.grid_crystal:
                      self.grid_crystal[idx] = Site(
                        chemical_specie=affected_site,
                        position=tuple(pos),
                        site_type="interstitial",
                        Act_E_dict=copy.deepcopy(self.Act_E_dict),
                        defects_config = self.defects_config,
                        reactions_config = self.reactions_config
                      )
                      interstitial_count += 1
                step3_time = time.perf_counter() - start_time
                print(f"Step 3 (Interstitial sites): {step3_time:.4f} seconds")
                
                print(f"Total sites created: {len(self.grid_crystal)} "
                  f"({len(self.structure)} host + {interstitial_count} interstitial)")
                
                
                # --- STEP 4: Initialize migration pathways from grid ---
                start_time = time.perf_counter()
                self._build_kdtree()
                self._initialize_migration_pathways(radius_neighbors)
                step4_time = time.perf_counter() - start_time
                print(f"Step 4 (Migration pathways): {step4_time:.4f} seconds")

              
                # --- STEP 5: Neighbor analysis (uses FULL grid) ---
                start_time = time.perf_counter()
                print('Starting sequencial neighbor')
                self._sequencial_neighbors_analysis()
                step5_time = time.perf_counter() - start_time
                print(f"Step 5 (Neighbor analysis): {step5_time:.4f} seconds") 
                end_time = time.perf_counter()
                
                total_time = time.perf_counter() - total_start_time
                print(f"\n{'='*60}")
                print(f"TOTAL INITIALIZATION TIME: {total_time:.4f} seconds")
                print(f"{'='*60}")
                print(f"Breakdown:")
                print(f"  Step 1: {step1_time:.4f}s ({step1_time/total_time*100:.1f}%)")
                print(f"  Step 2: {step2_time:.4f}s ({step2_time/total_time*100:.1f}%)")
                print(f"  Step 3: {step3_time:.4f}s ({step3_time/total_time*100:.1f}%)")
                print(f"  Step 4: {step4_time:.4f}s ({step4_time/total_time*100:.1f}%)")
                print(f"  Step 5: {step5_time:.4f}s ({step5_time/total_time*100:.1f}%)")
                      
        
                if use_mpi:
                    # Prepare data for broadcasting
                    grid_data = {
                        'grid_crystal': self.grid_crystal,
                        'domain_height': self.domain_height
                    }
                else:
                    grid_data = None
                
            else:
                # Only reached when using MPI and rank != 0
                grid_data = None
                self.migration_pathways = {}
                
            # Broadcast only if using MPI
            if use_mpi:
                grid_data = comm.bcast(grid_data,root = 0)
                
                if rank != 0:
                    self.grid_crystal = grid_data['grid_crystal']
                    self.domain_height = grid_data['domain_height']
                        
                
        else:
            # Loading existing grid
            #import copy
            self.grid_crystal = grid_crystal
            self.domain_height = self.crystal_size[2]
            
            # Initialize pathways for loaded grids too 
            self._build_kdtree()
            self._initialize_migration_pathways(radius_neighbors)  
            
        # If we include grain boundaries, we should modify the activation energies
        if hasattr(self, 'gb_configurations'):
          self.gb_model = GrainBoundary(self.crystal_size,self.gb_configurations)
          for site in self.grid_crystal.values():
            self.gb_model.modify_act_energy_GB(site,self.migration_pathways,self.defects_config,self.reactions_config)
            
     
    def _get_applicable_defects_for_site(self,site_type):
      """
      Determine which defect configurations apply to this site.
      Called once during site initialization.
      """
      applicable_defects = []  
      # Check all defect configurations
      for defect_name, cfg in self.defects_config.items():
        allowed_sublattices = cfg.get("allowed_sublattices",[])
        if site_type in allowed_sublattices:
          applicable_defects.append(defect_name)
          
      return applicable_defects
        
           
            
    def _generate_interstitial_sites(self,api_key):
      """
      Generate interstitial sites using the actual species being simulated.
      """
      
      # Get the interstitial species from defect_config
      interstitial_species = None
      for name, cfg in self.defects_config.items():
        if cfg.get('site_type') == 'interstitial':
          interstitial_species = cfg['symbol']
          break
          
      if interstitial_species is None:
        raise ValueError("No interstitial species found in defects_config")
        
      try:
        with MPRester(api_key) as mpr:
          chgcar = mpr.get_charge_density_from_material_id(self.id_material)
          cig = ChargeInterstitialGenerator()
          defects = cig.generate(chgcar, insert_species=[interstitial_species])
          
          defect_struct = next(defects).defect_structure
          interstitial_pos = None
          
          # Find the interstitial atom
          for site in reversed(defect_struct):
            if site.specie.symbol == interstitial_species:
              interstitial_pos = site.coords
              break
              
          if interstitial_pos is None:
            raise ValueError(f"Could not find {interstitial_species} interstitial position")
        
      except Exception as e:
        print(f'Warning: Could not get interstitial position for {interstitial_species}: {e}')
        # Fallback to reasonable default
        interstitial_pos = np.array(self.structure.lattice.abc) / 2.0
      
      unit_cell_lattice = self.structure_basic.lattice
          
      # Convert interstitial position to fractional coordinates in UNIT CELL
      frac_pos = unit_cell_lattice.get_fractional_coords(interstitial_pos)
      
      try:
       # Generate all symmetry-equivalent positions in the unit cell
        sga = SpacegroupAnalyzer(self.structure_basic)
        sg = sga.get_space_group_operations()
        equivalent_frac = set()
          
        for op in sg:
          transformed_frac = op.operate(frac_pos)
          # Wrap to [0,1)
          transformed_frac = np.mod(transformed_frac, 1.0)
          # Round to avoid floating point issues
          transformed_frac = tuple(np.round(transformed_frac, 6))
          equivalent_frac.add(transformed_frac)
        
        base_positions_unit_cell = [unit_cell_lattice.get_cartesian_coords(frac) for frac in equivalent_frac]

        
      except Exception:
        # Fallback: use the single position found
        base_positions_unit_cell = [interstitial_pos]
        
      # Replicate in supercell
      supercell_interstitials = []
      lattice = self.structure.lattice
      repetitions = np.ceil(np.array(self.crystal_size) / np.array(unit_cell_lattice.abc)).astype(int)
      
      tol_boundary = 1e-6
      for cart_pos in base_positions_unit_cell:
        for i in range(repetitions[0]):
          for j in range(repetitions[1]):
            for k in range(repetitions[2]):
              offset = (i * unit_cell_lattice.matrix[0] +
                        j * unit_cell_lattice.matrix[1] +
                        k * unit_cell_lattice.matrix[2])
              new_pos = cart_pos + offset
              
              if (-tol_boundary <= new_pos[0] <= self.crystal_size[0] + tol_boundary and
                  -tol_boundary <= new_pos[1] <= self.crystal_size[1] + tol_boundary and
                  -tol_boundary <= new_pos[2] <= self.crystal_size[2] + tol_boundary):
                  supercell_interstitials.append(new_pos)
                  
      # Remove duplicates
      unique_positions = []
      tol = 0.1
      for pos in supercell_interstitials:
        is_duplicate = False
        for existing in unique_positions:
          if np.linalg.norm(np.array(pos) - np.array(existing)) < tol:
            is_duplicate = True
            break
        if not is_duplicate:
          unique_positions.append(pos)
            
      return unique_positions

           
    def _handle_missing_neighbors(self,radius_neighbors, affected_site):
        """
        Identify and add missing neighbor sites to the grid_crystal.
        Some neighbor sites weren't initially created during grid initialization using structure. 
        This method finds these missing sites and adds them to ensure complete neighbor analysis.
        
        Args:
        radius_neighbors (float): Neighbor search radius
        affected_site: Site type for newly created sites
        """
        tol = 1e-6
        domain_height = tol

        for site in self.structure:
            # Neighbors for each idx in grid_crystal
            neighbors = self.structure.get_neighbors(site,radius_neighbors)

            # Some sites are not created with the dictionary comprenhension
            # If the sites have neighbors that are within the crystal dimension range
            # but not included, we included
            for neigh in neighbors:
              pos = neigh.coords
              idx = self.get_idx_coords(pos,self.basis_vectors)
              
              if (-tol <= pos[2] <= self.crystal_size[2] + tol): 
                # Select the highest point of the domain
                if pos[2] > domain_height:
                  domain_height = pos[2]
            
                if idx not in self.grid_crystal:
                  pos_aux = (pos[0] % self.crystal_size[0], 
                            pos[1] % self.crystal_size[1], 
                            pos[2])
                    
                    # If not in the boundary region, where we should apply periodic boundary conditions
                  if tuple(pos) == pos_aux:
                    self.grid_crystal[idx] = Site(
                      chemical_specie = neigh.specie.symbol,
                      position = tuple(pos),
                      site_type = neigh.specie.symbol,
                      Act_E_dict = copy.deepcopy(self.Act_E_dict),
                      defects_config = self.defects_config,
                      reactions_config = self.reactions_config
                    )
                        
        self.domain_height = domain_height
    
                
    def _parallel_neighbors_analysis(self, num_cores=4):
      """Parallel neighbor analysis using grid_crystal."""
      import concurrent.futures
      import math
      
      try:
        from itertools import batched
      except ImportError: # Python <3.12
        import itertools
        def batched(iterable,n):
          it = iter(iterable)
          while True:
            batch = list(itertools.islice(it,n))
            if not batch:
              break
            yield batch
      
      grid_keys = list(self.grid_crystal.keys())
      batch_size = math.ceil(len(grid_keys) / num_cores)
      batches = list(batched(grid_keys, batch_size))
      
      # Shared data (immutable)
      shared_data = {
        'crystal_size': self.crystal_size,
        'event_labels': self.event_labels,
        'radius_neighbors': self.radius_neighbors
      }
      
      with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [
          executor.submit(
            self._process_batch_sites_worker,
            batch,
            self.grid_crystal,
            shared_data
          )
          for batch in batches
        ]
        concurrent.futures.wait(futures)
 
                
    def _sequencial_neighbors_analysis(self):
      """Sequential neighbor analysis using FULL grid_crystal."""
      
      for site_idx in self.grid_crystal.keys():
        site = self.grid_crystal[site_idx]
        
        # Get neighbors using k-d tree
        neighbor_site_indices = self._get_neighbors_for_site(site_idx, self.radius_neighbors)
        
        if site_idx in neighbor_site_indices:
          neighbor_site_indices.remove(site_idx)
        
        # Convert to positions
        neighbor_positions = [self.grid_crystal[idx].position for idx in neighbor_site_indices]  
      
        site.neighbors_analysis(
          self.grid_crystal,
          neighbor_site_indices,
          neighbor_positions,
          self.crystal_size,
          self.event_labels,
          site_idx
        )  
            

    def get_num_cores(self,local_max_cores=6):
        # Get number of cores in SLURM, PBS or local machine
        
        # HPC: SLURM
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            try:
                cores = int(os.environ['SLURM_CPUS_PER_TASK'])
                return max(1, cores - 1)  # Reserve 1 for system/OpenMP
            except (ValueError, TypeError):
                pass
        
        # HPC: PBS
        if 'PBS_NUM_PPN' in os.environ:
            try:
                cores = int(os.environ['PBS_NUM_PPN'])
                # Reserve 1 core for OpenMP/system processes
                return max(1, cores - 1)
            except ValueError:
                pass
            
        # Fallback to OMP_NUM_THREADS if set
        if 'OMP_NUM_THREADS' in os.environ:
            try:
                cores = int(os.environ['OMP_NUM_THREADS'])
                return max(1, cores - 1)
            except ValueError:
                pass
            
        # Local machine:
        try:
            import psutil
            cores = psutil.cpu_count(logical=True)
            if cores is None:
                cores = os.cpu_count() or 1
                
        except ImportError:
            cores = os.cpu_count() or 1
        return min(cores, local_max_cores)
        
    def _process_batch_sites_worker(self, batch_keys, grid_crystal, shared_data):
      """
      Worker using FULL grid for neighbor search.
      
      Args:
            batch_keys: List of site indices to process
            batch_sites: Dict of sites to modify (copies)
            full_grid: Complete grid for neighbor lookups (read-only)
            shared_data: Dict with structure, etc.

        Returns:
            Dict of {idx: modified_site} for sites in this batch
      """
      crystal_size = shared_data['crystal_size']
      event_labels = shared_data['event_labels']
      radius_neighbors = shared_data['radius_neighbors']

      for idx in batch_keys:
        site = grid_crystal[idx]
        
        # Find neighbors within radius
        site_pos = np.array(site.position)
        neighbors_idx = []
        neighbors_positions = []
        
        for neigh_idx, neigh_site in grid_crystal.items():
          if neigh_idx == idx:
            continue
          dist = np.linalg.norm(np.array(neigh_site.position) - site_pos)
          if dist <= radius_neighbors:
            neighbors_idx.append(neigh_idx)
            neighbors_positions.append(neigh_site.position)
        
        # Modify original object in-place
        site.neighbors_analysis(
          grid_crystal,
          neighbors_idx,
          neighbors_positions,
          crystal_size,
          event_labels,
          idx  
        )

        
# =============================================================================
# Get coordinates of particles for solving Poisson and points to evaluate electric field
# =============================================================================
    def save_electric_bias(self,V):
      self.V = V
    
    def get_evaluation_points(self):
      # Get charge locations and charges from System_state
      particle_locations, charges = self._extract_particles_charges()
      gen_site_locations = self._extract_generation_site_location()
                    
      if len(particle_locations) > 0 : # In case there is no particles
        E_field_points = np.concatenate([particle_locations,gen_site_locations],axis = 0)
      else:
        E_field_points = gen_site_locations
        
      return particle_locations, charges, E_field_points
        
    def _extract_particles_charges(self):
        """
        Extract charge locations and magnitudes from System_state.
        """
        particle_locations = []
        charges = []
        
        for site in self.sites_occupied:
          particle_locations.append(self.grid_crystal[site].position)
          charges.append(self.grid_crystal[site].ion_charge * constants.e * self.screening_factor)
           
        if len(particle_locations) == 0:
          particle_locations = np.empty((0, 3), dtype=np.float64)
          charges = np.empty((0,), dtype=np.float64)
        else:
          particle_locations = np.array(particle_locations,dtype=np.float64)
          charges = np.array(charges, dtype=np.float64)
         
        return particle_locations, charges 
        
    def _extract_generation_site_location(self):
        """
        Extract generation site locations from System_state.
        """
        gen_site_locations = []
        
        for site in self.adsorption_sites:
            gen_site_locations.append(self.grid_crystal[site].position)
        
        if len(gen_site_locations) == 0:
          return np.empty((0, 3), dtype=np.float64)
        else:
          return np.array(gen_site_locations,dtype=np.float64)
            
                
    
    def get_idx_coords(self, coords,basis_vectors):
            # Check if the coordinates are already in the cache
            coords_tuple = tuple(coords)
            if coords_tuple not in self.coord_cache:
                # Calculate and cache the rounded coordinates
                # np.linalg.solve --> To obtain the linear combination of the basis vector for the site coordinate
                idx_coords = np.linalg.solve(basis_vectors.transpose(), coords)
                idx_coords = tuple(np.round(idx_coords).astype(int))
                self.coord_cache[coords_tuple] = idx_coords
            return self.coord_cache[coords_tuple]


    def Wulff_Shape(self,api_key):
        
        with MPRester(api_key=api_key) as mpr:
        
            # Check what attributes are available in mpr
            # available_attributes = [attr for attr in dir(mpr) if not attr.startswith('_')]
            # print(f"Available attributes in MPRester: {available_attributes}")
            # surface_properties_doc = mpr.surface_properties.search(material_ids=[self.id_material])

            surface_properties_doc = mpr.materials.surface_properties.search(
                material_ids=self.id_material
                )
        
        miller_indices = []
        surface_energies = []
        for surface in surface_properties_doc[0].surfaces:
            miller_indices.append(tuple(surface.miller_index))
            surface_energies.append(surface.surface_energy)
            
        self.wulff_shape = WulffShape(self.structure_basic.lattice, miller_indices,surface_energies)
        
        # We can show the Wulff Shape with the following:
        #self.wulff_shape.show()
        
        self.wulff_facets = [] # Miller index and normal vector
        for facet in self.wulff_shape.facets:
            self.wulff_facets.append([facet.miller,facet.normal])
            
        # I can still eliminate the parallel normal vectors
        self.wulff_facets = sorted(self.wulff_facets,key = lambda x:x[0][0])
        
        
    def create_edges(self,facets_type):
        
        """
        To obtain the relation between the migration, the edges and the facets we first need:
            1. Calculate the edges
            2. Calculate the facets parallel to each migration
            3. Calculate the edges parallel to each migration
            4. Associate each edge with a facet type
            5. Relation between migration direction, edges and facet type
            
            Create a dictionary:
                dir_edge_facets[migration_label] = aux_edge_facet
                aux_edge_facet - list of lists:
                    aux_edge_facet[0][0] = (4, 11) migrations defining the edge
                    aux_edge_facet[0][1] = (1,0,0) facets
            
        """
        
        for idx,site in self.grid_crystal.items():
            if ((self.crystal_size[0] * 0.45 < site.position[0] < self.crystal_size[0] * 0.55) 
                and (self.crystal_size[1] * 0.45 < site.position[1] < self.crystal_size[1] * 0.55)
                and site.position[2] < self.crystal_size[1] * 0.2):
                break            # Introduce specie in the site
        
        # Obtain the different edge in the plane
        # Neighbors only in plane
        neighbors = [[self.grid_crystal[neigh[0]].position,neigh[1]] for neigh in self.grid_crystal[idx].migration_paths['Plane']]
        # Minimum distance between neighbors
        min_dist = np.linalg.norm(np.array(self.grid_crystal[idx].position) - np.array(np.array(neighbors[4][0])))
        edges = {}
        
        for neighbor in neighbors:
            for j in range(len(neighbors)):
                if (math.isclose(np.linalg.norm(np.array(neighbor[0]) - np.array(np.array(neighbors[j][0]))), min_dist)) and ((neighbors[j][1],neighbor[1]) not in 
                                                                                                                              edges):
                    edges[(neighbor[1],neighbors[j][1])] = np.array(neighbor[0]) - np.array(np.array(neighbors[j][0]))
        
        # Calculate the facets that are parallel to each migration
        mig_directions = {neigh[1]:np.array(self.grid_crystal[neigh[0]].position) - np.array(self.grid_crystal[idx].position) for neigh in self.grid_crystal[idx].migration_paths['Plane']}
        mig_parallel_facets = {}
        #Search for the facets that are parallel to the migration direction
        for mig_direct,vector in mig_directions.items():
            facet_list = []
            for facet in self.wulff_facets:
                if (facet[1][2] > 0 and facet[1][2] != 1 and # Screen facets that are looking downward or parallel to the x-y plane
                    abs(np.dot(facet[1][:2],vector[:2])) < 1e-12): # Perpendicular between facet normal vector (x-y) and migration direction
                        facet_list.append(facet)        
                    
            mig_parallel_facets[mig_direct] = facet_list
            
                    
        # Calculate the edges parallel to the migration
        parallel_mig_direction_edges = {}
        for mig,vector_dir in mig_directions.items():
            list_edges = []

            for edge,vector in edges.items():
                if np.linalg.norm(np.cross(vector,vector_dir)) < 1e-12: # Migration vector is parallel to the edges
                    list_edges.append(edge)
                
            parallel_mig_direction_edges[mig] = list_edges
            

        # Associate edge with facets
        # Edge is defined by two migrations: we sum those vectors to obtain a vector that should be parallel to the facet normal vector
        self.dir_edge_facets = {}
        for mig,edges_2 in parallel_mig_direction_edges.items():
            aux_edge_facet = []
            for edge in edges_2:
                v1 = mig_directions[edge[0]] + mig_directions[edge[1]]
                
                for facet in mig_parallel_facets[mig]:
                    if np.dot(v1,facet[1]) > 0: # Pointing in the same direction
                       aux_edge_facet.append([edge,facet[0]])
                       
            self.dir_edge_facets[mig] = aux_edge_facet
            
        self.dir_edge_facets = {
            key: [sublist for sublist in value if sublist[1] in facets_type]
            for key, value in self.dir_edge_facets.items()
        }

        
    def available_generation_sites(self, support_update_sites = set()):
        
        update_gen_sites = set()
        # Generation of vacancy in the bulk
        if self.mode == 'vacancy': return


                
                    
        # Normal deposition process: gas-substrate interface
        elif self.mode == 'regular':
        
            """
            Deprecating: 2025/10/03
            
            if not update_supp_av:
              # The sites in contact with the substrate or supported by more than 2 particles
              self.adsorption_sites = [
                  idx for idx, site in self.grid_crystal.items()
                  if (self.sites_generation_layer in site.supp_by or len(site.supp_by) > 2) and site.chemical_specie == self.affected_site
              ]
            """
            
            adsorption_sites_set = set(self.adsorption_sites)

            for idx in support_update_sites:
                site = self.grid_crystal[idx]

                if idx in adsorption_sites_set:
                    if ((self.sites_generation_layer not in site.supp_by and len(site.supp_by) < 3) or (site.chemical_specie != self.affected_site)):
                        self.adsorption_sites.remove(idx)
                        site.remove_event_type(self.num_event-1)
                    
                else:
                    if (self.sites_generation_layer in site.supp_by or len(site.supp_by) > 2) and site.chemical_specie == self.affected_site:
                        self.adsorption_sites.append(idx)
                        site.deposition_event(self.TR_gen,idx,self.num_event-1,self.Act_E_gen)
           
           
                        
        # Oxidation reaction at the electrode-dielectric interface
        elif self.mode == 'interstitial':
                            
          adsorption_sites_set = set(self.adsorption_sites)

          for idx in support_update_sites:
                site = self.grid_crystal[idx]

                if idx in adsorption_sites_set:
                    if (site.chemical_specie != self.affected_site):
                        self.adsorption_sites.remove(idx)
                        site.remove_event_type(self.num_event-1)
                    
                else:
                    if (self.sites_generation_layer in site.supp_by) and (site.chemical_specie == self.affected_site):
                        self.adsorption_sites.append(idx)
                        site.ion_generation_interface(idx)
                        update_gen_sites.add(idx)
                        
          return update_gen_sites  
       
    
    # ====================================================================
    #  Super basin helper methods
    #
    #  REMOVE THE SUPERBASIN CHECK FROM THE MAIN SCRIPT 
    # ====================================================================
    
    def should_activate_superbasin(self,kmc_time_step):
        """
        Determine if superbasin approach should be activated based on system state
        
        Superbasins are only useful when:
          1. A stable, percolating filament exists
          2. The system is trapped in shallow wells
            
        Returns:
        --------
        bool : True if superbasin should be activated
        """ 
        
        # Condition 1: Must have a percolating filament
        if not self.is_filament_percolating():
          return False
        
        # Condition 2: Must be in a trapped regime
        if self.time_based_superbasin:
          # Memristor switching: time-based superbasin activation
          return self._check_time_based_superbasin(kmc_time_step)
        else:
          # Deposition: event-based superbasin activation
          return self._check_event_based_superbasin()
          
        return False
    
    def is_filament_percolating(self):
      return any(
        cluster.attached_layer.get('bottom_layer') and
        cluster.attached_layer.get('top_layer')
        for cluster in self.clusters.values()
      )
        
    def _check_event_based_superbasin(self):
        """
        Check superbasin activation for deposition (based on system changes)
        """
        # Track occupied sites count
        current_occupied = len(self.sites_occupied)
        self.superbasin_tracker.append(current_occupied)
        
        # Keep only recent history
        if len(self.superbasin_tracker) > self.n_search_superbasin:
          self.superbasin_tracker.pop(0)
          
        # Check if system has been static
        if len(self.superbasin_tracker) >= self.n_search_superbasin:
          recent_mean = np.mean(self.superbasin_tracker[-self.n_search_superbasin:])
          if abs(recent_mean - current_occupied) < 1e-10: # No change
            self.nothing_happen_count += 1
          else:
            self.nothing_happen_count = 0
            
            # Adjust energy minimum
            if self.E_min - self.energy_step > 0:
              self.E_min -= self.energy_step
            else:
              self.E_min = 0
              
        
        # Check if superbasin should be activated
        if self.nothing_happen_count == self.n_search_superbasin:
          return True
          
        elif (self.nothing_happen_count > 0 and self.nothing_happen_count % self.n_search_superbasin == 0):
          # Gradually increase E_min back
          if self.E_min_lim_superbasin >= self.E_min + self.energy_step:
            self.E_min += self.energy_step
          else:
            self.E_min = self.E_min_lim_superbasin
          return True
          
        return False
        
    def _check_time_based_superbasin(self,kmc_time_step):
      """
      Check superbasin activation for memristor switching (based on time)
      """
      self.superbasin_tracker.append(kmc_time_step)
      if len(self.superbasin_tracker) > self.n_search_superbasin:
        self.superbasin_tracker.pop(0)
        
      # Check if current step is slow
      is_slow_step = self._slow_timesteps()
      
      if is_slow_step:
        self.nothing_happen_count += 1
        
        if self.nothing_happen_count >= self.n_search_superbasin:
          self.nothing_happen_count = 0
          return True
      else:
        # Reset counter when we get a larger timestep
        self.nothing_happen_count = 0
      
      return False
      
      
    def _slow_timesteps(self):  
      """
      Determine if a timestep is considered "slow" (system barely advances)
      
      Small timesteps indicate the system is evolving slowly and may be 
      stuck in metastable states, making it a good candidate for superbasin.
      """
      
      # Need to ensure we have enough data points
      if len(self.superbasin_tracker) < self.n_search_superbasin:
        return False  # Not enough data yet
        
      recent_mean_timestep = np.mean(self.superbasin_tracker[-self.n_search_superbasin:])
      
      # Small timestep = slow evolution -> Candidate for superbasin
      # Option 1: Absolute threshold
      if recent_mean_timestep < self.time_step_limits:
        return True
        
      # Option 2: Relative threshold (alternative)
      # if recent_mean < 0.1 * self.voltage_update_time:  # 10% of voltage update interval
      #     return True
      return False
    
        
                    
    def transition_rate_adsorption(self,experimental_conditions):
# =============================================================================
#         Kim, S., An, H., Oh, S., Jung, J., Kim, B., Nam, S. K., & Han, S. (2022).
#         Atomistic kinetic Monte Carlo simulation on atomic layer deposition of TiN thin film. 
#         Computational Materials Science, 213. https://doi.org/10.1016/j.commatsci.2022.111620
# =============================================================================
        
        # Maxwell-Boltzman statistics for transition rate of adsorption rate
        sticking_coeff, partial_pressure, T = experimental_conditions['sticking_coeff'], experimental_conditions['partial_pressure'], experimental_conditions['T'] 
        self.mass_specie = Element(self.chemical_specie).atomic_mass

        # The mass in kg of a unit of the chemical specie
        self.mass_specie = self.mass_specie / constants.Avogadro / 1000
        
        
        n_sites_layer_0 = 0
        for site in self.grid_crystal.values():
            if site.position[2] <= 1e-6:
                n_sites_layer_0 += 1


        # Area per site = total area of the crystal / number of sites available at the bottom layer
        # Area in m^2
        area_specie = 1e-18 *self.crystal_size[0] * self.crystal_size[1] / n_sites_layer_0
        
        # Boltzmann constant (m^2 kg s^-2 K^-1)
        self.TR_gen = sticking_coeff * partial_pressure * area_specie / np.sqrt(2 * constants.pi * self.mass_specie * constants.Boltzmann * T)
    
        # Activation energy for deposition
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E12;  # nu0 (s^-1) bond vibration frequency
        self.Act_E_gen = -np.log(self.TR_gen/nu0) * kb * self.temperature
        
    def limit_kmc_timestep(self,P_limits):
        
        self.timestep_limits = -np.log(1-P_limits)/self.TR_gen
        
        
    def defect_gen(self,rng):
        
        sites_needing_support_update = set()
        sites_needing_event_update = set()
        
        for defect in self.defects_config.values():
          for idx,site in self.grid_crystal.items():
              
              if site.site_type in defect['allowed_sublattices'] and site.chemical_specie in defect["valid_target_species"]:
                if self.gb_model.is_site_in_grain_boundary(site.position):
                  probability = defect['initial_concentration_GB']
                else:
                  probability = defect['initial_concentration_bulk']
                  
                if rng.random() < probability:
                  chemical_specie = defect['symbol']
                  ion_charge = defect['charge']   
                  self._introduce_specie_site(idx,sites_needing_support_update, sites_needing_event_update,chemical_specie,ion_charge)

          # Update sites availables, the support to each site and available migrations
          self.update_sites(sites_needing_support_update, sites_needing_event_update)

    def deposition_specie(self,t,rng,test = 0):  

        support_update_sites = set()
        event_update_sites = set()
        
        if test == 0:
            
            P = 1-np.exp(-self.TR_gen*t) # Adsorption probability in time t
            # Indexes of sites availables: supported by substrates or other species
            for idx in self.adsorption_sites:
                if rng.random() < P:   
                    # Introduce specie in the site
                    update_specie_events,support_update_sites = self.introduce_specie_site(idx,support_update_sites, event_update_sites,self.grid_crystal[idx].ion_charge)
            
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,support_update_sites)

            

        # Single particle in a determined place
        elif test == 1:
            
            # Compute geometric center of the domain
            center = np.array(self.crystal_size) * [0.5,0.5,0.5]  # assumes crystal_size = [Lx, Ly, Lz]
            min_dist = float('inf')
            central_site = None
            central_idx = None
            
            for idx, site in self.grid_crystal.items():
                pos = np.array(site.position)
                dist = np.linalg.norm(pos - center)
                #print(f'Dist {dist} and min. dist. {min_dist}')
                if dist < min_dist:
                    min_dist = dist
                    central_site = site
                    central_idx = idx   
                    
                     
            # Introduce specie in the site
            defect = 'hydrogen_interstitial'
            migrating_charge = self.defects_config[defect]['charge']
            chemical_specie = self.defects_config[defect]['symbol']
            self._introduce_specie_site(central_idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,support_update_sites)
                
            print('Particle in position: ',central_site.position, ' is a ', central_site.chemical_specie)
            print('Neighbors of that particle: ', central_site.nearest_neighbors_idx)
            print('Neighbors are supported by: ')
            for idx_3 in central_site.nearest_neighbors_idx:
                print(self.grid_crystal[idx_3].supp_by)
                
                
        # Two adjacent particles
        elif test == 2:
            defect = 'hydrogen_interstitial'
            migrating_charge = self.defects_config[defect]['charge']
            chemical_specie = self.defects_config[defect]['symbol']  
            site_type = self.defects_config[defect]['site_type']
            
            for idx,site in self.grid_crystal.items():
              if ((self.crystal_size[0] * 0.45 < site.position[0] < self.crystal_size[0] * 0.55) 
                and (self.crystal_size[1] * 0.45 < site.position[1] < self.crystal_size[1] * 0.55)
                and (self.crystal_size[2] * 0.45 < site.position[2] < self.crystal_size[2] * 0.55)) and site.site_type == site_type: 
                break
            
            # Introduce specie in the site
            self._introduce_specie_site(idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)

            # Update sites availables, the support to each site and available migrations
            self.update_sites(support_update_sites, event_update_sites)
            
            for neighbor_idx in self.grid_crystal[idx].nearest_neighbors_idx:
              neighbor = self.grid_crystal[neighbor_idx]
              
              if neighbor.site_type == 'O':
                target_idx = neighbor_idx
            
            defect = 'oxygen_vacancy'
            migrating_charge = self.defects_config[defect]['charge']
            chemical_specie = self.defects_config[defect]['symbol']  
            # Introduce specie in the neighbor site
            self._introduce_specie_site(target_idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(support_update_sites, event_update_sites)
            

        # Full coordinated atom --> Cluster
        elif test == 3:
            
            # Compute geometric center of the domain
            center = np.array(self.crystal_size) * [0.5,0.5,0.5]  # assumes crystal_size = [Lx, Ly, Lz]
            min_dist = float('inf')
            central_site = None
            central_idx = None
            
            for idx, site in self.grid_crystal.items():
                pos = np.array(site.position)
                dist = np.linalg.norm(pos - center)
                #print(f'Dist {dist} and min. dist. {min_dist}')
                if dist < min_dist:
                    min_dist = dist
                    central_site = site
                    central_idx = idx   
                    
                     
            # Introduce specie in the site
            update_specie_events,support_update_sites = self.introduce_specie_site(central_idx,update_specie_events,support_update_sites,0)
            self.update_sites(update_specie_events,support_update_sites)
            self._add_metal_atom_to_clusters(central_idx)
            
            for migration_path in self.grid_crystal[central_idx].migration_paths.values():
              
              for neighbor in migration_path:
                  update_specie_events,support_update_sites = self.introduce_specie_site(neighbor[0],update_specie_events,support_update_sites,0)
                  self.update_sites(update_specie_events,support_update_sites)
                  print(f'Neighbor: {neighbor}')
                  self._add_metal_atom_to_clusters(neighbor[0])
                  

        # Cluster - particles in plane, one on bottom
        elif test == 4:
            
            min_dist_xy = float('inf')
            idx = None
            ion_charge = 0
          
            for site_idx in self.adsorption_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - self.crystal_size[0] * 0.5
              dy = pos[1] - self.crystal_size[1] * 0.5
              dist_xy = np.sqrt(dx**2 + dy**2)
              if dist_xy < min_dist_xy:
                min_dist_xy = dist_xy
                idx = site_idx

            # Introduce central atom at bottom center
            update_specie_events,support_update_sites = self.introduce_specie_site(idx,update_specie_events,support_update_sites,1)
            self.update_sites(update_specie_events,support_update_sites)

            # Add in-plane neighbors (same layer)
            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,support_update_sites = self.introduce_specie_site(neighbor[0],update_specie_events,support_update_sites,ion_charge)
                self.update_sites(update_specie_events,support_update_sites)
                
            idx_neighbor_top = self.grid_crystal[neighbor[0]].migration_paths['Down'][1][0]
            update_specie_events,support_update_sites = self.introduce_specie_site(idx_neighbor_top,update_specie_events,support_update_sites,ion_charge)
            self.update_sites(update_specie_events,support_update_sites)
        
        # Cluster - particles in plane, one on top
        elif test == 5:
            
            min_dist_xy = float('inf')
            idx = None
          
            for site_idx in self.adsorption_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - self.crystal_size[0] * 0.5
              dy = pos[1] - self.crystal_size[1] * 0.5
              dist_xy = np.sqrt(dx**2 + dy**2)
              if dist_xy < min_dist_xy:
                min_dist_xy = dist_xy
                idx = site_idx

            # Introduce central atom at bottom center
            update_specie_events,support_update_sites = self.introduce_specie_site(idx,update_specie_events,support_update_sites,1)
            self.update_sites(update_specie_events,support_update_sites)

            # Add in-plane neighbors (same layer)
            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,support_update_sites = self.introduce_specie_site(neighbor[0],update_specie_events,support_update_sites,1)
                self.update_sites(update_specie_events,support_update_sites)
                
            idx_neighbor_top = self.grid_crystal[neighbor[0]].migration_paths['Up'][1][0]
            update_specie_events,support_update_sites = self.introduce_specie_site(idx_neighbor_top,update_specie_events,support_update_sites,1)
            self.update_sites(update_specie_events,support_update_sites)
            
        # Cluster - 3 layers
        elif test == 6:


            min_dist_xy = float('inf')
            idx = None
            ion_charge = 0
          
            for site_idx in self.adsorption_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - self.crystal_size[0] * 0.5
              dy = pos[1] - self.crystal_size[1] * 0.5
              dist_xy = np.sqrt(dx**2 + dy**2)
              if dist_xy < min_dist_xy:
                min_dist_xy = dist_xy
                idx = site_idx
                
            # Introduce specie in the site
            update_specie_events,support_update_sites = self.introduce_specie_site(idx,update_specie_events,support_update_sites,ion_charge)
            self.update_sites(update_specie_events,support_update_sites)
            self._add_metal_atom_to_clusters(idx)

            # Add in-plane neighbors (same layer)
            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av,ion_charge)
                self.update_sites(update_specie_events,update_supp_av)
                self._add_metal_atom_to_clusters(neighbor[0])
                
            idx_neighbor_top = self.grid_crystal[idx].migration_paths['Down'][0][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_top,update_specie_events,update_supp_av,ion_charge)
            self.update_sites(update_specie_events,update_supp_av)
            self._add_metal_atom_to_clusters(idx_neighbor_top)
            
            for neighbor in self.grid_crystal[idx_neighbor_top].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av,ion_charge)
                self.update_sites(update_specie_events,update_supp_av)
                self._add_metal_atom_to_clusters(neighbor[0])
            
            idx_neighbor_top = self.grid_crystal[neighbor[0]].migration_paths['Down'][0][0]
            update_specie_events,update_supp_av = self.introduce_specie_site(idx_neighbor_top,update_specie_events,update_supp_av,ion_charge)
            self.update_sites(update_specie_events,update_supp_av)
            self._add_metal_atom_to_clusters(idx_neighbor_top)
            
            for neighbor in self.grid_crystal[idx_neighbor_top].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av,ion_charge)
                self.update_sites(update_specie_events,update_supp_av)
                self._add_metal_atom_to_clusters(neighbor[0])
            
        elif test == 7:
            
            update_supp_av = set()
            update_specie_events = set()
            
            for site_idx in self.adsorption_sites:
                if (self.crystal_size[0] * 0.45 < self.grid_crystal[site_idx].position[0] < self.crystal_size[0] * 0.55) and (self.crystal_size[1] * 0.45 < self.grid_crystal[site_idx].position[1] < self.crystal_size[1] * 0.55):
                    idx = site_idx
                    break
            # Introduce specie in the site
            update_specie_events,update_supp_av = self.introduce_specie_site(idx,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)

            # Cluster in contact with the substrate
            for neighbor in self.grid_crystal[idx].migration_paths['Plane']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
            # Particle next to the cluster --> Select a site that is not occupied
            for idx_neighbor_plane in self.grid_crystal[neighbor[0]].migration_paths['Plane']:
                if idx_neighbor_plane[0] not in self.sites_occupied: break
            
            update_specie_events,update_supp_av = self.introduce_specie_site(tuple(idx_neighbor_plane[0]),update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
                
            # Cluster over the copper
            for neighbor in self.grid_crystal[idx].migration_paths['Up']:
                update_specie_events,update_supp_av = self.introduce_specie_site(neighbor[0],update_specie_events,update_supp_av)
                self.update_sites(update_specie_events,update_supp_av)
                
        elif test == 8:
            from collections import deque

            # Create a deque object for the queue
            queue = deque()
            for site_idx in self.adsorption_sites:
                if (self.crystal_size[0] * 0.45 < self.grid_crystal[site_idx].position[0] < self.crystal_size[0] * 0.55) and (self.crystal_size[1] * 0.45 < self.grid_crystal[site_idx].position[1] < self.crystal_size[1] * 0.55):
                    idx = site_idx
                    break
            queue.append(idx)
            visited = set()
            cluster_size = 29
            
            self.bfs_cluster(queue,visited,cluster_size)
            
        elif test == 9:
            from collections import deque

            # Create a deque object for the queue
            queue = deque()
            for site_idx in self.adsorption_sites:
                if (self.crystal_size[0] * 0.45 < self.grid_crystal[site_idx].position[0] < self.crystal_size[0] * 0.55) and (self.crystal_size[1] * 0.45 < self.grid_crystal[site_idx].position[1] < self.crystal_size[1] * 0.55):
                    idx = site_idx
                    break
            queue.append(idx)
            visited = set()
            cluster_size = 29
            
            self.bfs_cluster(queue,visited,cluster_size)
            
            ad_sites_aux = self.adsorption_sites.copy()
            for site_idx in ad_sites_aux:
                if self.grid_crystal[site_idx].position[2] > 0.1:
                    update_specie_events,update_supp_av = self.introduce_specie_site(site_idx,update_specie_events,update_supp_av)
                    self.update_sites(update_specie_events,update_supp_av)
                    
            ad_sites_aux = self.adsorption_sites.copy()
            for site_idx in ad_sites_aux:
                if self.grid_crystal[site_idx].position[2] > 2.2:
                    update_specie_events,update_supp_av = self.introduce_specie_site(site_idx,update_specie_events,update_supp_av)
                    self.update_sites(update_specie_events,update_supp_av)
            
            

            
    def processes(self,chosen_event):
 
        # Execute kMC event and update system state
        
        sites_needing_support_update = set()
        sites_needing_event_update = {chosen_event[-1]}
        
        if isinstance(chosen_event[2], int): # Migration event
          self._handle_migration_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )
        elif chosen_event[2] == 'generation':
          self._handle_generation_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )
        elif chosen_event[2] in ['reduction','oxidation']:
          self._handle_redox_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )    
        elif any(chosen_event[2] == reaction['name'] for reaction in self.reactions_config.values()): 

          self._handle_reaction_event(
            chosen_event, 
            sites_needing_support_update, 
            sites_needing_event_update
          )
        
        self.update_sites(sites_needing_support_update, sites_needing_event_update)
     
            
    def _get_mobile_sites(self, site_indices):
      """Filter only sites that can have mobile defects"""
      mobile_sites = []
      for idx in site_indices:
        site = self.grid_crystal[idx]
        if any(site.site_type in cfg.get("allowed_sublattices", []) for cfg in self.defects_config.values()):
          mobile_sites.append(idx)
      return mobile_sites
      
    def _is_at_top_electrode(self, site_idx):
      """ Check if site is at top electrode """
      return np.isclose(self.grid_crystal[site_idx].position[2], self.crystal_size[2])
      
    def _handle_migration_event(self, chosen_event, support_update_sites, event_update_sites):
      """Handle migration events with multi-species support."""
      source_idx = chosen_event[-1]
      dest_idx = chosen_event[1]
      
      # Check for removal at electrode
      if (self._is_at_top_electrode(dest_idx) and self.V < 0 and
          self.grid_crystal[source_idx].ion_charge != 0 and self.allow_specie_removal):
          self._remove_species_at_site(source_idx, support_update_sites, event_update_sites)
          if self.poissonSolver_parameters['solve_Poisson']:
            event_update_sites.update(self._get_mobile_sites(self.sites_occupied))
          return
          
      # Perform migration
      migrating_charge = self.grid_crystal[source_idx].ion_charge
      chemical_specie = self.grid_crystal[source_idx].chemical_specie
      self._introduce_specie_site(dest_idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)
      self._remove_species_at_site(source_idx, support_update_sites, event_update_sites)
      
      # Update Poisson-relevant sites
      if self.poissonSolver_parameters['solve_Poisson']:
        event_update_sites.update(self._get_mobile_sites(self.sites_occupied))
        
      # Handle cluster updates for neutral metal atoms
      if self.grid_crystal[dest_idx].ion_charge == 0:
        self._add_metal_atom_to_clusters(dest_idx)
        self._remove_metal_atom_from_clusters(source_idx)  
        
    def _handle_generation_event(self, chosen_event, support_update_sites, event_update_sites):
      """ Handle defect generation events """
      dest_idx = chosen_event[1]
      self._introduce_specie_site(dest_idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)
      
    def _handle_redox_event(self, chosen_event, support_update_sites, event_update_sites):
      """Handle redox events with multi-species support."""
      site_idx = chosen_event[1]
      site = self.grid_crystal[site_idx]
      
      if chosen_event[2] == 'reduction':
        site.ion_charge -= 1
        event_update_sites.add(site_idx)
        self._add_metal_atom_to_clusters(site_idx)
      
      elif chosen_event[2] == 'oxidation': 
        if self._is_at_top_electrode(site_idx) and self.V < 0:
          self._remove_species_at_site(site_idx, support_update_sites, event_update_sites)
        else:
          site.ion_charge += 1
          event_update_sites.add(site_idx)
        self._remove_metal_atom_from_clusters(site_idx)   
        
    def _handle_reaction_event(self,chosen_event, support_update_sites, event_update_sites):
      """ 
      Handler for reaction events 
      Reads products definitions from reactions_config to update site states
      """
      reaction_name_chosen = chosen_event[2]
      source_idx = chosen_event[-1]
      dest_idx = chosen_event[1]
      sites_involved = [source_idx,dest_idx]
      
      # Reaction definitions
      for reaction_name, reaction in self.reactions_config.items():
        if reaction['name'] == reaction_name_chosen:
          products = reaction['products']
          
      # Track sites that need kMC update
      for i, product in enumerate(products):
        # Determine which site this product applies to
        site_index = product.get('site_index',i)
        
        # Handle unimolecular reactions (only 1 site involved)
        if site_index >= len(sites_involved):
          continue
        
        target_idx = sites_involved[site_index]
        site = self.grid_crystal[target_idx]
        
        if product['symbol'] != 'Empty':
          defect = self._defect_by_name(product['symbol'])
          
          if 'charge' in defect:
            ion_charge = defect['charge']
            
          # Introduce species
          self._introduce_specie_site(
            target_idx,
            support_update_sites,
            event_update_sites,
            product['symbol'],
            ion_charge
          )
          
          # Handle passivation increment
          if 'passivation_increment' in product:
            site.passivation_level += product['passivation_increment']
            
          # Handle charge variation
          if 'charge_per_passivation' in defect:
            site.ion_charge += defect['charge_per_passivation']
          
          
        else:
          self._remove_species_at_site(
            target_idx, 
            support_update_sites, 
            event_update_sites
          )
          
    def _defect_by_name(self,symbol):
    
      for defect in self.defects_config.values():
        if defect['symbol'] == symbol:
          return defect
            
# =============================================================================
# At every kMC step we have to check if we destroy any superbasin                      
# =============================================================================
   
    def update_superbasin(self,chosen_event):
            
        # We dismantle the superbasin if the chosen_event affect some of the states
        # that belong to any of the superbasin
        keys_to_delete = [idx for idx, sb in self.superbasin_dict.items()
                          if chosen_event[1] in sb.superbasin_environment or
                          chosen_event[-1] in sb.superbasin_environment]

        for key in keys_to_delete:
            del self.superbasin_dict[key]
            
            
# =============================================================================
# Update sites: supporting particles, available generation sites and possible events                      
# =============================================================================
    
    def update_sites(self,support_update_sites, event_update_sites):
        """ Update the sites """    
        
        # Update support relationship only for relevant sites
        if support_update_sites:
            mobile_support_sites = self._get_mobile_sites(support_update_sites)
            # There are new sites supported by the new species
            # For loop over neighbors
            for idx in mobile_support_sites:
                self.grid_crystal[idx].supported_by(
                  self.grid_crystal, self.wulff_facets, self.dir_edge_facets,
                  self.domain_height,idx
                )
        
        # Update generation sites
        for defect_name, defect in self.defects_config.items():
          if 'generation' in defect['enabled_events']:          
            generation_sites = self.available_generation_sites(support_update_sites)

        # Update event pathways for mobile sites
        if event_update_sites: 
            # Sites are not available because a particle has migrated there
            mobile_event_sites = self._get_mobile_sites(event_update_sites)
            for idx in mobile_event_sites:
                self.grid_crystal[idx].available_pathways(
                  self.grid_crystal,idx,self.facets_type
                )
                self.grid_crystal[idx].transition_rates()
        

        # Update generation site rates
        if not (self.poissonSolver_parameters['solve_Poisson'] and platform.system() == 'Linux'):
        # We need Linux to solve Poisson equation
        # If we don't solve the Poisson equation, we are not updating TR of generation sites 
        # We only detect generation sites with available_generation_sites 
            for site_idx in generation_sites:
              self.grid_crystal[site_idx].transition_rates()  
              
            
                
                
# =============================================================================
# Update transition rates with electric field                    
# =============================================================================

    def update_transition_rates_with_electric_field(self,E_field):
      # Update System_state based on electric field
      
        for site in (self.sites_occupied + self.adsorption_sites):
          #print(f'For the site {self.grid_crystal[site].position} ({self.grid_crystal[site].chemical_specie}) the electric field is {E_site_field})')
          E_field_key = tuple(np.round(self.grid_crystal[site].position, 6))
          self.grid_crystal[site].transition_rates(
            E_site_field = E_field[E_field_key], migration_pathways = self.migration_pathways, 
            clusters = self.clusters, atom_to_cluster = self.atom_to_cluster
          )   
    
# =============================================================================
#             Introduce particle
# =============================================================================
    def _introduce_specie_site(self,idx,support_update_sites, event_update_sites, chemical_specie, ion_charge = None):
        """Introduce species at site and track affected sites."""
        # Chemical specie deposited
        site = self.grid_crystal[idx]
        site.introduce_specie(chemical_specie, ion_charge)
        
        # Track sites occupied
        if idx not in self.sites_occupied:
          self.sites_occupied.append(idx) 

        event_update_sites.add(idx)
        support_update_sites.update(site.nearest_neighbors_idx)
        support_update_sites.add(idx) 
        
        for affected_site_idx in support_update_sites:
            affected_site = self.grid_crystal[affected_site_idx]
            
            # Add sites that support the affected site
            for supporting_site_idx in affected_site.supp_by:
              if(isinstance(supporting_site_idx, tuple) and
                 self.grid_crystal[supporting_site_idx].chemical_specie != self.affected_site):
                 event_update_sites.add(supporting_site_idx)
                 
            # Add the affected site itself if occupied
            if affected_site.chemical_specie != self.affected_site:
                event_update_sites.add(affected_site_idx)
                
    
# =============================================================================
#             Remove particle 
# =============================================================================
    def _remove_species_at_site(self,idx,support_update_sites, event_update_sites):
        """Remove species from site and track affected sites."""
        site = self.grid_crystal[idx]
        site.remove_specie(self.affected_site)

        if idx in self.sites_occupied:
          self.sites_occupied.remove(idx) 
          
        event_update_sites.discard(idx)
        support_update_sites.update(site.nearest_neighbors_idx)
        support_update_sites.add(idx) 
        
        # Include in update_specie_events all the particles that can migrate 
        # to the sites in update_supp_av --> It might change the available migrations
        # or the activation energy
        for affected_site_idx in support_update_sites:
            affected_site = self.grid_crystal[affected_site_idx]
            
            for supporting_site_idx in affected_site.supp_by:
              if(isinstance(supporting_site_idx, tuple) and
                   self.grid_crystal[supporting_site_idx].chemical_specie != self.affected_site):
                   event_update_sites.add(supporting_site_idx)
            
            # Add the affected site itself if occupied
            if affected_site.chemical_specie != self.affected_site:
                event_update_sites.add(affected_site_idx)
        

    def track_time(self,t):
        
        self.time += t
        
    def add_time(self):
        
        self.list_time.append(self.time)
        
    def get_timestep_limit(self):
        """
        Calculate maximum timestep based on next Poisson solve time
        """
        next_poisson_time = self.last_poisson_time + self.timestep_limits
        timestep_limit = next_poisson_time - self.time
        
        tolerance = 1.e-12 * self.timestep_limits
        
        if timestep_limit < tolerance:
          self.time = next_poisson_time
          timestep_limit = 0.0
        
        return next_poisson_time - self.time

# =============================================================================
# --------------------------- PLOTTING FUNCTIONS ------------------------------
#         
# =============================================================================


    def plot_lattice_points(self,azim = 60,elev = 45):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        positions_cartesian = [site.position for site in self.grid_crystal.values()]
        
        x,y,z = zip(*positions_cartesian)
        
        ax.scatter3D(x, y, z, c='blue', marker='o')
        ax.set_aspect('equal', 'box')
        ax.view_init(azim=azim, elev = elev)

        ax.set_xlabel('x-axis (nm)')
        ax.set_ylabel('y-axis (nm)')
        ax.set_zlabel('z-axis (nm)')
        
        plt.show()
        
        
        
    def plot_crystal(self,azim = 60,elev = 45,path = '',i = 0):
        
        if self.lammps_file == False:
            nr = 1
            nc = 2
            fig = plt.figure(constrained_layout=True,figsize=(15, 8),dpi=300)
            subfigs = fig.subfigures(nr, nc, wspace=0.1, hspace=7, width_ratios=[1,1])
            
            axa = subfigs[0].add_subplot(111, projection='3d')
            axb = subfigs[1].add_subplot(111, projection='3d')
    
            positions = np.array([self.grid_crystal[idx].position for idx in self.sites_occupied])
            if positions.size != 0:
                x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
                axa.scatter3D(x, y, z, c='blue', marker='o', alpha = 1)
                axb.scatter3D(x, y, z, c='blue', marker='o', alpha = 1)
            
            axa.set_xlabel('x-axis (Angstrom)')
            axa.set_ylabel('y-axis (Angstrom)')
            axa.set_zlabel('z-axis (Angstrom)')
            axa.view_init(azim=azim, elev = elev)
    
            axa.set_xlim([0, self.crystal_size[0]]) 
            axa.set_ylim([0, self.crystal_size[1]])
            axa.set_zlim([0, self.crystal_size[2]])
            axa.set_aspect('equal', 'box')
            
            axb.set_xlabel('x-axis (Angstrom)')
            axb.set_ylabel('y-axis (Angstrom)')
            axb.set_zlabel('z-axis (Angstrom)')
            axb.view_init(azim=45, elev = 10)
    
            axb.set_xlim([0, self.crystal_size[0]]) 
            axb.set_ylim([0, self.crystal_size[1]])
            axb.set_zlim([0, self.crystal_size[2]])
            axb.set_aspect('equal', 'box')
    
    
            if path == '':
                plt.show()
            else:
                fig_filename = path / f"{i}_t(s) = {round(self.time, 5)} .png"
                plt.savefig(fig_filename, dpi = 300)
                plt.clf()
            plt.show()
            
        else:
            
            #species_mapping = {self.chemical_specie: 1}  # Example species mapping
            charge_mapping = {0: 1, 1: 2}  # charge 0 -> type 1, charge 1 -> type 2
            
            # Get charge values for occupied sites
            charges = [self.grid_crystal[site].ion_charge for site in self.sites_occupied]  # Assuming charge attribute exists
            species_ids = [charge_mapping.get(charge) for charge in charges]
            
            sites_occupied_cart = [(self.idx_to_cart(site)) for site in self.sites_occupied]
            #species_ids = [species_mapping.get(self.grid_crystal[site].chemical_specie) for site in self.sites_occupied]
            # Define particle IDs
            particle_ids = list(range(1, len(sites_occupied_cart) + 1))  # Unique IDs for each particle
            
    
            base_path = Path(path)
            dump_file_path = base_path / f"{i}.dump"
            # Write the LAMMPS dump file
            with open(dump_file_path, 'w') as dump_file:
                dump_file.write(f"ITEM: TIMESTEP\n{self.time:.10f}\n")
                dump_file.write("ITEM: NUMBER OF ATOMS\n")
                dump_file.write(f"{len(sites_occupied_cart)}\n")
                dump_file.write("ITEM: BOX BOUNDS (Angstrom)\n")
                dump_file.write(f"0.0 {self.crystal_size[0]}\n")
                dump_file.write(f"0.0 {self.crystal_size[1]}\n")
                dump_file.write(f"0.0 {self.crystal_size[2]}\n")
                dump_file.write("ITEM: ATOMS id type x y z\n")
                for pid, sid, pos in zip(particle_ids, species_ids, sites_occupied_cart):
                    dump_file.write(f"{pid} {sid} {pos[0]} {pos[1]} {pos[2]}\n")
            
            
            
            # Write the metadata file
            if i == 1:
                metadata_path = base_path / "metadata.json"
                metadata_species = {charge_mapping[charge]: charge for charge in charge_mapping}
                metadata_experimental_conditions = {'Experiment':self.experiment,
                                                    'Temperature':self.temperature,
                                                    'Partial pressure':self.partial_pressure,
                                                    'Sticking coefficient':self.sticking_coefficient}
                metadata_simulation_setup = {'Simulation domain (Angstroms)':self.crystal_size,
                                             'Growth direction':self.miller_indices,
                                             'Material id (Materials Project)':self.id_material,
                                             'Search superbasin after n steps with small time steps': self.n_search_superbasin,
                                             'Time limitation to search superbasin': self.time_step_limits,
                                             'Minimum activation energy for building superbasins':self.E_min,
                                             'Activation energy set':self.Act_E_dict}
                with open(metadata_path, 'w') as metadata_file:
                    json.dump({
                        "experimental_conditions": metadata_experimental_conditions,
                        "species": metadata_species,
                        "simulation_setup": metadata_simulation_setup
                    }, metadata_file, indent=4)
                    
                    
    def plot_islands(self,path = '',i = 0):
        
        visited = set()
        sites_occupied_cart = []
        species_ids = []
        
        # Assign unique species ID per island
        for island in self.islands_list:
            for id_cluster, cluster in enumerate(island.cluster_list,start = 1):
                for site in cluster:
                    visited.add(site)
                    sites_occupied_cart.append(self.idx_to_cart(site))
                    species_ids.append(id_cluster)

        # Assign another species ID (e.g., island_id + 1) for remaining unclustered atoms
        remainder_id = id_cluster + 1
        for site in self.sites_occupied:
            if site not in visited:
                visited.add(site)
                sites_occupied_cart.append(self.idx_to_cart(site))
                species_ids.append(remainder_id)
                
        # species_mapping = {self.chemical_specie: 1}  # Example species mapping
        # sites_occupied_cart = [(self.idx_to_cart(site)) for site in self.sites_occupied]
        # species_ids = [species_mapping.get(self.grid_crystal[site].chemical_specie) for site in self.sites_occupied]
        # Define particle IDs
        particle_ids = list(range(1, len(sites_occupied_cart) + 1))  # Unique IDs for each particle
        

        base_path = Path(path)
        dump_file_path = base_path / f"{i}.dump"
        # Write the LAMMPS dump file
        with open(dump_file_path, 'w') as dump_file:
            dump_file.write(f"ITEM: TIMESTEP\n{self.time:.10f}\n")
            dump_file.write("ITEM: NUMBER OF ATOMS\n")
            dump_file.write(f"{len(sites_occupied_cart)}\n")
            dump_file.write("ITEM: BOX BOUNDS (Angstrom)\n")
            dump_file.write(f"0.0 {self.crystal_size[0]}\n")
            dump_file.write(f"0.0 {self.crystal_size[1]}\n")
            dump_file.write(f"0.0 {self.crystal_size[2]}\n")
            dump_file.write("ITEM: ATOMS id type x y z\n")
            for pid, sid, pos in zip(particle_ids, species_ids, sites_occupied_cart):
                dump_file.write(f"{pid} {sid} {pos[0]} {pos[1]} {pos[2]}\n")
                    

        
    def plot_crystal_surface(self):
        
        x,y,z = self.obtain_surface_coord()
                
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
        
        # Set labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_zlim([0, self.crystal_size[2]])
        
        ax.view_init(azim=45, elev = 45)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Show the plot
        plt.show()
        
# =============================================================================
# --------------------------- MEASUREMENTS ------------------------------------
#         
# =============================================================================

    def measurements_crystal(self):
        
        self.calculate_mass()
        self.sites_occupation()
        self.average_thickness()
        self.terrace_area()
        self.RMS_roughness()
        
    def calculate_mass(self):
        
        x_size, y_size = self.crystal_size[:2]
        density = len(self.sites_occupied) * self.mass_specie / (x_size * y_size)
        g_to_ng = 1e9
        nm_to_cm = 1e7
        
        self.mass_gained = nm_to_cm**2 * g_to_ng * density / constants.Avogadro # (ng/cm2)

# =============================================================================
# We calculate % occupy per layer
# Average the contribution of each layer to the thickness acording to the z step
# =============================================================================
    def average_thickness(self):
        
        grid_crystal = self.grid_crystal
        z_step = next((vec[2] * 2 for vec in self.basis_vectors if vec[2] > 1e-10), None)
        z_steps = round(self.crystal_size[2]/z_step + 1)
        layers = [0] * z_steps  # Initialize each layer separately

        for site in grid_crystal.values():
            z_idx = int(round(site.position[2] / z_step))
            layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0

        sites_per_layer = len(grid_crystal)/z_steps
        normalized_layers = [count / sites_per_layer for count in layers]
        # Number of sites occupied and percentage of occupation for each layer
        self.layers = [layers, normalized_layers]
        
        # Layer 0 is z = 0, so it doesn't contribute
        self.thickness = sum(normalized_layers) * z_step # (nm)    
        
    def sites_occupation(self):
        
        self.fraction_sites_occupied = len(self.sites_occupied) / len(self.grid_crystal) 
        
    def terrace_area(self):
        
        layers = self.layers[0]
        grid_crystal = self.grid_crystal
        z_step = next((vec[2] * 2 for vec in self.basis_vectors if vec[2] > 0), None)
        z_steps = round(self.crystal_size[2]/z_step + 1)
        sites_per_layer = len(grid_crystal)/z_steps

        area_per_site = self.crystal_size[0] * self.crystal_size[1] / sites_per_layer
        
        terraces = [(sites_per_layer - layers[0])* area_per_site]
        terraces.extend((layers[i-1] - layers[i]) * area_per_site for i in range(1,len(layers)))
        terraces.append(layers[-1] * area_per_site) # (nm2)
        
        self.terraces = terraces
        
    def RMS_roughness(self):
        
        x,y,z = self.obtain_surface_coord()
        z = np.array(z)
        z_mean = np.mean(z)
        self.Ra_roughness = sum(abs(z-z_mean))/len(z)
        self.z_mean = z_mean
        self.surf_roughness_RMS = np.sqrt(np.mean((z-z_mean)**2))
        
    def neighbors_calculation(self):
        
        grid_crystal = self.grid_crystal
        sites_occupied = self.sites_occupied
        
        if not hasattr(self, 'num_event'):
            if hasattr(self, 'structure'):
                self.num_event = len(self.structure.get_neighbors(self.structure[0],self.radius_neighbors)) + 2
            else:
                self.num_event = 14

        # Size of histogram: number of neighbors that a particle can have, plus particle without neighbors
        histogram_neighbors = [0] * (self.num_event - 1)
        
        for site in sites_occupied:
            if 'bottom_layer' in grid_crystal[site].supp_by or 'Substrate' in grid_crystal[site].supp_by: 
                histogram_neighbors[len(grid_crystal[site].supp_by)-1] += 1
            else:
                histogram_neighbors[len(grid_crystal[site].supp_by)] += 1
                
        self.histogram_neighbors = histogram_neighbors
        
        
    # ----------------------------------------------------
    #    Metal clusters and islands calculations
    # ----------------------------------------------------
    
    def _initialize_cluster_tracking(self):
      """ Call once at simulation start """
      self.atom_to_cluster = {} # Mapping from site_id to cluster_id
      self.clusters = {}
      self.next_cluster_id = 0
      
    def _add_metal_atom_to_clusters(self,site_id):
      """
      Add a newly reduced metal atom at site_id to cluster system.
      """
      
      # Get metal neighbors (only neutral atoms)
      metal_neighbors = [
        nb for nb in self.grid_crystal[site_id].supp_by 
        if not isinstance(nb, str) and self.grid_crystal[nb].ion_charge == 0
      ]
      
      # Separate neighbors into: in-cluster vs. singletons
      in_cluster_neighbors = []
      singleton_neighbors = []
      for nb in metal_neighbors:
        if nb in self.atom_to_cluster:
          in_cluster_neighbors.append(nb)
        else:
          singleton_neighbors.append(nb)
          
      # Get unique clusters from in-clusters neighbors
      neighbor_cluster_ids = {self.atom_to_cluster[nb] for nb in in_cluster_neighbors}  
          
      # Case 1: No in-cluster neighbors and no singletons --> Isolated atom
      if not neighbor_cluster_ids and not singleton_neighbors:
        return  
          
      # Case 2: Only singletons --> Create a new cluster
      if not neighbor_cluster_ids:
        all_atoms = [site_id] + singleton_neighbors
        
        # Create new cluster
        positions = [self.grid_crystal[atom].position for atom in all_atoms]
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        new_cluster = Cluster(all_atoms,positions,{},self.conductivity)
        new_cluster.update_electrode_contact(self.grid_crystal)
        self.clusters[cid] = new_cluster
        for atom in all_atoms:
          self.atom_to_cluster[atom] = cid
        return
        
      # Case 3: Connect atom to existing cluster
      if len(neighbor_cluster_ids) == 1 and not singleton_neighbors:
        # Attach to existing cluster
        cid = neighbor_cluster_ids.pop()
        cluster = self.clusters[cid]
        cluster.atoms_id.add(site_id)
        cluster.atoms_positions.append(self.grid_crystal[site_id].position)
        cluster.size += 1
        self.atom_to_cluster[site_id] = cid 
        cluster.update_electrode_contact(self.grid_crystal)
        return 
        
      # Case 4: Merging existing cluster
      # Merge clusters + new atom + singleton neighbors
      all_atoms = set([site_id] + singleton_neighbors)
      all_positions = [self.grid_crystal[site_id].position]
      all_positions.extend(
        self.grid_crystal[atom].position for atom in singleton_neighbors
      )
      
      
      # Absorb existing clusters
      for cid in neighbor_cluster_ids:
        old_cluster = self.clusters[cid]
        all_atoms.update(old_cluster.atoms_id)
        all_positions.extend(old_cluster.atoms_positions)
        # Remove old mappings
        for atom in old_cluster.atoms_id:
          del self.atom_to_cluster[atom]
        del self.clusters[cid]
        
      # Create merged cluster
      new_cid = self.next_cluster_id
      self.next_cluster_id += 1
      new_cluster = Cluster(all_atoms, all_positions, {},self.conductivity)
      new_cluster.update_electrode_contact(self.grid_crystal)
      self.clusters[new_cid] = new_cluster
      for atom in all_atoms:
        self.atom_to_cluster[atom] = new_cid     
      
      
    def _remove_metal_atom_from_clusters(self,site_id):
       if site_id not in self.atom_to_cluster:
         return
        

       # Identify the cluster id of the atom
       cid = self.atom_to_cluster[site_id]
       cluster = self.clusters[cid]
       
       # Remove atom from cluster
       cluster.atoms_id.discard(site_id)
       cluster.size -= 1
       del self.atom_to_cluster[site_id]
       
       # Remove the corresponding atom position
       cluster.atoms_positions = [
        pos for pos in cluster.atoms_positions
        if not np.allclose(pos, self.grid_crystal[site_id].position)
       ]
            
       # Reset the site's electrode flag (no longer in any cluster)
       self.grid_crystal[site_id].in_cluster_electrode = {'bottom_layer': False, 'top_layer': False}   
       if cluster.size <= 1:
         # Remove cluster
         for atom in list(cluster.atoms_id):
           del self.atom_to_cluster[atom]
           self.grid_crystal[atom].in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}
         del self.clusters[cid]
       else:
         # Update cluster contact with electrodes (may have lost electrode contact)
         
         cluster.update_electrode_contact(self.grid_crystal)  
      
    def metal_clusters_analysis(self):
      
        atoms_idx = self.get_atoms()
        self._find_clusters(atoms_idx)
      
    def get_atoms(self):
        """
        Extract only atoms (not ions) from system
        """
        atoms_idx = []
        atoms_cart = []
        for site in self.sites_occupied:
          if self.grid_crystal[site].ion_charge == 0:
            atoms_idx.append(site)
            atoms_cart.append(self.grid_crystal[site].position)
            
        return atoms_idx,atoms_cart
        
    def _find_clusters(self,atoms_idx):
      """
      Find clusters using nearest neighbors connection
      """
      visited = set()
      clusters = []
      
      for atom_id in atoms_idx:
      
        if atom_id not in visited:
          # Start new cluster with this atom
          cluster_atoms,atoms_positions,attached_layer = self._dfs_explore(atom_id, visited)
          
          if len(cluster_atoms) > 1: # Only consider clusters with > 1 atom
            clusters.append(Cluster(cluster_atoms,atoms_positions,attached_layer))
      
      self.clusters = clusters

            
    def _dfs_explore(self, start_atom, visited):
      """
      Depth-first search to find all connected atoms
      """
      
      stack = [start_atom]
      cluster_atoms = []
      atoms_positions = []
      attached_layer = {'bottom_layer': False, 'top_layer': False}
      
      while stack:
        atom_id = stack.pop()
        
        if atom_id not in visited:
          if isinstance(atom_id, tuple):
            visited.add(atom_id)
            cluster_atoms.append(atom_id)
            atoms_positions.append(self.grid_crystal[atom_id].position)
            
            for neighbor_id in self.grid_crystal[atom_id].supp_by:
              if neighbor_id not in visited:
                stack.append(neighbor_id)
          else: 
            attached_layer[atom_id] = True
          
          
            
              
      return cluster_atoms,atoms_positions,attached_layer
           
      
      
    
    
    # Island is the full structure, not only the part that growth over the mean thickness
    def islands_analysis(self):

        # visited = set()
        island_visited = set()
        total_visited = set()

        normalized_layers = self.layers[1]
        count_islands = [0] * len(normalized_layers)
        layers_no_complete = np.where(np.array(normalized_layers) != 1.0)
        count_islands[normalized_layers == 1] = 1
        z_step = next((vec[2] * 2 for vec in self.basis_vectors if vec[2] > 0), None)



        islands_list = []
        

        for z_idx in layers_no_complete[0]:    
            z_layer = round(z_idx * z_step,3)
            
            for idx_site in self.sites_occupied:   

                if np.isclose(self.grid_crystal[idx_site].position[2], z_layer,atol=1e-1): 

                    island_slice = set()
                    total_visited,island_slice = self.detect_islands(idx_site,total_visited,island_slice,self.chemical_specie)

                    if len(island_slice):
                        island_visited = island_slice.copy()
                        island_sites = island_slice.copy()
                        island_visited,island_sites = self.build_island(island_visited,island_sites,list(island_slice)[0],self.chemical_specie)
                        islands_list.append(Island(z_idx,z_layer,island_sites))
                        count_islands[z_idx] += 1
                        total_visited.update(island_visited)
                        


                 
        self.islands_list = islands_list
        
# =============================================================================
#     Function to detect island and the coordinates of the base
# =============================================================================
    def detect_islands(self,idx_site,visited,island_slice,chemical_specie):

        site = self.grid_crystal[idx_site] 
        
        if idx_site not in visited and site.chemical_specie == chemical_specie:
            visited.add(idx_site)
            island_slice.add(idx_site)
            # dfs_recursive
            for idx in site.migration_paths['Plane']:
                if idx[0] not in visited:
                    visited,island_slice = self.detect_islands(idx[0],visited,island_slice,chemical_specie)
                                       
        return visited,island_slice

# =============================================================================
#     Function to build the full island starting from the base obtained in detect_islands()
# =============================================================================
    def build_island_2(self,visited,island_sites,idx,chemical_specie):
          
        site = self.grid_crystal[idx]
            
        for element in site.migration_paths['Up'] + site.migration_paths['Plane']+site.migration_paths['Down']:
    
            if element[0] not in visited and self.grid_crystal[element[0]].chemical_specie == chemical_specie:
                visited.add(element[0])
                island_sites.add(element[0])
                visited,island_sites = self.build_island(visited,island_sites,element[0],chemical_specie)
                
        return visited,island_sites
    
    def build_island(self,visited,island_sites,start_idx,chemical_specie):
          
        stack = [start_idx]

        while stack:
            idx = stack.pop()
            site = self.grid_crystal[idx]
            
            for element in site.migration_paths['Up'] + site.migration_paths['Plane'] + site.migration_paths['Down']:
        
                if element[0] not in visited and self.grid_crystal[element[0]].chemical_specie == chemical_specie:
                    visited.add(element[0])
                    island_sites.add(element[0])
                    stack.append(element[0])

        return visited,island_sites
    
    # Peak are the part of the film that growth over the mean thickness level
    def peak_detection(self):
        
        chemical_specie = self.chemical_specie
        
        # Thickness can be the mean thickness
        #thickness = self.thickness
        # Or we can choose a reference layer when less than X% of the layer is occupied after the maximum occupation layer in the film
        idx_max_layer = np.where(np.array(self.layers[1]) == max(np.array(self.layers[1]))) # The position of the most occupied layer
        reference_layer = np.where(np.array(self.layers[1])[idx_max_layer[0][0]:] < 0.7) # The layer that is less than a X% occupied after the maximum
        z_step = next((vec[2] * 2 for vec in self.basis_vectors if vec[2] > 1e-10), None)
        thickness = z_step * reference_layer[0][0]

        sites_occupied = self.sites_occupied
    
        # Convert occupied sites to Cartesian coordinates and sort by z-coordinate in descending order
        sites_occupied_cart = sorted(
            ((self.idx_to_cart(site), site) for site in sites_occupied), 
            key=lambda coord: coord[0][2], 
            reverse=True
        )
    
        total_visited = set()
        peak_list = []
        
        for cart_coords, site in sites_occupied_cart:
            if site not in total_visited and cart_coords[2] > thickness:
                peak_sites = self.build_peak({site},site,chemical_specie,thickness)
                peak_list.append(Island(site,cart_coords,peak_sites))
                total_visited.update(peak_sites)
                
        self.peak_list = peak_list
    
    def build_peak(self,peak_sites,start_idx,chemical_specie,thickness):
         
        grid_crystal = self.grid_crystal
        stack = [start_idx]
    
        while stack:
            idx = stack.pop()
            site = grid_crystal[idx]
            
            for element in site.migration_paths['Up'] + site.migration_paths['Plane'] + site.migration_paths['Down']:
    
                if element[0] not in peak_sites and grid_crystal[element[0]].chemical_specie == chemical_specie:
                    peak_sites.add(element[0])
                    
                    if self.idx_to_cart(element[0])[2] > thickness:
                        stack.append(element[0])
    
        return peak_sites    
 
# =============================================================================
#     Auxiliary functions
#     
# =============================================================================
    def unit_vector(self,vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """Finds angle between two vectors"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # Function to rotate a vector
    def rotate_vector(self,vector, axis=None, theta=None, rotation_matrix=None):
        """
        Rotates a 3D vector around a specified axis or using a provided rotation matrix. 
        
        Parameters:
        - vector: The 3D vector to rotate.
        - axis: The axis of rotation ('x', 'y', or 'z'). Optional if rotation_matrix is provided.
        - theta: The rotation angle in radians. Optional if rotation_matrix is provided.
        - rotation_matrix: A 3x3 rotation matrix. Optional if axis and theta are provided.
    
        Returns:
        The rotated vector.
        """
        if rotation_matrix is not None:
            R = rotation_matrix
        
        elif axis is not None and theta is not None:
            if axis == 'x':
                R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            elif axis == 'y':
                R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            elif axis == 'z':
                R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            else:
                raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")
                
        else:
            raise ValueError("Either rotation_matrix or both axis and theta must be provided.")
        
        return np.dot(R, vector)
    
    # Depth-First Search - Traverse a network or a graph -> grid_crystal
    def dfs_recursive(self, idx_site, visited):
        # We calculate the cartesian coordinates of the site using the basis vectors
        cart_site = self.idx_to_cart(idx_site)
        # cart_site[2] >= -1e-3 to avoid that some sites in the zero layer get outside
        if idx_site not in visited and (cart_site[0] >= 0 and cart_site[0] <= self.crystal_size[0]) and (cart_site[1] >= 0 and cart_site[1] <= self.crystal_size[1]) and (cart_site[2] >= -1e-3 and cart_site[2] <= self.crystal_size[2]):
            # We track the created sites
            visited.add(idx_site)
            # We create the site with the cartesian coordinates
            self.grid_crystal[idx_site] = Site("Empty",
                tuple(cart_site),
                self.Act_E_dict)
            
            for neighbor in self.latt.get_neighbors(idx_site):
                self.dfs_recursive(tuple(neighbor[:3]), visited)
                
    def dfs_iterative(self, start_idx_site):
        visited = set()
        stack = [start_idx_site]
    
        while stack:
            current_idx_site = stack.pop()
            if current_idx_site in visited:
                continue
    
            # Calculate the cartesian coordinates of the site using the basis vectors
            cart_site = self.idx_to_cart(current_idx_site)
   
            
            if (
                0 <= cart_site[0] <= self.crystal_size[0]
                and 0 <= cart_site[1] <= self.crystal_size[1]
                and 0 <= cart_site[2] <= self.crystal_size[2]
            ):
                # Track the created site
                visited.add(current_idx_site)
                # Create the site with the cartesian coordinates
                self.grid_crystal[current_idx_site] = Site(
                    "Empty", tuple(cart_site), self.Act_E_dict
                )
    
                # Push neighbors onto the stack
                stack.extend(tuple(neighbor[:3]) for neighbor in self.latt.get_neighbors(current_idx_site))
    # Breadth-First Search (Recursive) - Traverse a network or a graph -> grid_crystal
    # to build a cluster of a certain size
    def bfs_cluster(self,queue,visited,cluster_size):
        
        if not queue or len(visited) >= cluster_size:
            return
        
        # Dequeue a site from the front of the queue
        #Starting point
        current_idx_site = queue.popleft()
        
        if current_idx_site not in visited:
            visited.add(current_idx_site)
            update_supp_av = set()
            update_specie_events = set()
            
            update_specie_events,update_supp_av = self.introduce_specie_site(current_idx_site,update_specie_events,update_supp_av)
            self.update_sites(update_specie_events,update_supp_av)
            
            # Enqueue all unvisited neighbors of the current site
            for neighbor in self.grid_crystal[current_idx_site].migration_paths['Plane']:
                if neighbor[0] not in visited:
                    queue.append(neighbor[0])
                
        # Recur to process the next site in the queue
        self.bfs_cluster(queue, visited, cluster_size)
        
    def idx_to_cart(self,idx):
        return tuple(round(element,3) for element in np.sum(idx * np.transpose(self.basis_vectors), axis=1))
    

    
    def obtain_surface_coord(self):
        
        grid_crystal = self.grid_crystal
        z_step = next((vec[2] * 2 for vec in self.basis_vectors if vec[2] > 0), None)

        x = []
        y = []
        z = []
        
        for site in grid_crystal.values():
            top_layer_empty_sites = 0
            for jump in site.migration_paths['Up']:
                if grid_crystal[jump[0]].chemical_specie == 'Empty': top_layer_empty_sites +=1
                     
            if (site.chemical_specie != 'Empty') and top_layer_empty_sites >= 2:
                x.append(site.position[0])
                y.append(site.position[1])
                z.append(site.position[2]+z_step)
                
            elif (site.chemical_specie == 'Empty') and ('bottom_layer' in site.supp_by) and top_layer_empty_sites >= 2:
                x.append(site.position[0])
                y.append(site.position[1])
                z.append(site.position[2])
                
        return x,y,z