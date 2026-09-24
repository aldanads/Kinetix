# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:19:06 2024

@author: samuel.delgado
"""
from __future__ import annotations

import matplotlib.pyplot as plt



from scipy import constants
import numpy as np
from itertools import product
import math
from matplotlib import cm
import time
import copy

from kinetix.lattice.site import Site
from kinetix.lattice.cluster import Cluster
from kinetix.lattice.grain_boundary import GrainBoundary
from kinetix.lattice.island import Island
from kinetix.utils.mpi_context import MPIContext
from kinetix.utils.balanced_tree import Node, build_tree, update_data, search_value
from kinetix.utils.superbasin import Superbasin
from kinetix.lattice.events import EventHandler

from collections import Counter

# Pymatgen for creating crystal structure and connect with Material Project
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.wulff import WulffShape
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.defects.generators import ChargeInterstitialGenerator, VoronoiInterstitialGenerator
from pymatgen.core import Structure, PeriodicSite


import json
import subprocess
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Type-checking only: the config classes appear solely in annotations, so
    # importing them here at runtime would pull the full config stack into this
    # hot lattice module for no benefit.
    from kinetix.configs.solver_config import SuperbasinConfig
    from kinetix.configs.simulation_config import (
        ExperimentalConditions,
        SimulationSettings,
    )
    from kinetix.configs.material_config import MaterialConfig
    from kinetix.configs.calculator_config import CalculatorConfig

import os
from pathlib import Path
import platform
import logging

logger = logging.getLogger(__name__)


class Crystal_Lattice():
    
    METAL_SPECIES = {'Ag', 'Cu', 'Pt', "Au", "Pd", "Ni"}
    
    def __init__(
      self,
      material_config: MaterialConfig,
      Act_E_dict,
      lammps_file,
      superbasin_config: SuperbasinConfig,
      experimental_config: ExperimentalConditions | None = None,
      settings: SimulationSettings | None = None,
      api_key: str | None = None,
      rng=None,
      cache_dir=None,
      calculator_config: CalculatorConfig | None = None,
      defects_config: dict | None = None,
      reactions_config: dict | None = None,
      gb_configurations: list | None = None,
      mpi_ctx = None,
      simulation_type: str | None = None,
      **kwargs
    ):
        
        # Handling MPI. mpi_ctx=None means TRULY serial: no MPI operations at
        # all (no barriers, no broadcasts). This is what allows Rank 0 to
        # create the grid alone in initialize_grid_crystal (Phase 1) while
        # ranks 1..N wait at an outer barrier.
        self.mpi_ctx = mpi_ctx
        self.rank = self.mpi_ctx.rank if self.mpi_ctx is not None else 0
        self.comm = self.mpi_ctx.comm if self.mpi_ctx is not None else None
        self.use_mpi = self.mpi_ctx.available if self.mpi_ctx is not None else False
        
        # --- Material features (typed MaterialConfig) ---
        self.id_material = material_config.selection.mp_id
        self.crystal_size = material_config.structure.size
        self.miller_indices = material_config.structure.miller_indices
        self.api_key = api_key
        self.facets_type = material_config.structure.facets_type
        self.affected_site = material_config.structure.affected_site
        self.radius_neighbors = material_config.selection.radius_neighbors
        self.sites_generation_layer = material_config.structure.sites_generation_layer
        self.chemical_formula = material_config.formula
        self.interstitial_generation = material_config.structure.interstitial_generation

        # --- Global settings (typed SimulationSettings) ---
        self.mode = settings.mode if settings is not None else ''
        self.technology = settings.technology if settings is not None else ''

        # --- Derived / non-config values ---
        self.rng = rng
        self.cache_dir = cache_dir
        self.calculator_config = calculator_config

        # --- Feature payloads deferred to Epic 2 (still dict/list based) ---
        self.gb_configurations = gb_configurations
        self.defects_config = defects_config if defects_config is not None else {}
        self._active_site_types = {
          stype for cfg in self.defects_config.values()
          for stype in cfg.get("allowed_sublattices", [])
        }
        self.scavenged_ions = {}

        self.reactions_config = reactions_config if reactions_config is not None else {}    
             
        # --- Experimental conditions ---
        self.sticking_coefficient = experimental_config.sticking_coeff
        self.partial_pressure = experimental_config.partial_pressure
        self.temperature = experimental_config.temperature  # dataclass field is `temperature` (dict key was 'T')
        self.simulation_type = simulation_type
        
        # --- Activation energies (structured by defect name) ---
        self.Act_E_dict = Act_E_dict
        
         # --- Superbasin ---
        self.enabled_superbasin = superbasin_config.enabled_superbasin
        self.n_search_superbasin = superbasin_config.n_search_superbasin
        self.time_step_limits = superbasin_config.time_step_limits
        self.E_min = superbasin_config.E_min
        self.energy_step = superbasin_config.energy_step
        self.superbasin_dict = {}
        self.superbasin_tracker = []
        self.nothing_happen_count = 0
        self.time_based_superbasin = superbasin_config.time_based_superbasin
        self.allow_specie_removal = True # We need this variable to desactivate specie removal during superbasin creation
        
        # --- Poisson solver ---
        self._fields_changed = False
        self._dirty_sites = set()
        # --- Solver configuration (typed objects; carried for cli.py) ---
        self.poisson_config = kwargs.get('poisson_config', None)
        self.heat_config = kwargs.get('heat_config', None)
        self.mesh_config = kwargs.get('mesh_config', None)
        self.material_config = kwargs.get('material_config', None)
        self.solver_mesh_file = kwargs.get('mesh_file', None)
        self.characteristic_length = kwargs.get('characteristic_length', None)

        # The device is globally neutral. Charged ions + electrons/dielectric reduced
        # Example: Ag/CeO2 (Ce4+) --> Ag+ + CeO2 (Ce3+)
        # Debye length:
        # There is a distance from a charged particle beyond which its electrostatic influence 
        # is effectively blocked due to the redistribution of neighboring charges
        if self.poisson_config is not None:
          self.screening_factor = self.poisson_config.screening_factor
          self.conductivity = self.poisson_config.conductivity
          

        # Time tracking
        self.time = 0
        self.list_time = []
        
        # Event tracking
        self.events_tracking = Counter()
        
        # --- Grid generation ---n
        self.lattice_model(api_key, self.mode, self.affected_site, self.miller_indices)
        grid_crystal = kwargs.get('grid_crystal', None)
        self.crystal_grid(grid_crystal,self.radius_neighbors,self.mode,self.affected_site,api_key)
        self.active_event_sites = [] # Sites occupy be a chemical specie
        self.generation_sites = [] # Sites availables for deposition or migration
        
        #Transition rate for adsortion of chemical species
        if self.simulation_type != 'electronic_device':
            self.transition_rate_adsorption(experimental_config)
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
        for site_idx in support_update_sites:
          for defect, cfg in self.defects_config.items():
            site = self.grid_crystal[site_idx]
            
            if site.defect.chemical_specie == cfg['symbol'] and cfg['enabled_events']:
              event_update_sites.add(site_idx)
              self.active_event_sites.append(site_idx)    

        self.update_sites_topology(support_update_sites,event_update_sites)
        
        all_initial_sites = (
          support_update_sites |
          event_update_sites |
          set(self.generation_sites)
        )
        self._dirty_sites.update(all_initial_sites)
        
        if self.simulation_type == 'electronic_device':
          self._fields_changed=True
        #self._update_rates_lazily()
        
        self.lammps_file = lammps_file
        
       
        
    # ============ Helper methods: cache ========================
    # =========================================================================
    # Lattice construction: delegates to LatticeBuilder (Phase 5)
    # All construction bodies live in kinetix/lattice/lattice_builder.py; the
    # builder holds no simulation state (everything goes through
    # ``self.system``). `_is_active_site` and `_minimum_image_vector` STAY on
    # this class (construction + runtime callers) - the builder reaches them
    # through ``self.system.…``.
    # =========================================================================
    @property
    def lattice_builder(self):
        if not hasattr(self, '_lattice_builder'):
            from kinetix.lattice.lattice_builder import LatticeBuilder
            self._lattice_builder = LatticeBuilder(self)
        return self._lattice_builder

    def _load_mp_cache(self, key):
        return self.lattice_builder._load_mp_cache(key)

    def _save_mp_cache(self, key, data):
        return self.lattice_builder._save_mp_cache(key, data)

    def lattice_model(self, api_key, mode, affected_site=None, miller_indices=(0, 0, 1)):
        return self.lattice_builder.lattice_model(api_key, mode, affected_site, miller_indices)

    def _is_inside_supercell(self, cart_pos, supercell_lattice):
        return self.lattice_builder._is_inside_supercell(cart_pos, supercell_lattice)

    def _apply_miller_orientation(self, structure, miller_indices):
        return self.lattice_builder._apply_miller_orientation(structure, miller_indices)

    def _get_rotation_matrix(self, vec1, vec2):
        return self.lattice_builder._get_rotation_matrix(vec1, vec2)

    def _create_supercell(self, unit_cell):
        return self.lattice_builder._create_supercell(unit_cell)

    def _compute_basis_vectors(self):
        return self.lattice_builder._compute_basis_vectors()

    def _initialize_migration_pathways(self, radius_neighbors, reset_energies=False):
        return self.lattice_builder._initialize_migration_pathways(radius_neighbors, reset_energies)

    def _validate_migration_network(self, radius=None):
        return self.lattice_builder._validate_migration_network(radius)

    def _build_kdtree(self):
        return self.lattice_builder._build_kdtree()

    def _get_neighbors_for_site(self, site_idx, radius):
        return self.lattice_builder._get_neighbors_for_site(site_idx, radius)

    def _generate_periodic_images(self, site_pos, radius):
        return self.lattice_builder._generate_periodic_images(site_pos, radius)

    def _check_percolation_at_radius(self, radius, site_type="interstitial"):
        return self.lattice_builder._check_percolation_at_radius(radius, site_type)

    def find_optimal_radius(self, site_type="interstitial", min_radius=1.5, max_radius=6.0, step=0.25, safety_margin=0.5):
        return self.lattice_builder.find_optimal_radius(site_type, min_radius, max_radius, step, safety_margin)

    def diagnose_steep_down(self, site_idx, radius_neighbors):
        return self.lattice_builder.diagnose_steep_down(site_idx, radius_neighbors)

    def diagnose_interstitial_presence(self, site_idx, radius_neighbors, z_window=3.0):
        return self.lattice_builder.diagnose_interstitial_presence(site_idx, radius_neighbors, z_window)

    def crystal_grid(self, grid_crystal, radius_neighbors, mode, affected_site, api_key):
        return self.lattice_builder.crystal_grid(grid_crystal, radius_neighbors, mode, affected_site, api_key)

    def _efficient_act_e_copy(self, base_dict):
        return self.lattice_builder._efficient_act_e_copy(base_dict)

    def _get_applicable_defects_for_site(self, site_type):
        return self.lattice_builder._get_applicable_defects_for_site(site_type)

    def _compute_interface_flags(self):
        return self.lattice_builder._compute_interface_flags()

    def _generate_interstitial_sites(self, api_key=None):
        return self.lattice_builder._generate_interstitial_sites(api_key)

    def _find_interstitials_voronoi(self, interstitial_species, min_distance=0.3):
        return self.lattice_builder._find_interstitials_voronoi(interstitial_species, min_distance)

    def _refine_interstitial_positions(self, voronoi_positions, interstitial_species):
        return self.lattice_builder._refine_interstitial_positions(voronoi_positions, interstitial_species)

    def _cluster_and_average(self, positions, threshold=0.7):
        return self.lattice_builder._cluster_and_average(positions, threshold)

    def _validate_interstitial_positions(self, positions, structure):
        return self.lattice_builder._validate_interstitial_positions(positions, structure)

    def create_ovito_xyz_file(self, interstitial_species, base_positions_unit_cell, filename="interstitials.xyz"):
        return self.lattice_builder.create_ovito_xyz_file(interstitial_species, base_positions_unit_cell, filename)

    def _handle_missing_neighbors(self, radius_neighbors, affected_site):
        return self.lattice_builder._handle_missing_neighbors(radius_neighbors, affected_site)

    def _parallel_neighbors_analysis(self, num_cores=4):
        return self.lattice_builder._parallel_neighbors_analysis(num_cores)

    def _sequencial_neighbors_analysis(self):
        return self.lattice_builder._sequencial_neighbors_analysis()

    def get_num_cores(self, local_max_cores=6):
        return self.lattice_builder.get_num_cores(local_max_cores)

    def _process_batch_sites_worker(self, batch_keys, grid_crystal, shared_data):
        return self.lattice_builder._process_batch_sites_worker(batch_keys, grid_crystal, shared_data)

    def get_idx_coords(self, coords, basis_vectors):
        return self.lattice_builder.get_idx_coords(coords, basis_vectors)

    def Wulff_Shape(self, api_key):
        return self.lattice_builder.Wulff_Shape(api_key)

    def create_edges(self, facets_type):
        return self.lattice_builder.create_edges(facets_type)

    def _initialize_cluster_tracking(self):
        return self.lattice_builder._initialize_cluster_tracking()

    def _minimum_image_vector(self, vec):
      """
      Apply minimum-image convention for lateral PBC (x, y only).
      z is an open boundary (electrodes), so it is NOT wrapped.
      """
      vec = np.array(vec, dtype=float)
      for dim in range(2): # x and y only
        L = self.crystal_size[dim]
        if vec[dim] > L/2:
          vec[dim] -= L
        elif vec[dim] < -L/2:
          vec[dim] += L
      return vec
              
              
# =============================================================================
# Get coordinates of particles for solving Poisson and points to evaluate electric field
# =============================================================================
    @property
    def solver_coordinator(self):
        if not hasattr(self, '_solver_coordinator'):
            from kinetix.solvers.coordinator import SolverCoordinator
            self._solver_coordinator = SolverCoordinator(self)
        return self._solver_coordinator

    def save_electric_bias(self, V):
        return self.solver_coordinator.save_electric_bias(V)

    def get_evaluation_points(self):
        return self.solver_coordinator.get_evaluation_points()

    def prepare_clusters_for_bcs(self):
        return self.solver_coordinator.prepare_clusters_for_bcs()
      
    
    # -------------------------------------------------------------------------
    # get_idx_coords / Wulff_Shape / create_edges moved to
    # kinetix/lattice/lattice_builder.py (Phase 5); delegates live under the
    # "Lattice construction" banner after __init__.
    # -------------------------------------------------------------------------

    def available_generation_sites(self, support_update_sites = set(), defect_name=None, defect=None):
        
        update_gen_sites = set()
        # Generation of vacancy in the bulk
        if self.mode == 'vacancy': return
                    
        # Normal deposition process: gas-substrate interface
        elif self.mode == 'regular':
            
            generation_sites_set = set(self.generation_sites)

            for idx in support_update_sites:
                site = self.grid_crystal[idx]

                if idx in generation_sites_set:
                    if ((self.sites_generation_layer not in site.supp_by and len(site.supp_by) < 3) or (site.defect.chemical_specie != self.affected_site)):
                        self.generation_sites.remove(idx)
                        site.remove_event_type('generation')
                    
                else:
                    if (self.sites_generation_layer in site.supp_by or len(site.supp_by) > 2) and site.defect.chemical_specie == self.affected_site:
                        self.generation_sites.append(idx)
                        site.deposition_event(self.TR_gen,idx,'generation',self.Act_E_gen)
           
           
                        
        # Oxidation reaction at the electrode-dielectric interface
        elif self.mode == 'interstitial':  
          sites_generation_layer = defect.get('sites_generation_layer')
          
          # Find interstitial sites within interface proximity
          if sites_generation_layer not in ['top_layer', 'bottom_layer']:
            logger.warning("Unknown sites_generation_layer: %s", sites_generation_layer)
                            
          generation_sites_set = set(self.generation_sites)
          
          can_generate = self._should_generate_at_electrode(defect_name)
          
          if not can_generate:
            for idx in list(generation_sites_set):
              site = self.grid_crystal[idx]
              self.generation_sites.remove(idx)
              site.remove_event_type('generation')
            return update_gen_sites
          
          for idx in support_update_sites:
                site = self.grid_crystal[idx]
                
                # Cleanup branch
                if idx in generation_sites_set:
                    if (site.defect.chemical_specie != self.affected_site):
                        self.generation_sites.remove(idx)
                        site.remove_event_type('generation')
                    continue # Already handled, move to the next site
                    
                if site.site_type != "interstitial":
                  continue
                  
                is_at_interface = (
                  (sites_generation_layer == 'top_layer' and site.is_at_top_interface) or
                  (sites_generation_layer == 'bottom_layer' and site.is_at_bottom_interface)
                )
                      
                if (is_at_interface and site.defect.chemical_specie == self.affected_site):
                  self.generation_sites.append(idx)
                  site.ion_generation_interface(idx)
                  update_gen_sites.add(idx)    
                        
          #if len(generation_sites_set) == 0 and len(update_gen_sites) == 0:
          #  print(f"Warning: No generation sites found for {defect.get('symbol')}")
 
          return update_gen_sites  
          
    def _should_generate_at_electrode(self, defect_name):
      """
      Check if a defect should be generated at the electrode.
      
      Parameters:
      -----------
      defect_name : str
          Name of the defect
      electrode : str
          Electrode name ('top' or 'bottom')
      
      Returns:
      --------
      bool : True if defect should be generated
      """
      
      if defect_name not in self.defects_config:
        return False
        
      defect_cfg = self.defects_config[defect_name]
      
      electrode_scavenging = defect_cfg.get('electrode_scavenging')
      
      if electrode_scavenging is None:
        return True

      # Bool form: True = scavenging enabled without mass conservation
      # (same convention as _should_scavenge), so generation is always allowed.
      if not isinstance(electrode_scavenging, dict):
        return True

      # Check mass conservation
      if electrode_scavenging.get('mass_conservation'):
        if self.scavenged_ions.get(defect_name, 0) <= 0:
          return False # No ions available to inject
      
      return True
    
       
    
    # -------------------------------------------------------------------------
    # Superbasin activation helpers moved to kinetix/lattice/kmc_loop.py
    # (Phase 4): should_activate_superbasin, is_filament_percolating,
    # _check_event_based_superbasin, _check_time_based_superbasin,
    # _slow_timesteps. Thin delegates live under the "KMC logic" banner.
    # -------------------------------------------------------------------------

    def transition_rate_adsorption(self, experimental_config: ExperimentalConditions):
# =============================================================================
#         Kim, S., An, H., Oh, S., Jung, J., Kim, B., Nam, S. K., & Han, S. (2022).
#         Atomistic kinetic Monte Carlo simulation on atomic layer deposition of TiN thin film. 
#         Computational Materials Science, 213. https://doi.org/10.1016/j.commatsci.2022.111620
# =============================================================================
        
        # Maxwell-Boltzman statistics for transition rate of adsorption rate
        sticking_coeff, partial_pressure, T = experimental_config.sticking_coeff, experimental_config.partial_pressure, experimental_config.temperature 
        self.mass_specie = Element(self.chemical_specie).atomic_mass

        # The mass in kg of a unit of the chemical specie
        self.mass_specie = self.mass_specie / constants.Avogadro / 1000
        
        lattice = self.structure.lattice

        # Collect fractional z-coordinates of all sites in the grid
        frac_z_list = [lattice.get_fractional_coords(site.position)[2] for site in self.grid_crystal.values()]
        # Cluster the z-values into distinct layers based on a threshold (e.g., 0.02)
        frac_z_list.sort()
        layer_tol = 0.02  # Tolerance for clustering
        layers = []
        
        for z in frac_z_list:
            if not layers or (z - layers[-1][-1]) > layer_tol:
                layers.append([z])
            else:
                layers[-1].append(z)

        # Bottom layer = first cluster; its size is exactly n_sites_layer_0
        n_sites_layer_0 = len(layers[0]) if layers else 0

        if n_sites_layer_0 == 0:
          raise ValueError("No sites found in the bottom layer; "
                          "check layer_threshold or grid construction")

        # --- Surface area: correct for any cell shape -------------------------
        # For the ab face, the area is |a × b|
        # Area in m^2
        a_vec, b_vec, _ = lattice.matrix
        surface_area_A2 = np.linalg.norm(np.cross(a_vec, b_vec))  # in angstroms^2
        surface_area_m2 = surface_area_A2 * 1e-20  # Convert to m^2
        area_specie = surface_area_m2 / n_sites_layer_0
        
        # Boltzmann constant (m^2 kg s^-2 K^-1)
        self.TR_gen = sticking_coeff * partial_pressure * area_specie / np.sqrt(2 * constants.pi * self.mass_specie * constants.Boltzmann * T)
    
        # Activation energy for deposition
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E12;  # nu0 (s^-1) bond vibration frequency
        self.Act_E_gen = -np.log(self.TR_gen/nu0) * kb * self.temperature
        
    def limit_kmc_timestep(self,P_limits):
        
        self.timestep_limits = -np.log(1-P_limits)/self.TR_gen
        
        
    def defect_gen(self):
        
        sites_needing_support_update = set()
        sites_needing_event_update = set()
        

        for defect in self.defects_config.values():
          for idx,site in self.grid_crystal.items():
              
              if site.site_type in defect['allowed_sublattices'] and site.defect.chemical_specie in defect["valid_target_species"]:
                if self.gb_model.get_site_gb_region(site.position) == 'inner_boundary':
                  probability = defect['initial_concentration_GB']
                else:
                  probability = defect['initial_concentration_bulk']
                  
                if self.rng.random() < probability:
                  chemical_specie = defect['symbol']
                  ion_charge = defect['charge'] 
                  self._introduce_specie_site(idx,sites_needing_support_update, sites_needing_event_update,chemical_specie,ion_charge)

          # Update sites availables, the support to each site and available migrations
          self.update_sites_topology(sites_needing_support_update, sites_needing_event_update)
          
          all_affected_sites = (
            sites_needing_support_update |
            sites_needing_event_update |
            set(self.generation_sites)
          )
          
          self._dirty_sites.update(all_affected_sites)
              
    def deposition_specie(self,t,test = 0):  

        support_update_sites = set()
        event_update_sites = set()
        
        if test == 0:
            
            P = 1-np.exp(-self.TR_gen*t) # Adsorption probability in time t
            # Indexes of sites availables: supported by substrates or other species
            for idx in self.generation_sites:
                if self.rng.random() < P:   
                    # Introduce specie in the site
                    update_specie_events,support_update_sites = self.introduce_specie_site(idx,support_update_sites, event_update_sites,self.grid_crystal[idx].defect.charge)
            
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,support_update_sites)

            

        # Single particle in a determined place
        elif test == 1:
            
            lattice = self.structure.lattice
            # Compute geometric center of the domain
            center = lattice.get_cartesian_coords([0.5,0.5,0.5])
            min_dist = float('inf')
            central_idx = None
            
            for idx, site in self.grid_crystal.items():
                pos = np.array(site.position)
                dist = np.linalg.norm(pos - center)
                #print(f'Dist {dist} and min. dist. {min_dist}')
                if dist < min_dist:
                    min_dist = dist
                    central_idx = idx   
                    
                     
            # Introduce specie in the site
            defect = 'hydrogen_interstitial'
            migrating_charge = self.defects_config[defect]['charge']
            chemical_specie = self.defects_config[defect]['symbol']
            self._introduce_specie_site(central_idx, support_update_sites, event_update_sites, chemical_specie, migrating_charge)
            # Update sites availables, the support to each site and available migrations
            self.update_sites(update_specie_events,support_update_sites)
                
            logger.debug('Particle in position: %s is a %s', central_site.position, central_site.defect.chemical_specie)
            logger.debug('Neighbors of that particle: %s', central_site.nearest_neighbors_idx)
            logger.debug('Neighbors are supported by:')
            for idx_3 in central_site.nearest_neighbors_idx:
                logger.debug(self.grid_crystal[idx_3].supp_by)
                
                
        # Two adjacent particles
        elif test == 2:
            defect = 'hydrogen_interstitial'
            migrating_charge = self.defects_config[defect]['charge']
            chemical_specie = self.defects_config[defect]['symbol']  
            site_type = self.defects_config[defect]['site_type']
            
            lattice = self.structure.lattice
            for idx,site in self.grid_crystal.items():
              frac = lattice.get_fractional_coords(site.position)
              if ((0.45 < frac[0] < 0.55) 
                and (0.45 < frac[1] < 0.55)
                and (0.45 < frac[2] < 0.55)) and site.site_type == site_type: 
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
            

        # Central atom with N neighbors
        elif test == 3:
            
            N_HYDROGENS = 3
            lattice = self.structure.lattice
            # 1. Compute geometric center of the domain
            center = lattice.get_cartesian_coords([0.5,0.5,0.5])  # assumes crystal_size = [Lx, Ly, Lz]
            min_dist = float('inf')
            central_idx = None
            
            for idx, site in self.grid_crystal.items():
                if site.site_type != 'O':
                  continue
                pos = np.array(site.position)
                dist = np.linalg.norm(pos - center)
                #print(f'Dist {dist} and min. dist. {min_dist}')
                if dist < min_dist:
                    min_dist = dist
                    central_idx = idx   
                            
            if central_idx is None:
              raise ValueError("No oxygen site found at center for V_O creation")
                     
            # 2. Introduce specie in the site
            central_config = self.defects_config['oxygen_vacancy']
            self._introduce_specie_site(
              central_idx,
              support_update_sites,
              event_update_sites,
              central_config['symbol'],
              central_config['charge']
            )
            
            # 3. Find neighboring sites
            site = self.grid_crystal[central_idx]
            neighbors = []
            
            for neighbor_idx in site.nearest_neighbors_idx:
              neighbor = self.grid_crystal[neighbor_idx]
              if neighbor.site_type == 'interstitial' and neighbor.defect.chemical_specie == 'Empty':
                neighbors.append(neighbor_idx)
            
            # 4. Place atoms in sites
            neighbor_config = self.defects_config['hydrogen_interstitial']
            n_placed = 0
            
            for neigh_idx in neighbors:
              if n_placed >= N_HYDROGENS:
                break
              
              self._introduce_specie_site(
                neigh_idx,
                support_update_sites,
                event_update_sites,
                neighbor_config['symbol'],
                neighbor_config['charge']
              )
              n_placed += 1
             
            # 5. Update
            self.update_sites(support_update_sites, event_update_sites)
             
            # 6. Verification output
            logger.debug("=== Test Case 3: V_O Passivation ===")
            logger.debug("V_O site index: %s", central_idx)
            logger.debug("V_O passivation_level: %s", site.defect.passivation_level)
            logger.debug("V_O charge: %s", site.defect.charge)
            logger.debug("H atoms placed: %s", n_placed)
            logger.debug("Expected charge after %s H: %s", n_placed, central_config['charge'] + n_placed * central_config['charge_per_passivation'])
            logger.debug("=====================================")  

        # Cluster - particles in plane, one on bottom
        elif test == 4:
            
            min_dist_xy = float('inf')
            idx = None
            ion_charge = 0
            center = self.structure.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
            
            for site_idx in self.generation_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - center[0]
              dy = pos[1] - center[1]
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
            center = self.structure.lattice.get_cartesian_coords([0.5, 0.5, 0.5])

            for site_idx in self.generation_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - center[0]
              dy = pos[1] - center[1]
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
            center = self.structure.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
            for site_idx in self.generation_sites:
              pos = np.array(self.grid_crystal[site_idx].position)
              # Compute distance to (center_x, center_y) in xy-plane
              dx = pos[0] - center[0]
              dy = pos[1] - center[1]
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
            lattice = self.structure.lattice

            for site_idx in self.generation_sites:
                frac = lattice.get_fractional_coords(self.grid_crystal[site_idx].position)
                if (0.45 < frac[0] < 0.55) and (0.45 < frac[1] < 0.55):
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
                if idx_neighbor_plane[0] not in self.active_event_sites: break
            
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
            lattice = self.structure.lattice
            for site_idx in self.generation_sites:
                frac = lattice.get_fractional_coords(self.grid_crystal[site_idx].position)
                if (0.45 < frac[0] < 0.55) and (0.45 < frac[1] < 0.55):
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
            lattice = self.structure.lattice
            for site_idx in self.generation_sites:
                frac = lattice.get_fractional_coords(self.grid_crystal[site_idx].position)
                if (0.45 < frac[0] < 0.55) and (0.45 < frac[1] < 0.55):
                    idx = site_idx
                    break
            queue.append(idx)
            visited = set()
            cluster_size = 29
            
            self.bfs_cluster(queue,visited,cluster_size)
            
            ad_sites_aux = self.generation_sites.copy()
            for site_idx in ad_sites_aux:
              frac = lattice.get_fractional_coords(self.grid_crystal[site_idx].position)
              if frac[2] > 0.02:
                  update_specie_events,update_supp_av = self.introduce_specie_site(site_idx,update_specie_events,update_supp_av)
                  self.update_sites(update_specie_events,update_supp_av)
                    
            ad_sites_aux = self.generation_sites.copy()
            for site_idx in ad_sites_aux:
                frac = lattice.get_fractional_coords(self.grid_crystal[site_idx].position)
                if frac[2] > 0.1:
                    update_specie_events,update_supp_av = self.introduce_specie_site(site_idx,update_specie_events,update_supp_av)
                    self.update_sites(update_specie_events,update_supp_av)
            
    
    # ================================================
    # KMC logic: algorithm and execution of processes  
    # ================================================
    # =========================================================================
    # kMC loop: delegates to KMCLoop (Phase 4)
    # All loop bodies live in kinetix/lattice/kmc_loop.py; the loop holds no
    # simulation state and every read/write goes through ``self.system``.
    # ``_kmc_step`` calls ``self.system.processes(...)`` - this delegate - so
    # the golden trace's instance-level wrapper still observes the catalog.
    # =========================================================================
    @property
    def kmc_loop(self):
        if not hasattr(self, '_kmc_loop'):
            from kinetix.lattice.kmc_loop import KMCLoop
            self._kmc_loop = KMCLoop(self)
        return self._kmc_loop

    def step_kmc(self, rng):
        return self.kmc_loop.step_kmc(rng)

    def _kmc_step(self, rng, E_field_dict, T_field_dict):
        return self.kmc_loop._kmc_step(rng, E_field_dict, T_field_dict)

    def _search_superbasin(self, kmc_time_step):
        return self.kmc_loop._search_superbasin(kmc_time_step)

    def update_superbasin(self, chosen_event):
        return self.kmc_loop.update_superbasin(chosen_event)

    def should_activate_superbasin(self, kmc_time_step):
        return self.kmc_loop.should_activate_superbasin(kmc_time_step)

    def is_filament_percolating(self):
        return self.kmc_loop.is_filament_percolating()

    def _check_event_based_superbasin(self):
        return self.kmc_loop._check_event_based_superbasin()

    def _check_time_based_superbasin(self, kmc_time_step):
        return self.kmc_loop._check_time_based_superbasin(kmc_time_step)

    def _slow_timesteps(self):
        return self.kmc_loop._slow_timesteps()

    # Field solving stays with the SolverCoordinator (Phase 2); ``step_kmc``
    # and ``_kmc_step`` reach it through this delegate.
    def _evaluate_fields_for_kmc(self):
        return self.solver_coordinator._evaluate_fields_for_kmc()


    # =========================================================================
    # kMC event execution: delegates to EventHandler (Phase 3)
    # All handler bodies live in kinetix/lattice/events.py; the handler is
    # stateless and every read/write goes through ``self`` (this system).
    # =========================================================================
    @property
    def event_handler(self):
        if not hasattr(self, '_event_handler'):
            self._event_handler = EventHandler(self)
        return self._event_handler

    # -------------------------------------------------------------------------
    # Delegates kept for callers outside this module: superbasin.py calls
    # ``System_state.processes``; state_loader.py calls ``_introduce_specie_site``
    # / ``update_sites_topology``; the golden trace and the kMC-loop tests wrap
    # or call ``processes`` / ``_update_rates_lazily``; lattice construction
    # calls ``_get_mobile_sites``. ``_kmc_step`` calls ``self.processes(...)`` -
    # through this delegate - so instance-level wrappers (golden trace) work.
    # -------------------------------------------------------------------------

    def processes(self, chosen_event):
        return self.event_handler.processes(chosen_event)

    def update_sites_topology(self, support_update_sites, event_update_sites):
        return self.event_handler.update_sites_topology(support_update_sites, event_update_sites)

    def _update_rates_lazily(self, E_field_dict, T_field_dict):
        return self.event_handler._update_rates_lazily(E_field_dict, T_field_dict)

    def _introduce_specie_site(self, idx, support_update_sites, event_update_sites, chemical_specie, ion_charge=None):
        return self.event_handler._introduce_specie_site(idx, support_update_sites, event_update_sites, chemical_specie, ion_charge)

    def _get_mobile_sites(self, site_indices):
        return self.event_handler._get_mobile_sites(site_indices)

    # -------------------------------------------------------------------------
    # Not moved: lattice construction (crystal_grid) also calls this predicate,
    # so it stays here (EventHandler calls it as ``self.system._is_active_site``)
    # -------------------------------------------------------------------------

    def _is_active_site(self, site_type: str) -> bool:
      """ Check if a site type can host defects or participate in kMC events"""
      return site_type in self._active_site_types



    def track_time(self,t):
        
        self.time += t
        
    def add_time(self):
        
        self.list_time.append(self.time)
        
        
    def get_timestep_limit(self):
        return self.solver_coordinator.get_timestep_limit()

    def should_continue_simulation(self,total_simulation_time):
      """
      Check if simulation should continue based on time criterion.
      
      Parameters:
      -----------
      total_simulation_time : float
          Maximum simulation time in seconds
      
      Returns:
      --------
      continue_sim : bool
          True if simulation should continue, False if terminated
      """
      return self.time < total_simulation_time

    def should_solve_fields_now(self, elec_controller, tol=1e-12):
        return self.solver_coordinator.should_solve_fields_now(elec_controller, tol)
      
    

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
    
            positions = np.array([self.grid_crystal[idx].position for idx in self.active_event_sites])
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
            base_path = Path(path)
            self._write_dump(base_path,i)
            
                    
                    
      
        
    def _species_id_gen(self):
    
      # Ensure deterministic ordering (e.g., sorted by name)
      sorted_defects = sorted(self.defects_config.items(), key=lambda x:x[0])
      
      self.SPECIES_TYPE_MAP = {}
      species_id = 1
      
      for name, defect in sorted_defects:
        symbol = defect['symbol']
        max_passivation = defect.get('max_passivation_level',0)
        
        if max_passivation > 0:
          for level in range(max_passivation + 1):
            # Create unique key
            species_key = f"{symbol}_{level}"
            self.SPECIES_TYPE_MAP[species_key] = species_id
            species_id += 1
        else:
          self.SPECIES_TYPE_MAP[symbol] = species_id
          species_id += 1
      
      
      # Store reverse mapping for lookup during dump writing
      self.SPECIES_ID_TO_TYPE = {v:k for k,v in self.SPECIES_TYPE_MAP.items()}
      
    def _get_species_key(self, site):
      """
      Get species key for dump file, including passivation level if applicable.
      
      Returns:
          str: Species key (e.g., "H", "V_O_0", "V_O_2", "H2")
      """
      symbol = site.defect.chemical_specie
      
      # Check if this defect supports passivation
      defect_name = site._get_current_defect_name()
      defect_config = self.defects_config.get(defect_name,{})
      max_passivation = defect_config.get('max_passivation_level', 0)
      
      if max_passivation > 0:
        return f"{symbol}_{site.defect.passivation_level}"
      else:
        return symbol
                    
    def _write_dump(self, base_path: str = '.', step: int = 0, include_charge: bool = True) -> None:
      """
      Write OVITO-compatible dump file
        
      Parameters
      ----------
      path : str
        Output directory
      step : int
        Simulation step (used in filename)
      include_charge : bool
        Whether to write charge state as custom property
      """
      base_path.mkdir(parents=True, exist_ok=True)
      filename = base_path / f"{step}.dump"
        
      atoms = self._collect_atom_data(include_charge)
      
      # Write file
      with open(filename, 'w') as f:
        self._write_dump_header(f,step,len(atoms), include_charge)
        self._write_dump_atoms(f, atoms, include_charge)
        
        
    
    def _collect_atom_data(self,include_charge: bool) -> List[Dict]:
      """Optimized data collection (separated from I/O for performance)."""
      atoms = []
      
      for idx in self.active_event_sites:
        site = self.grid_crystal[idx]
        species_key = self._get_species_key(site)
        species_id = self.SPECIES_TYPE_MAP.get(species_key)
        
        # Build atom record
        atom = {
          'id': len(atoms) + 1,
          'type': species_id,
          'pos': site.position,
        }
        
        if include_charge:
          atom['charge'] = site.defect.charge
          
        atoms.append(atom)
        
      return atoms
      
    def _write_dump_header(self,f,step:int, n_atoms:int, include_charge: bool) -> None:
      """Write LAMMPS header with correct newline formatting."""
      # Standard LAMMPS dump header
    
      f.write("ITEM: TIMESTEP\n")
      f.write(f'{self.time:.6e}\n')
      f.write('ITEM: NUMBER OF ATOMS\n')
      f.write(f'{n_atoms}\n')
      f.write(f'ITEM: BOX BOUNDS (Angstrom)\n')
      f.write(f'0.0 {float(self.crystal_size[0])}\n')
      f.write(f'0.0 {float(self.crystal_size[1])}\n')
      f.write(f'0.0 {float(self.crystal_size[2])}\n')
        
      # Column definition: add custom properties at the end
      columns = 'id type x y z'
      if include_charge:
        columns += ' charge'
      f.write(f'ITEM: ATOMS {columns}\n')
      
    def _write_dump_atoms(self,f,atoms: List[Dict], include_charge: bool) -> None:
    # Write atom lines
      for atom in atoms:
        x, y, z = atom['pos']
        line = f"{atom['id']} {atom['type']} {x:.5f} {y:.5f} {z:.5f}"
        if include_charge:
          line += f" {atom['charge']:.1f}"
        f.write(line + "\n")
        
                    
                    
    def plot_islands(self,path = '',i = 0):
        
        visited = set()
        active_event_sites_cart = []
        species_ids = []
        
        # Assign unique species ID per island
        for island in self.islands_list:
            for id_cluster, cluster in enumerate(island.cluster_list,start = 1):
                for site in cluster:
                    visited.add(site)
                    active_event_sites_cart.append(self.idx_to_cart(site))
                    species_ids.append(id_cluster)

        # Assign another species ID (e.g., island_id + 1) for remaining unclustered atoms
        remainder_id = id_cluster + 1
        for site in self.active_event_sites:
            if site not in visited:
                visited.add(site)
                active_event_sites_cart.append(self.idx_to_cart(site))
                species_ids.append(remainder_id)
                
        # species_mapping = {self.chemical_specie: 1}  # Example species mapping
        # active_event_sites_cart = [(self.idx_to_cart(site)) for site in self.active_event_sites]
        # species_ids = [species_mapping.get(self.grid_crystal[site].defect.chemical_specie) for site in self.active_event_sites]
        # Define particle IDs
        particle_ids = list(range(1, len(active_event_sites_cart) + 1))  # Unique IDs for each particle
        

        base_path = Path(path)
        dump_file_path = base_path / f"{i}.dump"
        # Write the LAMMPS dump file
        with open(dump_file_path, 'w') as dump_file:
            dump_file.write(f"ITEM: TIMESTEP\n{self.time:.10f}\n")
            dump_file.write("ITEM: NUMBER OF ATOMS\n")
            dump_file.write(f"{len(active_event_sites_cart)}\n")
            dump_file.write("ITEM: BOX BOUNDS (Angstrom)\n")
            dump_file.write(f"0.0 {self.crystal_size[0]}\n")
            dump_file.write(f"0.0 {self.crystal_size[1]}\n")
            dump_file.write(f"0.0 {self.crystal_size[2]}\n")
            dump_file.write("ITEM: ATOMS id type x y z\n")
            for pid, sid, pos in zip(particle_ids, species_ids, active_event_sites_cart):
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
        density = len(self.active_event_sites) * self.mass_specie / (x_size * y_size)
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
            layers[z_idx] += 1 if site.defect.chemical_specie != 'Empty' else 0

        sites_per_layer = len(grid_crystal)/z_steps
        normalized_layers = [count / sites_per_layer for count in layers]
        # Number of sites occupied and percentage of occupation for each layer
        self.layers = [layers, normalized_layers]
        
        # Layer 0 is z = 0, so it doesn't contribute
        self.thickness = sum(normalized_layers) * z_step # (nm)    
        
    def sites_occupation(self):
        
        self.fraction_active_event_sites = len(self.active_event_sites) / len(self.grid_crystal) 
        
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
        active_event_sites = self.active_event_sites
        
        if not hasattr(self, 'num_event'):
            if hasattr(self, 'structure'):
                self.num_event = len(self.structure.get_neighbors(self.structure[0],self.radius_neighbors)) + 2
            else:
                self.num_event = 14

        # Size of histogram: number of neighbors that a particle can have, plus particle without neighbors
        histogram_neighbors = [0] * (self.num_event - 1)
        
        for site in active_event_sites:
            if 'bottom_layer' in grid_crystal[site].supp_by or 'Substrate' in grid_crystal[site].supp_by: 
                histogram_neighbors[len(grid_crystal[site].supp_by)-1] += 1
            else:
                histogram_neighbors[len(grid_crystal[site].supp_by)] += 1
                
        self.histogram_neighbors = histogram_neighbors
        
        
    # ----------------------------------------------------
    #    Metal clusters and islands calculations
    # ----------------------------------------------------
    
    # _initialize_cluster_tracking moved to kinetix/lattice/lattice_builder.py
    # (Phase 5); its delegate lives under the "Lattice construction" banner.

    def _add_metal_atom_to_clusters(self,site_id):
      """
      Add a newly reduced metal atom at site_id to cluster system.
      """
      
      # Get metal neighbors (only neutral atoms)
      metal_neighbors = [
        nb for nb in self.grid_crystal[site_id].supp_by 
        if not isinstance(nb, str) and self.grid_crystal[nb].defect.charge == 0
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
       
       
       # 1. Remove atom from cluster
       cluster.atoms_id.discard(site_id)
       cluster.size = len(cluster.atoms_id)
       del self.atom_to_cluster[site_id]
       
       # Clean positions: Remove the corresponding atom position
       target_pos = self.grid_crystal[site_id].position
       cluster.atoms_positions = [
        pos for pos in cluster.atoms_positions
        if not np.allclose(pos, target_pos)
       ]
            
       # Reset the site's electrode flag (no longer in any cluster)
       self.grid_crystal[site_id].in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}   
       
       
       if cluster.size < 2:
         # Remove cluster
         for atom in list(cluster.atoms_id):
           del self.atom_to_cluster[atom]
           self.grid_crystal[atom].in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}
         del self.clusters[cid]
         return
         
       # 3. Check connectivity and split if ruptured
       components = self._dfs_find_components(list(cluster.atoms_id), self.grid_crystal)  
         
       if len(components) == 1:
         # Still connected: just update flags
         cluster.update_electrode_contact(self.grid_crystal)
         return
         
       # 4. Fragmented: delete old cluster, create valid fragments only
       del self.clusters[cid]
       
       for comp in components:
         if len(comp) < 2:
           # Dissolve small fragments, cluster should have at least 2 atoms
           for atom in comp:
             del self.atom_to_cluster[atom]
             self.grid_crystal[atom].in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}
           continue 
             
         # Create new cluster for valid fragment
         positions = [self.grid_crystal[a].position for a in comp]
         new_cid = self.next_cluster_id 
         self.next_cluster_id += 1
         
         new_cluster = Cluster(comp, positions, {}, self.conductivity)
         new_cluster.update_electrode_contact(self.grid_crystal)
         self.clusters[new_cid] = new_cluster
         
         for atom in comp:
           self.atom_to_cluster[atom] = new_cid  

    def _dfs_find_components(self, cluster_atoms, grid_crystal):
      """
      Find connected components within a single cluster using DFS.
      Returns list of components, where each component is a list of site IDs.
      """
      atom_set = set(cluster_atoms)
      visited = set()
      components = []
      
      for start_atom in cluster_atoms:
        if start_atom in visited:
          continue
          
        component = []
        stack = [start_atom]
          
        while stack:
          atom = stack.pop()
          if atom in visited:
            continue
            
          visited.add(atom)
          component.append(atom)
            
          # Filter neighbors: must be metal (charge 0) AND in the same cluster
          for nb in grid_crystal[atom].nearest_neighbors_idx:
            if nb in atom_set and grid_crystal[nb].defect.charge == 0 and nb not in visited:
              stack.append(nb)
              
        components.append(component) 
           
      return components 

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
            
            for idx_site in self.active_event_sites:   

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
        
        if idx_site not in visited and site.defect.chemical_specie == chemical_specie:
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
    
            if element[0] not in visited and self.grid_crystal[element[0]].defect.chemical_specie == chemical_specie:
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
        
                if element[0] not in visited and self.grid_crystal[element[0]].defect.chemical_specie == chemical_specie:
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

        active_event_sites = self.active_event_sites
    
        # Convert occupied sites to Cartesian coordinates and sort by z-coordinate in descending order
        active_event_sites_cart = sorted(
            ((self.idx_to_cart(site), site) for site in active_event_sites), 
            key=lambda coord: coord[0][2], 
            reverse=True
        )
    
        total_visited = set()
        peak_list = []
        
        for cart_coords, site in active_event_sites_cart:
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
    
                if element[0] not in peak_sites and grid_crystal[element[0]].defect.chemical_specie == chemical_specie:
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
        if idx_site not in visited and self._is_inside_supercell(cart_site, self.structure.lattice):
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
   
            
            if self._is_inside_supercell(cart_site, self.structure.lattice):
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
                if grid_crystal[jump[0]].defect.chemical_specie == 'Empty': top_layer_empty_sites +=1
                     
            if (site.defect.chemical_specie != 'Empty') and top_layer_empty_sites >= 2:
                x.append(site.position[0])
                y.append(site.position[1])
                z.append(site.position[2]+z_step)
                
            elif (site.defect.chemical_specie == 'Empty') and ('bottom_layer' in site.supp_by) and top_layer_empty_sites >= 2:
                x.append(site.position[0])
                y.append(site.position[1])
                z.append(site.position[2])
                
        return x,y,z