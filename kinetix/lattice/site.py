# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:12 2024

@author: samuel.delgado
"""
from scipy import constants
import numpy as np
from sklearn.decomposition import PCA
import os
from typing import NamedTuple


class Site():
    
    def __init__(self,chemical_specie,position,site_type=None,Act_E_dict=None, defects_config = None, reactions_config = None):
        """
        Initialize a site in the kMC grid.
        
        Parameters:
            chemical_specie (str): Current occupancy ('O', 'Hf', 'Ag', 'Empty', etc.)
            position (tuple): Cartesian coordinates (x, y, z)
            site_type (str, optional): Permanent identity ('O', 'Hf', 'interstitial', 'fcc_hollow')
            Act_E_dict (dict, optional): Activation energies for this site
        """
        # Core properties
        self.chemical_specie = chemical_specie
        self.position = position
        self.site_type = site_type if site_type is not None else chemical_specie
        
        # Neighbor information
        self.nearest_neighbors_idx = [] # Nearest neighbors indexes
        self.nearest_neighbors_cart = [] # Nearest neighbors cartesian coordinates
        
        # Activation energies
        self.Act_E_dict = Act_E_dict  
        self.defects_config = defects_config 
        self.reactions_config = reactions_config
        
        # Event tracking
        self.site_events = [] # Possible events corresponding to this node
        self.migration_paths = {'Plane':[],'Up':[],'Down':[]} # Possible migration sites with the corresponding label

        # Cache memory            
        self.cache_planes = {}
        self.cache_TR = {}
        self.cache_edges = {}
        self.cache_site_energy = {}
        self.cache_CN_contr_redox_energy = {}
        
        # Physical constants
        self.kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.nu0=7E12  # nu0 (s^-1) bond vibration frequency
        
        # Electrostatic properties
        self.ion_charge = 0
        self.in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}
        self.is_at_bottom_interface = False
        self.is_at_top_interface = False

        
        # Calculate applicable defects
        self.applicable_defects = self._get_applicable_defects()
        current_defect = self._get_current_defect_name()
        if current_defect is not None:
          self.sites_generation_layer = defects_config[current_defect]["sites_generation_layer"]
        
        if current_defect is not None and "passivation_level" in defects_config[current_defect]:
          self.passivation_level = defects_config[current_defect]["passivation_level"]   
          

# =============================================================================
#     Helper methods           
# =============================================================================        
    def _get_applicable_defects(self):
      """Determine which defects can occupy this site type."""
      if self.defects_config is None:
        return []
        
      applicable = []
      for defect_name, cfg in self.defects_config.items():
        allowed_sublattices = cfg.get("allowed_sublattices",[])
        if self.site_type in allowed_sublattices:
          applicable.append(defect_name)
      return applicable
      
    def _get_current_defect_name(self):
      """Determine which defect configuration applies to current state."""  
      for defect_name in self.applicable_defects or []:
        if defect_name in self.Act_E_dict:
          defect_site_type = self.defects_config[defect_name]["site_type"]
          if defect_site_type == self.site_type:
            return defect_name
      return None
      
    def _is_mobile_site(self,site):
      """ Check if site can host mobile defects """
      return any(site.site_type in cfg.get("allowed_sublattices",[])
                 for cfg in self.defects_config.values())
      
# =============================================================================
#     We only consider the neighbors within the lattice domain            
# =============================================================================
    def neighbors_analysis(self,grid_crystal,neigh_idx,neigh_cart,crystal_size,event_labels,idx_origin):
       
        tol = 1e-6

        for idx,pos in zip(neigh_idx,neigh_cart):
                    
          if self._is_mobile_site(grid_crystal[idx]):
            self.nearest_neighbors_idx.append(tuple(idx))             
            self.nearest_neighbors_cart.append(tuple(pos))
                
            # Migration in the plane
            if -tol <= (pos[2]-self.position[2]) <= tol:
              self.migration_paths['Plane'].append([tuple(idx),event_labels[tuple(idx - np.array(idx_origin))]])
            # Migration upward
            elif (pos[2]-self.position[2]) > tol:
              self.migration_paths['Up'].append([tuple(idx),event_labels[tuple(idx - np.array(idx_origin))]])     
            # Migration downward
            elif (pos[2]-self.position[2]) < -tol:
              self.migration_paths['Down'].append([tuple(idx),event_labels[tuple(idx - np.array(idx_origin))]])
              
        self.mig_paths_plane = {num_event:site_idx for site_idx, num_event in self.migration_paths['Plane']}   

# =============================================================================
#         Occupied sites supporting this node
# =============================================================================    
    def supported_by(self,grid_crystal,wulff_facets,dir_edge_facets,domain_height,idx_origin):
        
        # Initialize supp_by as an empty list
        self.supp_by = []
        
        # Position close to 0 are supported by the substrate
        tol = 1e-6
        if self.position[2] <= tol:
            self.supp_by.append('bottom_layer')
            self.is_at_bottom_interface = True
        if abs(self.position[2] - domain_height) < tol:
            self.supp_by.append('top_layer')
            self.is_at_top_interface = True

        current_defect = self._get_current_defect_name()
        if self.chemical_specie != "Empty" and self.defects_config[current_defect]["CN_matters"]:
          # Go over the nearest neighbors
          for idx in self.nearest_neighbors_idx:
            neighbor = grid_crystal[idx]
            # Select the occupied sites that support this node
            if (neighbor.chemical_specie == self.chemical_specie and idx != idx_origin):
              self.supp_by.append(idx)
        
        # Calculate destination coordination for empty sites          
        elif self.chemical_specie == "Empty" and self.applicable_defects:
          self.destination_CN = {}
          for defect_name in self.applicable_defects:
            defect_config = self.defects_config.get(defect_name)
            if not defect_config or not defect_config.get("CN_matters", False):
              continue
            
            if defect_name not in self.Act_E_dict:
              continue
              
            defect_specie = defect_config["symbol"]
            cn_count = 0
            
            for idx in self.nearest_neighbors_idx:
              neighbor = grid_crystal[idx]
              if (neighbor.chemical_specie == defect_specie and idx != idx_origin):
                cn_count += 1
            self.destination_CN[defect_name] = cn_count 
            
              
           
        # Convert supp_by to a tuple
        self.supp_by = tuple(self.supp_by)
        self.calculate_site_energy()
        
        # Check for redox-capable defects
        has_redox = False
        if current_defect in self.Act_E_dict:
          defect_energies = self.Act_E_dict[current_defect]
          if 'E_reduction' in defect_energies or 'E_oxidation' in defect_energies:
            has_redox = True
        
        if has_redox:
          self.calculate_CN_contribution_redox_energy()
        
        if wulff_facets is not None and dir_edge_facets is not None:
          self.detect_edges(grid_crystal,dir_edge_facets,chemical_specie)               
          self.detect_planes(grid_crystal,wulff_facets[:14])
        
                
    def calculate_site_energy(self,destination_CN=None, origin_idx=None, is_at_top_interface=False, is_at_bottom_interface=False):
        """
        Calculate the site energy based on coordination environment.
        
        Parameters:
            destination_support (tuple, optional): Support tuple for destination site calculation
            origin_idx (tuple, optional): Origin site index for migration energy difference
            
        Returns:
            float: Site energy (eV)
        """
        # If this site is supported by the substrate, we add the binding energy to the substrate
        # We reduce 1 if it is supported by the substrate
        # We add 1 because if the site is occupied
        
        # Determine current defect
        current_defect = self._get_current_defect_name()
        if current_defect is None:
          return 0.0

        defect_energies = self.Act_E_dict[current_defect]
        cn_energies = defect_energies['CN_clustering_energy']
        
        
        if destination_CN is None:
            # Calculate energy for current site
            # Check memory cache
            cache_key = self.supp_by
            if cache_key in self.cache_site_energy:
                self.energy_site = self.cache_site_energy[cache_key]
                return self.energy_site
                
            support_tuple = self.supp_by
            cn_index = len(support_tuple)

            if 'top_layer' in self.supp_by:
                energy = cn_energies[cn_index] + defect_energies.get('binding_energy_top_layer',0.0)
            elif 'bottom_layer' in self.supp_by:
                energy = cn_energies[cn_index] + defect_energies.get('binding_energy_bottom_layer',0.0)
            else:
                energy = cn_energies[cn_index]
                
            # Store the result in the cache
            self.cache_site_energy[cache_key] = energy
            self.energy_site = energy
            return energy
                
        # We should consider the particle that would migrate there to calculate
        # the energy difference with the origin site
        else:
            
            if is_at_top_interface:
              energy = cn_energies[destination_CN] + defect_energies.get('binding_energy_top_layer', 0.0)
            elif is_at_bottom_interface:
              energy = cn_energies[destination_CN] + defect_energies.get('binding_energy_bottom_layer', 0.0)
            else:
              energy = cn_energies[destination_CN]
                
            return energy    
            
    def calculate_CN_contribution_redox_energy(self):
      """
      Calculate redox energy contribution based on coordination environment.
      Higher CN imply higher activation barriers for oxidation and lower for reduction
      Uses current defect context for multi-species support 
      """
      # Determine current defect context
      current_defect = self._get_current_defect_name()
      if current_defect is None:
        return None
        
      defect_energies = self.Act_E_dict[current_defect]
      
      # Check if this defect has redox energies
      if "CN_redox_energy" not in defect_energies:
        return 0.0
        
      cn_redox_energies = defect_energies['CN_redox_energy']
      
      # Check memory cache
      cache_key = self.supp_by
      if cache_key in self.cache_CN_contr_redox_energy:
        self.CN_redox_energy = self.cache_CN_contr_redox_energy[cache_key]
        return self.CN_redox_energy
        
      # Calculate redox energy based on support environment
      cn_index = len(self.supp_by)
        
      print(f'CN index: {cn_index}, CN redox energies length: {len(cn_redox_energies)}')
      print(f'Supp by: {self.supp_by}')
      
      if 'bottom_layer' in self.supp_by: # Bottom interface
        energy = cn_redox_energies[cn_index] + defect_energies['redox_bottom_electrode']
      elif 'top_layer' in self.supp_by: # Top interface
        energy = cn_redox_energies[cn_index] + defect_energies['redox_top_electrode']
      else: # Bulk
        energy = cn_redox_energies[cn_index + 1]
        
      self.cache_CN_contr_redox_energy[cache_key] = energy
      self.CN_redox_energy = energy
      return energy
    

   
# =============================================================================
#       Calculate the possible events corresponding to this node
#       - Migration events
#       - Desorption events
# =============================================================================
    # Change chemical_specie status
    # Add the desorption process
    def introduce_specie(self,chemical_specie,ion_charge = None):
        self.chemical_specie = chemical_specie
        
        if ion_charge is None:
          current_defect = self._get_current_defect_name()
          ion_charge = self.defects_config[current_defect]['charge']
        self.ion_charge = ion_charge

    def remove_specie(self,affected_site):
        self.chemical_specie = affected_site
        self.ion_charge = 0
        #self.site_events.remove(['Desorption',self.num_event])
        self.site_events = []
        
    def available_pathways(self,grid_crystal,idx_origin, facets_type):
    
      self.site_events = []
      current_defect = self._get_current_defect_name()
      
      if current_defect == None:
        return
      
      enabled_events = self.defects_config[current_defect]["enabled_events"]
      
      if 'migration' in enabled_events:      
          self.available_migrations(grid_crystal,idx_origin,facets_type)
      if 'reduction' in enabled_events: 
          self.available_reduction(idx_origin)
      if 'oxidation' in enabled_events:
          self.available_oxidation(idx_origin)
      if 'reaction' in enabled_events:
          self.available_reactions(grid_crystal,idx_origin)
    

    # Calculate posible migration sites
    def available_migrations(self,grid_crystal,idx_origin,facets_type):
        
        # Deposition experiments
        if facets_type is not None:
    # =============================================================================
    #         Lü, B., Almyras, G. A., Gervilla, V., Greene, J. E., & Sarakinos, K. (2018). 
    #         Formation and morphological evolution of self-similar 3D nanostructures on weakly interacting substrates. 
    #         Physical Review Materials, 2(6). https://doi.org/10.1103/PhysRevMaterials.2.063401
    #         - Number of nearest neighbors needed to support a site so a particle can migrate there
    # =============================================================================
            
    
            # Plane migrations
            for site_idx, num_event in self.migration_paths['Plane']:
                if site_idx not in self.supp_by and (self.sites_generation_layer in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 2):
                    energy_site_destiny = self.calculate_site_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                    
                    # Migrating on the substrate
                    if self.sites_generation_layer in self.supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[0] + energy_change])
                        
                    # Migrating on the film (111)
                    elif grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        if self.edges_v[num_event] == None: 
                            self.site_events.append([site_idx, num_event, self.Act_E_dict[7] + energy_change])
                        elif self.edges_v[num_event] == facets_type[0]:
                            self.site_events.append([site_idx, num_event, self.Act_E_dict[10] + energy_change])
                        elif self.edges_v[num_event] == facets_type[1]:
                            self.site_events.append([site_idx, num_event, self.Act_E_dict[9] + energy_change])
                            
                    # Migrating on the film (100)
                    elif grid_crystal[site_idx].wulff_facet == facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[8] + energy_change])
    
    
    # =============================================================================
    #         Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2020). 
    #         The role of Ru passivation and doping on the barrier and seed layer properties of Ru-modified TaN for copper interconnects. 
    #         Journal of Chemical Physics, 152(14). https://doi.org/10.1063/5.0003852
    #   
    #         - Migration upward stable is supported by three particles??  
    # =============================================================================                      
            # Upward migrations
            for site_idx, num_event in self.migration_paths['Up']:
            
                    # First nearest neighbors: 1 jump upward
                    # Supported by at least 2 particles (excluding this site)
    
                if site_idx not in self.supp_by and len(grid_crystal[site_idx].supp_by) > 2:
                    energy_site_destiny = self.calculate_site_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                   
                    # Migrating upward from the substrate
                    if self.sites_generation_layer in self.supp_by and grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[1] + energy_change])
                    
                    elif self.sites_generation_layer in self.supp_by and grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[5] + energy_change])
                        
                    # Migrating upward from the film (111)
                    elif self.wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[3] + energy_change])
                        
                    # Migrating upward from the film (100)
                    elif self.wulff_facet ==  facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[8] + energy_change])
    
                    
            # Downward migrations
            for site_idx, num_event in self.migration_paths['Down']:
    
                    # First nearest neighbors: 1 jump downward
                    # Supported by at least 2 particles (excluding this site)
                if site_idx not in self.supp_by and (self.sites_generation_layer in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 1):
                    energy_site_destiny = self.calculate_site_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                    
                    # From layer 1 to substrate
                    if self.wulff_facet == facets_type[0] and self.sites_generation_layer in grid_crystal[site_idx].supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[2] + energy_change])
                    
                    elif self.wulff_facet == facets_type[1] and self.sites_generation_layer in grid_crystal[site_idx].supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[6] + energy_change])
                    
                    # Migrating downward from the film (111)
                    elif self.wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[4] + energy_change])
                        
                    # Migrating downward from the film (100)
                    elif self.wulff_facet == facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_dict[8] + energy_change])
                
            
            
        # Migration of interstitial sites
        else:
            # Get current defect name for energy lookup
            current_defect = self._get_current_defect_name()
            Act_E_mig = self.Act_E_dict[current_defect]['E_mig']
            allowed_sublattices = self.defects_config[current_defect]['allowed_sublattices']
            valid_target_species = self.defects_config[current_defect]['valid_target_species']
            
            # Migration types
            migration_types = ['Plane', 'Up', 'Down']
            
            
            for migration_type in migration_types:
              for site_idx, num_event in self.migration_paths[migration_type]:
                dest_site = grid_crystal[site_idx]
                
                # 1. Geometric Check: Is this lattice position valid?
                if dest_site.site_type not in allowed_sublattices:
                  continue
                  
                # 2. Chemical Check: Is the site available (Empty or Vacancy)?
                if dest_site.chemical_specie not in valid_target_species:
                  continue
                  
                # 3. Support/Topology Check
                if site_idx in self.supp_by: 
                  continue
                
                # 4. Energy Calculation: Calculate energy difference between sites
                energy_site_destiny = self.calculate_site_energy(
                  dest_site.destination_CN[current_defect],idx_origin,
                  dest_site.is_at_top_interface, dest_site.is_at_bottom_interface
                )
                
                # 5. Barrier Model   
                energy_change = max(energy_site_destiny - self.energy_site, 0)
                self.site_events.append([site_idx, num_event, Act_E_mig[num_event] + energy_change])

            
    def available_reduction(self,idx_origin):
        current_defect = self._get_current_defect_name()
        if self.ion_charge > 0:
          E_reduction = self.Act_E_dict[current_defect]['E_reduction']
          CN_redox = self.CN_redox_energy
          self.site_events.append([idx_origin, 'reduction', E_reduction - CN_redox])
            
    def available_oxidation(self,idx_origin):
        current_defect = self._get_current_defect_name()
        is_surface_atom = (len(self.supp_by) < len(self.nearest_neighbors_idx)) # Fully coordinated atoms can't oxidize
        at_electrode_interface = ('top_layer' in self.supp_by) or ('bottom_layer' in self.supp_by) # Atoms at the interface with the top or bottom electrode can oxidize
        can_oxidize = self.ion_charge == 0 # Neutral atoms can oxidize
        
        if can_oxidize and (is_surface_atom or at_electrode_interface):
          E_oxidation = self.Act_E_dict[current_defect]['E_oxidation']
          CN_redox = self.CN_redox_energy  
          self.site_events.append([idx_origin, 'oxidation', E_oxidation + CN_redox])
    
          
    def available_reactions(self,grid_crystal,idx_origin):
        """
        Check all possible reactions involving this site.
        """
        site = grid_crystal[idx_origin]
        current_specie = site.chemical_specie
        
        for reaction_name, reaction in self.reactions_config.items():
          if not reaction.get("enabled", False):
            continue
                     
          # 1. Check if this site can act as a reactant for this reaction
          if not self._site_can_participate(site,reaction):
            continue
          
          # 2. Handle reactions
          if reaction['type'] ==  "bimolecular_neighbor":
            # For example: H + H -> H2
            self._handle_bimolecular_neighbor_reaction(grid_crystal,site,reaction)
            
          elif reaction['type'] == "bimolecular_capture":
            # Example: H + V_O -> V_OH (H hops into a neighbor V_O)
            # Triggered from the H site, targeting the V_O neighbor
            self._handle_bimolecular_capture_reaction(grid_crystal,site,reaction)
            
          elif reaction['type'] == "unimolecular_escape":
            # Example: V_OH -> V_O + H (Depassivation)
            self._handle_unimolecular_reaction(site,idx_origin,reaction)
            
            
    def _site_can_participate(self,site,reaction):
        """
        Filter to check if site specie and state match reaction requeriments
        """
        
        for reactant in reaction["reactants"]:
          # Check if chemical_specie matches this reactant role
          if site.chemical_specie == reactant["symbol"]:
          
            # Check sublattice constraint
            if "sublattice" in reactant: 
              if site.site_type != reactant["sublattice"]:
                continue
            
            return True
        
        return False 
        
    def _site_matches_reactant(self,site,reactant):
        """
        Check if site matches a specific reactant role (symbol + sublattice).
        """
        # Check chemical specie
        if site.chemical_specie != reactant['symbol']:
          return False
          
        if 'sublattice' in reactant:
          if site.site_type != reactant['sublattice']:
            return False
        return True
        
    def _handle_bimolecular_neighbor_reaction(self,grid_crystal,site,reaction):
        """
        Handle reactions requiring an adjacent partner (e.g., H + H -> H2).
        """
        reactants = reaction["reactants"]
        current_defect = self._get_current_defect_name()
        Act_E = self.Act_E_dict[current_defect][reaction['name']]
             
        my_role_idx = -1
        # Find which reactant role the origin site fulfills (0 or 1)
        for i,reactant in enumerate(reactants):
          if self._site_matches_reactant(site,reactant):
            my_role_idx = i
            break
            
        if my_role_idx == -1:
          return # Origin doesn't match any reactant
          
        # Neighbor must fulfill the other role
        partner_role_idx = 1 - my_role_idx
        partner_requirements = reactants[partner_role_idx]  
        
        # Check neighbors
        for neighbor_idx in site.nearest_neighbors_idx:
          neighbor = grid_crystal[neighbor_idx]
          if self._site_matches_reactant(neighbor,partner_requirements):
            self.site_events.append([
              neighbor_idx, 
              reaction['name'], 
              Act_E
            ])
            
            
    def _handle_bimolecular_capture_reaction(self,grid_crystal,site,reaction):
      """
      Handle capture reactions (e.g., H interstitial hops into V_O site).
      """
      reactants = reaction["reactants"]
      current_defect = self._get_current_defect_name()
      
      # Check if this defect drives the reaction
      if reaction['name'] not in self.Act_E_dict[current_defect]:
        return
      
      Act_E = self.Act_E_dict[current_defect][reaction['name']]
      
    
      my_role_idx = -1
      # Find which reactant role the origin site fulfills (0 or 1)
      for i,reactant in enumerate(reactants):
        if self._site_matches_reactant(site,reactant):
          my_role_idx = i
          break
      
      if my_role_idx == -1:
        return
          
      # Neighbor must fulfill the other role
      partner_role_idx = 1 - my_role_idx
      partner_requirements = reactants[partner_role_idx]
      
      for neighbor_idx in site.nearest_neighbors_idx:
        neighbor = grid_crystal[neighbor_idx]
        
        if self._site_matches_reactant(neighbor,partner_requirements):
          # Check passivation level of the trap
          neighbor_defect = neighbor._get_current_defect_name()
          max_passivation = self.defects_config[neighbor_defect]["max_passivation_level"]
          
          if neighbor.passivation_level >= max_passivation:
            continue # Trap is full
            
          if isinstance(Act_E,dict):
            passivation_key = str(neighbor.passivation_level)
            Act_E_value = Act_E[passivation_key]
          else:
            Act_E_value = Act_E
          
          self.site_events.append([
            neighbor_idx,
            reaction['name'],
            Act_E_value
          ])     
            
    def _handle_unimolecular_reaction(self,site,idx_origin,reaction):
      """
      Handle reactions that happen on a single site (e.g., V_OH -> V_O + H).
      """
      reactants = reaction["reactants"]
      
      min_passivation = reactants[0]['min_passivation']
      if site.passivation_level < min_passivation:
        return
      
      current_defect = self._get_current_defect_name()
      Act_E = self.Act_E_dict[current_defect][reaction['name']]
      
      if isinstance(Act_E, dict):
        passivation_key = str(site.passivation_level)
        Act_E_value = Act_E[passivation_key]
      else:
        Act_E_value = Act_E
        
      self.site_events.append([
        idx_origin,
        reaction['name'],
        Act_E_value
      ])
        
        
    def deposition_event(self,TR,idx_origin,num_event,Act_E):
        self.site_events.append([TR,idx_origin, num_event, Act_E])
        
    def ion_generation_interface(self,idx_origin):
        
        current_defect = self._get_current_defect_name()
        if current_defect is None:
          return

        E_gen = self.Act_E_dict[current_defect]['E_gen_defect']
        self.site_events.append([idx_origin, 'generation', E_gen])
        
    def remove_event_type(self,num_event):
        
        for i, event in enumerate(self.site_events):
            if event[2] == num_event:
                del self.site_events[i]
                break
            
    def detect_planes_test(self,System_state):
        
        atom_coordinates = np.array([System_state.grid_crystal[idx].position for idx in self.supp_by if idx != self.sites_generation_layer])

        self.miller_index = System_state.structure.lattice.get_miller_index_from_coords(atom_coordinates, coords_are_cartesian=True, round_dp=0, verbose=True)
                
        return self.miller_index
    
    
# =============================================================================
#     Detect planes using PCA - We search the plane that contains most of the points
#     in supp_by  to know the surface where this site is attached 
# =============================================================================

    def detect_planes(self,grid_crystal,wulff_facets):
        
        atom_coordinates = np.array([grid_crystal[idx].position for idx in self.supp_by if idx != self.sites_generation_layer])
        # atom_coordinates = tuple([grid_crystal[idx].position for idx in self.supp_by if idx != 'Substrate'])
        # Order the coordinates according to the first value, then the second, etc.
        # We are ordering the row, not the elements of the coordinates (x,y,z)
        # sorted_atom_coordinates = sorted(atom_coordinates, key=lambda x: tuple(x))
        # sorted_atom_coordinates = tuple(map(tuple, sorted_atom_coordinates))
        cache_key = self.supp_by
        
        # Check if the result is already cached
        if self.sites_generation_layer in self.supp_by:
            self.wulff_facet = (1,1,1)
            return
        
        if cache_key in self.cache_planes:
            self.wulff_facet = self.cache_planes[cache_key]
            return
        
        elif len(atom_coordinates) > 2:
            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(atom_coordinates)
            
            # Get the eigenvectors and eigenvalues
            eigenvectors = pca.components_
            eigenvalues = pca.explained_variance_
            
            # Sort eigenvectors based on eigenvalues
            sorted_indices = np.argsort(eigenvalues)[::-1]
            principal_components = eigenvectors[sorted_indices]
            
            # Define the plane by the two principal components with the largest eigenvalues
            plane_normal = np.cross(principal_components[0], principal_components[1])
            self.plane_normal = plane_normal
            
            aux_min = 2
            for plane in wulff_facets:
                cross_product = np.cross(plane[1],plane_normal)
                norm_cross_product = np.linalg.norm(cross_product)
                if norm_cross_product < aux_min:
                    aux_min = norm_cross_product
                    self.wulff_facet = plane[0]
                    
        else:
            self.wulff_facet = (1,1,1)
            
        # Cache the result
        self.cache_planes[cache_key] = self.wulff_facet
           
            
    def detect_edges(self,grid_crystal,dir_edge_facets,chemical_specie):
        
        # cache_key = tuple(sorted(self.supp_by, key=lambda x: str(x)))
        cache_key = self.supp_by
        
        if cache_key in self.cache_edges:
            self.cache_edges[cache_key]
            return 

        self.edges_v = {i:None for i in self.mig_paths_plane.keys()}
        
        bottom_support = all(site_idx in self.supp_by for site_idx, num_event in self.migration_paths['Down'])
            
        # To be an edge it must be support by the substrate or the atoms from the down layer
        if self.sites_generation_layer in self.supp_by or bottom_support:
            # Check for each migration direction the edges that are parallel
            for num_event,site_idx in self.mig_paths_plane.items():
                edges = dir_edge_facets[num_event]
            
                # Check if one of the edges is occupied for the chemical speice (both sites)
                for edge in edges:
                    if (grid_crystal[self.mig_paths_plane[edge[0][0]]].chemical_specie == chemical_specie 
                        and grid_crystal[self.mig_paths_plane[edge[0][1]]].chemical_specie == chemical_specie):
                        self.edges_v[num_event] = edge[1] # Associate the edge with the facet
                    
        # Store the result in the cache
        self.cache_edges[cache_key] = self.edges_v
        
    def unit_vector(self,vector):
        """ Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)
                
    def angle_between(self,v1, v2):
        """Finds angle between two vectors"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    

            
            

# =============================================================================
#         Calculate transition rates    
# =============================================================================
    def transition_rates(self,**kwargs):
        
        T = kwargs.get("T", 300)
        E_site_field = kwargs.get("E_site_field", np.array([0.0, 0.0, 0.0]))
        migration_pathways =  kwargs.get("migration_pathways")
        current_defect = self._get_current_defect_name()
        relevant_field = np.any(abs(E_site_field) > 1e6)
        
        # Ions in contact with the virtual electrode will have the reduction rate affected
        clusters = kwargs.get("clusters")
        atom_to_cluster = kwargs.get("atom_to_cluster", {})
        self.in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}
              
        if len(atom_to_cluster) > 0:        
          for nb in self.supp_by:
            if (not isinstance(nb, str) and nb in atom_to_cluster):    
              cid = atom_to_cluster[nb]
              cluster = clusters[cid]
              if cluster.attached_layer['bottom_layer']:
                self.in_cluster_with_electrode['bottom_layer'] = True
              if cluster.attached_layer['top_layer']:
                self.in_cluster_with_electrode['top_layer'] = True
          
                
        
        # Iterate over site_events directly, no need to use range(len(...))
        for event in self.site_events:
          
            if relevant_field:
              
              if event[-2] == 'generation':
                Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_dict[current_defect]['E_min_gen'])
                  
                            
              elif event[-2] in ('reduction', 'oxidation'):
                base_energy = event[-1]
                process = event[-2]
                field_factor_top = -0.5 # Field opposes reduction at top
                field_factor_bottom = +0.5 # Field assists reduction at bottom
                
                # Determine field scaling and energy floor based in process type
                if process == "reduction":
                  min_energy = self.Act_E_dict[current_defect]['E_reduction_min']
                else: # oxidation
                  min_energy = self.Act_E_dict[current_defect]['E_min_gen']
                  
                # Compute field projection
                # For oxidation, effective field is reversed
                field_proj = np.dot(E_site_field, [0,0,1]) * 1e-10
                if process == 'oxidation':
                  field_proj *= -1
                  
                field_contribution = round(field_proj,3)
                
                Act_E = base_energy
                
                # Apply top electrode correction
                if self.in_cluster_with_electrode['top_layer'] or 'top_layer' in self.supp_by:
                  Act_E = max(base_energy + field_factor_top * field_contribution, min_energy)
                
                # Apply bottom electrode correction
                if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                  Act_E = max(base_energy + field_factor_bottom * field_contribution, min_energy)
                  
                  
              elif isinstance(event[-2], int):
                mig_vec = migration_pathways[event[-2]]['direction']
                Act_E = max(event[-1] - self.ion_charge * round(np.dot(E_site_field,mig_vec) * 1e-10,3),self.Act_E_dict[current_defect]['E_min_mig'])
                
              elif any(event[-2] == reaction['name'] for reaction in self.reactions_config.values()): # Reactions
                Act_E = event[-1]
                
            else:
              Act_E = event[-1]
              
            # Fallback: Act. energy should be >= 0
            Act_E = max(Act_E,0)
            
            if Act_E in self.cache_TR:
                tr_value = self.cache_TR[Act_E]

            else:
                tr_value = self.nu0 * np.exp(-Act_E / (self.kb * T))
                self.cache_TR[Act_E] = tr_value
                
            # Use the length of event to determine the appropriate action
            if len(event) == 3:
                # Insert at the beginning of the list for the binary tree
                event.insert(0, tr_value)
            elif len(event) == 4:
                event[0] = tr_value
                
                