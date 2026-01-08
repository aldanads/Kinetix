# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:46:12 2024

@author: samuel.delgado
"""
from scipy import constants
import numpy as np
from sklearn.decomposition import PCA
import os
import copy


class Site():
    
    def __init__(self,chemical_specie,position,Act_E_list):
        
        self.chemical_specie = chemical_specie
        self.position = position
        self.nearest_neighbors_idx = [] # Nearest neighbors indexes
        self.nearest_neighbors_cart = [] # Nearest neighbors cartesian coordinates
        self.Act_E_list = copy.deepcopy(Act_E_list) # Its own copy 
        self.site_events = [] # Possible events corresponding to this node
        self.migration_paths = {'Plane':[],'Up':[],'Down':[]} # Possible migration sites with the corresponding label

        # Cache memory            
        self.cache_planes = {}
        self.cache_TR = {}
        self.cache_edges = {}
        self.cache_clustering_energy = {}
        self.cache_CN_contr_redox_energy = {}
        
        self.kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.nu0=7E12  # nu0 (s^-1) bond vibration frequency
        
        self.ion_charge = 0
        self.in_cluster_with_electrode = {'bottom_layer': False, 'top_layer': False}

# =============================================================================
#     We only consider the neighbors within the lattice domain            
# =============================================================================
    def neighbors_analysis(self,grid_crystal,neigh_idx,neigh_cart,crystal_size,event_labels,idx_origin):
       
        tol = 1e-6

        for idx,pos in zip(neigh_idx,neigh_cart):
            
            if tuple(idx) in grid_crystal:
                    
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

                    
            # Establish boundary conditions for neighbors in xy plane
            # If pos is out of the boundary in xy but within z limits:
            elif (-tol <= pos[2] <= crystal_size[2] + tol):
                
    
                # Apply periodic boundary conditions in the xy plane
                pos = (pos[0] % crystal_size[0], pos[1] % crystal_size[1], pos[2])

                # Find the nearest neighbor within the grid
                min_dist, min_dist_idx = min(
                    ((np.linalg.norm(np.array(site.position) - np.array(pos)), idx) 
                     for idx, site in grid_crystal.items() 
                     if np.isclose(site.position[2], pos[2], atol=1e-9, rtol=1e-9)),
                     key=lambda x: x[0]
                )
                
                self.nearest_neighbors_idx.append(tuple(min_dist_idx))
                self.nearest_neighbors_cart.append(tuple(grid_crystal[min_dist_idx].position))

                # Migration in the plane
                if -tol <= (pos[2]-self.position[2]) <= tol:
                    self.migration_paths['Plane'].append([tuple(min_dist_idx),event_labels[tuple(idx - np.array(idx_origin))]])
                            
                # Migration upward
                elif (pos[2]-self.position[2]) > tol:
                    self.migration_paths['Up'].append([tuple(min_dist_idx),event_labels[tuple(idx - np.array(idx_origin))]])
                        
                # Migration downward
                elif (pos[2]-self.position[2]) < -tol:               
                    self.migration_paths['Down'].append([tuple(min_dist_idx),event_labels[tuple(idx - np.array(idx_origin))]])
              
        self.mig_paths_plane = {num_event:site_idx for site_idx, num_event in self.migration_paths['Plane']}  
        self.NN_dist = min(np.sqrt(np.sum((np.array(self.nearest_neighbors_cart) - self.position)**2, axis=1)))     

# =============================================================================
#         Occupied sites supporting this node
# =============================================================================    
    def supported_by(self,grid_crystal,wulff_facets,dir_edge_facets,chemical_specie,affected_site,domain_height,sites_generation_layer):
        
        # Initialize supp_by as an empty list
        self.supp_by = []
        
        self.sites_generation_layer = sites_generation_layer
        
        # Position close to 0 are supported by the substrate
        tol = 1e-6
        if self.position[2] <= tol:
            self.supp_by.append('bottom_layer')
            
        
        if abs(self.position[2] - domain_height) < tol:
            self.supp_by.append('top_layer')

        # Go over the nearest neighbors
        for idx in self.nearest_neighbors_idx:
            # Select the occupied sites that support this node
            if grid_crystal[idx].chemical_specie != affected_site:
                self.supp_by.append(idx)
                    
        # Convert supp_by to a tuple
        self.supp_by = tuple(self.supp_by)
        self.calculate_clustering_energy()
        if 'E_reduction' in self.Act_E_list or 'E_oxidation' in self.Act_E_list:
            self.calculate_CN_contribution_redox_energy()
        
        if wulff_facets is not None and dir_edge_facets is not None:
            self.detect_edges(grid_crystal,dir_edge_facets,chemical_specie)               
            self.detect_planes(grid_crystal,wulff_facets[:14])
        
                
    def calculate_clustering_energy(self,supp_by_destiny = 0,idx_origin = 0):
        
        # If this site is supported by the substrate, we add the binding energy to the substrate
        # We reduce 1 if it is supported by the substrate
        # We add 1 because if the site is occupied
        
        if supp_by_destiny == 0:
            
            # Check memory cache
            cache_key = self.supp_by
            if cache_key in self.cache_clustering_energy:
                self.energy_site = self.cache_clustering_energy[cache_key]
                return


            if 'top_layer' in self.supp_by:
                self.energy_site = self.Act_E_list['CN_clustering_energy'][len(self.supp_by)] + self.Act_E_list['binding_energy_top_layer']
            elif 'bottom_layer' in self.supp_by:
                self.energy_site = self.Act_E_list['CN_clustering_energy'][len(self.supp_by)] + self.Act_E_list['binding_energy_bottom_layer']
            else:
                self.energy_site = self.Act_E_list['CN_clustering_energy'][len(self.supp_by)+1]
                
                # Store the result in the cache
            self.cache_clustering_energy[cache_key] = self.energy_site
                
        # We should consider the particle that would migrate there to calculate
        # the energy difference with the origin site
        else:
            
            # Check memory cache
            cache_key = supp_by_destiny
            if cache_key in self.cache_clustering_energy:
                return self.cache_clustering_energy[cache_key]


            if 'top_layer' in supp_by_destiny and idx_origin in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)-1] + self.Act_E_list['binding_energy_top_layer']
            elif 'bottom_layer' in supp_by_destiny and idx_origin in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)-1] + self.Act_E_list['binding_energy_bottom_layer']
            elif 'top_layer' in supp_by_destiny and idx_origin not in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)] + self.Act_E_list['binding_energy_top_layer']
            elif 'bottom_layer' in supp_by_destiny and idx_origin not in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)] + self.Act_E_list['binding_energy_bottom_layer']
            elif 'top_layer' not in supp_by_destiny and 'bottom_layer' not in supp_by_destiny and idx_origin in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)]
            elif 'top_layer' not in supp_by_destiny and 'bottom_layer' not in supp_by_destiny and idx_origin not in supp_by_destiny:
                energy_site = self.Act_E_list['CN_clustering_energy'][len(supp_by_destiny)+1]
            # Store the result in the cache
            self.cache_clustering_energy[cache_key] = energy_site
            return energy_site
            
            
    def calculate_CN_contribution_redox_energy(self):
      # Redox reactions are modified by the local structure: the number of nearest neighbors (NN) or coordination number (CN)
      # Higher CN imply higher activation barriers for oxidation and lower for reduction
      
      cache_key = self.supp_by

      if cache_key in self.cache_CN_contr_redox_energy:
        self.CN_redox_energy = self.cache_CN_contr_redox_energy[cache_key]
        return
        
      if 'bottom_layer' in self.supp_by: # Bottom interface
        self.CN_redox_energy = self.Act_E_list['CN_redox_energy'][len(self.supp_by)] + self.Act_E_list['redox_bottom_electrode']
      elif 'top_layer' in self.supp_by: # Top interface
        self.CN_redox_energy = self.Act_E_list['CN_redox_energy'][len(self.supp_by)] + self.Act_E_list['redox_top_electrode']
      else: # Bulk
        self.CN_redox_energy = self.Act_E_list['CN_redox_energy'][len(self.supp_by) + 1]
        
      self.cache_CN_contr_redox_energy[cache_key] = self.CN_redox_energy
    

   
# =============================================================================
#       Calculate the possible events corresponding to this node
#       - Migration events
#       - Desorption events
# =============================================================================
    # Change chemical_specie status
    # Add the desorption process
    def introduce_specie(self,chemical_specie,ion_charge):
        self.chemical_specie = chemical_specie
        self.ion_charge = ion_charge

    def remove_specie(self,affected_site):
        self.chemical_specie = affected_site
        self.ion_charge = 0
        #self.site_events.remove(['Desorption',self.num_event])
        self.site_events = []
        
    def available_pathways(self,grid_crystal,idx_origin, facets_type,available_events):
    
      self.site_events = []
      
      if available_events['migration']:      
          self.available_migrations(grid_crystal,idx_origin,facets_type)
      if available_events['reduction']: 
          self.available_reduction(idx_origin)
      if available_events['oxidation']:
          self.available_oxidation(idx_origin)
    

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
                    energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                    
                    # Migrating on the substrate
                    if self.sites_generation_layer in self.supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[0] + energy_change])
                        
                    # Migrating on the film (111)
                    elif grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        if self.edges_v[num_event] == None: 
                            self.site_events.append([site_idx, num_event, self.Act_E_list[7] + energy_change])
                        elif self.edges_v[num_event] == facets_type[0]:
                            self.site_events.append([site_idx, num_event, self.Act_E_list[10] + energy_change])
                        elif self.edges_v[num_event] == facets_type[1]:
                            self.site_events.append([site_idx, num_event, self.Act_E_list[9] + energy_change])
                            
                    # Migrating on the film (100)
                    elif grid_crystal[site_idx].wulff_facet == facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])
    
    
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
                    energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                   
                    # Migrating upward from the substrate
                    if self.sites_generation_layer in self.supp_by and grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[1] + energy_change])
                    
                    elif self.sites_generation_layer in self.supp_by and grid_crystal[site_idx].wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[5] + energy_change])
                        
                    # Migrating upward from the film (111)
                    elif self.wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[3] + energy_change])
                        
                    # Migrating upward from the film (100)
                    elif self.wulff_facet ==  facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])
    
                    
            # Downward migrations
            for site_idx, num_event in self.migration_paths['Down']:
    
                    # First nearest neighbors: 1 jump downward
                    # Supported by at least 2 particles (excluding this site)
                if site_idx not in self.supp_by and (self.sites_generation_layer in grid_crystal[site_idx].supp_by or len(grid_crystal[site_idx].supp_by) > 1):
                    energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                    energy_change = max(energy_site_destiny - self.energy_site, 0)
                    
                    # From layer 1 to substrate
                    if self.wulff_facet == facets_type[0] and self.sites_generation_layer in grid_crystal[site_idx].supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[2] + energy_change])
                    
                    elif self.wulff_facet == facets_type[1] and self.sites_generation_layer in grid_crystal[site_idx].supp_by:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[6] + energy_change])
                    
                    # Migrating downward from the film (111)
                    elif self.wulff_facet == facets_type[0]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[4] + energy_change])
                        
                    # Migrating downward from the film (100)
                    elif self.wulff_facet == facets_type[1]:
                        self.site_events.append([site_idx, num_event, self.Act_E_list[8] + energy_change])
                
            
            
        # Migration of interstitial sites
        else:
            
            Act_E_mig = self.Act_E_list['E_mig']
            
            # Migration types
            migration_types = ['Plane', 'Up', 'Down']
            
            
            for migration_type in migration_types:
              for site_idx, num_event in self.migration_paths[migration_type]:
                if site_idx not in self.supp_by:
                  # Calculate energy difference between sites
                  energy_site_destiny = self.calculate_clustering_energy(grid_crystal[site_idx].supp_by,idx_origin)
                  energy_change = max(energy_site_destiny - self.energy_site, 0)
                  self.site_events.append([site_idx, num_event, Act_E_mig[num_event] + energy_change])

            
    def available_reduction(self,idx_origin):
    
        if self.ion_charge > 0:
            self.site_events.append([idx_origin, 'reduction', self.Act_E_list['E_reduction'] - self.CN_redox_energy])
            
    def available_oxidation(self,idx_origin):
        
        if self.ion_charge == 0:  
            self.site_events.append([idx_origin, 'oxidation', self.Act_E_list['E_oxidation'] + self.CN_redox_energy])

        
    def deposition_event(self,TR,idx_origin,num_event,Act_E):
        self.site_events.append([TR,idx_origin, num_event, Act_E])
        
    def ion_generation_interface(self,idx_origin):
        self.site_events.append([idx_origin, 'generation', self.Act_E_list['E_gen_defect']])
        
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
                Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen'])
                            
              elif event[-2] == 'reduction':
                
                Act_E = event[-1]
                
                # Cluster bridging both electrodes
                if self.in_cluster_with_electrode['top_layer'] and self.in_cluster_with_electrode['bottom_layer']:
                  #field_along_pathway = [round(np.dot(E_site_field,migration_pathways[event[2]]) * 1e-10,3) for event in self.site_events if not isinstance(event[2],str)]
                  
                  #field_along_pathway = field_along_pathway * 0
                  #if field_along_pathway:
                  #  Act_E_pathway = [event[-1] + 0.5 * field for field in field_along_pathway]
                  #  Act_E = max(min(Act_E_pathway), 0.3)
                  
                    #if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    # Positive bias facilitate reduction --> Field helps add electrons
                    #Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,1]) * 1e-10,3), 0.3)
                  if self.in_cluster_with_electrode['top_layer'] or 'top_layer' in self.supp_by: 
                    # Positive bias hinder reduction --> Field helps remove electrons
                    Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,1]) * 1e-10,3), 0.3)
                  if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    #Positive bias facilitate reduction --> Field helps add electrons
                    Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,1]) * 1e-10,3), 0.3)

                else:
                  if self.in_cluster_with_electrode['top_layer'] or 'top_layer' in self.supp_by: 
                    # Positive bias hinder reduction --> Field helps remove electrons
                    Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,1]) * 1e-10,3), 0.3)
                  if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    # Positive bias facilitate reduction --> Field helps add electrons
                    Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,1]) * 1e-10,3), 0.3)
                    
                
                  
                  
              elif event[-2] == 'oxidation':
                
                Act_E = event[-1]
                
                # Cluster bridging both electrodes
                if self.in_cluster_with_electrode['top_layer'] and self.in_cluster_with_electrode['bottom_layer']:
                  #field_along_pathway = [round(np.dot(E_site_field,migration_pathways[event[2]]) * 1e-10,3) for event in self.site_events if not isinstance(event[2],str)]
                  
                  #field_along_pathway = field_along_pathway * 0
                  
                  #if field_along_pathway:
                  #  Act_E_pathway = [event[-1] - 0.5 * field for field in field_along_pathway]
                  #  Act_E = max(min(Act_E_pathway), self.Act_E_list['E_min_gen'])
                  
                    #if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    #  Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen'])
                  if self.in_cluster_with_electrode['top_layer'] or 'top_layer' in self.supp_by: 
                    Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen'])
                  
                  if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen']) 
                    
                else:
                  if self.in_cluster_with_electrode['top_layer'] or 'top_layer' in self.supp_by: 
                    Act_E = max(event[-1] - 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen'])
                  if self.in_cluster_with_electrode['bottom_layer'] or 'bottom_layer' in self.supp_by:
                    Act_E = max(event[-1] + 0.5 * round(np.dot(E_site_field,[0,0,-1]) * 1e-10,3), self.Act_E_list['E_min_gen'])
                  
                  
                
              
              else:
                mig_vec = migration_pathways[event[-2]]
                Act_E = max(event[-1] - self.ion_charge * round(np.dot(E_site_field,mig_vec) * 1e-10,3),self.Act_E_list['E_min_mig'])
                
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
                
        #if any(event[-2] == 'reduction' for event in self.site_events):
        #  print(f'Pos: {self.position}, NN: {len(self.supp_by)}')
        #  print(f'Site events: {self.site_events}')
                
class Cluster:
    def __init__(self,cluster_atoms,atoms_positions,attached_layer,conductivity):
      self.atoms_id = set(cluster_atoms)
      self.atoms_positions = atoms_positions
      self.size = len(self.atoms_id)
      self.attached_layer = attached_layer
      self.conductivity = conductivity
      
    def update_electrode_contact(self,grid_crystal):
      """Update whether this cluster touches top/bottom electrode layers."""
      touches_bottom = False
      touches_top = False
      
      for site in self.atoms_id:
        site_obj = grid_crystal[site]
        if 'bottom_layer' in site_obj.supp_by:
          touches_bottom = True
        elif 'top_layer' in site_obj.supp_by:
          touches_top = True
          
        if touches_bottom and touches_top:
          break
        
      self.attached_layer = {'bottom_layer': touches_bottom, 'top_layer': touches_top}
      
      
      
      # Propagate to every atom in the cluster
      for site_id in self.atoms_id:
        grid_crystal[site_id].in_cluster_with_electrode['bottom_layer'] = touches_bottom
        grid_crystal[site_id].in_cluster_with_electrode['top_layer'] = touches_top
        
    def prepare_cluster_for_bcs(self,grid_crystal):
      """
      1. Identifies and stores positions of fully coordinated atoms within the cluster.
       These internal atoms are suitable for applying Dirichlet BCs in the Poisson solver.
      2. If the cluster bridges both the top and bottom layers, it performs slicing
       to prepare data for per-layer potential drop calculations.
      """
      
      # 1) For virtual electrodes
      internal_atom_positions = []
      internal_sites = set()
      for site in self.atoms_id:
        neighbors = grid_crystal[site].nearest_neighbors_idx
        # Count how many neighbors are metal atoms in this cluster
        in_cluster_neighbors = sum(1 for nb in neighbors if nb in self.atoms_id)
        # Fully coordinated = all neighbors are in the cluster
        if in_cluster_neighbors == len(neighbors):
          internal_atom_positions.append(grid_crystal[site].position)
          internal_sites.add(site)
        
      self.internal_atom_positions = internal_atom_positions
      self.internal_sites = internal_sites
      
      # 2) When filament is bridging the electrodes: calculate the voltage drops across the filament
      if self.attached_layer['bottom_layer'] and self.attached_layer['top_layer']:
        self._slice_cluster(grid_crystal)
        self._cluster_resistance(grid_crystal)
      
      
    def _slice_cluster(self,grid_crystal):
      """
      Slice the cluster in the z axis
      """
      #sites_occupied = System_state.sites_occupied
      sites_occupied = self.atoms_id
        
      # Convert occupied sites to Cartesian coordinates and sort by z-coordinate in descending order
      sites_occupied_cart = sorted(
        ((grid_crystal[site].position, site) for site in sites_occupied), 
          key=lambda coord: coord[0][2], 
          reverse=True
        )
        
      total_visited = set()
      slice_list = []
      slice_internal_positions = []
      
      for cart_coords, site in sites_occupied_cart:
        if site not in total_visited:
          slice_sites = self._build_slice(grid_crystal, {site},site)
                
          # Intersection between the new slice and the total_visited atoms. If some atoms are already in total_visited -> Overlap
          # Skip that atom
          if slice_sites & total_visited:
            continue
                
          slice_list.append(list(slice_sites))
          slice_internal_ids = slice_sites.intersection(self.internal_sites)
          slice_internal_pos = [grid_crystal[s_id].position for s_id in slice_internal_ids]
          
          slice_internal_positions.append(slice_internal_pos)
          
          total_visited.update(slice_sites)

      self.slice_list = slice_list
      self.slice_internal_positions_per_slice = slice_internal_positions

      
    def _build_slice(self,grid_crystal,slice_sites,start_idx):
      """
      Helper function to build the slice
      """
      stack = [start_idx]
        
      while stack:
        idx = stack.pop()
        site = grid_crystal[idx]
            
        for element in site.migration_paths['Plane']:
    
          if element[0] not in slice_sites and element[0] in self.atoms_id:
            slice_sites.add(element[0])
            stack.append(element[0])
                    
      return slice_sites 
      
    def _cluster_resistance(self,grid_crystal):
      """
      Calculates the cluster resistance based on morphology (sliced layers).
      Uses cached geometric data for efficiency.
      """
      
      # Ensure geometric data is available (calculate once if needed)
      self._ensure_geometry_data_cached(grid_crystal)
      
      if self._layer_thickness_z is None or self._area_per_site is None:
        return None
        
      # --- Calculate Total Resistance ---
      total_resistance = 0.0
      layers_resistance = []
      
      for layer_slice in self.slice_list:
        num_atoms_in_layer = len(layer_slice)
        effective_area = num_atoms_in_layer * self._area_per_site
        layer_resistance = self._layer_thickness_z / (effective_area * self.conductivity)
        layers_resistance.append(layer_resistance)
        total_resistance += layer_resistance
        
      self.total_resistance = total_resistance
      self.layers_resistance = layers_resistance
    
    
    def voltage_across_cluster(self,V_top, V_bottom):
    
      V_diff = V_top - V_bottom
      I_current = V_diff / self.total_resistance # Current through the whole filament
      voltage_drop_layer = np.array(self.layers_resistance) * I_current # V_drop for each layer
      
      potential_layer_BC = [(V_top - np.sum(voltage_drop_layer[:i])) for i in range(len(voltage_drop_layer)) ]
      
      return potential_layer_BC
    
      
      
    def _ensure_geometry_data_cached(self,grid_crystal):
      """
      Helper function to calculate and cache geometric data (layer thickness, area per site)
      only if it hasn't been calculated yet
      """
      
      # Check if we already have the data
      if not hasattr(self, '_layer_thickness_z'):
        # --- Calculate layer thickness ---
        pos_1st_layer = grid_crystal[self.slice_list[0][0]].position
        pos_2nd_layer = grid_crystal[self.slice_list[1][0]].position
        self._layer_thickness_z = np.linalg.norm(np.array(pos_1st_layer[2]) - np.array(pos_2nd_layer[2])) * 1e-10
      
      
        # --- Calculate area per site
        sites_per_layer = len(grid_crystal) / len(self.slice_list)
        
        x_max = -float('inf')
        x_min = float('inf')
        
        y_max = -float('inf')
        y_min = float('inf')
      
        for site in grid_crystal.values():
          x,y,z = site.position
          if x > x_max: x_max = x
          if x < x_min: x_min = x
          if y > y_max: y_max = y
          if y < y_min: y_min = y  
                     
        x_size = (x_max - x_min) * 1e-10
        y_size = (y_max - y_min) * 1e-10
        
        self._area_per_site = (x_size * y_size) / sites_per_layer
    
        
        
class GrainBoundary:
    def __init__(self,domain_size, gb_configurations: list[dict] = None):
        
      """
      Grain boundaries for memristive filament formation. Supporing:
      - Vertical planar boundaries (between 2 grains)
      - Cylindrical boundaries (triple junctions - 3+ grains)
          
      Parameters:
      -----------
      domain_size : list [lx, ly, lz] - Domain dimensions in Angstroms
      gb_configurations : list of dicts 
        List of grain boundary configurations
      """
      
      self.domain_size = np.array(domain_size)
      self.gb_configurations = gb_configurations or self._create_default_configurations()
      
      # Pre-process configurations for fast lookup
      self._process_configurations()
      
    def _create_default_configurations(self) -> list[dict]:
      """
      Create default grain boundary configurations
      """
      configs = []
      
      # Vertical planar boundaries (between 2 grains)
      configs.extend([
        {
          'type':'vertical_planar',
          'orientation':'yz', # YZ plane
          'position':self.domain_size[0] * 0.2, # Position in x
          'width':2.0, # GB width in Angstroms
          'Act_E_diff_GB': 1 # Difference of Act Energy outside GB
        },
        {
          'type':'vertical_planar',
          'orientation':'xz',
          'position':self.domain_size[1] * 0.8, # Position in y
          'width':2.0,
          'Act_E_diff_GB': 1
        }
      ])
      
      # Cylindrical boundaries
      configs.extend([
        {
          'type':'cylindrical',
          'center': [self.domain_size[0] * 0.5, self.domain_size[1] * 0.5],
          'radius': 2.0,
          'outer_radius': 10.0,
          'Act_E_diff_GB': 1
        }
      ])
      
      # Triple Junction Planes (3 planes intersecting, between 3 grains)
      configs.extend([
        {
          'type': 'triple_junction_planes',
          'center': [self.domain_size[0] * 0.5, self.domain_size[1] * 0.5],
          'width' : 2.0, # Width of each planar GB region
          'Act_E_diff_GB': 1 
        }
      ])
      
      return configs
      
    def _process_configurations(self):
        """
        Pre-process configurations
        """
        self.vertical_gbs = []
        self.cylindrical_gbs = []
        self.triple_junction_gbs = []
        
        for config in self.gb_configurations:
          # Add linear interpolation parameters for all GB types
          if 'outer_radius' in config or 'outer_width' in config:
            # Determine the distance metric for this GB type
            if config['type'] == 'vertical_planar':
              inner_boundary = config['width'] / 2
              outer_boundary = config.get('outer_width',config['width']) / 2
              distance_function = self._distance_to_planar_gb
            elif config['type'] == 'cylindrical':
              inner_boundary = config['radius']
              outer_boundary = config['outer_radius']
              distance_function = self._distance_to_cylindrical_gb  
            elif config['type'] == 'triple_junction_planes':
              inner_boundary = config['width'] / 2
              outer_boundary = config.get('outer_width',config['width']) / 2
              # NEED TO WRITE THE FUNCTION: self._distance_to_triple_junction_gb
              #distance_function = self._distance_to_triple_junction_gb
            
            # Calculate linear interpolation parameters
            act_e_diff = config['Act_E_diff_GB']
          
            # Linear function: Act_E = slope * distance + intercept
            # At radius: Act_E = act_e_diff (inside GB)
            # At outer_radius: Act_E = 0 (outside GB)
            if outer_boundary > inner_boundary:
              slope = -act_e_diff / (outer_boundary - inner_boundary)
              intercept = act_e_diff - slope * inner_boundary
            else:
              slope = 0
              intercept = act_e_diff
              
            config['linear_slope'] = slope
            config['linear_intercept'] = intercept
            config['inner_boundary'] = inner_boundary
            config['outer_boundary'] = outer_boundary
            config['distance_function'] = distance_function
            
          if config['type'] == 'vertical_planar':
            self.vertical_gbs.append(config)
          elif config['type'] == 'cylindrical':  
            self.cylindrical_gbs.append(config)
          elif config['type'] == 'triple_junction_planes':
            self.triple_junction_gbs.append(config)
            
    def _distance_to_planar_gb(self,site_pos,gb_config):
      """Calculate distance to planar GB core"""
      x, y, z = site_pos
      if gb_config['orientation'] == 'yz':
        return abs(x - gb_config['position'])
      elif gb_config['orientation'] == 'xz':
        return abs(y - gb_config['position'])
      elif gb_config['orientation'] == 'xy':
        return abs(z - gb_config['position'])
        
    def _distance_to_cylindrical_gb(self,site_pos,gb_config):
      """Calculate distance to cylindrical GB axis"""
      x, y, z = site_pos
      cx, cy = gb_config['center']
      return np.sqrt((x - cx)**2 + (y - cy)**2)
            
    def _is_site_in_grain_boundary(self, site_position: tuple) -> bool:
      """
      Check if site is in any grain boundary
      """
      x, y, z = np.array(site_position)
      
      # Check vertical planar boundaries
      for gb in self.vertical_gbs:
        if gb['orientation'] == 'yz':
          # Vertical YZ place at specific x-position
          if abs(x - gb['position']) <= gb['outer_boundary']: return True
          
        elif gb['orientation'] == 'xz':
          # Vertical XZ place at specific y-position
          if abs(y - gb['position']) <= gb['outer_boundary']: return True
          
        elif gb['orientation'] == 'xy':
          # Horizontal XY place at specific z-position
          if abs(z - gb['position']) <= gb['outer_boundary']: return True
          
      # Check cylindrical boundaries
      for gb in self.cylindrical_gbs:
        cx, cy = gb['center']
        #radius = gb['inner_boundary']
        outer_radius = gb['outer_boundary']
        
        # Distance from cylinder axis
        distance_from_axis = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        #if distance_from_axis <= radius: return True
        if distance_from_axis <= outer_radius: return True
        
      return False
      
    
    def get_activation_energy_GB(self, site_position:tuple) -> float:
      """
      Get activation energy with unified linear interpolation
      
      Parameters:
      -----------
      site_position : tuple (x, y, z)
        Site coordinates in Angstroms
            
      Returns:
      --------
      float : Activation energy at site
      """
      x, y, z = np.array(site_position)
      
      for gb_list in [self.vertical_gbs, self.cylindrical_gbs, self.triple_junction_gbs]:
        for gb in gb_list:
          if 'linear_slope' in gb: # Has transition region
            distance = gb['distance_function'](site_position, gb)
            
            if distance <= gb['inner_boundary']:
              # Inside GB core
              return gb['Act_E_diff_GB']
            elif distance <= gb['outer_boundary']:
              # In transition region
              energy = gb['linear_slope'] * distance + gb['linear_intercept']
              return max(energy,0)
            # Else: continue to next GB (not in this GB's region)
            
      return 0 # Not in any GB
      
    """
    DEPRECATING 2026/01/05
    """
    def get_activation_energy_GB_2(self, site_position: tuple) -> float:
      """
      Get additional activation energy outside GB
      
      Parameters:
      -----------
      site_position : tuple (x, y, z)
        Site coordinates in Angstroms
            
      Returns:
      --------
      float : Activation energy at site
      """
      x, y, z = np.array(site_position)
      
      # Check vertical planar boundaries
      for gb in self.vertical_gbs:
        if gb['orientation'] == 'yz':
          if abs(x - gb['position']) <= gb['width'] / 2: return gb['Act_E_diff_GB']
          
        elif gb['orientation'] == 'xz':
          if abs(y - gb['position']) <= gb['width'] / 2: return gb['Act_E_diff_GB']
          
        elif gb['orientation'] == 'xy':
          if abs(z - gb['position']) <= gb['width'] / 2: return gb['Act_E_diff_GB']
          
      # Check cylindrical boundaries
      for gb in self.cylindrical_gbs:
        cx, cy = gb['center']
        radius = gb['radius']
        outer_radius = gb['outer_radius']
        
        distance_from_axis = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        if distance_from_axis <= radius:
          # Inside the core GB 
          return gb['Act_E_diff_GB']
        elif radius <= distance_from_axis <= outer_radius:
          # In the transition region - linear interpolation
          slope = gb['linear_slope']
          intercept = gb['linear_intercept']
          return max(slope * distance_from_axis + intercept, 0)
          # Outside outer_radius: return 0 (no effect)
        
      return 0 # Not in any GB
          
      
    
    def modify_act_energy_GB(self,site,migration_pathways):
      """
      Modify the activation energy for the site if it is outside the GB
      """
      
      site_pos = site.position
      
      if self._is_site_in_grain_boundary(site_pos):
        site.Act_E_list["E_gen_defect"] -= self.get_activation_energy_GB(site_pos)
        #site.Act_E_list["E_mig_plane"] -= self.get_activation_energy_GB(site_pos)
        #site.Act_E_list["E_mig_upward"] -= self.get_activation_energy_GB(site_pos)
        #site.Act_E_list["E_mig_downward"] -= self.get_activation_energy_GB(site_pos)

      Act_E_mig = {}
      
      in_GB = False
      out_GB = False
            
      for key,migration_vector in migration_pathways.items():
        # Calculate destination position
        dest_pos = site_pos + migration_vector * site.NN_dist
        z_component = migration_vector[2]
        
        # Migration in plane
        if np.isclose(z_component, 0.0, atol=1e-9):
          base_energy = site.Act_E_list['E_mig_plane'] 
        # Migration upward
        elif z_component > 0:
          base_energy = site.Act_E_list['E_mig_upward']
        # Migration downward
        elif z_component < 0:
          base_energy = site.Act_E_list['E_mig_downward']
          
        # Modify only if destination is in GB
        if self._is_site_in_grain_boundary(dest_pos):
          in_GB = True
          # Lower barrier to enter GB (particles prefer GB)
          modified_energy = base_energy - self.get_activation_energy_GB(dest_pos)

        else:
          # No modification for bulk destinations (base case)
          out_GB = True
          modified_energy = base_energy
          
        Act_E_mig[key] = modified_energy
        
          
      site.Act_E_list['E_mig'] = Act_E_mig   
      
        
class Island:
    def __init__(self,z_starting_position,z_starting_pos_cart,island_sites):
        self.z_starting_position = z_starting_position
        self.z_starting_pos_cart = z_starting_pos_cart
        self.island_sites = island_sites
        
    def analyze_island(self, System_state):
        "Perform full analysis on the island"

        # self._attached_to_substrate(System_state)
        layers = self._layers_calculation(System_state)
        self.terraces_general = self._island_terrace(System_state,layers)
        self._slice_detection(System_state)
        self._build_cluster_with_slices(System_state)
        
        self.cluster_layers = []
        self.cluster_terraces = []
        
        for cluster in self.cluster_list:
            layers = self._layers_calculation(System_state,cluster)
            self.cluster_layers.append(layers)
            terraces = self._island_terrace(System_state,layers)
            self.cluster_terraces.append(terraces)
            
        self._aspect_ratio_clusters(System_state)
        
    def _layers_calculation(self,System_state,cluster_sites = None):
        
        if cluster_sites is None:
            cluster_sites = self.island_sites
        
        grid_crystal = System_state.grid_crystal
        z_step = next((vec[2] * 2 for vec in System_state.basis_vectors if vec[2] > 0), None)
        z_steps = int(System_state.crystal_size[2]/z_step + 1)
        layers = [0] * z_steps  # Initialize each layer separately

        for idx in cluster_sites:
            site = grid_crystal[idx]
            z_idx = int(round(site.position[2] / z_step))
            layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0
        
        return layers
    
    def _island_terrace(self,System_state,layers):
        
        grid_crystal = System_state.grid_crystal
        z_step = next((vec[2] * 2 for vec in System_state.basis_vectors if vec[2] > 0), None)
        z_steps = int(System_state.crystal_size[2]/z_step + 1)
        sites_per_layer = len(grid_crystal)/z_steps

        area_per_site = System_state.crystal_size[0] * System_state.crystal_size[1] / sites_per_layer
        
        terraces = [(sites_per_layer - layers[0])* area_per_site]
        terraces.extend([(layers[i-1] - layers[i]) * area_per_site 
                    if (layers[i-1] - layers[i]) > 0 else 0 
                    for i in range(1,len(layers))
                    ])
      
        
        return terraces  

    # Check if the island is attached to the substrate
    def _attached_to_substrate(self,System_state):
        
        for site in self.island_sites:
            if System_state.sites_generation_layer in System_state.grid_crystal[site].supp_by:
                self.attached_substrate = True
                return
        self.attached_substrate = False # Default if not attached
        
        
    # Slice the island --> Only atoms in the plane that are in contact belong to the slice
    def _slice_detection(self,System_state):
        
        z_step = next((vec[2] * 2 for vec in System_state.basis_vectors if vec[2] > 0), None)
        z_steps = round(System_state.crystal_size[2]/z_step + 1)
        slice_list = [[] for _ in range(z_steps)] # Initialize each layer separately
        
        #sites_occupied = System_state.sites_occupied
        sites_occupied = self.island_sites
        
        # Convert occupied sites to Cartesian coordinates and sort by z-coordinate in descending order
        sites_occupied_cart = sorted(
            ((System_state.idx_to_cart(site), site) for site in sites_occupied), 
            key=lambda coord: coord[0][2], 
            reverse=True
        )
        
        total_visited = set()
        
        for cart_coords, site in sites_occupied_cart:
            if site not in total_visited:
                slice_sites = self._build_slice(System_state, {site},site)
                
                # Intersection between the new slice and the total_visited atoms. If some atoms are already in total_visited -> Overlap
                # Skip that atom
                if slice_sites & total_visited:
                    continue
                
                z_index = round(cart_coords[2] / z_step)
                slice_list[z_index].append(list(slice_sites))
                total_visited.update(slice_sites)
                
        self.slice_list = slice_list
        
    def _build_slice(self,System_state,slice_sites,start_idx):
        
        grid_crystal = System_state.grid_crystal
        stack = [start_idx]
        
        while stack:
            idx = stack.pop()
            site = grid_crystal[idx]
            
            for element in site.migration_paths['Plane']:
    
                if element[0] not in slice_sites and grid_crystal[element[0]].chemical_specie == System_state.chemical_specie:
                    slice_sites.add(element[0])
                    stack.append(element[0])
                    
        return slice_sites 
    
    def _build_cluster_with_slices(self,System_state):

        grid_crystal = System_state.grid_crystal

        # Find the first layer (from bottom to top) with only one slice
        # It is the layer where the peaks merge

        merge_layer_index = next(
            (i for i, slices in enumerate(self.slice_list) if (len(slices) == 1 and len(self.slice_list[i+1]) > 1 and len(self.slice_list[i+2]) != 1)),
            None  # in case no such layer exists
        )
        
        if merge_layer_index == None:
            # System_state.layers --> Use global layers to check which layer is less than 80% populated
            merge_layer_index = next(
                (i for i, layer in enumerate(System_state.layers[1]) if layer < 0.8),
                None  # in case no such layer exists
            )
            
        if (merge_layer_index == None or merge_layer_index == 0): merge_layer_index = -1
            
        self.merge_layer_index = merge_layer_index
        
        end_layer_index = next(
            (i for i, slices in enumerate(self.slice_list) if len(slices) == 0),
            None  # in case no such layer exists
        )
        
        # Using set() make it faster to check if neighbor[0] in cluster --> O(1)
        cluster_list = [set(layer_slice) for layer_slice in self.slice_list[merge_layer_index+1]]
        
        # Track which slices have already been added to a cluster
        visited = [set() for _ in self.slice_list]
        
        # Add the base of each cluster
        for i in np.arange(len(self.slice_list[merge_layer_index+1])):
            visited[merge_layer_index+1].add(i)
        
        for cluster in cluster_list:
            # For loop over layers
            for layer_idx in np.arange(merge_layer_index+1,end_layer_index):
                # For loop over slices
                for i, layer_slice in enumerate(self.slice_list[layer_idx]):
                    if i in visited[layer_idx]:
                        continue # Skip if already processed
                    
                    added_to_cluster = False
                    # For loop for atoms in the slice
                    for atom_idx in layer_slice:
                        site = grid_crystal[atom_idx]
                        
                        for neighbor in site.migration_paths['Down']:
                            if neighbor[0] in cluster:
                                cluster.update(layer_slice)
                                visited[layer_idx].add(i)
                                added_to_cluster = True
                                break  # Break atom loop
                                
                        if added_to_cluster:
                            break  # Break slice loop
                            
                        
                        
        self.cluster_list = cluster_list
        
                            
    def _aspect_ratio_clusters(self,System_state):
        grid_crystal = System_state.grid_crystal
        z_step = next((vec[2] * 2 for vec in System_state.basis_vectors if vec[2] > 0), None)
        z_steps = int(System_state.crystal_size[2]/z_step + 1)
        sites_per_layer = len(grid_crystal)/z_steps

        area_per_site = System_state.crystal_size[0] * System_state.crystal_size[1] / sites_per_layer

        self.cluster_aspect_ratio = []
        
        for cluster in self.cluster_layers:
            height_cluster = np.sum(np.array(cluster) > 0) * z_step
            first_nonzero_layer = next(i for i, layer in enumerate(cluster) if layer != 0)
            cluster_diameter = 2 * np.sqrt(cluster[first_nonzero_layer] * area_per_site / np.pi) 
            
            self.cluster_aspect_ratio.append(height_cluster/cluster_diameter)