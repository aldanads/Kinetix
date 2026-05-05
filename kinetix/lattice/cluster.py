# -*- coding: utf-8 -*-
"""Cluster class for filament analysis."""
import numpy as np

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
        
    def prepare_cluster_for_bcs(self,grid_crystal,crystal_size):
      """
      1. Identifies and stores positions of fully coordinated atoms within the cluster.
       These internal atoms are suitable for applying Dirichlet BCs in the Poisson solver.
      2. If the cluster bridges both the top and bottom layers, it performs slicing
       to prepare data for per-layer potential drop calculations.
      """
      
      # 1) For virtual electrodes
      self._identify_internal_atoms(grid_crystal)
      
      # 2) When filament is bridging the electrodes: calculate the voltage drops across the filament
      if self.attached_layer['bottom_layer'] and self.attached_layer['top_layer']:
        self._slice_cluster(grid_crystal)
        self._cluster_resistance(grid_crystal)
      else:
        self._get_distance_to_electrode(crystal_size)
        
    def _identify_internal_atoms(self,grid_crystal):
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
      
    def _get_distance_to_electrode(self,crystal_size):
     """
     Calculate minimum distance from cluster to electrode
     """
     min_distance = float("inf")
     
     for pos in self.atoms_positions:
       if self.attached_layer['bottom_layer']:
         distance = crystal_size[2] - pos[2]
       elif self.attached_layer['top_layer']:
         distance = pos[2]
       else:
         distance = crystal_size[2]
         
       min_distance = min(min_distance,distance)
       
     self.distance_electrode = min_distance
      
      
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