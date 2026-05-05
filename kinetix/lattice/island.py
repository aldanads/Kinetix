# -*- coding: utf-8 -*-
"""Island class for terrace/morphology analysis."""
import numpy as np

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