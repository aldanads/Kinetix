# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:31:14 2022

@author: ALDANADS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA
import seaborn as sns




"""
    -Nucleation time?
"""
class Island:
    def __init__(self,z_starting_position,z_starting_pos_cart,island_sites):
        self.z_starting_position = z_starting_position
        self.z_starting_pos_cart = z_starting_pos_cart
        self.island_sites = island_sites
        
    def layers_calculation(self,System_state):
        
        grid_crystal = System_state.grid_crystal
        z_step = System_state.basis_vectors[0][2]
        z_steps = int(System_state.crystal_size[2]/z_step + 1)
        layers = [0] * z_steps  # Initialize each layer separately
        
        for idx in self.island_sites:
            site = grid_crystal[idx]
            z_idx = int(round(site.position[2] / z_step))
            layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0
        
        self.layers = layers
        
        self.island_terrace(System_state)
    
    def island_terrace(self,System_state):
        
        grid_crystal = System_state.grid_crystal
        z_step = System_state.basis_vectors[0][2]
        z_steps = int(System_state.crystal_size[2]/z_step + 1)
        sites_per_layer = len(grid_crystal)/z_steps

        area_per_site = System_state.crystal_size[0] * System_state.crystal_size[1] / sites_per_layer
        
        terraces = [(self.layers[i-1] - self.layers[i]) * area_per_site for i in range(1,len(self.layers))
                    if (self.layers[i-1] - self.layers[i]) * area_per_site > 0]
        terraces.append(self.layers[-1] * area_per_site) # (nm2)
        
        self.terraces = terraces  
    

def calculate_mass(System_state):
    n_particles = len(System_state.sites_occupied)
    mass_specie = 63.546 # (mass of Copper in u) 
    x_size, y_size = System_state.crystal_size[:2]
    density = n_particles * mass_specie / (x_size * y_size)
    g_to_ng = 1e9
    nm_to_cm = 1e7

    return nm_to_cm**2 * g_to_ng * density / constants.Avogadro # (ng/cm2)


# =============================================================================
# We calculate % occupy per layer
# Average the contribution of each layer to the thickness acording to the z step
# =============================================================================
def average_thickness(System_state):
    
    grid_crystal = System_state.grid_crystal
    z_step = System_state.basis_vectors[0][2]
    z_steps = int(System_state.crystal_size[2]/z_step + 1)
    layers = [0] * z_steps  # Initialize each layer separately
    
    for site in grid_crystal.values():
        z_idx = int(round(site.position[2] / z_step))
        layers[z_idx] += 1 if site.chemical_specie != 'Empty' else 0
        
    sites_per_layer = len(grid_crystal)/z_steps
    normalized_layers = [count / sites_per_layer for count in layers]
    # Layer 0 is z = 0, so it doesn't contribute
    thickness = sum(normalized_layers) * z_step # (nm)    
        
    return thickness, normalized_layers,layers

def terrace_area(System_state,layers):
    
    grid_crystal = System_state.grid_crystal
    z_step = System_state.basis_vectors[0][2]
    z_steps = int(System_state.crystal_size[2]/z_step + 1)
    sites_per_layer = len(grid_crystal)/z_steps

    area_per_site = System_state.crystal_size[0] * System_state.crystal_size[1] / sites_per_layer
    
    terraces = [(sites_per_layer - layers[0])* area_per_site]
    terraces.extend((layers[i-1] - layers[i]) * area_per_site for i in range(1,len(layers)))
    terraces.append(layers[-1] * area_per_site) # (nm2)
    
    return terraces
    
def RMS_roughness(z):
    
    z_mean = np.mean(z)
    return np.sqrt(np.mean((np.array(z)-z_mean)**2))

def plot_crystal_surface(System_state,i):
    
    grid_crystal = System_state.grid_crystal
    z_step = System_state.basis_vectors[0][2]

    
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
            
        elif (site.chemical_specie == 'Empty') and ('Substrate' in site.supp_by) and top_layer_empty_sites >= 2:
            x.append(site.position[0])
            y.append(site.position[1])
            z.append(site.position[2])
            
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
    
    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_zlim([0, System_state.crystal_size[2]])
    
    ax.view_init(azim=45, elev = 45)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.savefig('Surface' + str(i) + '.png',dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', format = 'png')

    # Show the plot
    plt.show()

    return z

def island_calculations(System_state):   
    grid_crystal = System_state.grid_crystal
    normalized_layers = System_state.layers[1]
    chemical_specie = System_state.chemical_specie
    island_visited = set()
    total_visited = set()
    sites_occupied = System_state.sites_occupied

    count_islands = [0] * len(normalized_layers)
    layers_no_complete = np.where(np.array(normalized_layers) != 1.0)
    count_islands[normalized_layers == 1] = 1
    z_step = System_state.basis_vectors[0][2]

    islands_list = []

    for z_idx in layers_no_complete[0]:    
        z_layer = round(z_idx * z_step,3)
        for idx_site in sites_occupied:     
            if grid_crystal[idx_site].position[2] == z_layer: 
                island_slice = set()
                total_visited,island_slice = detect_islands(grid_crystal,idx_site,total_visited,island_slice,chemical_specie)

                if len(island_slice):
                    island_visited = set()
                    island_sites = island_slice.copy()
                    island_visited,island_sites = build_island(grid_crystal,island_visited,island_sites,list(island_slice)[0],chemical_specie)
                    islands_list.append(Island(z_idx,z_layer,island_sites))
                    count_islands[z_idx] += 1
                    total_visited.update(island_visited)

                    
    return islands_list
    
    
def detect_islands(grid_crystal,idx_site,visited,island_slice,chemical_specie):

    site = grid_crystal[idx_site] 
    
    if idx_site not in visited and site.chemical_specie == chemical_specie:
        visited.add(idx_site)
        island_slice.add(idx_site)
        # dfs_recursive
        for idx in site.migration_paths['Plane']:
            if idx[0] not in visited:
                visited,island_slice = detect_islands(grid_crystal,idx[0],visited,island_slice,chemical_specie)
                                   
    return visited,island_slice

def build_island_2(grid_crystal,visited,island_sites,island_slice,chemical_specie):
    
    for site in island_slice:
        
        for element in grid_crystal[site].migration_paths['Up']:

            if element[0] not in visited and grid_crystal[element[0]].chemical_specie == chemical_specie:
                visited.add(element[0])
                island_sites.add(element[0])
                visited,island_sites = build_island(grid_crystal,visited,island_sites,island_slice,chemical_specie)
                
    return visited,island_sites

def build_island(grid_crystal,visited,island_sites,idx,chemical_specie):
    
    site = grid_crystal[idx]
        
    for element in site.migration_paths['Up'] + site.migration_paths['Plane']:

        if element[0] not in visited and grid_crystal[element[0]].chemical_specie == chemical_specie:
            visited.add(element[0])
            island_sites.add(element[0])
            visited,island_sites = build_island(grid_crystal,visited,island_sites,element[0],chemical_specie)
            
    return visited,island_sites
    
    
def crystallographic_planes(center,System_state,plane_selection,azim=45, elev = 45,plane_normal = []):
    
    grid_crystal = System_state.grid_crystal
    
    x = [grid_crystal[center].position[0]]
    y = [grid_crystal[center].position[1]]
    z = [grid_crystal[center].position[2]]
    x_in_plane = []
    y_in_plane = []
    z_in_plane = []        

    basis_vectors = System_state.basis_vectors
    selected_positions = []
    if plane_selection == 'ab_111':
        plane = np.cross(basis_vectors[0],basis_vectors[1])
    elif plane_selection == 'ac_111':
        plane = np.cross(basis_vectors[0],basis_vectors[2])
    elif plane_selection == 'bc_111':
        plane = np.cross(basis_vectors[1],basis_vectors[2])
    elif plane_selection == 'a_100':
        v1 = basis_vectors[0]
        # for neighbor in grid_crystal[center].migration_paths['Down']:
        #     in_plane = np.cross(plane_aux,np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position))
        #     if np.linalg.norm(in_plane) < 1e-4:
        #         v1 =  np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
                
        for neighbor in grid_crystal[center].migration_paths['Plane']:
            v_aux = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
            if round(angle_between(basis_vectors[0],v_aux) - np.pi /2,2) == 0.0:
                v2 = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)

        plane = np.cross(v1,v2)

    elif plane_selection == 'b_100':
        plane_aux = basis_vectors[1]
        for neighbor in grid_crystal[center].migration_paths['Down']:
            in_plane = np.cross(plane_aux,np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position))
            if np.linalg.norm(in_plane) < 1e-4:
                v1 =  np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
                
        for neighbor in grid_crystal[center].migration_paths['Plane']:
            v_aux = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
            if round(angle_between(basis_vectors[1],v_aux) - np.pi /2,2) == 0.0:
                v2 = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
        
        plane = np.cross(v1,v2)

    elif plane_selection == 'c_100':
        plane_aux = basis_vectors[2]
        for neighbor in grid_crystal[center].migration_paths['Down']:
            in_plane = np.cross(plane_aux,np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position))
            if np.linalg.norm(in_plane) < 1e-3:
                v1 =  np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
                
        for neighbor in grid_crystal[center].migration_paths['Plane']:
            v_aux = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
            if round(angle_between(basis_vectors[2],v_aux) - np.pi /2,2) == 0.0:
                v2 = np.array(System_state.idx_to_cart(neighbor[0])) - np.array(grid_crystal[center].position)
        
        plane = np.cross(v1,v2)
        
        
    selected_positions.append(tuple(plane))
    for neighbor in grid_crystal[center].nearest_neighbors_cart:
        in_plane = np.dot(plane,np.array(neighbor) - np.array(grid_crystal[center].position))
        #print(in_plane, grid_crystal[center].position, '->', neighbor)
    
        if abs(in_plane) <= 1e-4:
            x_in_plane.append(neighbor[0])
            y_in_plane.append(neighbor[1])
            z_in_plane.append(neighbor[2])    
            selected_positions.append(neighbor)
        else:
            x.append(neighbor[0])
            y.append(neighbor[1])
            z.append(neighbor[2])
        
            
    plane /= np.linalg.norm(plane)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, c='blue', marker='o')
    ax.scatter3D(x_in_plane, y_in_plane,z_in_plane, c='red', marker='o')
    if len(plane_normal) > 0:
        # Plot the normal vectors
        origin = np.array(System_state.idx_to_cart(center))
        ax.quiver(*origin, *plane, color='r', label='Normal Vector 1')
        ax.quiver(*origin, *plane_normal, color='b', label='Normal Vector 2')
        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set plot limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.legend()
        
    ax.set_aspect('equal', 'box')
    ax.view_init(azim=azim, elev = elev)
    return selected_positions,plane
    
def calculate_millex_index(point1, point2, point3):
    # Calculate two vectors
    vector1 = point2 - point1
    vector2 = point3 - point1
    
    # Compute the cross product of the two vectors
    cross_product = np.cross(vector1, vector2)
    print(cross_product,np.linalg.norm(cross_product))
    # Normalize the resulting vector to obtain the reciprocal lattice vector
    reciprocal_lattice_vector = cross_product / np.linalg.norm(cross_product)
    
    return reciprocal_lattice_vector

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Finds angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def detect_planes(supp_by,System_state):
    atom_coordinates = np.array([System_state.idx_to_cart(idx) for idx in supp_by if idx != 'Substrate'])
    
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
    # Optionally, normalize the plane normal vector
    #plane_normal /= np.linalg.norm(plane_normal)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Plot the origin
    # ax.scatter(0, 0, 0, color='k', marker='o', label='Origin')
    
    # # Plot the normal vectors
    # origin = np.array([0, 0, 0])
    # ax.quiver(*origin, *plane1, color='r', label='Normal Vector 1')
    
    return plane_normal

def plot_vectors(plane1,plane2):
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the origin
    ax.scatter(0, 0, 0, color='k', marker='o', label='Origin')
    
    # Plot the normal vectors
    origin = np.array([0, 0, 0])
    ax.quiver(*origin, *plane1, color='r', label='Normal Vector 1')
    ax.quiver(*origin, *plane2, color='b', label='Normal Vector 2')
    
    # Set plot limits
    ax.set_xlim([0, 1.2])
    ax.set_ylim([-1, 0])
    ax.set_zlim([-1, 1])
    
    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show plot
    plt.show()
    
def plot_crystal(grid_crystal,sites_occupied,crystal_size,azim = 60,elev = 45,path = '',i = 0,size = 20):
    num_colors = 3  # Change this as needed
    colors_palette = sns.color_palette("tab10")

    nr = 1
    nc = 1
    fig = plt.figure(constrained_layout=True,figsize=(15, 8),dpi=300)
    subfigs = fig.subfigures(nr, wspace=0.1, hspace=7)
    
    axa = subfigs.add_subplot(111, projection='3d')
    positions = np.array([grid_crystal[idx].position for idx in sites_occupied])
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    # Define the range of z values
    z_min, z_max = np.min(z), np.max(z)
    # Compute the range of each color segment
    z_ranges = np.linspace(z_min, z_max, num_colors + 1)
    # Assign a color to each particle based on its z position
    colors = [colors_palette[np.digitize(z_pos, z_ranges) - 1] for z_pos in z]
    axa.scatter3D(x, y, z, c=colors, marker='o',s =size)
    
    axa.set_xlabel('x-axis (nm)')
    axa.set_ylabel('y-axis (nm)')
    axa.set_zlabel('z-axis (nm)')
    axa.view_init(azim=azim, elev = elev)

    axa.set_xlim([1, crystal_size[0]]) 
    axa.set_ylim([0, crystal_size[1]])
    #axa.set_zlim([0, crystal_size[2]])
    axa.set_aspect('equal', 'box')
    
def plot_atom_neighbors(grid_crystal,sites_occupied):
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors_palette = sns.color_palette(n_colors=4)
    
    size = 200
    nr = 1
    nc = 1
    fig = plt.figure(constrained_layout=True,figsize=(15, 8),dpi=300)
    subfigs = fig.subfigures(nr, wspace=0.1, hspace=7)
    
    axa = subfigs.add_subplot(111, projection='3d')
    
    mig_paths = grid_crystal[sites_occupied[1000]].migration_paths
    
    center = grid_crystal[sites_occupied[1000]].position
    neigh_plane = [grid_crystal[mig_paths['Plane'][i][0]].position for i in range(len(mig_paths['Plane']))]
    neigh_up = [grid_crystal[mig_paths['Up'][i][0]].position for i in range(len(mig_paths['Up']))]
    neigh_down = [grid_crystal[mig_paths['Down'][i][0]].position for i in range(len(mig_paths['Down']))]
        
    axa.scatter3D(center[0], center[1], center[2], c=colors_palette[0], marker='o',s =size)
    neigh_plane = [neigh_plane[0],neigh_plane[5],neigh_plane[3], neigh_plane[2],neigh_plane[1],neigh_plane[4]]
    x, y, z = np.array(neigh_plane)[:,0], np.array(neigh_plane)[:,1], np.array(neigh_plane)[:,2]
    axa.scatter3D(x, y, z, marker='o',s =size)
    plane_poly = Poly3DCollection([list(zip(x, y, z))], alpha=0.5)
    axa.add_collection3d(plane_poly)
    
    
    x, y, z = np.array(neigh_up)[:,0], np.array(neigh_up)[:,1], np.array(neigh_up)[:,2]
    axa.scatter3D(x, y, z, marker='o',s =size)
    plane_poly = Poly3DCollection([list(zip(x, y, z))],color=colors_palette[1], alpha=0.5)
    axa.add_collection3d(plane_poly)
    
    x, y, z = np.array(neigh_down)[:,0], np.array(neigh_down)[:,1], np.array(neigh_down)[:,2]
    axa.scatter3D(x, y, z, marker='o',s =size)
    plane_poly = Poly3DCollection([list(zip(x, y, z))],color=colors_palette[2], alpha=0.5)
    axa.add_collection3d(plane_poly)
    
            
plt.rcParams["figure.dpi"] = 300
system = ['Windows','Linux']
choose_system = system[1]
file_variables = ['variables','variables2']

for i in range(1):
    if choose_system == 'Windows':
        import shelve
        
        filename = file_variables[i]
        
        my_shelf = shelve.open(filename)
        for key in my_shelf:
            globals()[key]=my_shelf[key]
        my_shelf.close()
        
        
    elif choose_system == 'Linux':
        
        import pickle
        filename = file_variables[i]+'.pkl'
        
        # Open the file in binary mode
        with open(filename, 'rb') as file:
          
            # Call load method to deserialze
            myvar = pickle.load(file)
            
        System_state = myvar['System_state']
    
    
    mass_gained = calculate_mass(System_state)
    thickness, normalized_layers,layers = average_thickness(System_state)
    
    terraces = terrace_area(System_state,layers)
    
    fraction_sites_occupied = len(System_state.sites_occupied) / len(System_state.grid_crystal) 
    z = plot_crystal_surface(System_state,i)
    surf_roughness_RMS = RMS_roughness(z)
    
    
    islands_list = island_calculations(System_state)

                
#particles_in_plane = grid_crystal[sites_occupied[-5]].supp_by
# particles_in_plane = [idx[0] for idx in grid_crystal[sites_occupied[-5]].migration_paths['Plane']]
# plane_normal = detect_planes(particles_in_plane,System_state)

# center = (5,5,-8)
# center = System_state.adsorption_sites[round(len(System_state.adsorption_sites)/2)]
# plane = ['ab_111','ac_111','bc_111','a_100','b_100','c_100']
# selected_positions,n_plane = crystallographic_planes(center,System_state,plane[5],60,30, -plane_normal)

# cross_product = np.cross(plane_normal,n_plane)

# print(angle_between(plane_normal, n_plane) / np.pi * 180 )
# print(np.linalg.norm(cross_product))


# for idx in sites_occupied:
#     update_specie_events.append(idx)
    
#     # CAREFUL! We don't update 2nd nearest neighbors
#     # Nodes we need to update
#     update_supp_av.update(self.grid_crystal[idx].nearest_neighbors_idx)
#     update_supp_av.add(idx) # Update the new specie to calculate its supp_by





# for event in grid_crystal[sites_occupied[-5]].site_events:
#     v1 = np.array(System_state.idx_to_cart(event[1])) - np.array(grid_crystal[sites_occupied[-5]].position)
#     print(np.dot(v1,n_plane),event[1],sites_occupied[-5])
# =============================================================================
# # point1 = np.array((1.645, 0.95, 0.413))
# # point2 = np.array((1.645, 0.804, 0.207))
# # point3 = np.array((1.772, 0.877, 0.62))
# point1 = np.array((6, 5, -8))
# point2 = np.array((3, 6, -8))
# point3 = np.array((4, 6, -10))
# 
# miller_indices = calculate_millex_index(point1, point2, point3)
# print("Miller indices:", miller_indices)
# =============================================================================
