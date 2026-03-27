# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:35:07 2024

@author: samuel.delgado
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from pathlib import Path
import pickle
import shelve


class SimulationResults:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        # Initialize a CSV file with headers
        with open(excel_filename, 'w') as f:
            f.write('Material,Thickness,Ra_roughness,z_mean,RMS_roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,substrate_exposure,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace\n')
    
    def measurements_crystal(self, Material,thickness,Ra_roughness,z_mean,roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,substrate_exposure,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{Material},{thickness},{Ra_roughness},{z_mean},{roughness},{n_peaks},{mean_size_peak},{std_size_peak},{max_size_peak},{substrate_exposure},{mean_island_terraces},{std_island_terraces},{max_island_terrace},{general_terrace},{std_terrace},{max_terrace}\n')




# Base path for the simulations
path = Path(r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Control of fcc metal morphology via substrate interaction\Simulations\5nm')

materials = ['Ag', 'Au', 'Cu', 'Ni', 'Pd', 'Pt']


# Subfolders in the base path
folder_subs = [path / material for material in materials]
path_2 = Path('Program')  # Subpath for variables
growth_type = ['Homoepitaxial','Substrate_range','Substrate_range_downward','Substrate_range_v2','Substrate_range_downward_v2','annealing']
file_variables = ['variables.pkl']
root_parts = 3

# Data saved in a OS
system = ['Windows','Linux']
choose_system = system[1]

# Store the data of each file in a list
dfs = []
temperature = {'Sim_0': 431}

name_generated_file = 'Figure.csv'
Results = SimulationResults(name_generated_file)

name_histogram_size_file_terraces = 'Histogram_terraces.csv'
dfs_histogram_terraces = []

name_histogram_size_file_neighbors = 'Histogram_neighbors.csv'
dfs_histogram_neighbors = []

name_histogram_island_size = "Histogram_island_size.csv"
dfs_histogram_islands = []

name_peak_base_area = "peak_base_area.csv"
dfs_peak_base_area = []

occ_rate_filename = 'Occupation_rate.csv'
dfs_occ_rate = []

aspect_ratio_filename = 'Aspect_ratio.csv'
dfs_aspect_ratio = []

for sub in folder_subs:

    folder_P = sub / growth_type[5]
    i_count = 0
    for root, dirs, files in os.walk(folder_P):
      
        if ("Pd" in root and "descent" in root and "weak" in root and "E_min_limit_02" in root and
            "E_min_limit_025" not in root): continue

        dirs.sort()
        if 'Program' in dirs:
            
            if growth_type[5] == 'annealing': root_parts = 6
            key_parts = root.split(os.sep)[-root_parts:]  # Extract the parts "Ru25", "P=0.1", "Sim_1"
            key = os.path.join(*key_parts)
            filename = os.path.join(root, 'Program', file_variables[0])
            print(root)
                

            if choose_system == 'Windows':
                variables_file = folder_P / 'variables2'
                
                # Read shelve file
                with shelve.open(str(variables_file)) as my_shelf:
                    for key in my_shelf:
                        globals()[key] = my_shelf[key]
            
            elif choose_system == 'Linux':
                with open(filename, 'rb') as file:
                 
                   # Call load method to deserialze
                   myvar = pickle.load(file)
              
        
            System_state = myvar['System_state']

            System_state.affected_site = "Empty"
            System_state.mode = "regular"

            System_state.islands_analysis()
            
            atoms_deposited = []
            
            mass = 0

            new_islands_list = []
            for island in System_state.islands_list:

                island._attached_to_substrate(System_state)
                # Remove islands that have been detached from the substrate
                if island.attached_substrate:
                    # islands_terraces.append(np.mean(np.array(island.terraces)[np.array(island.terraces) != 0]))
                    
                    for site in island.island_sites:
                        atoms_deposited.append(site)
                    new_islands_list.append(island)  # keep it
            
            System_state.islands_list = new_islands_list
                    
            

            # Remove atoms separated from the substrate from the system
            count = 0
            for site in System_state.sites_occupied.copy(): # Need to copy, otherwise it modifies the same list it is reading
                if site not in atoms_deposited:
                    count += 1
                    update_supp_av = set()
                    update_specie_events = {site}
                    
                    update_specie_events,update_supp_av = System_state.remove_specie_site(site,update_specie_events,update_supp_av)
                    System_state.update_sites(update_specie_events,update_supp_av)
                    
            
            for island in System_state.islands_list:
                island.analyze_island(System_state)
                

            """
            Peak detection calculated only with atoms_deposited -> Removed the rest
            """
            System_state.peak_detection()
            System_state.sites_occupied = set(System_state.sites_occupied)
            System_state.neighbors_calculation()
            System_state.measurements_crystal()

            #System_state.RMS_roughness()
        
            # Create files with the particles
            
            # if 'Pt' in root:
            #     print(root)
            #     System_state.plot_crystal(45,45,'',i_count)
            #     i_count += 1

            #     if i_count == 8: exit()
            
            
            """
            Calculate area per site
            """
            z_step = next((vec[2] * 2 for vec in System_state.basis_vectors if vec[2] > 0), None)
            z_steps = round(System_state.crystal_size[2]/z_step + 1)
            sites_per_layer = len(System_state.grid_crystal)/z_steps
            area_per_site = System_state.crystal_size[0] * System_state.crystal_size[1] / sites_per_layer
            
    
            peak_size = []
            peak_base_area = []
            all_terraces = []
            aspect_ratio_island = []
            seen_clusters = set()

            for island in System_state.islands_list:
                

                for i in reversed(range(len(island.cluster_list))):
                    cluster = island.cluster_list[i]
                    cluster_key = frozenset(cluster)
                    
                    if cluster_key in seen_clusters:
                        del island.cluster_list[i]
                        del island.cluster_layers[i]
                        del island.cluster_terraces[i]
                        del island.cluster_aspect_ratio[i]
                    else:
                        seen_clusters.add(cluster_key)
                        peak_size.append(len(cluster))
                    
                for cluster_layer in island.cluster_layers:
                    # We select the base of the cluster/island
                    first_non_zero = next((i for i,x in enumerate(cluster_layer) if x != 0), None)
                    if first_non_zero is not None:
                        peak_base_area.append(cluster_layer[first_non_zero] * area_per_site)
                       
                if island.merge_layer_index > 0 and island.cluster_terraces:
                    terraces = np.array(System_state.terraces[1:island.merge_layer_index])
                    all_terraces.extend(terraces[terraces > 0])
                    
                for terraces in island.cluster_terraces:
                    terraces = np.array(terraces[1:])
                    all_terraces.extend(terraces[terraces > 0])
                    
                aspect_ratio_island.extend(island.cluster_aspect_ratio)

                    
            total_area = System_state.crystal_size[0] * System_state.crystal_size[1]
            substrate_exposure = (System_state.crystal_size[0] * System_state.crystal_size[1] - max(System_state.layers[0]) * area_per_site) / total_area

            peak_mean_size = np.mean(peak_size)
            peak_std_size = np.std(peak_size)
            
            mean_terraces = np.mean(all_terraces)
            std_terraces = np.std(all_terraces)
            if all_terraces:
                max_terraces = max(all_terraces) / total_area
            else:
                max_terraces = max(System_state.terraces) / total_area
            
            peak_base_area = np.array(peak_base_area) / total_area

            
# =============================================================================
#             for peak in System_state.peak_list:
#                 peak.analyze_island(System_state)
#                 if len(peak.island_sites) > 0:
#                     peak_size.append(len(peak.island_sites))
#                     index_peak_layers = np.where(np.array(peak.layers) != 0)
#                     # We select the first non-zero element
#                     peak_base_area.append(peak.layers[index_peak_layers[0][0]] * area_per_site)
#                                     
#             peak_mean_size = np.mean(peak_size)
#             peak_std_size = np.std(peak_size)
# =============================================================================
            
            
            # System_state.terrace_area()
        
# =============================================================================
#             if islands_terraces:
#                 islands_terraces_filtered = np.array(islands_terraces)
#                 mean_islands = np.mean(islands_terraces_filtered)
#                 std_islands = np.std(islands_terraces_filtered)
#                 max_islands = max(islands_terraces_filtered)
#             else:
#                 mean_islands = np.nan
#                 std_islands = np.nan
#                 max_islands = np.nan
# =============================================================================
            # Results: thickness, roughness, islands, terraces
            peak_size_max = max(peak_size) if peak_size else 0
            
            for i,island in enumerate(System_state.islands_list):
                print("Island: ",i, "merge_Layer: ", island.merge_layer_index)
                for j,slice_layer in enumerate(island.slice_list):
                    print("Layer: ", j, "n slices: ",len(slice_layer))
                    
                    if j == (island.merge_layer_index + 6): break
                    
            # if "descent" in root and "weak" in root and "E_min_limit_025" in root: exit()
            # if ("Ag" in root and 'homoepitaxial' in root and "E_min_limit_025" in root): exit()
                
            Results.measurements_crystal(key,
                                         System_state.thickness,
                                         System_state.Ra_roughness,
                                         System_state.z_mean,
                                         System_state.surf_roughness_RMS,
                                         len(peak_size),
                                         np.mean(peak_size),
                                         np.std(peak_size),
                                         peak_size_max,
                                         substrate_exposure,
                                         mean_terraces,
                                         std_terraces,
                                         max_terraces,
                                         np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),
                                         np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),
                                         max(System_state.terraces)
                                         )
            
            # Size of terrace per layer
            df_histogram_terraces = pd.DataFrame({key+"_terrace_area": all_terraces})   
            dfs_histogram_terraces.append(df_histogram_terraces)
            
            # Histogram of neighbors
            df_histogram_neighbors = pd.DataFrame({key+"_neighbors": System_state.histogram_neighbors})   
            dfs_histogram_neighbors.append(df_histogram_neighbors)
            
            # Ocuppation rate per layer
            df_occ_rate = pd.DataFrame({key+"_occupation_rate": System_state.layers[1]})   
            dfs_occ_rate.append(df_occ_rate)
            
            # Island size
            df_histogram_island = pd.DataFrame({key+"_island_size": peak_size})
            dfs_histogram_islands.append(df_histogram_island)
            
            # Peak base area
            df_peak_base_area = pd.DataFrame({key+"_peak_base_area": peak_base_area})
            dfs_peak_base_area.append(df_peak_base_area)
            
            # Aspect ratio islands
            df_aspect_ratio = pd.DataFrame({key+"_aspect_ratio": aspect_ratio_island})
            dfs_aspect_ratio.append(df_aspect_ratio)

    df_combined = pd.concat(dfs_histogram_terraces, axis=1)    
    df_combined.to_csv(name_histogram_size_file_terraces, index=False)
    
    df_combined = pd.concat( dfs_histogram_neighbors, axis=1)    
    df_combined.to_csv(name_histogram_size_file_neighbors, index=False)
    
    df_combined = pd.concat(dfs_occ_rate, axis=1)    
    df_combined.to_csv(occ_rate_filename, index=False)
    
    df_combined = pd.concat(dfs_histogram_islands, axis=1)    
    df_combined.to_csv(name_histogram_island_size, index=False)
    
    df_combined = pd.concat(dfs_peak_base_area, axis=1)    
    df_combined.to_csv(name_peak_base_area, index=False)
    
    df_combined = pd.concat(dfs_aspect_ratio, axis=1)    
    df_combined.to_csv(aspect_ratio_filename, index=False)
    
    
