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
            f.write('Material,Thickness,Ra_roughness,z_mean,RMS_roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace\n')
    
    def measurements_crystal(self, Material,thickness,Ra_roughness,z_mean,roughness,n_peaks,mean_size_peak,std_size_peak,max_size_peak,mean_island_terraces,std_island_terraces,max_island_terrace,general_terrace,std_terrace,max_terrace):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{Material},{thickness},{Ra_roughness},{z_mean},{roughness},{n_peaks},{mean_size_peak},{std_size_peak},{max_size_peak},{mean_island_terraces},{std_island_terraces},{max_island_terrace},{general_terrace},{std_terrace},{max_terrace}\n')




# Base path for the simulations
path = Path(r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Material deposition exploration\Simulations\5nm')

materials = ['Ag', 'Au', 'Cu', 'Ni', 'Pd', 'Pt']
# Subfolders in the base path
folder_subs = [path / material for material in materials]
path_2 = Path('Program')  # Subpath for variables
growth_type = ['Homoepitaxial','Substrate_range']
file_variables = ['variables.pkl']

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

occ_rate_filename = 'Occupation_rate.csv'
dfs_occ_rate = []


for sub in folder_subs:
    folder_P = sub / growth_type[1] #/ 'Program'
    for root, dirs, files in os.walk(folder_P):
        dirs.sort()
        if 'Program' in dirs:
            key_parts = root.split(os.sep)[-3:]  # Extract the parts "Ru25", "P=0.1", "Sim_1"
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
            System_state.peak_detection()
            System_state.islands_analysis()
            System_state.neighbors_calculation()
            # System_state.RMS_roughness()
            System_state.measurements_crystal()
    
            peak_size = []
            for peak in System_state.peak_list:
                if len(peak.island_sites) > 10:
                    peak_size.append(len(peak.island_sites))
                                    
            peak_mean_size = np.mean(peak_size)
            peak_std_size = np.std(peak_size)
            islands_terraces = []
            System_state.terrace_area()
            for island in System_state.islands_list:
                island.layers_calculation(System_state)
                islands_terraces.append(np.mean(np.array(island.terraces)[np.array(island.terraces) != 0]))
            
            if islands_terraces:
                islands_terraces_filtered = np.array(islands_terraces)
                mean_islands = np.mean(islands_terraces_filtered)
                std_islands = np.std(islands_terraces_filtered)
                max_islands = max(islands_terraces_filtered)
            else:
                mean_islands = np.nan
                std_islands = np.nan
                max_islands = np.nan
            # Results: thickness, roughness, islands, terraces
            peak_size_max = max(peak_size) if peak_size else 0
            Results.measurements_crystal(key,
                                         System_state.thickness,
                                         System_state.Ra_roughness,
                                         System_state.z_mean,
                                         System_state.surf_roughness_RMS,
                                         len(peak_size),
                                         np.mean(peak_size),
                                         np.std(peak_size),
                                         peak_size_max,
                                         mean_islands,
                                         std_islands,
                                         max_islands,
                                         np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),
                                         np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),
                                         max(System_state.terraces)
                                         )
            
            # Size of terrace per layer
            df_histogram_terraces = pd.DataFrame({key+"_terrace_area": System_state.terraces})   
            dfs_histogram_terraces.append(df_histogram_terraces)
            
            # Histogram of neighbors
            df_histogram_neighbors = pd.DataFrame({key+"_neighbors": System_state.histogram_neighbors})   
            dfs_histogram_neighbors.append(df_histogram_neighbors)
            
            # Ocuppation rate per layer
            df_occ_rate = pd.DataFrame({key+"_occupation_rate": System_state.layers[1]})   
            dfs_occ_rate.append(df_occ_rate)
        

    df_combined = pd.concat(dfs_histogram_terraces, axis=1)    
    df_combined.to_csv(name_histogram_size_file_terraces, index=False)
    
    df_combined = pd.concat( dfs_histogram_neighbors, axis=1)    
    df_combined.to_csv(name_histogram_size_file_neighbors, index=False)
    
    df_combined = pd.concat(dfs_occ_rate, axis=1)    
    df_combined.to_csv(occ_rate_filename, index=False)
    
