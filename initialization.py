# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:03:03 2024

@author: samuel.delgado
"""
import numpy as np
import matplotlib.pyplot as plt
import platform
import shutil
from crystal_lattice import Crystal_Lattice
from ElectricalController import ElectricalController
from superbasin import Superbasin
from pymatgen.ext.matproj import MPRester
# from mp_api.client import MPRester
import json
from pathlib import Path

import os
import pickle
import shelve
import time
import warnings


def initialization(n_sim):
    
# =============================================================================
#         Simulation parameters
#         
# =============================================================================
    seed = 1
    # Random seed as time
    rng = np.random.default_rng(seed) # Random Number Generator (RNG) object
    
    save_data = True
    lammps_file = True
    snapshoots_steps = int(4e1)
    total_steps = int(snapshoots_steps * 25)
    
    experiments = ['deposition','annealing','ECM memristor']
    experiment = experiments[2]
    
    simulation_parameters = {
      'save_data':save_data, 'snapshoots_steps':snapshoots_steps,
      'total_steps':total_steps
    }    

    # Default resolution for figures
    plt.rcParams["figure.dpi"] = 100 # Default value of dpi = 300
    
    if save_data:
        files_copy = ['initialization.py', 'crystal_lattice.py','Site.py','main.py','KMC.py',
                      'balanced_tree.py','analysis.py','superbasin.py', 'PoissonSolver.py' , 'ElectricalController.py',
                      'activation_energies_deposition.json', 'activation_energies_memristors.json']
        
        if platform.system() == 'Windows': # When running in laptop
            dst = Path(r'\\FS1\Docs2\samuel.delgado\My Documents\Publications\Memristor ECM\Simulations\Tests')
        elif platform.system() == 'Linux': # HPC works on Linux
            dst = Path(r'/home/Docs2/samuel.delgado/linuxhome/Documents/Simulators/test')
            
        paths,Results = save_simulation(files_copy,dst,n_sim,experiment) # Create folders and python files
        
    else:
        paths = {'data': ''}
        Results = []
        

    





    if experiment == 'deposition':         
# =============================================================================
#         Experimental conditions
#         
# =============================================================================
# =============================================================================
#        Partial pressure and deposition temperature
#         Lee, Won-Jun, Sa-Kyun Rha, Seung-Yun Lee, Dong-Won Kim, and Chong-Ook Park. 
#         "Effect of the pressure on the chemical vapor deposition of copper from copper hexafluoroacetylacetonate trimethylvinylsilane." 
#         Thin Solid Films 305, no. 1-2 (1997): 254-258.
# 
#       "Chemical vapor deposition of Cu films from copper(I) cyclopentadienyl triethylphophine: Precursor
#       characteristics and interplay between growth parameters and films morphology"
# =============================================================================
        sticking_coeff = 1        
        partial_pressure = 113 # (Pa = N m^-2 = kg m^-1 s^-2)
        #partial_pressure = 100
        # p = 0.1 - 10 typical values 
        # T = 573 + n_sim * 100 # (K)
        temp = 431
        T = temp # (K)
        
        experimental_conditions = [sticking_coeff,partial_pressure,T,experiment]
    
# =============================================================================
#         Crystal structure
#         
# =============================================================================
        material_selection = {"Ni":"mp-23","Cu":"mp-30", "Pd": "mp-2","Ag":"mp-124","Pt":"mp-126","Au":"mp-81", "PbZrO3":"mp-1068577"}
        id_material_Material_Project = material_selection['Au']
        crystal_size = (20,20,20) # (angstrom (Å))
        orientation = ['001','111']
        use_parallel = None
        facets_type = [(1,1,1),(1,0,0)]
        affected_site = 'Empty'
        mode = ['regular']
        radius_neighbors = 3
        sites_generation_layer = ['bottom_layer','top_layer']


        script_directory = Path(__file__).parent        # Get the config path from the environment variable or fallback to the current directory
        config_path = script_directory / 'config.json'
        
        
        # Create a config.json file with the API key -> To avoid uploading to Github
        with open(config_path) as config_file:
            config = json.load(config_file)
            api_key = config['api_key']
        

        # Retrieve material data
        with MPRester(api_key) as mpr:
            # Retrieve material summary information
            # material_summary = mpr.summary.search(material_ids=[id_material_Material_Project])
            # formula = material_summary[0].get('formula_pretty')
            
            material_summary = mpr.materials.summary.search(material_ids=[id_material_Material_Project])
            formula = material_summary[0].formula_pretty
        
        crystal_features = {
          'id_material_Material_Project': id_material_Material_Project,
          'crystal_size': crystal_size,
          'orientation': orientation[1],
          'api_key': api_key,
          'use_parallel': use_parallel,
          'facets_type': facets_type,
          'affected_site': affected_site,
          'mode': mode[0],
          'radius_neighbors': radius_neighbors,
          'sites_generation_layer': sites_generation_layer[0]
        }
        
# =============================================================================
#             Superbasin parameters
#     
# =============================================================================
        n_search_superbasin = 25 # If the time step is very small during 10 steps, search for superbasin
        time_step_limits = 1e-7 # Time needed for efficient evolution of the system
        E_min = 0.0
        energy_step = 0.05
        superbasin_parameters = [n_search_superbasin,time_step_limits,E_min,energy_step]
# =============================================================================
#       Different surface Structures- fcc Metals
#       https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Surface_Science_(Nix)/01%3A_Structure_of_Solid_Surfaces/1.03%3A_Surface_Structures-_fcc_Metals
#       Activation energies
#       Nies, C. L., Natarajan, S. K., & Nolan, M. (2022). 
#       Control of the Cu morphology on Ru-passivated and Ru-doped TaN surfaces-promoting growth of 2D conducting copper for CMOS interconnects. 
#       Chemical Science, 13(3), 713–725. https://doi.org/10.1039/d1sc04708f
#           - Migrating upward/downward one layer - It seems is promoted by other atoms surrounding
#           - Migrating upward/downward two layers in one jump
# 
#       Jamnig, A., Sangiovanni, D. G., Abadias, G., & Sarakinos, K. (2019). 
#       Atomic-scale diffusion rates during growth of thin metal films on weakly-interacting substrates. 
#       Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-43107-8
#           - Migration of Cu on graphite - 0.05-0.13 eV
# 
#       Kondati Natarajan, S., Nies, C. L., & Nolan, M. (2019). 
#       Ru passivated and Ru doped ϵ-TaN surfaces as a combined barrier and liner material for copper interconnects: A first principles study. 
#       Journal of Materials Chemistry C, 7(26), 7959–7973. https://doi.org/10.1039/c8tc06118a
#           - TaN (111) - Activation energy for Cu migration - [0.85 - 1.26] (ev)
#           - Ru(0 0 1) - Activation energy for Cu migration - [0.07 - 0.11] (ev)
#           - 1ML Ru - Activation energy for Cu migration - [0.01, 0.21, 0.45, 0.37] (ev)
#           - 2ML Ru - Activation energy for Cu migration - [0.46, 0.44] (ev)
#           - Information about clustering two Cu atoms on TaN and Ru surfaces
# 
#       ACTIVATION ENERGIES
#       Kim, Sung Youb, In-Ho Lee, and Sukky Jun. 
#       "Transition-pathway models of atomic diffusion on fcc metal surfaces. I. Flat surfaces." 
#       Physical Review B 76, no. 24 (2007): 245407.
# 
#       Kim, Sung Youb, In-Ho Lee, and Sukky Jun. 
#       "Transition-pathway models of atomic diffusion on fcc metal surfaces. II. Stepped surfaces." 
#       Physical Review B 76, no. 24 (2007): 245408.
# =============================================================================
        select_dataset = 3   
        Act_E_dataset = ['TaN','Ru25','Ru50','homoepitaxial','template_upward']  
        
        # Retrieve the activation energies
        activation_energy_file = script_directory / 'activation_energies_deposition.json'
        with open(activation_energy_file, 'r') as file:
            data = json.load(file)
            
        E_dataset = []
        for element in data['elements']:
            # Search the selected element we retrieved from Materials Project
            if element['name'] == formula:
                
                #Search the activation energies
                for key,activation_energies in element.items():
                    if 'activation_energies' in key and Act_E_dataset[select_dataset] in key:
                        # Select the dataset
                        for act_energy in activation_energies.values():
                            if isinstance(act_energy, (int, float)):
                                E_dataset.append(act_energy)
        
        E_mig_sub = 0.5
        #E_mig_sub = E_dataset[0] # (eV)
        E_mig_upward_subs_layer111 = E_dataset[1] * (0.1 + 0.2 * n_sim)
        E_mig_downward_layer111_subs = E_dataset[2] * (1.6 - 0.2 * n_sim)
        E_mig_upward_layer1_layer2_111 = E_dataset[3] * (0.1 + 0.2 * n_sim)
        E_mig_downward_layer2_layer1_111 = E_dataset[4] * (1.6 - 0.2 * n_sim)
        E_mig_upward_subs_layer100 = E_dataset[5] * (0.1 + 0.2 * n_sim)
        E_mig_downward_layer100_subs = E_dataset[6] * (1.6 - 0.2 * n_sim)
        E_mig_111_terrace_Cu = E_dataset[7]
        E_mig_100_terrace_Cu = E_dataset[8] * (1.6 - 0.2 * n_sim)
        E_mig_edge_100 = E_dataset[9]
        E_mig_edge_111 = E_dataset[10]

        # =============================================================================
        #     Papanicolaou, N. 1, & Evangelakis, G. A. (n.d.). 
        #     COMPARISON OF DIFFUSION PROCESSES OF Cu AND Au ADA TOMS ON THE Cu(1l1) SURFACE BY MOLECULAR DYNAMICS.
        #     
        #     Mińkowski, Marcin, and Magdalena A. Załuska-Kotur. 
        #     "Diffusion of Cu adatoms and dimers on Cu (111) and Ag (111) surfaces." 
        #     Surface Science 642 (2015): 22-32. 10.1016/j.susc.2015.07.026
        # =============================================================================

        # Binding energy | Desorption energy: https://doi.org/10.1039/D1SC04708F
        binding_energy = E_dataset[-2] * (0.1 + 0.2 * n_sim)

             

# =============================================================================
#     Kim, Sung Youb, In-Ho Lee, and Sukky Jun. 
#     "Transition-pathway models of atomic diffusion on fcc metal surfaces. II. Stepped surfaces." 
#     Physical Review B 76, no. 24 (2007): 245408.
# 
#     Extract the contribution of the coordination number from the atoms migrating to the step corner   
# =============================================================================
        clustering_energy = E_dataset[-1]
        E_clustering = [0,0,clustering_energy * 2,clustering_energy * 3,clustering_energy * 4,clustering_energy * 5,clustering_energy * 6,clustering_energy * 7,clustering_energy * 8,clustering_energy * 9,clustering_energy * 10,clustering_energy * 11,clustering_energy * 12,clustering_energy * 13] 


        Act_E_list = [E_mig_sub,
                      E_mig_upward_subs_layer111,E_mig_downward_layer111_subs,
                      E_mig_upward_layer1_layer2_111,E_mig_downward_layer2_layer1_111,
                      E_mig_upward_subs_layer100,E_mig_downward_layer100_subs,
                      E_mig_111_terrace_Cu,E_mig_100_terrace_Cu,
                      E_mig_edge_100,E_mig_edge_111,
                      binding_energy,E_clustering]
        
        
        filename = 'grid_'+ formula + "_" + str(int(max(crystal_size) / 10)) + "nm"
        System_state = initialize_grid_crystal(filename,crystal_features,experimental_conditions,Act_E_list, 
              lammps_file,superbasin_parameters,save_data)  

        # The minimum energy to select transition pathways to create a superbasin should be smaller
        # than the adsorption energy
        print(f"Minimum energy for superbasin {superbasin_parameters[2]} and activation energy for adsorption {System_state.Act_E_gen}")
        if superbasin_parameters[2] > System_state.Act_E_gen:
            raise ValueError(f"Minimum energy for superbasin {superbasin_parameters[2]} is greater than activation energy for adsorption {System_state.Act_E_ad}")
            import sys
            sys.exit(1)
            
        # Maximum probability per site for deposition to establish a timestep limits
        # The maximum timestep is that one that occupy X% of the site during the deposition process
        P_limits = 0.05
        System_state.limit_kmc_timestep(P_limits)

# =============================================================================
#     - test[0] - Normal deposition
#     - test[1] - Introduce a single particle in a determined site
#     - test[2] - Introduce and remove a single particle in a determined site 
#     - test[3] - Introduce two adjacent particles
#     - test[4] - Hexagonal seed - 7 particles in plane + 1 particle in plane
#     - test[5] - Hexagonal seed - 7 particles in plane and 1 on the top of the layer
#     - test[6] - 2 hexagonal seeds - 2 layers and one particle on the top 
#     - test[7] - 2 hexagonal seeds - 2 layers and one particle attach to the lateral
#     - test[8] - cluster
#     - test[9] - 3 Cu layers

# =============================================================================
        test_selected = 0
        test = [0,1,2,3,4,5,6,7,8,9]

        # Deposition process of chemical species
        if System_state.timestep_limits < float('inf'):
            System_state.deposition_specie(System_state.timestep_limits,rng,test[test_selected])
            System_state.track_time(System_state.timestep_limits) 
            System_state.add_time()
        else:
            System_state.deposition_specie(0,rng,test[test_selected])
            System_state.track_time(0) 
            System_state.add_time()
            
    elif experiment == 'annealing':
        
        script_directory = Path(__file__).parent
        filename = script_directory / 'variables_AsDeposited.pkl'
        
        # Open the file in binary mode
        with open(filename, 'rb') as file:
          
            # Call load method to deserialze
            myvar = pickle.load(file)
            
        System_state = myvar['System_state']
        
        temp = [723] #(K)
    
        System_state.temperature = temp[n_sim]
        System_state.experiment = experiment
        P_limits = 1
        System_state.TR_gen = 0;
        System_state.Act_E_gen = 0
        System_state.limit_kmc_timestep(P_limits)
        System_state.time = 0
        System_state.list_time = []
        System_state.E_min = 0.0
        System_state.E_min_lim_superbasin = 0.25
        #System_state.n_search_superbasin = 25
        #System_state.time_step_limits = 1e-10
        #System_state.domain_height = System_state.crystal_size[2]
        #System_state.sites_generation_layer = 'bottom_layer'
        #System_state.facets_type = [(1,1,1),(1,0,0)]
        
        
        for site in System_state.adsorption_sites:
            if System_state.grid_crystal[site].site_events:
                System_state.grid_crystal[site].site_events[0][0] = System_state.TR_gen
                System_state.grid_crystal[site].site_events[0][-1] = System_state.Act_E_gen

        
    elif experiment == 'ECM memristor':
        # =============================================================================
        #         Experimental conditions
        #         
        # =============================================================================
        sticking_coeff = None       
        partial_pressure = None # (Pa = N m^-2 = kg m^-1 s^-2)
        temp = 300
        T = temp # (K)
        
        experimental_conditions = {'sticking_coeff':sticking_coeff,'partial_pressure':partial_pressure,'T':T,'experiment':experiment}
        
        
        # =============================================================================
        #         Crystal structure
        #         
        # =============================================================================
        material_selection = {"CeO2":"mp-20194", "ZrPbO3":"mp-1068577"}
        nearest_neighbors_distance = {"CeO2":4, "ZrPbO3": 5}
        technologies = ['ECM','PZT']
        technology = technologies[0]
        id_material_Material_Project = material_selection["CeO2"]
        radius_neighbors = nearest_neighbors_distance["CeO2"]
        crystal_size = (50,50,50) # (angstrom (Å))
        miller_indices = (0,0,1)
        use_parallel = None
        facets_type = None
        affected_sites = ['Empty','O']
        affected_site = affected_sites[0]
        interstitial_specie = 'Ag'
        available_events = {'migration': True, 'reduction': True, 'oxidation': True}

        mode = ['interstitial', 'vacancy']

        sites_generation_layer = ['bottom_layer','top_layer']
        
        # -----------------
        # Grain boundaries
        # -----------------
        
        gb_configurations = [
          {
          'type':'cylindrical',
          'center': [crystal_size[0] * 0.5, crystal_size[1] * 0.5],
          'radius': 8.0,
          'outer_radius': 8.0,
          'Act_E_diff_GB': 2.7
          }
        ]
        
        """
        gb_configurations = [
          {
          'type':'vertical_planar',
          'orientation':'xz',
          'position':crystal_size[1] * 0.5 + 2.0, # Position in y
          'width':4.0,
          'outer_width':5.0,
          'Act_E_diff_GB': 2.7
          }
        ]
        """

        script_directory = Path(__file__).parent        # Get the config path from the environment variable or fallback to the current directory
        config_path = script_directory / 'config.json'
        
        
        # Create a config.json file with the API key -> To avoid uploading to Github
        with open(config_path) as config_file:
            config = json.load(config_file)
            api_key = config['api_key']
        

        # Retrieve material data
        with MPRester(api_key) as mpr:
            # Retrieve material summary information
            material_summary = mpr.materials.summary.search(material_ids=[id_material_Material_Project])
            formula = material_summary[0].formula_pretty
                            
        crystal_features = {
          'id_material_Material_Project': id_material_Material_Project,
          'crystal_size': crystal_size,
          'miller_indices': miller_indices,
          'api_key': api_key,
          'use_parallel': use_parallel,
          'facets_type': facets_type,
          'affected_site': affected_site,
          'mode': mode[0],
          'radius_neighbors': radius_neighbors,
          'sites_generation_layer': sites_generation_layer[1],
          'available_events': available_events,
          'gb_configurations': gb_configurations
        }


        # =============================================================================
        #             Superbasin parameters
        #     
        # =============================================================================
        n_search_superbasin = 50 # If the time step is very small during n_search_superbasin steps, search for superbasin
        time_step_limits = 1e-4 # Time needed for efficient evolution of the system
        E_min = 0.5
        energy_step = 0.05
        time_based_superbasin = True
        superbasin_parameters = {
          'n_search_superbasin':n_search_superbasin, 
          'time_step_limits':time_step_limits, 
          'E_min':E_min, 
          'energy_step':energy_step,
          'time_based_superbasin':time_based_superbasin
        }
        
        # =============================================================================
        #             Electric field parameters: Required for the Poisson Solver
        #     
        # =============================================================================
        mesh_file = formula + "_" + str(int(max(crystal_size) / 10)) + "nm" + "_mesh.msh"  # Adjust filename as needed
        
        # Parameters for Poisson solver
        active_dipoles = 4
        poisson_solve_frequency = int(snapshoots_steps)  # Solve Poisson every N KMC steps
        solve_Poisson = True
        save_Poisson = False
        
        screening_factor = 0.01
        ion_charge = 1
        conductivity_CF  = 6.3e7 * 0.1
        conductivity_dielectric = 1e-1
        

        
        # Extract data from Materials Project
        with MPRester(api_key) as mpr:
            # Get the material with chemenv data specifically: chemical environment: valence, local symmetry
            material_data = mpr.materials.chemenv.search(material_ids=[id_material_Material_Project])
            dielectric_data = mpr.materials.dielectric.search(material_ids=[id_material_Material_Project])
        
        chem_env_symmetry = material_data[0].chemenv_name[0]
        metal_valence = material_data[0].valences[0]
        
        # Bond lengths
        central_atom = material_data[0].mol_from_site_environments[0][0]
        d_metal_O = central_atom.distance(material_data[0].mol_from_site_environments[0][1])
        
        # Dielectric constant
        try:
            if dielectric_data and len(dielectric_data) > 0 and hasattr(dielectric_data[0], 'e_total'):
                epsilon_r = dielectric_data[0].e_total
            else:
                warnings.warn("No dielectric data available for this material. Using manual-introduced value.")
                epsilon_r = 23
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Could not extract dielectric constant: {e}. Using manual-introduced value.")
            epsilon_r = 23
            
            

        poissonSolver_parameters = {'mesh_file':mesh_file,
                                    'epsilon_r':epsilon_r,'chem_env_symmetry':chem_env_symmetry,'metal_valence':metal_valence,'d_metal_O':d_metal_O,'active_dipoles':active_dipoles,
                                    'poisson_solve_frequency':poisson_solve_frequency,'solve_Poisson':solve_Poisson,'save_Poisson':save_Poisson, 'screening_factor':screening_factor,
                                    'ion_charge':ion_charge,
                                    'conductivity_CF':conductivity_CF, 'conductivity_dielectric':conductivity_dielectric
        
        }
        
        # =============================================================================
        #             Electrical parameters
        #     
        # =============================================================================
        initial_voltage=0.0 
        initial_time = 0.0   
        series_resistance = 2e2  
        area_experimental_device = np.pi * (50*1e-6)**2
        
        max_voltage = 2.6
        min_voltage = -1.5
        ramp_rate = 1
        num_cycles=1
        voltage_update_time = 0.1
        
        Elec_controller = ElectricalController(initial_voltage,initial_time,series_resistance, current_model = 'schottky', crystal_size = crystal_size)
        
        Elec_controller.initialize_ramp_voltage_cycle(
          max_voltage, 
          min_voltage, 
          ramp_rate, 
          voltage_update_time,
          num_cycles
        )
        
        # Schottky emission parameters
        phi_b = 0.53
        
        Elec_controller.initialize_current_parameters(
          barrier_height=phi_b,      # eV
          temperature=T,         # K  
          area= area_experimental_device,             # m²
          epsilon_r=epsilon_r            # HfO2
        )
        
        # =============================================================================
        #             Activation energies
        #     
        # =============================================================================
        # Retrieve the activation energies
        activation_energy_file = script_directory / 'activation_energies_memristors.json'
        with open(activation_energy_file, 'r') as file:
            data = json.load(file)
            
        Act_E_list = {}
        for defect in data[technology]:
            # Search the selected element we retrieved from Materials Project
            if defect['specie'] == interstitial_specie:

                #Search the activation energies
                for key,activation_energies in defect.items():
                    if 'activation_energies' in key:
                        # Select the dataset
                        for key_1,act_energy in activation_energies.items():
                            if isinstance(act_energy, (int, float)):
                                #E_dataset.append(act_energy)
                                Act_E_list[key_1] = act_energy
                
        
        CN_clustering_energy = Act_E_list['CN_clustering_energy']
        E_clustering = [0,0,CN_clustering_energy * 2,CN_clustering_energy * 3,CN_clustering_energy * 4,CN_clustering_energy * 5,CN_clustering_energy * 6,CN_clustering_energy * 7,CN_clustering_energy * 8,CN_clustering_energy * 9,CN_clustering_energy * 10,CN_clustering_energy * 11,CN_clustering_energy * 12,CN_clustering_energy * 13] 
        Act_E_list['CN_clustering_energy'] = E_clustering
        
        if 'CN_redox_energy' in Act_E_list:
          CN_redox_energy = Act_E_list['CN_redox_energy']
          E_redox = [0,0,CN_redox_energy * 2,CN_redox_energy * 3,CN_redox_energy * 4,CN_redox_energy * 5,CN_redox_energy * 6,CN_redox_energy * 7,CN_redox_energy * 8,CN_redox_energy * 9,CN_redox_energy * 10,CN_redox_energy * 11,CN_redox_energy * 12,CN_redox_energy * 13] 
          Act_E_list['CN_redox_energy'] = E_redox
        

        # =============================================================================
        #             Filename
        #     
        # =============================================================================
        
        filename = 'grid_' + formula + "_" + str(int(max(crystal_size) / 10)) + "nm"
        
        
        # =============================================================================
        #             Crystal structure generation
        #     
        # =============================================================================
        System_state = initialize_grid_crystal(filename,crystal_features,experimental_conditions,Act_E_list, 
              lammps_file,superbasin_parameters,save_data, interstitial_specie,poissonSolver_parameters) 
              
        Elec_controller.crystal_size = System_state.crystal_size #  The crystal_size after the generation of the lattice may differ from the parameter provided in a NN points separation
                
        # =============================================================================
        #             Initialization of defects
        #     
        # =============================================================================
        #P = 0.01
        #System_state.defect_gen(rng,P)
        #System_state.deposition_specie(0,rng,test = 6)
        
        # This timestep_limits will depend on the V/s ratio
        System_state.timestep_limits = voltage_update_time

    return System_state,rng,paths,Results, simulation_parameters,Elec_controller

    # =============================================================================
    #     Initialize the crystal grid structure - nodes with empty spaces
    # =============================================================================    
def initialize_grid_crystal(filename,crystal_features,experimental_conditions,Act_E_list, 
    lammps_file,superbasin_parameters,save_data, interstitial_specie = None, poissonSolver_parameters = None):
      
        # If grid_crystal exists: we loaded
        # Otherwise: we create it (very expensive for larger systems ~100 anstrongs)
        current_directory = Path(__file__).parent
        # Check for .dat and .pkl extensions       
        # Dynamically append extensions for checks
        dat_file = current_directory / filename
        dat_file_with_ext = dat_file.with_suffix('.dat')
        pkl_file_with_ext = dat_file.with_suffix('.pkl')
        
        # Prepare keyword arguments
        crystal_kwargs = {}
        
        # Add interstitial_specie if provided
        if interstitial_specie is not None:
            crystal_kwargs['interstitial_specie'] = interstitial_specie
            
        if poissonSolver_parameters is not None:
            crystal_kwargs['poissonSolver_parameters'] = poissonSolver_parameters
        
        if dat_file_with_ext.exists():
            print('Loading ' + filename + ".dat")
            # Load from .dat
            dat_file = current_directory / f"{filename}"
            with shelve.open(dat_file) as my_shelf:
                grid_crystal = my_shelf.get(filename)
                
            # Add grid_crystal to kwargs
            crystal_kwargs['grid_crystal'] = grid_crystal
            
            System_state = Crystal_Lattice(
                crystal_features,
                experimental_conditions,
                Act_E_list,
                lammps_file,
                superbasin_parameters,
                **crystal_kwargs
                )

        elif pkl_file_with_ext.exists():
            print('Loading ' + filename + '.pkl')
            # Load from .pkl
            with open(pkl_file_with_ext, 'rb') as file:
                # Call load method to deserialze
                data = pickle.load(file)
            grid_crystal = data.get(filename)
            
            # Add grid_crystal to kwargs
            crystal_kwargs['grid_crystal'] = grid_crystal

            System_state = Crystal_Lattice(
                crystal_features,
                experimental_conditions,
                Act_E_list,
                lammps_file,
                superbasin_parameters,
                **crystal_kwargs
                )

            
        else:
            # Create new grid_crystal
            print('Creating ' + filename)
            System_state = Crystal_Lattice(
                crystal_features,
                experimental_conditions,
                Act_E_list,
                lammps_file,
                superbasin_parameters,
                **crystal_kwargs # This will only contain interstitial_specie if provided
                )
            
            # Save the newly created data
            if save_data:
                print('Saving ' + filename)
                save_variables(current_directory, {filename : System_state.grid_crystal}, filename)

        return System_state
        
        
def search_superbasin(System_state,kmc_time_step):
          
    # We need a deepcopy? System_state.sites_occupied will be modified on site
    # when calling Superbasin() and it will change the order of sites_occupied
    # sites_occupied = copy.deepcopy(System_state.sites_occupied)
    
    if not System_state.should_activate_superbasin(kmc_time_step):
        return  # Early exit: no action needed
    
    # This approach should be more efficient and memory-friendly
    sites_occupied = System_state.sites_occupied[:] 

    start_time = time.time()
    
    for idx in sites_occupied:
        for event in System_state.grid_crystal[idx].site_events:
            if (idx not in System_state.superbasin_dict) and (event[3] <= System_state.E_min) and isinstance(event[2],int):
                print('Creating superbasin')
                superbasin = Superbasin(idx, System_state, System_state.E_min,sites_occupied)
                print(f'Is superbasin valid? {superbasin.valid}')
                if superbasin.valid:    
                    System_state.superbasin_dict.update({idx: superbasin})
    

    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    if elapsed_time > 300 and System_state.E_min_lim_superbasin > System_state.energy_step:
        System_state.E_min -= System_state.energy_step
    print(f"Elapsed time superbasin: {elapsed_time} seconds")    
    print("Superbasins generated: ",len(System_state.superbasin_dict))
        

def save_simulation(files_copy,dst,n_sim,experiment):
    
    # Create the simulation directory
    parent_dir = f'Sim_{n_sim}'
    sim_dir = dst / parent_dir
    sim_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
    
    # Define subdirectories
    program_directory = sim_dir / 'Program'
    data_directory = sim_dir / 'Crystal evolution'
    
    # Create directories
    program_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    
    # Return paths as a dictionary
    paths = {
        'data': data_directory,
        'program': program_directory,
        'results': sim_dir
    }
    
    # Copy the files
    current_directory = Path(__file__).parent  # Get the current directory of the script
    for file in files_copy:
        source_file = current_directory / file  # Path of the source file
        destination_file = paths['program'] / file  # Path for the destination file
        shutil.copyfile(source_file, destination_file)  # Copy the file
    

    if experiment in ['deposition','annealing']:
      # Create and return results object
      excel_filename = paths['results'] / 'Results.csv'  # Define the path to the results CSV file
      Results = SimulationResults(excel_filename)
      
    else:
      Results = None
      
    return paths, Results


def save_variables(paths,variables,filename):
    
    
    # Convert paths to Path object if it's a string (if it's not already)
    paths = Path(paths)  # Ensure paths is a Path object
    
    # Full file path
    file_path = paths / filename

    if platform.system() == 'Windows':  # When running on Windows
        with shelve.open(str(file_path), 'n') as my_shelf:
            for key, value in variables.items():
                my_shelf[key] = value

    elif platform.system() == 'Linux':  # When running on Linux
        filename += '.pkl'
        file_path = file_path.with_name(filename)  # Ensure the filename ends with .pkl

        # Open the file and use pickle.dump()
        with open(file_path, 'wb') as file:
            pickle.dump(variables, file)
    

class SimulationResults:
    def __init__(self, excel_filename):
        self.excel_filename = excel_filename
        # Initialize a CSV file with headers
        with open(excel_filename, 'w') as f:
            f.write('Time,Mass,Sites Occupation,Average Thickness,Terrace Area,std_terrace,max_terrace,RMS Roughness,Performance time\n')
    
    def measurements_crystal(self, time, mass_gained, sites_occupation, thickness, avg_terrace,std_terrace,max_terrace, surf_roughness_RMS,performance_time):
            # Append measurements to the CSV file
            with open(self.excel_filename, 'a') as f:
                f.write(f'{time},{mass_gained},{sites_occupation},{thickness},{avg_terrace},{std_terrace},{max_terrace},{surf_roughness_RMS},{performance_time}\n')