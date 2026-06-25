# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:03:03 2024

@author: samuel.delgado
"""
import numpy as np
import matplotlib.pyplot as plt
import platform
import shutil

from kinetix.lattice.crystal import Crystal_Lattice
from kinetix.solvers.electrical import ElectricalController
from kinetix.utils.mpi_context import MPIContext
from kinetix.material_fetcher import MaterialDataFetcher
from kinetix.configs.electrical_config import ElectricalConfig, VoltageConfig, CurrentConfig, VoltageMode, CurrentModel
from kinetix.configs.config_loader import get_api_key,load_activation_energies, get_grids_root,get_mesh_root,get_parameters_root
from kinetix.configs.material_config import MaterialConfig, MaterialSelection, CrystalStructure
from kinetix.configs.defect_config import DefectsConfig, DefectConfig
from kinetix.configs.reaction_config import ReactionsConfig, ReactionConfig, ReactionSpecies
from kinetix.configs.solver_config import PoissonSolverConfig, SuperbasinConfig, HeatSolverConfig
from kinetix.configs.simulation_config import SimulationConfig, ExperimentalConditions, SimulationSettings
from kinetix.configs.grain_boundary_config import GrainBoundariesConfig

from pymatgen.ext.matproj import MPRester
# from mp_api.client import MPRester
import json
from pathlib import Path

import os
import pickle
import shelve
import time
import warnings



from typing import Any, Dict, List, Tuple


def initialization(n_sim,params):
    
# =============================================================================
#         Simulation parameters
#         
# =============================================================================  
    # === Initialize MPI ===
    mpi_ctx = MPIContext.get_instance()
    
    parameters_root = get_parameters_root()
    preset_name = 'ECM_CeO2.yaml'
    preset_path = parameters_root / 'presets' / preset_name
    config = SimulationConfig.from_yaml(preset_path)
    
    seed = config.settings.seed_rng
    # Random seed as time
    rng = np.random.default_rng(seed) # Random Number Generator (RNG) object
    
    save_data = config.settings.save_data
    lammps_file = config.settings.lammps_output
    snapshoots_steps = int(4e1)
    total_steps = int(snapshoots_steps * 25)
    
    simulation_type = config.settings.simulation_type
    
    simulation_parameters = {
      'save_data':save_data, 'snapshoots_steps':snapshoots_steps,
      'total_steps':total_steps
    }    

    # Default resolution for figures
    plt.rcParams["figure.dpi"] = 100 # Default value of dpi = 300
    
    if save_data:
        files_copy = ['run_simulation.py', 
                      'data/parameters','kinetix']
        
        output_path = Path(config.settings.output_path)
            
        if mpi_ctx.rank == 0:
          paths,Results = save_simulation(files_copy,output_path,n_sim,simulation_type) # Create folders and python files
        else:
          # Other ranks create empty paths structure (same keys)
          sim_dir = output_path / f'Sim_{n_sim}'
          paths = {
            'data': sim_dir / 'output',
            'program': sim_dir / 'program',
            'results': sim_dir
          }
          Results = None
        
        mpi_ctx.barrier()
          
    else:
        paths = {'data': ''}
        Results = []
        

    





    if simulation_type == 'deposition':         
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
        
        experimental_conditions = [sticking_coeff,partial_pressure,T,simulation_type]
    
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
            
    elif simulation_type == 'annealing':
        
        script_directory = Path(__file__).parent
        filename = script_directory / 'variables_AsDeposited.pkl'
        
        # Open the file in binary mode
        with open(filename, 'rb') as file:
          
            # Call load method to deserialze
            myvar = pickle.load(file)
            
        System_state = myvar['System_state']
        
        temp = [723] #(K)
    
        System_state.temperature = temp[n_sim]
        System_state.experiment = simulation_type
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

        
    elif simulation_type == 'electronic_device':        
        
        ### ----------------- PARAMETER SWEEP ----------------- ###
        #config.defects.defects["oxygen_vacancy"].initial_concentration_bulk = params["vo_initial_concentration"]
        #config.defects.defects["oxygen_vacancy"].initial_concentration_GB = params["vo_initial_concentration"] 
        #config.experimental.temperature = params["temperature"]
    
        
        # 1. Fetch Material Data from Materials Project
        api_key = get_api_key()
        fetcher = MaterialDataFetcher(api_key,mpi_ctx)
        material_data = fetcher.get_all_material_data(config.material.selection.mp_id)
        
        
        # Resolve epsilon_r: override user value with MP data if exists
        mp_epsilon = material_data.get('epsilon_r')
        if mp_epsilon is not None:
          config.material.epsilon_r = mp_epsilon
        
        # 2. Update config with fetched data
        config.material.formula = material_data['formula']
        config.material.chem_env_symmetry = material_data.get('chem_env_symmetry')
        config.material.metal_valence = material_data.get('metal_valence')
        config.material.bond_length = material_data.get('bond_length_metal_O')
        
        if config.electrical and config.electrical.current:
          config.electrical.current.epsilon_r = config.material.epsilon_r
          
        
        Elec_controller = ElectricalController.from_config(config.electrical)
        
        # 3. Experimental conditions
        experimental_conditions = {
          'sticking_coeff': config.experimental.sticking_coeff,
          'partial_pressure':config.experimental.partial_pressure,
          'T':config.experimental.temperature,
          'simulation_type':simulation_type
        }
        
        # 4. Prepare parameters for grid initialization
        crystal_size = config.material.structure.size # (angstrom)
        formula = config.material.formula
        
        defects_config = config.defects.to_dict()
        
        if config.reactions:
          reactions_config = config.reactions.to_dict()
        else:
          reactions_config = None
        
        gb_configurations = [grainboundary.to_dict() for grainboundary in config.grain_boundaries]                    
        crystal_features = {
          'chemical_formula': formula,
          'id_material_Material_Project': config.material.selection.mp_id,
          'crystal_size': crystal_size,
          'miller_indices': config.material.structure.miller_indices,
          'api_key': api_key,
          'facets_type': config.material.structure.facets_type,
          'affected_site': config.material.structure.affected_site,
          'mode': config.settings.mode,
          'radius_neighbors': config.material.selection.radius_neighbors,
          'sites_generation_layer': config.material.structure.sites_generation_layer,
          'defects_config': defects_config,
          'reactions_config': reactions_config,
          'gb_configurations': gb_configurations,
          'technology': config.settings.technology,
          'rng': rng
        }
        
        # 5. Superbasin parameters
        superbasin_parameters = config.superbasin.to_dict()
        
        # 6. Poisson solver parameters
        mesh_file = f"{formula}_{int(max(crystal_size) / 10)}nm_mesh.msh"
        poissonSolver_parameters = {
          'mesh_file': mesh_file,
          'epsilon_r': config.material.epsilon_r,
          'chem_env_symmetry': config.material.chem_env_symmetry,
          'metal_valence': config.material.metal_valence,
          'd_metal_O': config.material.bond_length,
          'active_dipoles': config.poisson.active_dipoles,
          'solve_Poisson': config.poisson.solve_Poisson,
          'save_Poisson': config.poisson.save_Poisson, 
          'screening_factor': config.poisson.screening_factor,
          'conductivity_CF': config.poisson.conductivity_CF, 
          'conductivity_dielectric':config.poisson.conductivity_dielectric,
          'defects_config':defects_config,
          'mesh_config': config.mesh.to_dict()
        }

        # 7. Activation energies
        ae_data = load_activation_energies(preset_path, config.settings)
        ### ----------------- PARAMETER SWEEP ----------------- ###
        #ae_data['PZT'][1]['activation_energies']['E_gen_defect'] = params['h_generation']

        Act_E_dict = _process_activation_energies(
          defects_config,
          ae_data,
          config.settings.technology
        )
        
        # 8. Initialize crystal lattice
        filename = f'grid_{formula}_{int(max(crystal_size) / 10)}nm'

        System_state = initialize_grid_crystal(
          filename,
          mpi_ctx,
          crystal_features,
          experimental_conditions,
          Act_E_dict, 
          lammps_file,
          superbasin_parameters,
          save_data,
          poissonSolver_parameters
        ) 
        
        
        # 9. Post initialization steps
        # Write metadata
        System_state.write_metadata(paths['data'])   

        Elec_controller.crystal_size = System_state.crystal_size #  The crystal_size after the generation of the lattice may differ from the parameter provided in a NN points separation
        System_state.timestep_limits = Elec_controller.voltage_update_time  
        
        # Initialize defects
        System_state.defect_gen()
        

    return System_state,rng,paths,Results, simulation_parameters,Elec_controller

    # =============================================================================
    #     Initialize the crystal grid structure - nodes with empty spaces
    # =============================================================================    
def initialize_grid_crystal(
  filename,
  mpi_ctx,
  crystal_features,
  experimental_conditions,
  Act_E_dict, 
  lammps_file,
  superbasin_parameters,
  save_data, 
  poissonSolver_parameters = None
):
        """
        Initialize or load a crystal lattice state for kMC simulation.
        
        Parameters
        ----------
        filename : str
            Base name for saved grid files (without extension).
        crystal_features : dict
            Contains material, defect config, size, etc.
        ... (other params)
        
        Returns
        -------
        Crystal_Lattice
        """
        # If grid_crystal exists: we loaded
        # Otherwise: we create it (very expensive for larger systems ~100 anstrongs)
        grid_directory = get_grids_root()
        dat_path = grid_directory / f"{filename}.dat"
        pkl_path = grid_directory / f"{filename}.pkl"

        # Determine if we can load an existing grid
        grid_crystal = None
        
        if dat_path.exists():
          print('Loading ' + filename + ".dat")
          with shelve.open(str(grid_directory / filename)) as shelf:
            grid_crystal = shelf.get(filename)
            
        elif pkl_path.exists():
          print('Loading ' + filename + '.pkl')
          with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            grid_crystal = data.get(filename)
        
        
        # Prepare keyword arguments
        crystal_kwargs = {}
        if poissonSolver_parameters is not None:
          crystal_kwargs['poissonSolver_parameters'] = poissonSolver_parameters
            
        if grid_crystal is not None:
          crystal_kwargs['grid_crystal'] = grid_crystal
            
        # Instantiate system (loads or creates grid internally)
        System_state = Crystal_Lattice(
          crystal_features = crystal_features,
          experimental_conditions = experimental_conditions,
          Act_E_dict = Act_E_dict,
          lammps_file = lammps_file,
          superbasin_parameters = superbasin_parameters,
          mpi_ctx = mpi_ctx,
          **crystal_kwargs 
        )
            
            # Save the newly created data
        if save_data and grid_crystal is None:
          print(f'Saving {filename}')
          save_variables(grid_directory, {filename : System_state.grid_crystal}, filename)

        return System_state
           
           
def _process_activation_energies(defects_config, ae_data:Dict, technology: str) -> Dict:
  """
  Process activation energies from JSON into Act_E_dict format.
  Handles CN-dependent energy expansion.
  """   
  def expand_clustering_energy(base_energy: float, max_cn: int = 15) -> list:
    """ 
    Returns list where index = CN, values = base_energy * CN for CN >= 2
    """
    energies = [0.0, 0.0] # CN = 0,1 -> No clustering
    for cn in range(2,max_cn+1):
      energies.append(base_energy * cn)
    return energies
  
  # Container: Act_E_dict[defect_name] = {energy_key: value or list}
  Act_E_dict = {}
          
  for defect_name, defect_cfg in defects_config.items():
    key = defect_cfg["activation_energies_key"]  
          
    matching_entry = None
    for entry in ae_data[technology]:
      if entry.get("specie") == key:
        matching_entry = entry
        break
        
    if matching_entry is None:
      warnings.warn(f'No activation energy data found for "{key}" in technology "{technology}". Skipping.')
      continue
            
    # Extract all activation energies from this entry
    energies = {}
    for field_name, value in matching_entry.items():
      if "activation_energies" in field_name and isinstance(value, dict):
      # This block contains named energies (e.g., "migration", "clustering")
        for energy_name, energy_val in value.items():
          if isinstance(energy_val, (int,float,dict)):
            energies[energy_name] = energy_val
            
    # Expand clustering and redox energies into CN-dependent lists
    if "CN_clustering_energy" in energies:
      base = energies["CN_clustering_energy"]
      energies["CN_clustering_energy"] = expand_clustering_energy(base)
            
    if "CN_redox_energy" in energies:
      base = energies["CN_redox_energy"]
      energies["CN_redox_energy"] = expand_clustering_energy(base)
            
    Act_E_dict[defect_name] = energies
    
  return Act_E_dict        
    

def save_simulation(files_copy,dst,n_sim,simulation_type):
    
     # === Exclusion patterns - don't copy these ===
    from shutil import ignore_patterns 
      
    # Create the simulation directory
    parent_dir = f'Sim_{n_sim}'
    sim_dir = dst / parent_dir
    sim_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
    
    # Define subdirectories
    program_directory = sim_dir / 'program'
    data_directory = sim_dir / 'output'
    
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
    project_root = current_directory.parent
    
    for item in files_copy:
        source_path = project_root / item  # Path of the source file
        dest_path = program_directory / item
        
        if not source_path.exists():
          print(f"SKIP: {item} not found at {source_path}")
          continue
        
        # === Check for symlinks (common cause of this error) ===
        if source_path.is_symlink():
          print(f"SKIP: {item} is a symlink")
          continue
          
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
          shutil.copy2(source_path, dest_path)  # Copy the file
          print(f"Copied file: {item}")
        elif source_path.is_dir():
          if dest_path.exists():
            shutil.rmtree(dest_path)
            print(f"Removed existing: {item}")
            
          try:
            shutil.copytree(
              source_path,
              dest_path, 
              ignore=ignore_patterns('__pycache__', '*.pyc', '.git'),
              dirs_exist_ok=False
            )
            print(f"Copied directory: {item}")
          except shutil.Error as e:
            print(f"Failed to copy {item}: {e}")
            # List what was copied before failure
            if dest_path.exists():
              files = list(dest_path.rglob('*'))
              print(f"Partially copied {len(files)} items")
            raise
          
    

    if simulation_type in ['deposition','annealing']:
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