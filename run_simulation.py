# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""




import cProfile
import sys
from kinetix.initialization import initialization,save_variables
import numpy as np
import time
import platform

def get_parameters_from_sim_id(sim_id: int) -> dict:
   """Map SIM_ID to simulation parameters."""
   # Define parameters
   v0_initial_concentrations = [1.0e-3, 1.0e-2, 2.0e-2, 3.0e-2,  4.0e-2,  5.0e-2]
   temperatures = [293.0, 310.0, 323.0]
   h_generation = [0.45,0.48,0.50, 0.52, 0.55]
   
   idx = sim_id
   
   i_vo = idx % 6
   i_temp = (idx // 6) % 3
   i_gen_h = (idx // 18) % 5
   
   return {
     'vo_initial_concentration': v0_initial_concentrations[i_vo],
     'temperature': temperatures[i_temp],
     'h_generation': h_generation[i_gen_h]
   }
   

def main(sim_id):
        
        params = get_parameters_from_sim_id(sim_id)
        System_state,rng,paths,Results,simulation_parameters,Elec_controller = initialization(sim_id, params)
        
        if System_state.rank == 0:
          print(f'System size: {System_state.crystal_size}')
          total_start_time = time.time()
          System_state.plot_crystal(45,45,paths['data'],0)    
          
        System_state.add_time()
            
        
        j = 0
        snapshots_steps = simulation_parameters['snapshoots_steps']
        total_steps = simulation_parameters['total_steps']
        save_data = simulation_parameters['save_data']
        
        starting_time = time.time()
    # =============================================================================
    #     Deposition
    # 
    # =============================================================================
        if System_state.simulation_type == 'deposition':   
    
            nothing_happen = 0
            # list_time_step = []
            list_sites_occu = []
            thickness_limit = 10 # (1 nm)
            System_state.measurements_crystal()
            i = 0
            while System_state.thickness < thickness_limit:
                i+=1
          
                System_state,KMC_time_step, _ = KMC(System_state,rng)
                                
                list_sites_occu.append(len(System_state.sites_occupied))
                
                if np.mean(list_sites_occu[-System_state.n_search_superbasin:]) == len(System_state.sites_occupied):
                # if np.mean(list_time_step[-System_state.n_search_superbasin:]) <= System_state.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if System_state.E_min - System_state.energy_step > 0:
                        System_state.E_min -= System_state.energy_step
                    else:
                        System_state.E_min = 0
                
                if System_state.n_search_superbasin == nothing_happen:
                    search_superbasin(System_state)
                elif nothing_happen> 0 and nothing_happen % System_state.n_search_superbasin == 0:
                    if System_state.E_min_lim_superbasin >= System_state.E_min + System_state.energy_step:
                        System_state.E_min += System_state.energy_step
                    else:
                        System_state.E_min = System_state.E_min_lim_superbasin
                    search_superbasin(System_state)
                    
    
                    
                # print('Superbasin E_min: ',System_state.E_min)
            
                if i%snapshots_steps== 0:
                    System_state.add_time()
                    
                    j+=1
                    System_state.measurements_crystal()
                    print(str(System_state.thickness/thickness_limit * 100) + ' %','| Thickness: ', System_state.thickness, '| Total time: ',System_state.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                      System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                      System_state.surf_roughness_RMS,end_time-starting_time)
        
                    System_state.plot_crystal(45,45,paths['data'],j)
                    
    
    # =============================================================================
    #     Annealing  
    #            
    # =============================================================================
        elif System_state.simulation_type == 'annealing':
            i = 0
            
            nothing_happen = 0

            System_state.measurements_crystal()
            list_time_step = []
    
            while j*snapshots_steps < total_steps:
    
                i+=1
                System_state,KMC_time_step, _ = KMC(System_state,rng)
                list_time_step.append(KMC_time_step)
                
    # =============================================================================
    #                 Search of superbasin
    # =============================================================================
                if np.mean(list_time_step[-System_state.n_search_superbasin:]) <= System_state.time_step_limits:
                # if np.mean(list_time_step[-4:]) <= System_state.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if System_state.E_min - System_state.energy_step > 0:
                        System_state.E_min -= System_state.energy_step
                    else:
                        System_state.E_min = 0
                        
                if System_state.n_search_superbasin == nothing_happen:
                    search_superbasin(System_state)
                elif nothing_happen > 0 and nothing_happen % System_state.n_search_superbasin == 0:
                    if System_state.E_min_lim_superbasin >= System_state.E_min + System_state.energy_step:
                        System_state.E_min += System_state.energy_step
                    else:
                        System_state.E_min = System_state.E_min_lim_superbasin
                    search_superbasin(System_state)
                    
    # =============================================================================
    #                     Finish search superbasin
    # =============================================================================
                
                if i%snapshots_steps== 0:
                    
                    System_state.sites_occupied = list(set(System_state.sites_occupied))
                                        
                    System_state.add_time()
                    j+=1
                    System_state.measurements_crystal()
                    print(str(j)+"/"+str(int(total_steps/snapshots_steps)),'| Total time: ',System_state.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                      System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                      System_state.surf_roughness_RMS,end_time-starting_time)
                        
                    System_state.plot_crystal(45,45,paths['data'],j)
                    
    # =============================================================================
    #     Devices: PZT, memristors  
    #            
    # =============================================================================
                    
        elif System_state.simulation_type == 'electronic_device':
            
            from collections import Counter
            solve_Poisson = System_state.poissonSolver_parameters['solve_Poisson']
            save_Poisson = System_state.poissonSolver_parameters['save_Poisson']
            
            solve_heat = System_state.heat_parameters.get('solve_heat', False)
            save_heat = System_state.heat_parameters.get('save_heat', False)
            
            V_top = Elec_controller.apply_voltage(System_state.time)
            System_state.save_electric_bias(V_top)
       
            # Dolfinx only works in Linux
            if solve_Poisson and platform.system() == 'Linux':
                from kinetix.solvers.poisson import PoissonSolver
                from kinetix.solvers.heat import HeatSolver
                from mpi4py import MPI
                
                # Initialize Poisson solver on all MPI ranks
                poisson_solver = PoissonSolver(
                  System_state.poissonSolver_parameters, 
                  grid_crystal=System_state.grid_crystal,
                  path_results = paths["results"],
                  mpi_ctx = System_state.mpi_ctx
                )
                poisson_solver.set_boundary_conditions(top_value=V_top, bottom_value=0.0)  # Set appropriate BCs
                
                if solve_heat:
                  heat_solver = HeatSolver(
                    System_state.heat_parameters,
                    grid_crystal=System_state.grid_crystal,
                    path_results = paths["results"],
                    mpi_ctx=System_state.mpi_ctx
                  )
            
                  heat_solver.set_boundary_conditions(
                    top_value=heat_solver.T_ambient,
                    bottom_value=heat_solver.T_ambient
                  )
            
            while System_state.should_continue_simulation(Elec_controller.total_simulation_time):
            
                     
                if solve_Poisson and platform.system() == 'Linux': 
                  should_solve_fields_now, snapshots = System_state.should_solve_fields_now(Elec_controller)
                       
                  particle_locations, charges, evaluation_points = System_state.get_evaluation_points()
                    
                  if should_solve_fields_now:
                        # Every time we change the applied voltage, we should calculate Poisson
                        V_top = Elec_controller.apply_voltage(System_state.time)
                        System_state.save_electric_bias(V_top)
                        clusters = System_state.prepare_clusters_for_bcs()
                        # We need the cluster to know what is the effective gap for calculating the Schottky emission
                        V_eff, _ = Elec_controller.calculate_current(clusters) # Obtain effective voltage after voltage drop of series resistance
                          
                        poisson_solver.set_boundary_conditions(top_value=V_eff, bottom_value=0.0,clusters = clusters)
                        
                        
                        run_start_time = MPI.Wtime()
                        uh = poisson_solver.solve(particle_locations,charges) 
                        run_time = MPI.Wtime() - run_start_time
                        
                        if System_state.rank == 0: print(f'Run time to solve Poisson: {run_time}')

                        if save_Poisson:
                          poisson_solver.save_potential(System_state.time,j+1)
                          
                        if solve_heat:
                         heat_start_time = MPI.Wtime()
                         
                         # Update temperature with thermal relaxation
                         # dt = time since last heat solve
                         dt_heat = Elec_controller.voltage_update_time
                         
                         T_solution = heat_solver.update_temperature(
                           dt=dt_heat,
                           poisson_solver=poisson_solver,
                           recompute_steady=True
                         )
                         
                         heat_run_time = MPI.Wtime() - heat_start_time
                         
                         Avg_T = heat_solver.get_average_temperature()
                         if System_state.rank == 0: 
                           print(f'Run time to solve Heat: {heat_run_time}', flush=True)
                           print(f'Avg temperature: {Avg_T:.10f} K', flush=True)
                         
                         # Save temperature
                         if save_heat:
                           heat_solver.save_temperature(System_state.time, j+1)
                           

                          
                        run_time = 0
                        
                  # === Evaluate fields at site positions ===
                  E_field = poisson_solver.evaluate_electric_field_at_points(evaluation_points)
                  
                  # Evaluate temperature at same points (if heat solver enabled)
                  if solve_heat:
                    T_field = heat_solver.evaluate_temperature_at_points(evaluation_points)
                  else:
                    T_field = None # Will use ambient temperature
                                   
                  System_state.update_transition_rates(E_field)
                        
                
                System_state.step_kmc(rng)
                
                """
                for cluster in clusters.values():
                  if cluster.attached_layer['bottom_layer'] and cluster.attached_layer['top_layer']:
                    print('')
                    
                """
                
                if snapshots:
                
                    j+=1
                    # Continue with serial processing on rank 0
                    if System_state.rank == 0:
                        System_state.add_time()
    
                        # System_state.measurements_crystal()
                        print(str(j)+"/"+str(int(Elec_controller.total_simulation_time/Elec_controller.voltage_update_time)),'| Total time: ',System_state.list_time[-1],'| Voltage: ',V_top)
                        print(f'Events at step {j}: {System_state.events_tracking}')
                        if Elec_controller.current_enabled:
                          print(f"Current: {Elec_controller.measurements['current'][-1]}")
    
                        end_time = time.time()
                        System_state.plot_crystal(45,45,paths['data'],j)        
                        
    
        if System_state.rank == 0:
          
          total_end_time = time.time()
          print(f"==================================================")
          print(f"SUCCESS: Simulation {sim_id} completed in {total_end_time - total_start_time:.2f} seconds.")
          print(f"==================================================")
          
          # Variables to save
          variables = {'System_state' : System_state}
          filename = 'variables'
          if save_data: save_variables(paths['program'],variables,filename)
          
          
        Elec_controller.save_IV_csv(paths['results'])
        Elec_controller.plot_V_I(paths['results'])

    
        return System_state

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
      sim_id = int(sys.argv[1])
    else:
      sim_id = 0
    
    System_state = main(sim_id)
    
    
# Use cProfile to profile the main function
#     cProfile.run('main()', 'profile_output.prof')    

# import pstats

# # Load and analyze the profiling results
# p = pstats.Stats('profile_output.prof')
# p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions