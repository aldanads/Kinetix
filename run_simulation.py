# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""

import sys
import argparse
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
   
def parse_arguments():
  """Parse command-line arguments"""
  parser = argparse.ArgumentParser(
    description="Kinetix: Kinetic Monte Carlo simulator for materials and memristive devices.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python run_simulation.py 42
  python run_simulation.py 42 --config PZT_ZrTi(PbO3)2.yaml
  python run_simulation.py 42 --config VCM_HfO2_cylindrical_gb.yaml --profile
  python run_simulation.py --config PZT_ZrTi(PbO3)2_annealing.yaml     
    """
  )
  
  parser.add_argument(
    'sim_id',
    type=int,
    nargs='?',
    default=0,
    help='Simulation ID for parameter sweep indexing (default: 0)'
  )
  
  parser.add_argument(
    '--config', '-c',
    type=str,
    default='PZT_ZrTi(PbO3)2.yaml',
    help='Preset configuration file name or path (default: PZT_ZrTi(PbO3)2.yaml)'
  )
  
  parser.add_argument(
    '--profile',
    action='store_true',
    help='Enable cProfile profiling and save results to kmc_profile.prof'
  )
  
  parser.add_argument(
    '--allow-multi-rank-profile',
    action='store_true',
    help='Allow profiling with >1 MPI rank (results may be misleading due to synchronization distortion)'

  )
  
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Print resolved configuration and exit without running'
  )
  
  return parser.parse_args()
  
def _enforce_single_rank_profiling(args):
    """
    Enforce single-rank execution when profiling is active.

    cProfile is a single-process profiler. With multiple MPI ranks:
    - Only rank 0 is instrumented; other ranks run unprofiled.
    - Rank 0 is slowed by profiler overhead, causing other ranks to
      accumulate artificial wait time at MPI barriers/collectives.
    - The resulting profile is dominated by MPI synchronization
      artifacts rather than real computational bottlenecks.

    Raises
    ------
    SystemExit
        If world_size > 1 and --allow-multi-rank-profile is not set.
    """
    if not args.profile:
      return

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
    except ImportError:
        # No mpi4py installed ? single-process execution, nothing to check
        return

    if world_size > 1 and not args.allow_multi_rank_profile:
        msg = (
            f"\n{'='*60}\n"
            f"ERROR: --profile active with {world_size} MPI ranks.\n"
            f"\n"
            f"cProfile only instruments rank 0. The other {world_size - 1} rank(s)\n"
            f"run unprofiled, causing:\n"
            f"  - Artificial MPI wait times in the profile\n"
            f"  - Misleading bottleneck identification\n"
            f"  - Wasted compute resources\n"
            f"\n"
            f"Recommendation: rerun with a single core:\n"
            f"  python run_simulation.py {args.sim_id} --profile --config {args.config}\n"
            f"  mpiexec -n 1 python run_simulation.py {args.sim_id} --profile\n"
            f"\n"
            f"To override (not recommended), add --allow-multi-rank-profile.\n"
            f"{'='*60}\n"
        )
        if rank == 0:
            print(msg, file=sys.stderr)
        comm.Barrier()  # Ensure all ranks see the message before abort
        comm.Abort(1)
   

def main(sim_id, config_name='PZT_ZrTi_PbO3_2.yaml'):
        
        params = get_parameters_from_sim_id(sim_id)
        System_state,rng,paths,Results,simulation_parameters,Elec_controller = initialization(sim_id, params, config_name)
        
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
                System_state._poisson_solver = poisson_solver
                
                poisson_solver.set_boundary_conditions(top_value=V_top, bottom_value=0.0)  # Set appropriate BCs
                
                if solve_heat:
                  heat_solver = HeatSolver(
                    System_state.heat_parameters,
                    grid_crystal=System_state.grid_crystal,
                    path_results = paths["results"],
                    mpi_ctx=System_state.mpi_ctx
                  )
                  System_state._heat_solver = heat_solver
            
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
                        System_state._fields_changed = True
                              
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
                        print(str(j)+"/"+str(int(Elec_controller.total_simulation_time/Elec_controller.voltage_update_time)),'| Total time: ',System_state.list_time[-1],'| Voltage: ',V_top, flush=True)
                        print(f'Events at step {j}: {System_state.events_tracking}', flush=True)
                        print(f'Scavenged ions: {System_state.scavenged_ions}', flush=True)
                        if Elec_controller.current_enabled:
                          print(f"Current: {Elec_controller.measurements['current'][-1]}", flush=True)
    
                        end_time = time.time()
                        System_state.plot_crystal(45,45,paths['data'],j)        
                        
    
        if System_state.rank == 0:
          
          total_end_time = time.time()
          print(f"==================================================")
          print(f"SUCCESS: Simulation {sim_id} completed in {total_end_time - total_start_time:.2f} seconds.")
          print(f"==================================================")
          
          # Variables to save
          
          if save_data: 
          
            if hasattr(System_state, '_poisson_solver'):
              del System_state._poisson_solver
            if hasattr(System_state, '_heat_solver'):
              del System_state._heat_solver
              
            variables = {'System_state' : System_state}
            filename = 'variables'
            save_variables(paths['program'],variables,filename)
          
          
        Elec_controller.save_IV_csv(paths['results'])
        Elec_controller.plot_V_I(paths['results'])

    
        return System_state

if __name__ == '__main__':
    import atexit
    
    args = parse_arguments()
    
    sim_id = args.sim_id
    config_name = args.config
    profile_mode = args.profile
    
    if args.dry_run:
      print(f"[DRY RUN] sim_id={sim_id}, config={config_name}, profile={profile_mode}")
      sys.exit(0)
      
    _enforce_single_rank_profiling(args)
    
    if profile_mode:
        import pstats
        import cProfile
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        atexit.register(lambda: profiler.dump_stats('kmc_profile.prof'))
        
        try:
          System_state = main(sim_id)
        finally:
          profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\n" + "="*60)
        print("PROFILING RESULTS (top 15 functions by cumulative time)")
        print("="*60)
        stats.print_stats(15)
        print("Full profile saved to 'kmc_profile.prof'")
    else:
        System_state = main(sim_id, config_name)