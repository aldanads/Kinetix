# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado

Command-line interface for Kinetix, moved verbatim from run_simulation.py in
the repository root. The argparse interface and dispatch behavior are
unchanged; main() additionally serves as the console-script entry point
(`kinetix` / `python -m kinetix`) when called without arguments.
"""

import sys
import argparse
from kinetix.initialization import initialization,save_variables
import numpy as np
import time
import platform
import logging

from kinetix.logging_config import setup_logging

logger = logging.getLogger(__name__)

def get_parameters_from_sim_id(sim_id: int) -> dict:
   """Map SIM_ID to simulation parameters."""
   # Define parameters
   v0_initial_concentrations = [1.0e-2, 2.0e-2, 3.0e-2,  4.0e-2,  5.0e-2]
   temperatures = [293.0, 310.0, 373.0, 473.0, 573.0]
   h_generation = [0.45,0.48,0.50, 0.52, 0.55]
   
   idx = sim_id
   
   i_vo = idx % 5
   i_temp = (idx // 5) % 5
   i_gen_h = (idx // 25) % 5
   
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
  python -m kinetix 42
  python -m kinetix 42 --config PZT_ZrTi(PbO3)2.yaml
  python -m kinetix 42 --config VCM_HfO2_cylindrical_gb.yaml --profile
  python -m kinetix --config PZT_ZrTi(PbO3)2_annealing.yaml     
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
    default='PZT_ZrTi_PbO3_2.yaml',
    help='Preset configuration file name or path (default: PZT_ZrTi_PbO3_2.yaml)'
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
            f"  python -m kinetix {args.sim_id} --profile --config {args.config}\n"
            f"  mpiexec -n 1 python -m kinetix {args.sim_id} --profile\n"
            f"\n"
            f"To override (not recommended), add --allow-multi-rank-profile.\n"
            f"{'='*60}\n"
        )
        if rank == 0:
            print(msg, file=sys.stderr)
        comm.Barrier()  # Ensure all ranks see the message before abort
        comm.Abort(1)
   

def main(sim_id=None, config_name='PZT_ZrTi_PbO3_2.yaml'):
        # ------------------------------------------------------------------
        # CLI dispatch: when called without an explicit sim_id (console
        # script `kinetix` or `python -m kinetix`), parse the command line
        # exactly as run_simulation.py did in its __main__ block.
        # ------------------------------------------------------------------
        if sim_id is None:
            import atexit

            args = parse_arguments()

            sim_id = args.sim_id
            config_name = args.config
            profile_mode = args.profile

            if args.dry_run:
                logger.info("[DRY RUN] sim_id=%s, config=%s, profile=%s", sim_id, config_name, profile_mode)
                sys.exit(0)

            _enforce_single_rank_profiling(args)

            if profile_mode:
                import pstats
                import cProfile

                profiler = cProfile.Profile()
                profiler.enable()

                atexit.register(lambda: profiler.dump_stats('kmc_profile.prof'))

                try:
                    simulator = main(sim_id)
                finally:
                    profiler.disable()

                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                logger.info("PROFILING RESULTS (top 15 functions by cumulative time)")
                stats.print_stats(15)
                logger.info("Full profile saved to 'kmc_profile.prof'")
            else:
                simulator = main(sim_id, config_name)

            # Entry-point wrappers invoke this function as sys.exit(main());
            # a returned KMCSimulator object would be read as a truthy
            # non-int status, so end CLI mode explicitly with code 0.
            sys.exit(0)
        
        # Configure logging with defaults before config is loaded
        setup_logging()
        params = get_parameters_from_sim_id(sim_id)
        simulator,rng,paths,Results,simulation_parameters,Elec_controller = initialization(sim_id, params, config_name)
        
        if simulator.rank == 0:
          logger.info('System size: %s', simulator.crystal_size)
          total_start_time = time.time()
          simulator.plot_crystal(45,45,paths['data'],0)    
          
        simulator.add_time()
            
        
        j = 0
        snapshots_steps = simulation_parameters['snapshoots_steps']
        total_steps = simulation_parameters['total_steps']
        save_data = simulation_parameters['save_data']
        
        starting_time = time.time()

    # =============================================================================
    #     Deposition
    # 
    # =============================================================================
        if simulator.simulation_type == 'deposition':   
    
            nothing_happen = 0
            # list_time_step = []
            list_sites_occu = []
            thickness_limit = 10 # (1 nm)
            simulator.measurements_crystal()
            i = 0
            while simulator.thickness < thickness_limit:
                i+=1
          
                simulator,KMC_time_step, _ = KMC(simulator,rng)
                                
                list_sites_occu.append(len(simulator.sites_occupied))
                
                if np.mean(list_sites_occu[-simulator.n_search_superbasin:]) == len(simulator.sites_occupied):
                # if np.mean(list_time_step[-simulator.n_search_superbasin:]) <= simulator.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if simulator.E_min - simulator.energy_step > 0:
                        simulator.E_min -= simulator.energy_step
                    else:
                        simulator.E_min = 0
                
                if simulator.n_search_superbasin == nothing_happen:
                    search_superbasin(simulator)
                elif nothing_happen> 0 and nothing_happen % simulator.n_search_superbasin == 0:
                    if simulator.E_min_lim_superbasin >= simulator.E_min + simulator.energy_step:
                        simulator.E_min += simulator.energy_step
                    else:
                        simulator.E_min = simulator.E_min_lim_superbasin
                    search_superbasin(simulator)
                    
    
                    
                # print('Superbasin E_min: ',simulator.E_min)
            
                if i%snapshots_steps== 0:
                    simulator.add_time()
                    
                    j+=1
                    simulator.measurements_crystal()
                    logger.info('%s %% | Thickness: %s | Total time: %s', str(simulator.thickness/thickness_limit * 100), simulator.thickness, simulator.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(simulator.list_time[-1],simulator.mass_gained,simulator.fraction_sites_occupied,
                                                      simulator.thickness,np.mean(np.array(simulator.terraces)[np.array(simulator.terraces) > 0]),np.std(np.array(simulator.terraces)[np.array(simulator.terraces) > 0]),max(simulator.terraces),
                                                      simulator.surf_roughness_RMS,end_time-starting_time)
        
                    simulator.plot_crystal(45,45,paths['data'],j)
                    
    
    # =============================================================================
    #     Annealing  
    #            
    # =============================================================================
        elif simulator.simulation_type == 'annealing':
            i = 0
            
            nothing_happen = 0

            simulator.measurements_crystal()
            list_time_step = []
    
            while j*snapshots_steps < total_steps:
    
                i+=1
                simulator,KMC_time_step, _ = KMC(simulator,rng)
                list_time_step.append(KMC_time_step)
                
    # =============================================================================
    #                 Search of superbasin
    # =============================================================================
                if np.mean(list_time_step[-simulator.n_search_superbasin:]) <= simulator.time_step_limits:
                # if np.mean(list_time_step[-4:]) <= simulator.time_step_limits:
                    nothing_happen +=1    
                else:
                    nothing_happen = 0
                    if simulator.E_min - simulator.energy_step > 0:
                        simulator.E_min -= simulator.energy_step
                    else:
                        simulator.E_min = 0
                        
                if simulator.n_search_superbasin == nothing_happen:
                    search_superbasin(simulator)
                elif nothing_happen > 0 and nothing_happen % simulator.n_search_superbasin == 0:
                    if simulator.E_min_lim_superbasin >= simulator.E_min + simulator.energy_step:
                        simulator.E_min += simulator.energy_step
                    else:
                        simulator.E_min = simulator.E_min_lim_superbasin
                    search_superbasin(simulator)
                    
    # =============================================================================
    #                     Finish search superbasin
    # =============================================================================
                
                if i%snapshots_steps== 0:
                    
                    simulator.sites_occupied = list(set(simulator.sites_occupied))
                                        
                    simulator.add_time()
                    j+=1
                    simulator.measurements_crystal()
                    logger.info('%s/%s | Total time: %s', str(j), str(int(total_steps/snapshots_steps)), simulator.list_time[-1])
                    end_time = time.time()
                    if save_data:
                        Results.measurements_crystal(simulator.list_time[-1],simulator.mass_gained,simulator.fraction_sites_occupied,
                                                      simulator.thickness,np.mean(np.array(simulator.terraces)[np.array(simulator.terraces) > 0]),np.std(np.array(simulator.terraces)[np.array(simulator.terraces) > 0]),max(simulator.terraces),
                                                      simulator.surf_roughness_RMS,end_time-starting_time)
                        
                    simulator.plot_crystal(45,45,paths['data'],j)
                    
    # =============================================================================
    #     Devices: PZT, memristors  
    #            
    # =============================================================================
                    
        elif simulator.simulation_type == 'electronic_device':
            
            from collections import Counter
            solve_Poisson = (simulator.poisson_config is not None
                             and simulator.poisson_config.solve_Poisson)
            save_Poisson = (simulator.poisson_config is not None
                            and simulator.poisson_config.save_Poisson)

            solve_heat = (simulator.heat_config is not None
                          and simulator.heat_config.solve_heat)
            save_heat = (simulator.heat_config is not None
                         and simulator.heat_config.save_heat)
            
            V_top = Elec_controller.apply_voltage(simulator.time)
            simulator.save_electric_bias(V_top)
            
            # Dolfinx only works in Linux
            if solve_Poisson and platform.system() == 'Linux':
                from kinetix.solvers.poisson import PoissonSolver
                from kinetix.solvers.heat import HeatSolver
                from mpi4py import MPI
                
                # Initialize Poisson solver on all MPI ranks
                poisson_solver = PoissonSolver(
                  mesh_file=simulator.solver_mesh_file,
                  mesh_config=simulator.mesh_config,
                  poisson_config=simulator.poisson_config,
                  material_config=simulator.material_config,
                  defects_config=simulator.defects_config,
                  grid_crystal=simulator.grid_crystal,
                  path_results = paths["results"],
                  mpi_ctx = simulator.mpi_ctx
                )
                simulator._poisson_solver = poisson_solver
                
                poisson_solver.set_boundary_conditions(top_value=V_top, bottom_value=0.0)  # Set appropriate BCs
                
                if solve_heat:
                  heat_solver = HeatSolver(
                    mesh_file=simulator.solver_mesh_file,
                    mesh_config=simulator.mesh_config,
                    heat_config=simulator.heat_config,
                    ambient_temperature=simulator.temperature,
                    characteristic_length=simulator.characteristic_length,
                    defects_config=simulator.defects_config,
                    grid_crystal=simulator.grid_crystal,
                    path_results = paths["results"],
                    mpi_ctx=simulator.mpi_ctx
                  )
                  simulator._heat_solver = heat_solver
            
                  heat_solver.set_boundary_conditions(
                    top_value=heat_solver.T_ambient,
                    bottom_value=heat_solver.T_ambient
                  )
                  
            
            
            while simulator.should_continue_simulation(Elec_controller.total_simulation_time):
            
                     
                if solve_Poisson and platform.system() == 'Linux': 
                  should_solve_fields_now, snapshots = simulator.should_solve_fields_now(Elec_controller)
                       
                  particle_locations, charges, evaluation_points = simulator.get_evaluation_points()
                    
                  if should_solve_fields_now:
                        # Every time we change the applied voltage, we should calculate Poisson
                        V_top = Elec_controller.apply_voltage(simulator.time)
                        simulator.save_electric_bias(V_top)
                        clusters = simulator.prepare_clusters_for_bcs()
                        # We need the cluster to know what is the effective gap for calculating the Schottky emission
                        V_eff, _ = Elec_controller.calculate_current(clusters) # Obtain effective voltage after voltage drop of series resistance
                          
                        poisson_solver.set_boundary_conditions(top_value=V_eff, bottom_value=0.0,clusters = clusters)
                        
                        
                        run_start_time = MPI.Wtime()
                        uh = poisson_solver.solve(particle_locations,charges) 
                        run_time = MPI.Wtime() - run_start_time
                        
                        if simulator.rank == 0: logger.info('Run time to solve Poisson: %s', run_time)

                        if save_Poisson:
                          poisson_solver.save_potential(simulator.time,j+1)
                          
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
                         if simulator.rank == 0: 
                           logger.info('Run time to solve Heat: %s', heat_run_time)
                           logger.info('Avg temperature: %.10f K', Avg_T)
                         
                         # Save temperature
                         if save_heat:
                           heat_solver.save_temperature(simulator.time, j+1)
                           
                        run_time = 0     
                        simulator._fields_changed = True
                              
                simulator.step_kmc(rng)
                
                if snapshots:
                
                    j+=1
                    # Continue with serial processing on rank 0
                    if simulator.rank == 0:
                        simulator.add_time()
    
                        # simulator.measurements_crystal()
                        logger.info('%s/%s | Total time: %s | Voltage: %s', str(j), str(int(Elec_controller.total_simulation_time/Elec_controller.voltage_update_time)), simulator.list_time[-1], V_top)
                        logger.info('Events at step %s: %s', j, simulator.events_tracking)
                        logger.info('Scavenged ions: %s', simulator.scavenged_ions)
                        if Elec_controller.current_enabled:
                          logger.info("Current: %s", Elec_controller.measurements['current'][-1])
    
                        end_time = time.time()
                        simulator.plot_crystal(45,45,paths['data'],j)        
                        
    
        if simulator.rank == 0:
          
          total_end_time = time.time()
          logger.info("SUCCESS: Simulation %s completed in %.2f seconds.", sim_id, total_end_time - total_start_time)
          
          # Variables to save
          
          if save_data: 
          
            if hasattr(simulator, '_poisson_solver'):
              del simulator._poisson_solver
            if hasattr(simulator, '_heat_solver'):
              del simulator._heat_solver
              
            variables = {'simulator' : simulator}
            filename = 'variables'
            save_variables(paths['program'],variables,filename)
          
          
        Elec_controller.save_IV_csv(paths['results'])
        Elec_controller.plot_V_I(paths['results'])

    
        return simulator


if __name__ == '__main__':
    main()
