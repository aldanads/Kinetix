# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:12:23 2024

@author: samuel.delgado
"""


"""
Profiling - Reference: Grown 0.4 nm in a 2x2nm box

3588847061 function calls (3583118442 primitive calls) in 4531.376 seconds

   Ordered by: internal time
   List reduced from 3094 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 82083948  963.713    0.000 1404.471    0.000 Site.py:208(available_migrations)
 74649433  416.469    0.000 1025.775    0.000 Site.py:89(supported_by)
 82083948  289.458    0.000  426.047    0.000 Site.py:475(transition_rates)
328902104  278.530    0.000  278.578    0.000 Site.py:139(calculate_clustering_energy)
 81337137  215.623    0.000  215.644    0.000 {built-in method numpy.array}
  3820373  192.869    0.000 3213.572    0.001 crystal_lattice.py:659(update_sites)
    12054  188.685    0.016  408.147    0.034 superbasin.py:107(transition_matrix)
514541761  176.782    0.000  176.787    0.000 {built-in method builtins.max}
576148287  149.388    0.000  149.388    0.000 {method 'append' of 'list' objects}
 74649433  148.408    0.000  404.024    0.000 Site.py:379(detect_planes)
  3820122  146.402    0.000  230.872    0.000 crystal_lattice.py:748(remove_specie_site)
599989208/599983845  122.053    0.000  122.056    0.000 {built-in method builtins.len}
    12054  115.540    0.010 3761.155    0.312 superbasin.py:43(trans_absorbing_states)
  3820373  115.057    0.000  164.150    0.000 crystal_lattice.py:322(available_adsorption_sites)
128842061  112.088    0.000  184.653    0.000 {method 'update' of 'set' objects}
"""


import cProfile
import sys
from initialization import initialization,save_variables,search_superbasin
from KMC import KMC
import numpy as np
import time

save_data = True
lammps_file = True

# def main():

for n_sim in range(0,1):
    
    System_state,rng,paths,Results = initialization(n_sim,save_data,lammps_file)
    
    System_state.add_time()
        
    System_state.plot_crystal(45,45,paths['data'],0)    
    j = 0
    
    snapshoots_steps = int(5e1)
    starting_time = time.time()
# =============================================================================
#     Deposition
# 
# =============================================================================
    if System_state.experiment == 'deposition':   

        nothing_happen = 0
        # list_time_step = []
        list_sites_occu = []
        thickness_limit = 10 # (1 nm)
        System_state.measurements_crystal()
        i = 0
        while System_state.thickness < thickness_limit:
            i+=1
      
            System_state,KMC_time_step = KMC(System_state,rng)
            
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
        
            if i%snapshoots_steps== 0:
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
                
                # print('j = ',j)
                # if j == 5:
                #     sys.exit()
                    


# =============================================================================
#     Annealing  
#            
# =============================================================================
    elif System_state.experiment == 'annealing':
        i = 0
        #otal_steps = int(2.5e6)
        total_steps = int(100)
        System_state.measurements_crystal()
        
        while j*snapshoots_steps < total_steps:

            i+=1
            System_state,KMC_time_step = KMC(System_state,rng)
            
            if i%snapshoots_steps== 0:
                System_state.add_time()
                j+=1
                System_state.measurements_crystal()
                print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                end_time = time.time()
                if save_data:
                    Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                                                  System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                                                  System_state.surf_roughness_RMS,end_time-starting_time)
                    
                System_state.plot_crystal(45,45,paths['data'],j)
                
    elif System_state.experiment == 'ECM memristor':
        
        i = 0
        total_steps = int(500)
        # System_state.measurements_crystal()
        while j*snapshoots_steps < total_steps:
            
            i+=1
            System_state,KMC_time_step = KMC(System_state,rng)
            
            if i%snapshoots_steps== 0:
                System_state.add_time()
                j+=1
                # System_state.measurements_crystal()
                print(str(j)+"/"+str(int(total_steps/snapshoots_steps)),'| Total time: ',System_state.list_time[-1])
                end_time = time.time()
                # if save_data:
                    # Results.measurements_crystal(System_state.list_time[-1],System_state.mass_gained,System_state.fraction_sites_occupied,
                    #                               System_state.thickness,np.mean(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),np.std(np.array(System_state.terraces)[np.array(System_state.terraces) > 0]),max(System_state.terraces),
                    #                               System_state.surf_roughness_RMS,end_time-starting_time)
                    
                System_state.plot_crystal(45,45,paths['data'],j)


    System_state.plot_crystal(45,45)
    
    # Variables to save
    variables = {'System_state' : System_state}
    filename = 'variables'
    if save_data: save_variables(paths['program'],variables,filename)

#     return System_state

# if __name__ == '__main__':
#     System_state = main()
# Use cProfile to profile the main function
#     cProfile.run('main()', 'profile_output.prof')    

# import pstats

# # Load and analyze the profiling results
# p = pstats.Stats('profile_output.prof')
# p.strip_dirs().sort_stats('time').print_stats(15)  # Show top 10 time-consuming functions