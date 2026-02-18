# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:43:50 2024

@author: samuel.delgado
"""
from balanced_tree import Node, build_tree, update_data, search_value
import numpy as np

def KMC(System_state,rng):
        
    grid_crystal = System_state.grid_crystal
    superbasin_dict = System_state.superbasin_dict

# =============================================================================
#     TR_catalog store:
#      - TR_catalog[0] = TR
#      - TR_catalog[1] = Arrival site
#      - TR_catalog[2] = Event label - Migration, desorption, etc
#      - TR_catalog[3] = Starting site
# =============================================================================
    # Build TR catalog
    TR_catalog = []
    for idx in System_state.sites_occupied + System_state.adsorption_sites:
        
        if idx not in superbasin_dict:
            TR_catalog.extend([(item[0],item[1],item[2],idx) for item in grid_crystal[idx].site_events])
        else:
            TR_catalog.extend([(item[0],item[1],item[2],idx) for item in superbasin_dict[idx].site_events_absorbing])

    if not TR_catalog:
      # No events possible
      timestep_limit = System_state.get_timestep_limit()
      System_state.track_time(timestep_limit)
      return System_state, timestep_limit, None
      

    # Build a balanced tree structure
    TR_tree = build_tree(TR_catalog)
    # Each node is the sum of their children, starting from the leaf
    sumTR = update_data(TR_tree)

    if sumTR == None or sumTR == 0: 
      timestep_limit = System_state.get_timestep_limit()
      System_state.track_time(timestep_limit)
      return System_state,timestep_limit, None # Exit if there is not possible event
    
    # When we only have one node in the tree, it returns a tuple
    if type(sumTR) is tuple: 
      sumTR = sumTR[0]
    
    #Calculate the time step
    time_step = -np.log(rng.random())/sumTR
    
    # Calculate maximum allowed timestep
    timestep_limit = System_state.get_timestep_limit()
    print(f'Time step limits: {timestep_limit}, time step: {time_step}, Time: {System_state.time}')
    
    if time_step <= timestep_limit:
        # We search in our binary tree the event that happen
        chosen_event = search_value(TR_tree,sumTR*rng.random())
        System_state.processes(chosen_event)
        System_state.update_superbasin(chosen_event)
        System_state.track_time(time_step)  
        return System_state,time_step,chosen_event
    else:
        # No event occurs within timestep_limits
        System_state.track_time(timestep_limit)
        return System_state,timestep_limit, None 
        
        
    

    