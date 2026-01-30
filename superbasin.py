# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:31:56 2024

@author: samuel.delgado
"""
import numpy as np
from scipy import constants


# =============================================================================
# Fichthorn, Kristen A., and Yangzheng Lin. 
# "A local superbasin kinetic Monte Carlo method." 
# The Journal of chemical physics 138, no. 16 (2013).

# Novotny, M. A. 
# "A tutorial on advanced dynamic Monte Carlo methods for systems with discrete state spaces." 
# Annual Reviews Of Computational PhysicsIX (2001): 153-210.
# =============================================================================

class Superbasin():
    
    def __init__(self,idx, System_state,E_min,sites_occupied):
        
        original_allow_specie_removal = System_state.allow_specie_removal 
        System_state.allow_specie_removal = False  # disable during virtual moves
        
        try:
          self.particle_idx = idx
          self.E_min = E_min
          self.epsilon_min_decrement = 0.1  # Step to decrease E_min per retr       
          self.retry_limit = max(round(E_min / self.epsilon_min_decrement),2)  # Maximum retry attempts
  
          # Core workflow
          self.trans_absorbing_states(idx,System_state,sites_occupied)
  
          if not self.absorbing_states or not self.transient_states:
              self.valid = False  # Mark the object as invalid
              return
          
          self.valid = True  # Mark the object as valid if absorbing states exist
          print('Absorbing states exist')
          
          self.transition_matrix()
          self.markov_matrix()
          
          if not self.absorption_probability_matrix():
              self.valid = False  # Mark as invalid if poor conditioning is detected
              return
              
          print('No poor conditioning detected')
          
          self.calculate_transition_rates_absorbing_states(System_state.num_event)
          self.calculate_superbasin_environment(System_state.grid_crystal)
  
        finally:      
          System_state.allow_specie_removal = original_allow_specie_removal

   
    def trans_absorbing_states(self,start_idx,System_state,sites_occupied):
        
        stack = [start_idx]
        visited = set()

        grid_crystal = System_state.grid_crystal
        self.absorbing_states = []
        self.transient_states_transitions = []
        self.absorbing_states_transitions = []
        aux_transitions = []
        last_transition = None
        
        while stack:
            idx = stack.pop()

            if idx not in visited:
                site = grid_crystal[idx]
                is_absorbing = True # Assume that it is an absorbing state
                site_has_migrations = False
                
                for transition in site.site_events:
                    
                    if not isinstance(transition[2],int):
                      continue # Skip redox, generation, etc. Keep only migrations
                
                    site_has_migrations = True
                    last_transition = transition # update last known transition
                    transition_with_idx = transition + [idx]
                    
                    if transition[3] <= self.E_min:
                        is_absorbing = False # If it is a easy way out, it is a transient state (< E_min)
                        if transition_with_idx not in self.transient_states_transitions:
                            self.transient_states_transitions.append(transition_with_idx)
                    else:
                        aux_transitions.append(transition_with_idx)
                    
                if is_absorbing and site_has_migrations:
                    self.absorbing_states.append(idx)

                elif not is_absorbing:
                    # Explore neighbors via migration
                    for transition in site.site_events:
                        if not isinstance(transition[2], int):
                          continue
                        dest = transition[1]
                        # Visit all the transitions from a transient state, even those
                        # with larger Act. Energy than E_min
                        if dest not in visited and dest not in stack:
                            stack.append(dest)
                
                # Control of the states visited
                visited.add(idx)
                                
            
            # Prepare system state for evaluating the next site in the stack
            if stack and last_transition is not None:
                next_site = stack[-1]
                # Perform a virtual move of the ion to `next_site` so that
                # activation energies for its outgoing transitions are up to date.
                System_state.processes((last_transition[0], next_site, last_transition[2], idx))
              
        # Return to the original state
        if last_transition is not None and isinstance(last_transition[2],int):            
            System_state.processes((last_transition[0], start_idx, last_transition[2], idx)) 
            

        # Construct the transitions to the absorbing states
        for absorbing_state in self.absorbing_states: 
            i = 0
            while i < len(aux_transitions):
                transition = aux_transitions[i]
                if transition[1] == absorbing_state:
                    self.absorbing_states_transitions.append(transition)
                    aux_transitions.pop(i)  # Remove the processed transition
                else:
                    i += 1
                
                
        self.transient_states = list({transition[-1] for transition in self.transient_states_transitions})
        self.superbasin_idx = self.absorbing_states + self.transient_states 
        
        # Check that grid_crystal is in the original state
        self.verify_grid_crystal(System_state,sites_occupied) 
   
    def transition_matrix(self):
        
        transitions = self.absorbing_states_transitions + self.transient_states_transitions
        
        # Create a dictionary for quick lookup
        transition_dict = {(transition[1], transition[-1]): transition[0] 
                           for transition in transitions}
        # Initialize the transition matrix with zeros
        n = len(self.superbasin_idx)
        A_transitions = np.zeros((n, n))
        
        state_origins = self.superbasin_idx
        state_destinations = self.superbasin_idx
        
        # Fill the diagonal elements
        for i, state_origin in enumerate(state_origins):
            
            if state_origin in self.absorbing_states:
                A_transitions[i, i] = 0
            else:
                tau = sum(transition[0] for transition in transitions if 
                          state_origin == transition[-1])
                A_transitions[i, i] = tau
                
        # Fill the off-diagonal elements
        for i, state_origin in enumerate(state_origins):
            for j, state_destination in enumerate(state_destinations):
                if i != j:
                    rate = -transition_dict.get((state_origin, state_destination), 0)
                    A_transitions[i, j] = max(rate, 0)
                    
        self.A_transitions = A_transitions
        
    def transition_matrix_2(self):
        self.A_transitions_2 = []
        transitions = self.absorbing_states_transitions + self.transient_states_transitions

        # Create a dictionary for quick lookup
        transition_dict = {}
        for transition in transitions:
            key = (transition[1], transition[-1])
            transition_dict[key] = transition[0]

        for i,state_origin in enumerate(self.superbasin_idx):
            row = [] # Each row include the transition from each state_origin
            for j,state_destination in enumerate(self.superbasin_idx):
                
                # Aii = 0 for the absorbing border states
                
                if i == j:
                    if state_origin in self.absorbing_states:
                        row.append(0)
                    else:
                        # For the diagonal in transient states: sum runs over all k superbasin states
                        tau = sum(transition[0] for transition in transitions if
                                  state_origin == transition [-1])
                        row.append(tau)
                        
                else:
                    
                    rate = -transition_dict.get((state_origin, state_destination), 0)
                    rate = rate if rate > 0 else 0
                    row.append(rate)
                    
            self.A_transitions_2.append(row)
        self.A_transitions_2 = np.array(self.A_transitions_2)
    
     
    def markov_matrix(self):
        
        self.M_Markov = []
        transitions = self.absorbing_states_transitions + self.transient_states_transitions

        for i,state_origin in enumerate(self.superbasin_idx):
            row = [0] * len(self.superbasin_idx)  # Initialize row with zeros
            
            if state_origin in self.absorbing_states:
                row[i] = 1 # Absorbing states have a self-transition probability of 1
                
            else:
                for transition in transitions:
                    origin, destination, rate = transition[-1], transition[1], transition[0]
                    if state_origin == origin:
                        j = self.superbasin_idx.index(destination)
                        row[j] = rate / self.A_transitions[i][i]                  
                    
            self.M_Markov.append(row)
        self.M_Markov = np.array(self.M_Markov)
      
         
    def absorption_probability_matrix(self):
        
        self.B_absorption = []
        
        n_transient = len(self.transient_states)
        n_absorbing = len(self.absorbing_states)
        

        T_transient = self.M_Markov[n_absorbing:,n_absorbing:]# Transient matrix
        R_recurrent = self.M_Markov[n_absorbing:,:n_absorbing] # Recurrent matrix
        
        # Calculate the fundamental matrix N
        I = np.eye(n_transient)
        I_reg = I - T_transient
        
        # Check for poor conditioning 
        cond_number = np.linalg.cond(I_reg)
        #print('Conditioning number: ', cond_number)
        if cond_number > 1e10: # In this case, the matrix is almost singular --> Large errors
            #print(f"Matrix is poorly conditioned (cond = {cond_number}). Adjust regularization or inspect T_transient.")
            return False
        
        # Check for NaN or infinity values
        if np.any(np.isnan(I_reg)) or np.any(np.isinf(I_reg)):
            #raise ValueError("I_reg = I - T_transient contains NaN or infinity values.")
            return False
        
            # Regularize the matrix
        epsilon = 1e-10
        I_reg += epsilon * np.eye(I_reg.shape[0])
        # Calculate the pseudo-inverse with error handling
        try:
            self.N = np.linalg.pinv(I - T_transient) # Pseudo-inverse
        except np.linalg.LinAlgError:
            print("SVD did not converge")
            return False
        
        # Calculate the absorption probabilities matrix B
        self.B_absorption = np.dot(self.N, R_recurrent) 
       
        return True
        
      
    def calculate_transition_rates_absorbing_states(self,num_event,T=300):
        
        # Mean time to absorbing state j from the transient state i
        # (first passage time - FPT)
        FPT = np.sum(self.N, axis=1)
        # Calculate transition_rates
        
        # Avoid division by zero or very small values
        epsilon = 1e-10  # Small value to avoid division by zero
        FPT[FPT < epsilon] = epsilon
        transition_rates = self.B_absorption / FPT[:, None]
        
        # We are only interested in the absorbing states
        # Not necessary the probability from each transient state to each absorbing state
        sum_transition_rates = np.sum(transition_rates,axis=0)
        self.transition_rates = np.where(sum_transition_rates < 0, 0, sum_transition_rates)
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        nu0=7E12;  # nu0 (s^-1) bond vibration frequency
        # The np.where() is included to handle transition_rates = 0
        with np.errstate(divide='ignore'):
            self.EAct = np.where(self.transition_rates > 0, -kb * T * np.log(self.transition_rates / nu0), 
                                 np.inf)
                
        self.site_events_absorbing = [
            (transition_r, absorbing_state, num_event - 2, EAct, self.particle_idx)
            for transition_r, absorbing_state, EAct 
            in zip(self.transition_rates, self.absorbing_states, self.EAct)
            ]
        
  
    def calculate_superbasin_environment(self,grid_crystal):
        
        self.superbasin_environment = set()
        
        for idx in self.superbasin_idx:
            
            self.superbasin_environment = self.superbasin_environment.union(grid_crystal[idx].supp_by)
            
        self.superbasin_environment = set(list(self.superbasin_environment) + self.superbasin_idx)
        
        self.superbasin_environment.discard('bottom_layer') 
        self.superbasin_environment.discard('top_layer') 

    # Verify that we leave grid_crystal in the original state
    def verify_grid_crystal(self,System_state,sites_occupied):       
        
        for site in sites_occupied:
            # It should be occupied, but it is not
            if System_state.grid_crystal[site].chemical_specie != System_state.chemical_specie:
                # Select deposition event
                # event = System_state.grid_crystal[site].site_events[0]
                # Remove the site from sites_occupied
                # System_state.sites_occupied.remove(event[1])
                System_state.sites_occupied.remove(site)
                # Introduce the particle
                # System_state.processes((event[0], event[1], event[2], event[1])) 
                System_state.processes((0, site, System_state.num_event-1, site)) 

        
      