# -*- coding: utf-8 -*-
"""GrainBoundary class for GB physics."""
import numpy as np

class GrainBoundary:
    def __init__(self,domain_size, gb_configurations: list[dict] = None):
        
      """
      Grain boundaries for memristive filament formation. Supporing:
      - Vertical planar boundaries (between 2 grains)
      - Cylindrical boundaries (triple junctions - 3+ grains)
          
      Parameters:
      -----------
      domain_size : list [lx, ly, lz] - Domain dimensions in Angstroms
      gb_configurations : list of dicts 
        List of grain boundary configurations
      """
      
      self.domain_size = np.array(domain_size)
      self.gb_configurations = gb_configurations or self._create_default_configurations()
      
      # Pre-process configurations for fast lookup
      self._process_configurations()
      
    def _create_default_configurations(self) -> list[dict]:
      """
      Create default grain boundary configurations
      """
      configs = []
      
      # Vertical planar boundaries (between 2 grains)
      configs.extend([
        {
          'type':'vertical_planar',
          'orientation':'yz', # YZ plane
          'position':self.domain_size[0] * 0.2, # Position in x
          'width':2.0, # GB width in Angstroms
          'Act_E_diff_GB': 1 # Difference of Act Energy outside GB
        },
        {
          'type':'vertical_planar',
          'orientation':'xz',
          'position':self.domain_size[1] * 0.8, # Position in y
          'width':2.0,
          'Act_E_diff_GB': 1
        }
      ])
      
      # Cylindrical boundaries
      configs.extend([
        {
          'type':'cylindrical',
          'center': [self.domain_size[0] * 0.5, self.domain_size[1] * 0.5],
          'radius': 2.0,
          'outer_radius': 10.0,
          'Act_E_diff_GB': 1
        }
      ])
      
      # Triple Junction Planes (3 planes intersecting, between 3 grains)
      configs.extend([
        {
          'type': 'triple_junction_planes',
          'center': [self.domain_size[0] * 0.5, self.domain_size[1] * 0.5],
          'width' : 2.0, # Width of each planar GB region
          'Act_E_diff_GB': 1 
        }
      ])
      
      return configs
      
    def _process_configurations(self):
        """
        Pre-process configurations
        """
        self.vertical_gbs = []
        self.cylindrical_gbs = []
        self.triple_junction_gbs = []
        
        for config in self.gb_configurations:
          # Determine the distance metric for this GB type
          if config['type'] == 'vertical_planar':
            inner_boundary = config.get('width',0) / 2
            outer_boundary = config.get('outer_width',config['width']) / 2
            distance_function = self._distance_to_planar_gb
          elif config['type'] == 'cylindrical':
            inner_boundary = config.get('radius',0)
            outer_boundary = config.get('outer_radius',config['radius'])
            distance_function = self._distance_to_cylindrical_gb  
          elif config['type'] == 'triple_junction_planes':
            inner_boundary = config.get('width',0) / 2
            outer_boundary = config.get('outer_width',config['width']) / 2
            # NEED TO WRITE THE FUNCTION: self._distance_to_triple_junction_gb
            #distance_function = self._distance_to_triple_junction_gb
          else:
            continue
            
          config['inner_boundary'] = inner_boundary
          config['outer_boundary'] = outer_boundary
          config['distance_function'] = distance_function
          
          # Calculate event-specific interpolation parameters
          event_modifications = config.get('event_modifications', {})
          
          for event_type, event_config in event_modifications.items():  
            act_e_diff = event_config['Act_E_diff_GB']
            
            # Event-specific boundaries
            event_inner = event_config.get('inner_boundary', inner_boundary)
            event_outer = event_config.get('outer_boundary', outer_boundary)
          
            # Linear function: Act_E = slope * distance + intercept
            # At radius: Act_E = act_e_diff (inside GB)
            # At outer_radius: Act_E = 0 (outside GB)
            if event_outer > event_inner:
              slope = -act_e_diff / (event_outer - event_inner)
              intercept = act_e_diff - slope * event_inner
            else:
              slope = 0
              intercept = act_e_diff
            
              
            event_config['linear_slope'] = slope
            event_config['linear_intercept'] = intercept
            event_config['inner_boundary'] = inner_boundary
            event_config['outer_boundary'] = outer_boundary
            
          if config['type'] == 'vertical_planar':
            self.vertical_gbs.append(config)
          elif config['type'] == 'cylindrical':  
            self.cylindrical_gbs.append(config)
          elif config['type'] == 'triple_junction_planes':
            self.triple_junction_gbs.append(config)
            
    def _distance_to_planar_gb(self,site_pos,gb_config):
      """Calculate distance to planar GB core"""
      x, y, z = site_pos
      if gb_config['orientation'] == 'yz':
        return abs(x - gb_config['position'])
      elif gb_config['orientation'] == 'xz':
        return abs(y - gb_config['position'])
      elif gb_config['orientation'] == 'xy':
        return abs(z - gb_config['position'])
        
    def _distance_to_cylindrical_gb(self,site_pos,gb_config):
      """Calculate distance to cylindrical GB axis"""
      x, y, z = site_pos
      cx, cy = gb_config['center']
      return np.sqrt((x - cx)**2 + (y - cy)**2)
            
    def is_site_in_grain_boundary(self, site_position: tuple) -> bool:
      """
      Check if site is in any grain boundary (inner or outer region)
      Wrapper method for backward compatibility
      """
      return self.get_site_gb_region(site_position) != 'bulk'
      
    def get_site_gb_region(self,site_position: tuple) -> str:
      """
      Determine which GB region a site belongs to.
      
      Returns:
        str: 'inner_boundary', 'outer_boundary', or 'bulk'
      """
      x, y, z = np.array(site_position)
      
      # Check vertical planar boundaries
      for gb in self.vertical_gbs:
        if gb['orientation'] == 'yz':
          # Vertical YZ place at specific x-position
          distance = abs(x - gb['position'])
        elif gb['orientation'] == 'xz':
          # Vertical XZ place at specific y-position
          distance = abs(y - gb['position'])
        elif gb['orientation'] == 'xy':
          # Horizontal XY place at specific z-position
          distance = abs(z - gb['position'])
        else:
          continue
        
        inner_boundary = gb.get('inner_boundary', gb.get('width', 0))
        outer_boundary = gb.get('outer_boundary', gb.get('outer_width', 0))
        
        if distance <= inner_boundary:
          return 'inner_boundary'
        elif distance <= outer_boundary:
          return 'outer_boundary'
          
      # Check cylindrical boundaries
      for gb in self.cylindrical_gbs:
        cx, cy = gb['center']
        # Distance from cylinder axis
        distance_from_axis = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        inner_radius = gb.get('inner_boundary', gb.get('radius', 0))
        outer_radius = gb.get('outer_boundary', gb.get('outer_radius', 0))
        
        
        #if distance_from_axis <= radius: return True
        if distance_from_axis <= inner_radius:
          return 'inner_boundary'
        elif distance_from_axis <= outer_radius:
          return 'outer_boundary'
          
      return 'bulk'
      
    
    def get_activation_energy_GB(self, site_position:tuple, event_type='migration') -> float:
      """
      Get activation energy with unified linear interpolation
      
      Parameters:
      -----------
      site_position : tuple (x, y, z)
        Site coordinates in Angstroms
            
      Returns:
      --------
      float : Activation energy at site
      """
      x, y, z = np.array(site_position)
      
      for gb_list in [self.vertical_gbs, self.cylindrical_gbs, self.triple_junction_gbs]:
        for gb in gb_list:
          # Get event-specific config
          event_mods = gb.get('event_modifications',{})
          event_config = event_mods.get(event_type)
          
          if event_config is None:
            continue
          
          # Get event-specific interpolation parameters
          slope = event_config.get('linear_slope', 0)
          intercept = event_config.get('linear_intercept', 0)
          inner_boundary = event_config.get('inner_boundary', 0)
          outer_boundary = event_config.get('outer_boundary', 0)
          
          distance_function = gb.get('distance_function')
          if distance_function:
            distance = distance_function(site_position,gb)
          else:
            continue
            
          # Apply interpolation  
          if distance <= inner_boundary:
            # Inside GB core
            return event_config.get('Act_E_diff_GB', 0.0)
          elif distance <= outer_boundary:
            # In transition region
            energy = slope * distance + intercept
            return max(energy,0)
          # Else: continue to next GB (not in this GB's region)
            
      return 0.0 # Not in any GB
      
    def _region_matches(self, site_region, required_region):
      """Check if site region satisfies the requirement."""
      if required_region == 'inner_boundary':
        return site_region == 'inner_boundary'
      elif required_region == 'outer_boundary':
        return site_region in ['inner_boundary', 'outer_boundary']
      return False
       
    
    def modify_act_energy_GB(self,site,migration_pathways,defects_config,reactions_config):
      """
      Modify activation energies for defects that can migrate through this site type.
      Uses 'allowed_sublattices' from defects_config.
      """
      
      # Find defect that matches this site's type
      applicable_defects = site.applicable_defects
        
      if not applicable_defects: 
        return # No defects can occupy this site type
        
      # GB configuration
      gb_config = self.gb_configurations[0]
      event_modifications = gb_config.get('event_modifications',{})
      
      if not event_modifications:
        return
      
      site_pos = site.position
      # Check if site is in GB
      site_gb_region = self.get_site_gb_region(site_pos)
      
      # Iterate through all applicable defects
      for defect_name in applicable_defects:
        base_energies = site.Act_E_dict[defect_name]
      
        # 1. Handle generation
        if "generation" in event_modifications:
          gen_config = event_modifications['generation']
          required_region = gen_config.get('region', 'outer_boundary')
          
          if self._region_matches(site_gb_region, required_region):
              
            if defect_name in gen_config.get('affected_defects', []):
              if "E_gen_defect" in base_energies:
                gb_reduction = self.get_activation_energy_GB(site_pos,"generation")
                base_energies["E_gen_defect"] -= gb_reduction
                
          
        # 2. Handle migration    
        if "migration" in event_modifications:
          mig_config = event_modifications['migration']
          required_region = mig_config.get('region', 'outer_boundary')
          
          if defect_name in mig_config.get('affected_defects', []):
            Act_E_mig = {}
                
            for key,migration_vector in migration_pathways.items():
              # Calculate destination position
              dest_pos = np.array(site_pos) + migration_vector['direction'] * migration_vector['distance']
              z_component = migration_vector['direction'][2]
              
              # Migration in plane
              if np.isclose(z_component, 0.0, atol=1e-9):
                base_energy = base_energies['E_mig_plane'] 
              # Migration upward
              elif z_component > 0:
                base_energy = base_energies['E_mig_upward']
              # Migration downward
              else:
                base_energy = base_energies['E_mig_downward']
                
              dest_gb_region = self.get_site_gb_region(dest_pos)
              
              # Modify only if destination is in GB
              if self._region_matches(dest_gb_region, required_region):
                # Lower barrier to enter GB (particles prefer GB)
                gb_reduction = self.get_activation_energy_GB(dest_pos,'migration')
                modified_energy = base_energy - gb_reduction
              else:
                # No modification for bulk destinations (base case)
                modified_energy = base_energy
              
              Act_E_mig[key] = modified_energy
                
            site.Act_E_dict[defect_name]['E_mig'] = Act_E_mig
          
        # 3. Handle reactions
        if "reaction" in event_modifications:
          rxn_config = event_modifications['reaction']
          required_region = rxn_config.get('region', 'inner_boundary')
          
          if self._region_matches(site_gb_region, required_region):
               
            for reaction_name, reaction in reactions_config.items():
              if reaction_name in rxn_config.get('affected_reactions',[]):
                if reaction['name'] in base_energies:
                    gb_reduction = self.get_activation_energy_GB(site_pos, 'reaction')
                    base_energies[reaction['name']] -= gb_reduction