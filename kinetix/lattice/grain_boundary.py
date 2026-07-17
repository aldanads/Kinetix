# -*- coding: utf-8 -*-
"""GrainBoundary class for GB physics."""
import numpy as np
import math

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
        self.max_gb_influence = 0.0
        
        for config in self.gb_configurations:
          gb_type = config['type']
          # Determine the distance metric for this GB type
          if gb_type == 'vertical_planar':
            inner_boundary = config.get('width',0) / 2.0
            outer_boundary = config.get('outer_width',config['width']) / 2.0
            distance_function = self._distance_to_planar_gb
            self.vertical_gbs.append(config)
          elif gb_type == 'cylindrical':
            inner_boundary = config.get('radius',0)
            outer_boundary = config.get('outer_radius',config['radius'])
            distance_function = self._distance_to_cylindrical_gb  
            self.cylindrical_gbs.append(config)
          elif gb_type == 'triple_junction_planes':
            inner_boundary = config.get('width',0) / 2
            outer_boundary = config.get('outer_width',config['width']) / 2
            # NEED TO WRITE THE FUNCTION: self._distance_to_triple_junction_gb
            #distance_function = self._distance_to_triple_junction_gb
            distance_function = None
            self.triple_junction_gbs.append(config)
          else:
            continue
            
          config['inner_boundary'] = inner_boundary
          config['outer_boundary'] = outer_boundary
          config['distance_function'] = distance_function
          
          if outer_boundary > self.max_gb_influence:
            self.max_gb_influence = outer_boundary
          
          # Calculate event-specific interpolation parameters
          event_modifications = config.get('event_modifications', {})
          
          # Pre-compute event configs
          config['mig_cfg'] = event_modifications.get('migration', {})
          config['gen_cfg'] = event_modifications.get('generation', {})
          config['rxn_cfg'] = event_modifications.get('reaction', {})
          
          for cfg_key in ['mig_cfg', 'gen_cfg', 'rxn_cfg']:
            cfg = config[cfg_key]
            if cfg: 
              if 'affected_defects' in cfg:
                cfg['affected_defects_set'] = set(cfg['affected_defects'])
              if 'affected_reactions' in cfg:
                cfg['affected_reactions_set'] = set(cfg['affected_reactions'])
          
          
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
            event_config['inner_boundary'] = event_inner
            event_config['outer_boundary'] = event_outer
            
    def _distance_to_planar_gb(self,site_pos,gb_config):
      """Calculate distance to planar GB core"""
      x, y, z = site_pos
      
      orient = gb_config['orientation']
      if orient == 'yz':
        return abs(x - gb_config['position'])
      if orient == 'xz':
        return abs(y - gb_config['position'])
      if orient == 'xy':
        return abs(z - gb_config['position'])
        
    def _distance_to_cylindrical_gb(self,site_pos,gb_config):
      """Calculate distance to cylindrical GB axis"""
      x, y, z = site_pos
      cx, cy = gb_config['center']
      return math.sqrt((x - cx)**2 + (y - cy)**2)
            
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
      all_gbs = self.vertical_gbs + self.cylindrical_gbs + self.triple_junction_gbs
      
      for gb in all_gbs:
        dist_func = gb.get('distance_function')
        if not dist_func: continue
        
        dist = dist_func(site_position, gb)
        if dist <= gb['inner_boundary']:
          return 'inner_boundary'
        elif dist <= gb['outer_boundary']:
          return 'outer_boundary'
      return 'bulk'
      
    def _get_gb_reduction_for_site(self, pos, gb, event_type):
      """Helper to calculate the exact energy reduction for a specific GB and event."""
      event_cfg = gb.get('event_modifications',{}).get(event_type)
      if not event_cfg: return 0.0
      
      dist_func = gb.get('distance_function')
      if not dist_func: return 0.0
      
      dist = dist_func(pos, gb)
      inner = event_cfg['inner_boundary']
      outer = event_cfg['outer_boundary']
      
      if dist <= inner:
        return event_cfg['Act_E_diff_GB']
      elif dist <= outer:
        energy = event_cfg['linear_slope'] * dist + event_cfg['linear_intercept']
        return max(energy, 0.0)
      return 0.0
      
    def _calculate_dest_pos(self, origin_pos, mig_vector):
      """Calculate destination coordinates for a migration step."""
      ox, oy, oz = origin_pos
      dx = ox + mig_vector['direction'][0] * mig_vector['distance']
      dy = oy + mig_vector['direction'][1] * mig_vector['distance']
      dz = oz + mig_vector['direction'][2] * mig_vector['distance']
      return (dx, dy, dz)
    

      
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
        
      if not applicable_defects or not self.is_site_in_grain_boundary(site.position): 
        return # No defects can occupy this site type or outside GB
        
      site_pos = site.position
      # Check if site is in GB
      site_gb_region = self.get_site_gb_region(site_pos)
      
      # Track maximum reductions to handle overlapping GBs
      max_mig_reductions = {defect: {} for defect in applicable_defects}
      max_gen_reductions = {defect: 0.0 for defect in applicable_defects}
      max_rxn_reductions = {defect: {} for defect in applicable_defects}
      
      all_gbs = self.vertical_gbs + self.cylindrical_gbs + self.triple_junction_gbs
      
      for gb in all_gbs:
      
        if not gb.get('distance_function'):
          continue
        
        mig_cfg = gb['mig_cfg']
        gen_cfg = gb['gen_cfg']
        rxn_cfg = gb['rxn_cfg']
        
        # Iterate through all applicable defects
        for defect_name in applicable_defects:
          base_energies = site.Act_E_dict[defect_name]
        
          # 1. Handle generation
          if gen_cfg and defect_name in gen_cfg.get('affected_defects_set', set()):
            if self._region_matches(site_gb_region, gen_cfg.get('region')):
              red = self._get_gb_reduction_for_site(site_pos, gb, 'generation')
              max_gen_reductions[defect_name] = max(max_gen_reductions[defect_name], red)
              
          # 2. Handle migration    
          if mig_cfg and defect_name in mig_cfg.get('affected_defects_set', set()):
            mig_req_region = mig_cfg.get('region')
            for key,migration_vector in migration_pathways.items():
              dest_pos = self._calculate_dest_pos(site_pos, migration_vector)
              dest_gb_region = self.get_site_gb_region(dest_pos)
              
              if self._region_matches(dest_gb_region, mig_req_region):
                red = self._get_gb_reduction_for_site(dest_pos,gb, 'migration')
                current_max = max_mig_reductions[defect_name].get(key, 0.0)
                max_mig_reductions[defect_name][key] = max(current_max, red)
            
          # 3. Handle reactions
          if rxn_cfg:
            affected_rxns_set = rxn_cfg.get('affected_reactions_set',set())
            if affected_rxns_set and self._region_matches(site_gb_region, rxn_cfg.get('region')):
              red = self._get_gb_reduction_for_site(site_pos, gb, 'reaction')
              
              for rxn_key in affected_rxns_set:
                if rxn_key in reactions_config: 
                  rxn_name = reactions_config[rxn_key]['name']
                  if rxn_name in base_energies:
                    current_max = max_rxn_reductions[defect_name].get(rxn_name, 0.0)
                    max_rxn_reductions[defect_name][rxn_name] = max(current_max, red)
                    
                    
      # Apply the calculated maximum reductions to the site's Act_E_dict      
      for defect_name in applicable_defects:
        base_energies = site.Act_E_dict[defect_name]
        
        # Apply generation
        if "E_gen_defect" in base_energies and max_gen_reductions[defect_name] > 0:
          base_energies["E_gen_defect"] -= max_gen_reductions[defect_name]
          
        # Apply migration
        if migration_pathways:
          new_mig_energies = {}
          for key, mig_vector in migration_pathways.items():
            z_comp = mig_vector['direction'][2]
            if abs(z_comp) < 1e-9: base_e = base_energies.get('E_mig_plane', 0)
            elif z_comp > 0:       base_e = base_energies.get('E_mig_upward',0)
            else:                  base_e = base_energies.get('E_mig_downward', 0)
            
            reduction = max_mig_reductions[defect_name].get(key,0.0)
            new_mig_energies[key] = base_e - reduction
          base_energies['E_mig'] = new_mig_energies
          
        # Apply reactions
        for rxn_name, reduction in max_rxn_reductions[defect_name].items():
          if rxn_name in base_energies and reduction > 0:
            base_energies[rxn_name] -= reduction
                  