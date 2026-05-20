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
        self.max_gb_influence = 0.0
        
        for config in self.gb_configurations:
          # Determine the distance metric for this GB type
          if config['type'] == 'vertical_planar':
            inner_boundary = config.get('width',0) / 2.0
            outer_boundary = config.get('outer_width',config['width']) / 2.0
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
            distance_function = None
          else:
            continue
            
          config['inner_boundary'] = inner_boundary
          config['outer_boundary'] = outer_boundary
          config['distance_function'] = distance_function
          
          if outer_boundary > self.max_gb_influence:
            self.max_gb_influence = outer_boundary
          
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
      x, y, z = site_position
      
      # Check vertical planar boundaries
      for gb in self.vertical_gbs:
        orient = gb['orientation']
        if orient == 'yz':
          # Vertical YZ place at specific x-position
          distance = abs(x - gb['position'])
        elif orient == 'xz':
          # Vertical XZ place at specific y-position
          distance = abs(y - gb['position'])
        elif orient == 'xy':
          # Horizontal XY place at specific z-position
          distance = abs(z - gb['position'])
        else:
          continue
        
        if distance <= gb['inner_boundary']:
          return 'inner_boundary'
        elif distance <= gb['outer_boundary']:
          return 'outer_boundary'
          
      # Check cylindrical boundaries
      for gb in self.cylindrical_gbs:
        cx, cy = gb['center']
        # Distance from cylinder axis
        distance_from_axis = math.sqrt((x - cx)**2 + (y - cy)**2)
        
        #if distance_from_axis <= radius: return True
        if distance_from_axis <= gb['inner_boundary']:
          return 'inner_boundary'
        elif distance_from_axis <= gb['outer_boundary']:
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
      x, y, z = site_position
      
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
        
      sx, sy, sz = site.position
      
      if not self._is_site_near_gb(sx, sy, sz):
        return
        
        
      # GB configuration
      gb_config = self.gb_configurations[0]
      event_modifications = gb_config.get('event_modifications',{})
      if not event_modifications:
        return
        
      # ========================================================================
      # CACHE CONFIGURATION (avoids repeated dict lookups)
      # ========================================================================
      mig_cfg = event_modifications.get('migration', {})
      gen_cfg = event_modifications.get('generation', {})
      rxn_cfg = event_modifications.get('reaction', {})
      
      if mig_cfg:
        req_mig = mig_cfg.get('region')
        affected_mig = set(mig_cfg.get('affected_defects', []))
        slope_m = mig_cfg.get('linear_slope', 0.0)
        intercept_m = mig_cfg.get('linear_intercept', 0.0)
        inner_m = mig_cfg.get('inner_boundary', 0.0)
        outer_m = mig_cfg.get('out_boundary', 0.0)
        
        v_gbs = self.vertical_gbs
        c_gbs_sq = [
          (gb['center'][0], gb['center'][1], gb['inner_boundary']**2, gb['outer_boundary']**2)
          for gb in self.cylindrical_gbs
        ]
      
      # Check if site is in GB
      #site_gb_region = self.get_site_gb_region(site.position)
      
      # Iterate through all applicable defects
      for defect_name in applicable_defects:
        base_energies = site.Act_E_dict[defect_name]
      
        # 1. Handle generation
        if gen_cfg and defect_name in gen_cfg.get('affected_defects', []) and "E_gen_defect" in base_energies:
          req_gen = gen_cfg.get('region')
          region = self.get_site_gb_region(site.position)
          
          if (req_gen == 'inner_boundary' and region == 'inner_boundary') or (req_gen == 'outer_boundary' and region != 'bulk'):
            base_energies["E_gen_defect"] -= self.get_activation_energy_GB(site.position,"generation")

        # 2. Handle migration    
        if mig_cfg and defect_name in affected_mig:
          Act_E_mig = {}
                
          for key,vec in migration_pathways.items():
            # Raw math destination 
            dx = sx + vec['direction'][0] * vec['distance']
            dy = sy + vec['direction'][1] * vec['distance']
            dz = sz + vec['direction'][2] * vec['distance']
            
            z_comp = vec['direction'][2]
            # Base energy selection
            if abs(z_comp) < 1e-9: base_e = base_energies['E_mig_plane'] 
            elif z_comp > 0: base_e = base_energies['E_mig_upward']
            else: base_e = base_energies['E_mig_downward']
              
            dist = 0.0; in_gb = False; is_inner = False
            
            # Check Planar GBs
            for gb in v_gbs:
              o = gb['orientation']
              if o == 'yz': d = abs(dx - gb['position'])
              elif o == 'xz': d = abs(dy - gb['position'])
              elif o == 'xy': d = abs(dz - gb['position'])
              else: continue
              
              if d <= inner_m: dist = d; in_gb = True; is_inner = True; break
              if d <= outer_m: dist = d; in_gb = True; is_inner = False; break
              
            # Check cylindrical GBs (if not already in planar)  
            if not in_gb:  
              for cx, cy, in_sq, out_sq in c_gbs_sq:
                d_sq = (dx - cx)**2 + (dy - cy)**2
                if d_sq <= in_sq: dist = d_sq**0.5; in_gb = True; is_inner = True; break
                if d_sq <= out_sq: dist = d_sq**0.5; in_gb = True; is_inner = False; break
                
            # Apply modification if region matches
            if in_gb:
              if (req_mig == 'inner_boundary' and is_inner) or (req_mig == 'outer_boundary'):
                red = slope_m * dist + intercept_m
                Act_E_mig[key] = base_e - (red if red > 0 else 0.0)
              else:
                Act_E_mig[key] = base_e
            else:
              Act_E_mig[key] = base_e
            
            site.Act_E_dict[defect_name]['E_mig'] = Act_E_mig
          
        # 3. Handle reactions
        if rxn_cfg:
          req_rxn = rxn_cfg.get('region')
          region = self.get_site_gb_region(site.position)
          if (req_rxn == 'inner_boundary' and region == 'inner_boundary') or (req_rxn == 'outer_boundary' and region != 'bulk'):
            affected_rxns = rxn_cfg.get('affected_reactions',[])
            for reaction_name, reaction in reactions_config.items():
              if reaction_name in affected_rxns and reaction['name'] in base_energies:
                base_energies[reaction['name']] -= self.get_activation_energy_GB(site.position, 'reaction')
                    
                    
    def _is_site_near_gb(self, x, y, z):
      """Fast bounding check: Is this site within max influence of ANY GB?"""
      # Quick planar check
      for gb in self.vertical_gbs:
        orient = gb['orientation']
        pos = gb['position']
        outer = gb['outer_boundary']
        if orient == 'yz' and abs(x - pos) > outer: continue
        if orient == 'xz' and abs(y - pos) > outer: continue
        if orient == 'xy' and abs(z - pos) > outer: continue
        return True
        
      # Quick cylindrical check
      for gb in self.cylindrical_gbs:
        cx, cy = gb['center']
        outer = gb['outer_boundary']
        if ((x - cx)**2 + (y - cy)**2)**0.5 <= outer:
          return True
          
      return False