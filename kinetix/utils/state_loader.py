import logging

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_lammps_dump(dump_path: str) -> dict:
  """
  Extract the timestep and atom data (id, type, x, y, z, charge) from a LAMMPS
  dump file.

  Returns {'timestep': float, 'atoms': [dict, ...]}. Box bounds are parsed only
  to be skipped: the simulation domain comes from the crystal grid, so they are
  intentionally not returned.

  Only one frame is returned. For a multi-frame dump the atom list is restarted
  at every 'ITEM: ATOMS' block, so the atoms of the LAST frame win while
  'timestep' keeps that same last frame's value.

  Coordinates are used as written (no periodic-image folding), and a row with
  fewer columns than the header raises ValueError with its 1-based line number.
  """
  
  with open(dump_path, 'r') as f:
    lines = f.readlines()
    
  timestep = 0.0
  atoms = []
  headers = []
  
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    if line == "ITEM: TIMESTEP":
      timestep = float(lines[i+1].strip())
      i += 2
    elif line.startswith("ITEM: NUMBER OF ATOMS"):
      i += 2
    elif line.startswith("ITEM: BOX BOUNDS"):
      i += 4 # Skip 3 lines of bounds + the header line
    elif line.startswith("ITEM: ATOMS"):
      headers = line.replace("ITEM: ATOMS", "").strip().split()
      # Each frame restarts the atom list: parse_lammps_dump returns a
      # single frame, so the LAST frame of a multi-frame dump wins.
      atoms = []
      i += 1
      # Read atoms until EOF or next ITEM
      while i < len(lines) and not lines[i].startswith("ITEM:"):
         parts = lines[i].strip().split()
         if parts:
           if len(parts) < len(headers):
             raise ValueError(
               f"Malformed row at line {i + 1}: expected {len(headers)} "
               f"columns ({' '.join(headers)}), got {len(parts)}"
             )
           atom = {}
           for j, header in enumerate(headers):
             val = parts[j] 
             if header in ['id', 'type']:
               atom[header] = int(val)
             else:
               atom[header] = float(val)
           atoms.append(atom)
         i += 1
    else:
      i += 1
    
  return {
      "timestep": timestep,
      "atoms": atoms
  }
    
def decode_species_key_passivation(species_key: str, defects_config: dict) -> tuple:
  """
  Reverses the logic of _species_id_gen to map 'V_O_2' -> ('V_O', 2, 'oxygen_vacancy').
  """
  # 1. Check direct symbol matches (e.g., H, H2, Ag)
  for defect_name, cfg in defects_config.items():
    if cfg['symbol'] == species_key:
      return cfg['symbol'], 0, defect_name
      
  # 2. Check passivation levels (e.g., V_O_1, V_O_2)
  for defect_name, cfg in defects_config.items():
    symbol = cfg['symbol']
    max_pass = cfg.get('max_passivation_level', 0)
    if max_pass > 0:
      for level in range(max_pass + 1):
        if f"{symbol}_{level}" == species_key:
          return symbol, level, defect_name
          
  return species_key, 0, None
    

def load_state_from_dump(system_state, dump_path: str, tolerance: float = 0.1):
  """
  Loads a kMC state from a LAMMPS dump file onto the pristine grid_crystal.
    
  Parameters:
  -----------
  system_state : KMCSimulator
    The initialized system with a pristine grid_crystal.
  dump_path : str
    Path to the LAMMPS dump file.
  tolerance : float
    Maximum distance (in angstroms) to map a dump coordinate to a grid site.
  """
  logger.info('Loading state from dump: %s', dump_path)
  
  # 1. Parse dump
  dump_data = parse_lammps_dump(dump_path)
  timestep = dump_data['timestep']
  atoms = dump_data['atoms']
  
  # 2. Ensure species mapping is available 
  if not hasattr(system_state, 'SPECIES_ID_TO_TYPE') or not system_state.SPECIES_ID_TO_TYPE:
    system_state._species_id_gen()
    
  # 3. Build KDTree for robust coordinate mapping 
  grid_indices = list(system_state.grid_crystal.keys())
  grid_positions = np.array([system_state.grid_crystal[idx].position for idx in grid_indices])
  kdtree = cKDTree(grid_positions)
  
  # 4. Prepare update sets for topology rebuild
  support_update_sites = set()
  event_update_sites = set()
  
  loaded_count = 0
  skipped_count = 0
  
  
  # 5. Map atoms to grid
  for atom in atoms:
    atom_type = atom['type']
    x,y,z = atom['x'], atom['y'], atom['z']
    pos = np.array([x,y,z])
    
    species_key = system_state.SPECIES_ID_TO_TYPE.get(atom_type)
    if species_key is None or species_key == 'Empty':
      skipped_count += 1
      continue
      
    chemical_specie, passivation_level, defect_name = decode_species_key_passivation(
      species_key, system_state.defects_config
    )
    
    # Skip host lattice atoms (already in the grid)
    if defect_name is None:
      skipped_count += 1
      continue
    
    # Find nearest grid site using KDTree
    dist, nearest_idx_in_array = kdtree.query(pos)
    if dist > tolerance:
      logger.warning('No grid site found near (%.3f, %.3f, %.3f) Distance: %.3f angstroms', x, y, z, dist)
      skipped_count += 1
      continue
    
    idx = grid_indices[nearest_idx_in_array]
    
    # Get charge (default to neutral when the dump has no charge column)
    ion_charge = atom.get('charge', 0)
    
    # Introduce species using Kinetix infrastructure
    system_state.event_handler._introduce_specie_site(
      idx, support_update_sites, event_update_sites,
      chemical_specie, ion_charge
    )

    # Apply passivation level
    site = system_state.grid_crystal[idx]
    if passivation_level > 0:
      site.defect.passivation_level = passivation_level
     
    loaded_count += 1      
    
  # 6. Rebuild topology at the end
  logger.info('Rebuilding topology for %d active sites...', len(event_update_sites))
  system_state.event_handler.update_sites_topology(support_update_sites, event_update_sites)
  
  # 7. Update time
  system_state.time = timestep
  system_state.list_time = [timestep]
  
  logger.info('State loaded successfully at t=%s', timestep)
  logger.info('  - Loaded %d defect/interstitial atoms', loaded_count)
  logger.info('  - Skipped %d host lattice/empty atoms', skipped_count)
  
            
