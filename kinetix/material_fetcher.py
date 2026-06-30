# kinetix/material_fetcher.py
"""
Materials Project data fetching with caching.
Separates API concerns from simulation configuration.
"""
from pathlib import Path
import json
from typing import Optional, Dict, Any, Tuple
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure


class MaterialDataFetcher:
  """Fetch and cache material data from Materials Project."""
  
  def __init__(self, api_key:str, mpi_ctx, cache_path: str = None):
    """Initialize with MP API key"""
    self.api_key = api_key
    self.rank = mpi_ctx.rank
    self.mpi_ctx = mpi_ctx
    self.cache_path = Path(cache_path) if cache_path else None
    
  # ------------ Helper methods --------------------
  def _load_from_cache(self) -> Dict[str, Any]:
    """ Load material data from local cache file """
    if self.cache_path and self.cache_path.exists():
      with open(self.cache_path, 'r') as f:
        return json.load(f)
    return None
    
  def _save_to_cache(self, data: Dict[str, Any]):
    """Save material data to local cache file (rank 0 only)"""
    if self.rank == 0 and self.cache_path:
      self.cache_path.parent.mkdir(parents=True, exist_ok=True)
      with open(self.cache_path, 'w') as f:
        json.dump(data, f, indent=2)
  # ------------------------------------------------
    
  
  def _fetch_material_summary(self, mp_id: str) -> Dict[str, Any]:
    """Fetch material summary (formula, density, etc.)"""
    with MPRester(self.api_key) as mpr:
      summary = mpr.materials.summary.search(material_ids=[mp_id])
      
      if not summary:
        raise ValueError(f"Material {mp_id} not found in Materials Project")
        
      s = summary[0]
      
      return {
        'formula': s['formula_pretty'],
        'density': s['density'],
        'volume': s['volume'],
        'space_group':s['symmetry']['symbol'],
      }
      
  def _fetch_chemenv_data(self, mp_id: str) -> Dict[str, Any]:
    """Fetch chemical environment data"""  
    with MPRester(self.api_key) as mpr:
      chem_data = mpr.materials.chemenv.search(material_ids=[mp_id])
      if not chem_data:
        print(f"Warning: No chemenv data for {mp_id}")
        return {
          'chem_env_symmetry': None,
          'metal_valence': None,
          'bond_length_metal_O': None,
        }
        
      c = chem_data[0]
      
      chemenv_names = c.get('chemenv_name', [])
      valences = c.get('valences', [])
      mol_envs = c.get('mol_from_site_environments', [])
      
      chem_env_symmetry = chemenv_names[0] if chemenv_names else None
      metal_valence = valences[0] if valences else None
      bond_length = None
      
      if mol_envs:
        mol_data = mol_envs[0]
        
        if len(mol_data.sites) >= 2:
          bond_length = mol_data.sites[0].distance(mol_data.sites[1])
          
      if not bond_length or not metal_valence or not chem_env_symmetry:
        print(f" Chemenv API data incomplete for {mp_id}. Using fallback bulk Structure and BVAnalyzer.")
        
        struct = mpr.get_structure_by_material_id(mp_id)
        fallback_data = self._calculate_fallback_chemenv(struct)
        
        bond_length = bond_length or fallback_data['bond_length_metal_O']
        metal_valence = metal_valence or fallback_data['metal_valence']
        chem_env_symmetry = chem_env_symmetry or fallback_data['chem_env_symmetry']
        
        print(f"Generalized fallback successful: {chem_env_symmetry}, Valence={metal_valence}, d={bond_length:.3f} angstroms")
      
      return {
        'chem_env_symmetry': chem_env_symmetry,
        'metal_valence': metal_valence,
        'bond_length_metal_O': bond_length,
      }
      
  def _calculate_fallback_chemenv(self, struct: Structure) -> Dict[str, Any]:
    """
    Generalized physics-based calculation of chemenv parameters from a bulk Structure.
    """
    # Find the first non-oxygen atom (metal cation)
    metal_sites = [site for site in struct if site.specie.symbol != 'O']
    if not metal_sites:
      raise ValueError("Structure contains no non-Oxygen atoms to analyze.")
    
    metal_site = metal_sites[0]
    metal_element = metal_site.specie
    
    # 1. Calculate bond length and coordination number
    # Use 3 angstroms as cutoff
    neighbors = struct.get_neighbors(metal_site, 3.0)
    o_neighbors = [n for n in neighbors if n.species_string == 'O']
    
    if o_neighbors:
      bond_length = sum(n.nn_distance for n in o_neighbors) / len(o_neighbors)
      coord_num = len(o_neighbors)
    else:
      # Fallback to ionic radii if geometry is distorted
      r_metal = metal_element.atomic_radius or 0.7
      r_oxygen = 1.4
      bond_length = r_metal + r_oxygen
      coord_num = 6
      
    geom_map = {
      3: 'Trigonal',
      4: 'Tetrahedron',
      6: 'Octahedron',
      7: 'Disheptahedral',
      8: 'Cube',
      12: 'Cuboctahedral'
    }
    chem_env_symmetry = geom_map.get(coord_num, 'Octahedron')
    
    # 2. Calculate valence using BVAnalyzer
    metal_valence = None
    # BVAnalyzer estimate method to determine oxidation states in a structure
    bva = BVAnalyzer()
    val_struct = bva.get_oxi_state_decorated_structure(struct)
    
    # Extract the calculated valence for our metal
    for site in val_struct:
      if site.specie.symbol == metal_element.symbol:
        metal_valence = abs(site.specie.oxi_state)
        break
        
    if metal_valence is None:
      common_oxi = metal_element.common_oxidation_states
      metal_valence = abs(common_oxi[0]) if common_oxi else 2.0
      
    return {
      'chem_env_symmetry': chem_env_symmetry,
      'metal_valence': metal_valence,
      'bond_length_metal_O': bond_length
    }
    
      
  def _fetch_dielectric_data(self, mp_id: str) -> Dict[str, Any]:
    """Fetch dielectric properties"""
    with MPRester(self.api_key) as mpr:
      diel_data = mpr.materials.dielectric.search(material_ids=[mp_id])
      
      if diel_data and len(diel_data) > 0 and hasattr(diel_data[0], 'e_total'):
        return {'epsilon_r': float(diel_data[0].e_total), 'source': 'MP'}
      else:
        print(f"Warning: No dielectric data for {mp_id}, using value provided by user")
        return {'epsilon_r': None}
        
  def get_all_material_data(self, mp_id: str) -> Dict[str, Any]:
    """Fetch all material data from Materials Project."""
    
    # === Try to load from cache first (all ranks can do this) ===
    cached_data = self._load_from_cache()
    
    if cached_data is not None:
      if self.rank == 0:
        print(f'Loaded material from cache: {self.cache_path}')
      return cached_data
    
    # Reuse methods
    if self.rank == 0:
      summary = self._fetch_material_summary(mp_id)
      chemenv = self._fetch_chemenv_data(mp_id)
      dielectric = self._fetch_dielectric_data(mp_id)
      data = {**summary, **chemenv, **dielectric}
      
      self._save_to_cache(data)
      print(f'Saved material cache: {self.cache_path}')
    else:
      data = None
      
    data = self.mpi_ctx.bcast(data, root=0)
    
    # Merge all dictionaries into one
    return data
    