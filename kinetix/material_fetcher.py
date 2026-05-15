# kinetix/material_fetcher.py
"""
Materials Project data fetching with caching.
Separates API concerns from simulation configuration.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pymatgen.ext.matproj import MPRester

class MaterialDataFetcher:
  """Fetch and cache material data from Materials Project."""
  
  def __init__(self, api_key:str):
    """Initialize with MP API key"""
    self.api_key = api_key
  
  def fetch_material_summary(self, mp_id: str) -> Dict[str, Any]:
    """Fetch material summary (formula, density, etc.)"""
    with MPRester(self.api_key) as mpr:
      summary = mpr.materials.summary.search(material_ids=[mp_id])
      
      if not summary:
        raise ValueError(f"Material {mp_id} not found in Materials Project")
        
      s = summary[0]
      return {
        'formula': s.formula_pretty,
        'density': s.density,
        'volume': s.volume,
        'space_group':s.symmetry.symbol,
      }
      
  def fetch_chemenv_data(self, mp_id: str) -> Dict[str, Any]:
    """Fetch chemical environment data"""  
    with MPRester(self.api_key) as mpr:
      chem_data = mpr.materials.chemenv.search(material_ids=[mp_id])
      if not chem_data:
        print(f"Warning: No chemenv data for {mp_id}")
        
      c = chem_data[0]
      mol = c.mol_from_site_environments[0]
      bond_length = mol.sites[0].distance(mol.sites[1])
      
      return {
        'chem_env_symmetry': c.chemenv_name[0],
        'metal_valence': c.valences[0],
        'bond_length_metal_O': bond_length,
      }
      
  def fetch_dielectric_data(self, mp_id: str) -> Dict[str, Any]:
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
    # Reuse methods
    summary = self.fetch_material_summary(mp_id)
    chemenv = self.fetch_chemenv_data(mp_id)
    dielectric = self.fetch_dielectric_data(mp_id)
    
    # Merge all dictionaries into one
    return {**summary, **chemenv, **dielectric}
    