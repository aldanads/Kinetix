# kinetix/configs/material_config.py
"""Material and crystal structure configuration."""
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class MaterialSelection:
  """Material identification and MP database info."""
  name: str
  mp_id: str
  radius_neighbors: float = 4.0
    
@dataclass
class CrystalStructure:
  """Crystal geometry and orientation."""
  size: Tuple[float, float, float] = (50.0, 50.0, 50.0)  # Angstroms
  miller_indices: Tuple[int, int, int] = (0, 0, 1)
  sites_generation_layer: str = 'top_layer'  # or 'bottom_layer'
  
@dataclass
class MaterialConfig:
  """Complete material configuration."""
  selection: MaterialSelection
  structure: CrystalStructure
  formula: str = ""  # Filled by MaterialDataFetcher
  epsilon_r: float = 23.0  # Filled by MaterialDataFetcher
  chem_env_symmetry: str = "Unknown"  # Filled by MaterialDataFetcher
  metal_valence: float = 0.0  # Filled by MaterialDataFetcher
  bond_length_metal_O: float = 2.0  # Filled by MaterialDataFetcher
  
  # Derived properties
  @property
  def mesh_filename(self) -> str:
    """Generate mesh filename from material properties"""
    max_dim = max(self.structure.size)
    size_nm = int(max_dim / 10)
    return f"{self.formula}_{size_nm}nm_mesh.msh"
    
  @property
  def grid_filename(self) -> str:
    """Generate grid filename from material properties"""
    max_dim = max(self.structure.size)
    size_nm = int(max_dim / 10)
    return f"grid_{formula}_{size_nm}nm"