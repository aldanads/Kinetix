# kinetix/configs/material_config.py
"""Material and crystal structure configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class MaterialSelection:
  """Material identification and Materials Project database info.

  Attributes:
      name: Human-readable material name (e.g. "PZT").
      mp_id: Materials Project database identifier (e.g. "mp-1234").
      radius_neighbors: Cutoff radius in Angstroms for neighbor searches.
  """
  name: str
  mp_id: str
  radius_neighbors: float = 4.0
    
@dataclass
class CrystalStructure:
  """Crystal geometry and orientation.

  Attributes:
      size: Simulation box dimensions (x, y, z) in Angstroms.
      miller_indices: Surface orientation as Miller (hkl) indices.
      sites_generation_layer: Surface layer used for site generation
          ("top_layer" or "bottom_layer").
      affected_site: Identifier of the site affected by defect generation.
      facets_type: Optional facet descriptor (structure-dependent; may be None).
      interstitial_generation: Optional parameters for interstitial site
          generation, keyed by name (may be None).
  """
  size: tuple[float, float, float] = (50.0, 50.0, 50.0)  # Angstroms
  miller_indices: tuple[int, int, int] = (0, 0, 1)
  sites_generation_layer: str = 'top_layer'  # or 'bottom_layer'
  affected_site: str = ''
  facets_type: Any | None = None
  interstitial_generation: dict[str, Any] | None = None
  
@dataclass
class MaterialConfig:
  """Complete material configuration.

  Attributes:
      selection: Material identification and database info.
      structure: Crystal geometry and orientation.
      formula: Chemical formula (filled by MaterialDataFetcher).
      epsilon_r: Relative permittivity (filled by MaterialDataFetcher).
      chem_env_symmetry: Local chemical environment symmetry
          (filled by MaterialDataFetcher).
      metal_valence: Metal cation valence (filled by MaterialDataFetcher).
      bond_length_metal_O: Metal-oxygen bond length in Angstroms
          (filled by MaterialDataFetcher).
  """
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
    """Generate the mesh filename from material properties.

    Returns:
        Mesh filename of the form "{formula}_{size}nm_mesh.msh", where
        size is the largest box dimension converted to nm.
    """
    max_dim = max(self.structure.size)
    size_nm = int(max_dim / 10)
    return f"{self.formula}_{size_nm}nm_mesh.msh"
    
  @property
  def grid_filename(self) -> str:
    """Generate the grid filename from material properties.

    Returns:
        Grid filename of the form "grid_{formula}_{size}nm".

    Raises:
        NameError: When accessed before `formula` is set, because this
            implementation references the bare name `formula` rather
            than `self.formula` (pre-existing behavior).
    """
    max_dim = max(self.structure.size)
    size_nm = int(max_dim / 10)
    return f"grid_{formula}_{size_nm}nm"