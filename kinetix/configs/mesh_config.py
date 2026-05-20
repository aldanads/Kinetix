# kinetix/configs/mesh_config.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

@dataclass
class MeshConfig:
  """Mesh generation parameters for Gmsh."""  
  
  # Domain settings
  gdim: int = 3
  gmsh_model_rank: int = 0
  
  # Global mesh sizing
  mesh_size: float = 3.0 # Angstroms
  bounding_box_padding: float = 3.0 # Angstroms
  
  # Charge spreading (physical parameter)
  epsilon_gaussian_charge: float = 0.8 # Angstroms
  
  # Adaptive refinement
  activate_mesh_refinement: bool = True
  fine_mesh_size: float = 1.0 # Angstroms
  refinement_radius: float = 2.5 # Angstroms
  
  def __post_init__(self):
    """Validate mesh parameters after initialization."""
    # Charge spreading vs mesh resolution
    if self.fine_mesh_size > 2.0 * self.epsilon_gaussian_charge:
      raise ValueError(
        f"fine_mesh_size ({self.fine_mesh_size} Angstroms) must be = 2xepsilon_gaussian_charge "
        f"({2*self.epsilon_gaussian_charge:.1f} Angstroms) to resolve charges"
      )
      
    if self.mesh_size > 4.0 * self.epsilon_gaussian_charge:
      print(f"Warning: mesh_size ({self.mesh_size} Angstroms) > 4xepsilon_gaussian_charge")
      print(f"Coarse regions may not fully resolve charge distributions.")
      
    # Refinement radius should cover charge distribution
    if self.refinement_radius < 2.0 * self.epsilon_gaussian_charge:
      print(f"Warning: refinement_radius ({self.refinement_radius} Angstroms) < 2xepsilon_gaussian_charge")
      print(f"Refinement zone may not fully contain charge distribution.")
      
    # Refinement radius must be = fine mesh size
    if self.refinement_radius < self.fine_mesh_size:
      raise ValueError(
        f"refinement_radius ({self.refinement_radius} Angstroms) must be = fine_mesh_size "
        f"({self.fine_mesh_size} Angstroms)"
      )
      
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization."""
    return {
      'gdim': self.gdim,
      'gmsh_model_rank': self.gmsh_model_rank,
      'mesh_size': self.mesh_size,
      'bounding_box_padding': self.bounding_box_padding,
      'epsilon_gaussian_charge': self.epsilon_gaussian_charge,
      'activate_mesh_refinement': self.activate_mesh_refinement,
      'fine_mesh_size': self.fine_mesh_size,
      'refinement_radius': self.refinement_radius,
    }
    
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'MeshConfig':
    """Create MeshConfig from dictionary (e.g., loaded YAML)."""
    return cls(
      gdim=data.get('gdim', 3),
      gmsh_model_rank=data.get('gmsh_model_rank', 0),
      mesh_size=data.get('mesh_size', 3.0),
      bounding_box_padding=data.get('bounding_box_padding', 3.0),
      epsilon_gaussian_charge=data.get('epsilon_gaussian_charge', 0.8),
      activate_mesh_refinement=data.get('activate_mesh_refinement', True),
      fine_mesh_size=data.get('fine_mesh_size', 1.0),
      refinement_radius=data.get('refinement_radius', 2.5),
    )
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> 'MeshConfig':
    """Load mesh config from YAML file."""
    yaml_path = Path(yaml_path)
    
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
      
    # Extract mesh section
    mesh_data = data.get('mesh', {})
    return cls.from_dict(mesh_data)