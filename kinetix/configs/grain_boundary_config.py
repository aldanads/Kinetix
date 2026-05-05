# kinetix/configs/grain_boundary_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

@dataclass
class GrainBoundaryConfig:
  """
  Configuration for a single grain boundary.
    
  Supports multiple GB types:
  - vertical_planar: Planar GB perpendicular to a crystal axis
  - horizontal_planar: Planar GB parallel to crystal layers
  - cylindrical: Cylindrical GB (e.g., columnar grains)
  """
  # Basic identification
  type: str                 # [REQUIRED] - 'vertical_planar', 'horizontal_planar', 'cylindrical'
  enabled: bool = True
  
  # Geometry parameters
  orientation: Optional[str] = None # [REQUIRED for planar] - 'xz', 'xy', 'yz'
  position: Optional[float] = None # [REQUIRED for planar] - Position in Å
  width: Optional[float] = None # [REQUIRED for planar] - GB width in Å
  outer_width: Optional[float] = None # [OPTIONAL] - Transition region width
  
  # Cylindrical GB parameters
  center: Optional[List[float]] = None # [REQUIRED for cylindrical] - [x, y] center position
  radius: Optional[float] = None # [REQUIRED for cylindrical] - Inner radius in Å
  outer_radius: Optional[float] = None # [OPTIONAL for cylindrical] - Outer radius
  
  # Event modifications (affect migration/reaction barriers)
  event_modifications: Dict[str, Any] = field(default_factory=dict)
  
  # Description
  description: str = ""
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'type': self.type,
      'enabled': self.enabled,
      'orientation': self.orientation,
      'position': self.position,
      'width': self.width,
      'outer_width': self.outer_width,
      'center': self.center,
      'radius': self.radius,
      'outer_radius': self.outer_radius,
      'event_modifications': self.event_modifications,
      'description': self.description,
    }
    
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'GrainBoundaryConfig':
    """
    Create GrainBoundaryConfig from dictionary (loaded from YAML).
    Validates required fields based on GB type.
    """
    gb_type = _get_required(data, 'type', None, 'grain_boundary.type')
    enabled = data.get('enabled', True)
    
    # Validate required fields based on type
    if gb_type in ['vertical_planar', 'horizontal_planar']:
      # Planar GB requires: orientation, position, width
      orientation = _get_required(data, 'orientation', None, f'grain_boundary.orientation (for {gb_type})')
      position = _get_required(data, 'position', None, f'grain_boundary.position (for {gb_type})')
      width = _get_required(data,  'width', None, f'grain_boundary.width (for {gb_type})')
      
      center = None
      radius = None
      outer_radius = None
    
    elif gb_type == 'cylindrical':
      # Cylindrical GB requires: center, radius
      center = _get_required(data, 'center', None, 'grain_boundary.center (for cylindrical)')
      radius = _get_required(data, 'radius', None, 'grain_boundary.radius (for cylindrical)')
      
      orientation = None
      position = None
      width = None
      outer_radius = data.get('outer_radius')
    
    else:
      raise ValueError(f"Unkown grain boundary type: '{gb_type}'. Supported: vertical_planar, horizontal_planar, cylindrical")
    
    return cls(
      type=gb_type,
      enabled=enabled,
      orientation=orientation,
      position=position,
      width=width,
      outer_width=data.get('outer_width'),
      center=center,
      radius=radius,
      outer_radius=outer_radius,
      event_modifications=data.get('event_modifications',{}),
      description=data.get('description',''),
    )
    
# =============================================================================
# Container for Multiple Grain Boundaries
# =============================================================================
@dataclass
class GrainBoundariesConfig:
  """
  Container for multiple grain boundary configurations.
  Loaded from a single YAML file.
  """
  grain_boundaries: List[GrainBoundaryConfig] = field(default_factory=list)
  description: str = ""
  
  def add_gb(self, gb: GrainBoundaryConfig):
    """Add a grain boundary configuration"""
    self.grain_boundaries.append(gb)
    
  def to_dict(self) -> List[Dict[str, Any]]:
    """Convert all GBs to list of dictionaries"""
    return [gb.to_dict() for gb in self.grain_boundaries]
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> 'GrainBoundariesConfig':
    """
    Load grain boundary configurations from YAML file.
        
    Args:
      yaml_path: Path to grain boundaries YAML file
        
    Returns:
      GrainBoundariesConfig with all GBs loaded
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
      raise FileNotFoundError(f"Grain boundary config file not found: {yaml_path}")
      
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
      
    config = cls(
      description=data.get('description', '')
    )
    
    # Load each grain boundary from YAML
    gb_list = data.get('grain_boundaries', [])
    for gb_data in gb_list:
      gb = GrainBoundaryConfig.from_dict(gb_data)
      config.add_gb(gb)
    
    return config

# =============================================================================
# Helper Function: Strict Field Validation
# =============================================================================
def _get_required(data: dict, key: str, yaml_path: Optional[Path], field_name: str) -> Any:
  """
  Get required field from dictionary, raise clear error if missing.
  """
  
  value = data.get(key)
  
  if value is None:
    location = f" in {yaml_path}" if yaml_path else ""
    raise ValueError(
      f"Missing required field '{field_name}'{location}\n\n"
      f" Please add '{key}: <value>' to your YAML file"
    )
  
  return value