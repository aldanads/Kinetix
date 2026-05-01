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
      'radius'
    
    }
