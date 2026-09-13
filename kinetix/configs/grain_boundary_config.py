"""Configuration dataclasses for grain boundaries.

Defines :class:`GrainBoundaryConfig` (a single grain boundary) and
:class:`GrainBoundariesConfig` (a container loaded from one YAML file),
plus the strict-field helper :func:`_get_required`.
"""

# kinetix/configs/grain_boundary_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
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
  orientation: str | None = None # [REQUIRED for planar] - 'xz', 'xy', 'yz'
  position: float | None = None # [REQUIRED for planar] - Position in �
  width: float | None = None # [REQUIRED for planar] - GB width in �
  outer_width: float | None = None # [OPTIONAL] - Transition region width
  
  # Cylindrical GB parameters
  center: list[float] | None = None # [REQUIRED for cylindrical] - [x, y] center position
  radius: float | None = None # [REQUIRED for cylindrical] - Inner radius in �
  outer_radius: float | None = None # [OPTIONAL for cylindrical] - Outer radius
  
  # Event modifications (affect migration/reaction barriers)
  event_modifications: dict[str, Any] = field(default_factory=dict)
  
  # Description
  description: str = ""
  
  def to_dict(self) -> dict[str, Any]:
    """Convert to a plain dictionary for backwards compatibility.

    Returns:
      Mapping with all configuration fields keyed by their YAML names.
    """
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
  def from_dict(cls, data: dict[str, Any]) -> GrainBoundaryConfig:
    """
    Create GrainBoundaryConfig from dictionary (loaded from YAML).
    Validates required fields based on GB type: planar types require
    'orientation', 'position' and 'width'; 'cylindrical' requires
    'center' and 'radius'.

    Args:
      data: Mapping describing a single grain boundary.

    Returns:
      GrainBoundaryConfig populated from ``data``.

    Raises:
      ValueError: If a required field is missing or the type is unknown.
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
  grain_boundaries: list[GrainBoundaryConfig] = field(default_factory=list)
  description: str = ""
  
  def add_gb(self, gb: GrainBoundaryConfig) -> None:
    """Add a grain boundary configuration to the container.

    Args:
      gb: The grain boundary configuration to append.
    """
    self.grain_boundaries.append(gb)
    
  def to_dict(self) -> list[dict[str, Any]]:
    """Convert all grain boundaries to a list of dictionaries.

    Returns:
      One dictionary per configured grain boundary, in order.
    """
    return [gb.to_dict() for gb in self.grain_boundaries]
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> GrainBoundariesConfig:
    """
    Load grain boundary configurations from YAML file.
        
    Args:
      yaml_path: Path to grain boundaries YAML file
        
    Returns:
      GrainBoundariesConfig with all GBs loaded

    Raises:
      FileNotFoundError: If ``yaml_path`` does not exist.
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
def _get_required(data: dict[str, Any], key: str, yaml_path: Path | None, field_name: str) -> Any:
  """Get a required field from a dictionary, raising a clear error if missing.

  Args:
    data: Mapping loaded from a YAML configuration file.
    key: Key to look up in ``data``.
    yaml_path: Optional path of the source file, included in the error
      message when provided.
    field_name: Human-readable field name used in the error message.

  Returns:
    The value stored under ``key``.

  Raises:
    ValueError: If the key is missing or its value is None.
  """
  
  value = data.get(key)
  
  if value is None:
    location = f" in {yaml_path}" if yaml_path else ""
    raise ValueError(
      f"Missing required field '{field_name}'{location}\n\n"
      f" Please add '{key}: <value>' to your YAML file"
    )
  
  return value