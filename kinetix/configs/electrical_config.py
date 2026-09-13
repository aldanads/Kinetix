"""Configuration classes for the electrical (voltage/current) subsystem.

Defines the voltage protocol (:class:`VoltageConfig`, :class:`VoltageMode`),
the current measurement model (:class:`CurrentConfig`, :class:`CurrentModel`)
and the aggregate :class:`ElectricalConfig` consumed by
:class:`kinetix.solvers.electrical.ElectricalController`.
"""

# =============================================================================
# config.py
# Configuration classes for Kinetic Monte Carlo Resistive Switching Simulator
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from pathlib import Path
import yaml

class VoltageMode(Enum):
  """Supported voltage application modes for the simulator"""
  NONE = auto()         # Default: no profile initialized
  RAMP_CYCLE = auto()   # Triangular ramp cycle
  ZERO_HOLD = auto()    # Constant 0V for ralaxation studies 
  CONSTANT = auto()     # Constant V  
  
class CurrentModel(Enum):
  """Supported conduction models for current measurement."""

  OHMIC = auto()
  SCHOTTKY = auto()
  
# ---- Dataclasses for electrical configuration ----

@dataclass
class VoltageConfig:
  """Voltage protocol configuration."""
  mode: VoltageMode = VoltageMode.RAMP_CYCLE
  initial_voltage: float = 0.0
  max_voltage: float = 2.0
  min_voltage: float = -2.0
  ramp_rate: float = 1.0
  num_cycles: int = 1
  
  # CONSTANT / ZERO_HOLD parameters
  constant_voltage: float = 0.0
  total_time: float | None = None # Optional: auto-calculated from RAMP, explicit for others
  voltage_update_time: float = 0.1
  
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> VoltageConfig:
    """Create a config from a dictionary (loaded from YAML).

    Missing keys fall back to the dataclass defaults.

    Args:
      data: Mapping with optional keys 'mode', 'initial_voltage',
        'max_voltage', 'min_voltage', 'ramp_rate', 'constant_voltage',
        'total_time', 'num_cycles' and 'voltage_update_time'.

    Returns:
      VoltageConfig populated from ``data``.

    Raises:
      KeyError: If 'mode' is missing or not a name of :class:`VoltageMode`.
    """
    mode_str = data.get('mode')
    mode = VoltageMode[mode_str]
    
    return cls(
      mode=mode,
      initial_voltage=data.get('initial_voltage'),
      max_voltage=data.get('max_voltage'),
      min_voltage=data.get('min_voltage'),
      ramp_rate=data.get('ramp_rate'),
      constant_voltage=data.get('constant_voltage'),
      total_time=data.get('total_time'),
      num_cycles=data.get('num_cycles'),
      voltage_update_time=data.get('voltage_update_time')
    )
  
@dataclass
class CurrentConfig:
  """Current measurement model parameters."""
  model: CurrentModel = CurrentModel.SCHOTTKY
  barrier_height: float = 0.5
  temperature: float = 300.0
  area: float = 1.e-10
  epsilon_r: float = 23.0
  
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> CurrentConfig:
    """Create a config from a dictionary (loaded from YAML).

    Missing keys fall back to the dataclass defaults.

    Args:
      data: Mapping with optional keys 'model', 'barrier_height',
        'temperature', 'area' and 'epsilon_r'.

    Returns:
      CurrentConfig populated from ``data``.

    Raises:
      KeyError: If 'model' is missing or not a name of :class:`CurrentModel`.
    """
    model_str = data.get('model')
    model = CurrentModel[model_str]
    
    return cls(
      model=model,
      barrier_height=data.get('barrier_height'),
      temperature=data.get('temperature'),
      area=data.get('area'),
      epsilon_r=data.get('epsilon_r')
    )
  
@dataclass
class ElectricalConfig:
  """
  Complete electrical configuration
  This is the main config object you'll pass to ElectricalController
  """
  initial_time: float = 0.0
  series_resistance: float = 0.0
  crystal_size: tuple[int, int, int] = (50,50,50)
  voltage: VoltageConfig = field(default_factory=VoltageConfig)
  current: CurrentConfig | None = None
  
  @classmethod
  def from_yaml(cls, yaml_path: Path, crystal_size: tuple[float, float, float] | None = None) -> ElectricalConfig:
    """Load an electrical configuration from a YAML file.

    Args:
      yaml_path: Path to the electrical configuration YAML file.
      crystal_size: Accepted for API compatibility; not used when loading.

    Returns:
      ElectricalConfig built from the YAML contents. The 'voltage' section
      is parsed into a VoltageConfig; the 'current' section (when present)
      into a CurrentConfig, otherwise ``current`` stays None.

    Raises:
      FileNotFoundError: If ``yaml_path`` does not exist.
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
      raise FileNotFoundError(f"Electrical config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
      
    voltage_data = data.get('voltage',{})
    voltage = VoltageConfig.from_dict(voltage_data)
    
    current_data = data.get('current')
    if current_data is not None:
      current = CurrentConfig.from_dict(current_data)
    else:
      current = None
    
    config = cls(
      initial_time=float(data.get('initial_time')),
      series_resistance=float(data.get('series_resistance')),
      voltage=voltage,
      current=current
    )
    
    return config
  