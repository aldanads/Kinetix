# =============================================================================
# config.py
# Configuration classes for Kinetic Monte Carlo Resistive Switching Simulator
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import yaml

class VoltageMode(Enum):
  """Supported voltage application modes for the simulator"""
  NONE = auto()         # Default: no profile initialized
  RAMP_CYCLE = auto()   # Triangular ramp cycle
  ZERO_HOLD = auto()    # Constant 0V for ralaxation studies 
  CONSTANT = auto()     # Constant V  
  
class CurrentModel(Enum):
  OHMIC = auto()
  SCHOTTKY = auto()
  
# ---- Dataclasses for electrical configuration ----

@dataclass
class VoltageConfig:
  """Voltage protocol configuration."""
  mode: VoltageMode = VoltageMode.RAMP_CYCLE
  max_voltage: float = 2.0
  min_voltage: float = -2.0
  ramp_rate: float = 1.0
  constant_voltage: float = 0.0
  total_time: float = 100.0
  num_cycles: int = 1
  voltage_update_time: float = 0.1
  
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'VoltageConfig':
    """Create from dictionary (loaded from YAML) """
    mode_str = data.get('mode')
    mode = VoltageMode[mode_str]
    
    return cls(
      mode=mode,
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
  epsilon_r: float = 25.0
  
  @classmethod
  def from_dict(cls, data: Dict[str,Any]) -> 'CurrentConfig':
    """Create from dictionary (loaded from YAML)"""
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
  initial_voltage: float = 0.0
  initial_time: float = 0.0
  series_resistance: float = 0.0
  crystal_size: tuple = (50,50,50)
  voltage: VoltageConfig = field(default_factory=VoltageConfig)
  current: Optional[CurrentConfig] = None
  
  @classmethod
  def from_yaml(cls, yaml_path: Path, crystal_size: Optional[Tuple[float, float, float]] = None) -> 'ElectricalConfig':
    """Load electrical configuration from YAML file."""
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
      initial_voltage=data.get('initial_voltage'),
      initial_time=data.get('initial_time'),
      series_resistance=data.get('series_resistance'),
      voltage=voltage,
      current=current
    )
    
    return config
  