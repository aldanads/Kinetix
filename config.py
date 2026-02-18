# =============================================================================
# config.py
# Configuration classes for Kinetic Monte Carlo Resistive Switching Simulator
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum, auto

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
  
@dataclass
class CurrentConfig:
  """Current measurement model parameters."""
  model: CurrentModel = CurrentModel.SCHOTTKY
  barrier_height: float = 0.5
  temperature: float = 300.0
  area: float = 1.e-10
  epsilon_r: float = 25.0
  
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
  current: CurrentConfig = field(default_factory=CurrentConfig)