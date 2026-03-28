# kinetix/configs/simulation_config.py
"""Master simulation configuration combining all sub-configs."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from kinetix.configs.material_config import MaterialConfig
from kinetix.configs.defect_config import DefectsConfig
from kinetix.configs.reaction_config import ReactionsConfig
from kinetix.configs.solver_config import PoissonSolverConfig, HeatSolverConfig, SuperbasinConfig
from kinetix.configs.electrical_config import ElectricalConfig

@dataclass
class ExperimentalConditions:
  """Experimental environment parameters."""
  temperature: float = 300.0  # K
  sticking_coeff: Optional[float] = None
  partial_pressure: Optional[float] = None
  experiment: str = "ECM memristor"

@dataclass
class SimulationSettings:
  """General simulation parameters."""
  technology: str = "PZT"
  mode: str = "interstitial"  # or 'vacancy'
  use_parallel: bool = True
  save_data: bool = True
  snapshoots_steps: int = 40
  total_steps: int = 1000
  seed: int = 1

@dataclass
class GrainBoundaryConfig:
  """Grain boundary configuration."""
  type: str = "vertical_planar"
  orientation: str = "xz"
  position: float = 0.0
  width: float = 4.0
  outer_width: float = 25.0
  event_modifications: Dict[str, Any] = field(default_factory=dict)
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary"""
    return {
      'type': self.type,
      'orientation': self.orientation,
      'position': self.position,
      'width': self.width,
      'outer_width': self.outer_width,
      'event_modifications': self.event_modifications,
    }

@dataclass
class SimulationConfig:
  """
  Master configuration for kMC simulation.
  Combines all sub-configurations into one object.
  """
  # Core configurations
  material: MaterialConfig
  experimental: ExperimentalConditions = field(default_factory=ExperimentalConditions)
  settings: SimulationSettings = field(default_factory=SimulationSettings)
  defects: DefectsConfig = field(default_factory=DefectsConfig)
  reactions: ReactionsConfig = field(default_factory=ReactionsConfig)
  poisson: PoissonSolverConfig = field(default_factory=PoissonSolverConfig)
  heat: HeatSolverConfig = field(default_factory=HeatSolverConfig)
  superbasin: SuperbasinConfig = field(default_factory=SuperbasinConfig)
  electrical: Optional[ElectricalConfig] = None
  grain_boundaries: List[GrainBoundaryConfig] = field(default_factory=list)
    
  # Runtime objects (not serialized)
  rng: Any = None
  mpi_ctx: Any = None
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert entire config to dictionary for backwards compatibility"""
    return {
      'material': {
        'formula': self.material.formula,
        'mp_id': self.material.selection.mp_id,
        'epsilon_r': self.material.epsilon_r,
        'crystal_size': self.material.structure.size,
        'miller_indices': self.material.structure.miller_indices,
      },
      'experimental': {
        'temperature': self.experimental.temperature,
        'experiment': self.experimental.experiment,
      },
      'settings': {
        'technology': self.settings.technology,
        'mode': self.settings.mode,
        'save_data': self.settings.save_data,
        'snapshoots_steps': self.settings.snapshoots_steps,
        'total_steps': self.settings.total_steps,
      },
      'defects_config': self.defects.to_dict(),
      'reactions_config': self.reactions.to_dict(),
      'poisson': self.poisson.to_dict(),
      'heat': self.heat.to_dict(),
      'superbasin': self.superbasin.to_dict(),
      'gb_configurations': [gb.to_dict() for gb in self.grain_boundaries],
    }