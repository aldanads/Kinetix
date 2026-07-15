# kinetix/configs/solver_config.py
"""Poisson and Heat solver configuration."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class PoissonSolverConfig:
  """Poisson equation solver parameters."""
  mesh_file: str = ""
  epsilon_r: float = 23.0
  chem_env_symmetry: str = "Unknown"
  metal_valence: float = 0.0
  d_metal_O: float = 2.0
  active_dipoles: int = 4
  poisson_solve_frequency: int = 100
  solve_Poisson: bool = True
  save_Poisson: bool = False
  screening_factor: float = 0.01
  conductivity: Dict[str, float] = field(default_factory=dict)
  epsilon_r: float = 23.0
  defects_config: Dict[str, Any] = field(default_factory=dict)
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'mesh_file': self.mesh_file,
      'epsilon_r': self.epsilon_r,
      'chem_env_symmetry': self.chem_env_symmetry,
      'metal_valence': self.metal_valence,
      'd_metal_O': self.d_metal_O,
      'active_dipoles': self.active_dipoles,
      'poisson_solve_frequency': self.poisson_solve_frequency,
      'solve_Poisson': self.solve_Poisson,
      'save_Poisson': self.save_Poisson,
      'screening_factor': self.screening_factor,
      'conductivity': self.conductivity,
      'defects_config': self.defects_config,
    }

@dataclass
class HeatSolverConfig:
  """Heat equation solver parameters."""
  solve_heat: bool = False
  save_heat: bool = False
  thermal_conductivity: Dict[str, float] = field(default_factory=dict)  # W/m-K
  specific_heat: float = 500.0  # J/kg-K
  density: float = 5000.0  # kg/m³
  heat_solve_frequency: int = 100
  tau_thermal: float = 1.0 # s
  use_thermal_inertia: bool = True
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary"""
    return {
      'solve_heat': self.solve_heat,
      'save_heat': self.save_heat,
      'thermal_conductivity': self.thermal_conductivity,
      'specific_heat': self.heat_capacity,
      'density': self.density,
      'heat_solve_frequency': self.heat_solve_frequency,
      'tau_thermal': self.tau_thermal,
      'use_thermal_inertia': use_thermal_inertia
    }

@dataclass
class SuperbasinConfig:
  """Superbasin acceleration parameters."""
  enabled_superbasin: bool = False
  n_search_superbasin: int = 50
  time_step_limits: float = 1e-4
  E_min: float = 0.5
  energy_step: float = 0.05
  time_based_superbasin: bool = True
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary"""
    return {
      'enabled_superbasin': self.enabled_superbasin,
      'n_search_superbasin': self.n_search_superbasin,
      'time_step_limits': self.time_step_limits,
      'E_min': self.E_min,
      'energy_step': self.energy_step,
      'time_based_superbasin': self.time_based_superbasin,
    }