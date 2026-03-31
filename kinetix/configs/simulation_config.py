# kinetix/configs/simulation_config.py
"""Master simulation configuration combining all sub-configs."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import yaml

from kinetix.configs.material_config import MaterialConfig, MaterialSelection, CrystalStructure
from kinetix.configs.defect_config import DefectsConfig
from kinetix.configs.reaction_config import ReactionsConfig
from kinetix.configs.solver_config import PoissonSolverConfig, HeatSolverConfig, SuperbasinConfig
from kinetix.configs.electrical_config import ElectricalConfig, VoltageConfig, VoltageMode


class ConfigValidationError(Exception):
    """Raised when required configuration field is missing"""
    pass

@dataclass
class ExperimentalConditions:
  """Experimental environment parameters."""
  temperature: float = 300.0  # K
  sticking_coeff: Optional[float] = None
  partial_pressure: Optional[float] = None # Pa

@dataclass
class SimulationSettings:
  """General simulation parameters."""
  # Core settings
  
  total_steps: Optional[int] = None  
  seed_rng: Optional[int] = None 
  simulation_type: str = ""
  technology: str = ""
  mode: str = ""  # or 'vacancy'
  save_data: bool = True
  snapshoots_steps: int = 40
  lammps_output: bool = True

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
  # Metadata
  name: str = ""
  description: str = ""
  author: str = ""
  
  # Core configurations
  material: Optional[MaterialConfig] = None
  experimental: Optional[ExperimentalConditions] = None
  settings: Optional[SimulationSettings] = None  
  defects: Optional[DefectsConfig] = None        
  reactions: Optional[ReactionsConfig] = None    
  poisson: Optional[PoissonSolverConfig] = None  
  heat: Optional[HeatSolverConfig] = None        
  superbasin: Optional[SuperbasinConfig] = None  
  electrical: Optional[ElectricalConfig] = None
  grain_boundaries: List[GrainBoundaryConfig] = field(default_factory=list)
    
  # Runtime objects
  rng: Any = None
  mpi_ctx: Any = None
  base_path: Optional[Path] = None # For resolving relative component paths
    
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
        'sticking_coeff': self.experimental.sticking_coeff,
        'partial_pressure': self.experimental.partial_pressure
      },
      'settings': {
        'simulation_type': self.simulation_type.simulation_type,
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
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> 'SimulationConfig':
    """
    Load complete simulation configuration from YAML preset file.
    Automatically loads referenced component files (defects first, reactions/GB later).
        
    Args:
      yaml_path: Path to preset YAML file (e.g., presets/PZT_ZrPbO3.yaml)
        
    Returns:
      SimulationConfig with all components loaded
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
      raise FileNotFoundError(f"Preset file not found: {yaml_path}")
      
    # Store base path for resolving relative component paths
    # yaml_path is like: data/parameters/presets/PZT_ZrPbO3.yaml
    # base_path should be: data/parameters
    base_path =  yaml_path.parent.parent
    
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
      
    # Create config object with metadata
    config = cls(
      name=_get_required(data.get('metadata', {}), 'name', yaml_path, 'metadata.name'),
      description=data.get('metadata',{}).get('description', ''),
      author=data.get('metadata', {}).get('author',''),
      base_path=base_path,
    )
    
    # =========================================================================
    # Load Material Configuration
    # =========================================================================
    material_data = _get_required(data, 'material', yaml_path, 'material')
    crystal_data = _get_required(data, 'crystal', yaml_path, 'crystal')
    
    config.material = MaterialConfig(
      selection=MaterialSelection(
        name=_get_required(material_data, 'name', yaml_path, 'material.name'),
        mp_id=_get_required(material_data, 'mp_id', yaml_path, 'material.mp_id'),
        radius_neighbors=_get_required(material_data, 'radius_neighbors', yaml_path, 'material.radius_neighbors')
      ),
      structure=CrystalStructure(
        size=tuple(_get_required(crystal_data, 'size', yaml_path, 'crystal.size')),
        miller_indices=tuple(_get_required(crystal_data, 'miller_indices', yaml_path, 'crystal.miller_indices')),
        sites_generation_layer=_get_required(crystal_data, 'sites_generation_layer', yaml_path, 'crystal.sites_generation_layer')
      )
    )
    
    # =========================================================================
    # Load Component Files
    # =========================================================================
    components = _get_required(data, 'components', yaml_path, 'components')
    
    # --- DEFECTS ---
    defects_path = _get_required(components, 'defects', yaml_path, 'components.defects')
    defects_path = base_path / defects_path
    try: 
      config.defects = DefectsConfig.from_yaml(defects_path)
      print(f"Loaded defects from {defects_path}")
      print(f"{len(config.defects.defects)} defect species")
    except Exception as e:
      print(f"Failed to load defects: {e}")

    # --- REACTIONS ---
    if 'reactions' in components:
      reactions_path = base_path / components['reactions']
      
    # --- GRAIN BOUNDARIES ---
    if 'grain_boundaries' in components:
      gb_path = base_path / components['grain_boundaries']
      
    # =========================================================================
    # Load Electrical Configuration
    # =========================================================================
    if 'electrical' in components:
      electrical_path = base_path / components['electrical']
      try:
        crystal_size = config.material.structure.size
        config.electrical = ElectricalConfig.from_yaml(
          electrical_path,
          crystal_size=crystal_size
        ) 
        print(f"Loaded electrical from {electrical_path}")
      except Exception as e:
        print(f"Failed to load electrical: {e}")
    
    # =========================================================================
    # Load Poisson Solver Configuration (OPTIONAL)
    # =========================================================================
    poisson_data = data.get('poisson')
    if poisson_data:
      config.poisson = PoissonSolverConfig(
        solve_Poisson=_get_required(poisson_data, 'solve_Poisson', yaml_path, 'poisson.solve_Poisson'),
        save_Poisson=poisson_data.get('save_Poisson', False),
        active_dipoles=_get_required(poisson_data, 'active_dipoles', yaml_path, 'poisson.active_dipoles'),
        screening_factor=poisson_data.get('screening_factor',0.01),
        conductivity_CF=_get_required(poisson_data, 'conductivity_CF', yaml_path, 'poisson.conductivity_CF'),
        conductivity_dielectric=_get_required(poisson_data, 'conductivity_dielectric', yaml_path, 'poisson.conductivity_dielectric')
      )
      print("Poisson solver config loaded")
    else:
      print("Poisson solver: Not configured")
      
    # =========================================================================
    # Load Heat Solver Configuration (OPTIONAL)
    # =========================================================================
    heat_data = data.get('heat')
    if heat_data:
      config.heat = HeatSolverConfig(
        solve_heat=_get_required(heat_data,'solve_heat', yaml_path, 'heat.solve_heat'),
        thermal_conductivity=_get_required(heat_data,'thermal_conductivity',yaml_path, 'heat.thermal_conductivity'),
        heat_capacity=_get_required(heat_data,'heat_capacity',yaml_path, 'heat.heat_capacity'),
        density=_get_required(heat_data,'density',yaml_path, 'heat.density'),
      
      )
      
    # =========================================================================
    # Load Heat Solver Configuration (OPTIONAL)
    # =========================================================================
    superbasin_data = data.get('superbasin')
    if superbasin_data:
      config.superbasin = SuperbasinConfig(
        enabled_superbasin=_get_required(superbasin_data, 'enabled_superbasin', yaml_path, 'superbasin.enabled_superbasin'),
        n_search_superbasin=_get_required(superbasin_data, 'n_search_superbasin', yaml_path, 'superbasin.n_search_superbasin'),
        time_step_limits=_get_required(superbasin_data, 'time_step_limits', yaml_path, 'superbasin.time_step_limits'),
        E_min=_get_required(superbasin_data, 'E_min', yaml_path, 'superbasin.E_min'),
        energy_step=_get_required(superbasin_data, 'energy_step', yaml_path, 'superbasin.energy_step'),
        time_based_superbasin=_get_required(superbasin_data, 'time_based_superbasin', yaml_path, 'superbasin.time_based_superbasin'),
      )
      
    # =========================================================================
    # Load Simulation Settings (REQUIRED)
    # =========================================================================
    settings_data = _get_required(data, 'settings', yaml_path, 'settings')
    print(settings_data)
    config.settings = SimulationSettings(
      simulation_type=settings_data.get('simulation_type'),
      technology=settings_data.get('technology'),
      mode=settings_data.get('mode'),
      save_data=_get_required(settings_data, 'save_data', yaml_path, 'settings.save_data'),
      snapshoots_steps=settings_data.get('snapshoots_steps'),
      seed_rng=settings_data.get('seed_rng'),
      lammps_output=settings_data.get('lammps_output', True),
    )
    
    # =========================================================================
    # Load Experimental Conditions
    # =========================================================================
    experimental_data = _get_required(data, 'experimental', yaml_path, 'experimental')
    config.experimental = ExperimentalConditions(
            temperature=_get_required(experimental_data, 'temperature', yaml_path, 'experimental.temperature')
    )
    
    return config
    
        
def _get_required(data: dict, key: str, yaml_path: Path, field_name: str) -> Any:
  """
  Get required field from dictionary, raise clear error if missing.
    
  Args:
    data: Dictionary to search
    key: Key to look for
    yaml_path: Path to YAML file (for error message)
    field_name: Human-readable field name (for error message)
    
  Returns:
    Value if found
    
  Raises:
    ConfigValidationError: If field is missing or None
  """
  value = data.get(key)
    
  if value is None:
    raise ConfigValidationError(
      f"Missing required field '{field_name}' in {yaml_path}\n\n"
      f"  Please add '{key}: <value>' to your YAML file"
    )
    
  return value
    