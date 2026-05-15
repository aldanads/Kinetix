# kinetix/config_loader.py
"""
Configuration file loader for runtime settings (API keys, paths, etc.)
"""

from pathlib import Path
import json
from typing import Any, Dict

# =============================================================================
# Path Helpers
# =============================================================================

def get_project_root() -> Path:
  """Get the project root directory (parent of kinetix/)"""
  return Path(__file__).parent.parent.parent
  
def get_data_root() -> Path:
  """Get the data directory where parameter files are stored"""
  return get_project_root() / 'data'

def get_parameters_root() -> Path:
  """Get the parameters subdirectory"""
  return get_data_root() / 'parameters'
  
def get_grids_root() -> Path:
  """Get the grids subdirectory"""
  return get_data_root() / 'grids'
  
def get_mesh_root() -> Path:
  """Get the mesh subdirectory"""
  return get_data_root() / 'mesh'
  
# =============================================================================
# Config File Loading (config.json)
# =============================================================================
  
def get_config_path() -> Path:
  """Get the path to config.json"""
  return get_project_root() / 'config.json'
  
def load_config() -> Dict[str, Any]:
  """Load and return the configuration dictionary"""
  config_path = get_config_path()
  
  if not config_path.exists():
    raise FileNotFoundError(
      f'Config file not found at {config_path}\n'
      f'Please create config.json in the project root with your API key'
    )
    
  with open(config_path, 'r') as f:
    config = json.load(f)
    
  # Validate required keys
  required_keys = ['api_key']
  missing = [key for key in required_keys if key not in config]
  if missing:
    raise KeyError(f'Missing required config keys: {missing}')
  
  return config
  
def get_api_key() -> str:
    """Convenience function to get just the API key"""
    config = load_config()
    return config['api_key']
    
    
# =============================================================================
# Activation Energy Loading
# =============================================================================
def load_activation_energies(preset_path: Path, config_settings) -> Dict[str, Any]:
  """
    Load activation energies from the file specified in the simulation settings.
    
  Args:
    preset_path: Path to the loaded YAML preset (used to resolve relative paths)
    config_settings: SimulationSettings dataclass instance
    
  Returns:
    Dictionary with structure: {"PZT": [{"specie": "H", ...}, ...]}
  """
  if not config_settings.activation_energies:
    raise ValueError("No activation energies file specified in settings")
    
  # Resolve relative to data/parameters/ (parent of presets/)
  base_path = preset_path.parent.parent
  ae_path = base_path / config_settings.activation_energies

  if not ae_path.exists():
    raise FileNotFoundError(f'Activation energies file not found: {ae_path}')
    
  with open(ae_path, 'r') as f:
    ae_data = json.load(f)
    
  return ae_data