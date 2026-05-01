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
def load_activation_energies(file_type: str = 'memristors') -> Dict[str, Any]:
  """
  Load activation energy parameters.
    
  Args:
    file_type: 'memristors' or 'deposition'
    
  Returns:
    Dictionary with activation energy data
  """
  parameters_root = get_parameters_root()
  
  activation_energies_root = parameters_root / 'activation_energies'
  
  file_mapping = {
    'memristors': 'activation_energies_memristors.json',
    'deposition': 'activation_energies_deposition.json'
  }
  
  if file_type not in file_mapping:
    raise ValueError(f"Unknown file_type: {file_type}. Choose from {list(file_mapping.keys())}")
    
  filename = file_mapping[file_type]
  file_path = activation_energies_root / filename
  
  if not file_path.exists():
    raise FileNotFoundError(
      f"Activation energy file not found at {file_path}\n"
      f"Please ensure the file exists in data/parameters/"
    )
    
  with open(file_path, 'r') as f:
    data = json.load(f)
    
  return data
  
def get_activation_energies_memristors() -> Dict[str,Any]:
  """Convenience function for memristor activation energies"""
  return load_activation_energies('memristors')
  
def get_activation_energies_deposition() -> Dict[str, Any]:
  """Convenience function for deposition activation energies"""
  return load_activation_energies('deposition')