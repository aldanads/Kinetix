# kinetix/configs/calculator_config.py
"""Activation-energy calculator configuration (MACE CI-NEB / tabulated).

Pure dataclasses only - no torch/mace imports at module level.
Actual MACE model loading happens lazily in kinetix/calculators/mace_neb.py.
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class InterstitialRefinementConfig:
  """Parameters for refining interstitial sites before grid generation."""
  enabled: bool = False
  displacement_threshold: float = 0.5  # Angstroms

  def to_dict(self) -> Dict[str, object]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'enabled': self.enabled,
      'displacement_threshold': self.displacement_threshold,
    }

@dataclass
class CalculatorConfig:
  """Activation-energy calculator parameters.

  type: "mace_neb" or "tabulated"
  model: Local path or HF repo ID
  cluster: {"R_active": 5.0, "R_shell": 7.0} or None for full periodic NEB
  """
  type: str = "tabulated"              # "mace_neb" or "tabulated"
  model: str = ""                      # Local path or HF repo ID
  n_images: int = 5
  fmax: float = 0.05                   # eV/Å
  max_steps: int = 300
  device: str = "cpu"                  # "cpu" or "cuda"
  default_dtype: str = "float64"
  cluster: Optional[Dict[str, float]] = None  # {"R_active": 5.0, "R_shell": 7.0}
  cache_dir: str = "data/cache/neb_cache"
  interstitial_refinement: Optional[InterstitialRefinementConfig] = None

  def to_dict(self) -> Dict[str, object]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'type': self.type,
      'model': self.model,
      'n_images': self.n_images,
      'fmax': self.fmax,
      'max_steps': self.max_steps,
      'device': self.device,
      'default_dtype': self.default_dtype,
      'cluster': self.cluster,
      'cache_dir': self.cache_dir,
      'interstitial_refinement': (
        self.interstitial_refinement.to_dict() if self.interstitial_refinement else None
      ),
    }