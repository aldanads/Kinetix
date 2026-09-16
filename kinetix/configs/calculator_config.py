# kinetix/configs/calculator_config.py
"""Activation-energy calculator configuration (MACE CI-NEB / tabulated).

Pure dataclasses only - no torch/mace imports at module level.
Actual MACE model loading happens lazily in kinetix/calculators/mace_neb.py.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class InterstitialRefinementConfig:
  """Parameters for refining interstitial sites before grid generation.

  Attributes:
      enabled: Whether interstitial refinement is performed.
      displacement_threshold: Displacement cutoff in Angstroms.
  """
  enabled: bool = False
  displacement_threshold: float = 0.5  # Angstroms

  def to_dict(self) -> dict[str, object]:
    """Convert to a plain dictionary for backwards compatibility.

    Returns:
        Dictionary with the keys "enabled" and "displacement_threshold".
    """
    return {
      'enabled': self.enabled,
      'displacement_threshold': self.displacement_threshold,
    }

@dataclass
class CalculatorConfig:
  """Activation-energy calculator parameters.

  Attributes:
      type: Calculator type, either "mace_neb" or "tabulated".
      model: Local model path or Hugging Face repository ID.
      model_filename: Filename of the model inside the HF repo, used only
          when ``model`` is a repository ID.
      n_images: Number of NEB images.
      fmax: Force convergence criterion in eV/Angstrom.
      max_steps: Maximum optimizer steps for the NEB relaxation.
      device: Torch device, "cpu" or "cuda".
      default_dtype: Torch default dtype name (e.g. "float64").
      cluster: Cluster radii such as {"R_active": 5.0, "R_shell": 7.0},
          or None for a full periodic NEB.
      cache_dir: Directory for caching computed barriers.
      interstitial_refinement: Optional interstitial refinement settings.
  """
  type: str = "tabulated"              # "mace_neb" or "tabulated"
  model: str = ""                      # Local path or HF repo ID
  model_filename: str = "model.model"  # Model filename in case it is fetching from HF repo
  n_images: int = 5
  fmax: float = 0.05                   # eV/Å
  max_steps: int = 300
  device: str = "cpu"                  # "cpu" or "cuda"
  default_dtype: str = "float64"
  cluster: dict[str, float] | None = None  # {"R_active": 5.0, "R_shell": 7.0}
  cache_dir: str = "data/cache/neb_cache"
  interstitial_refinement: InterstitialRefinementConfig | None = None

  def to_dict(self) -> dict[str, object]:
    """Convert to a plain dictionary for backwards compatibility.

    Returns:
        Dictionary with all calculator fields; `interstitial_refinement`
        is nested as its own dictionary (or None).
    """
    return {
      'type': self.type,
      'model': self.model,
      'model_filename': self.model_filename,
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