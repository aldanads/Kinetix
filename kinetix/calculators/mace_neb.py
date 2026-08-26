"""mace_neb.py — MACE CI-NEB barrier calculator with model & barrier caching."""
from pathlib import Path
import numpy as np

class MACENEBBarrierCalculator:
  """Loads a MACE model once, builds NEB bands (periodic or cluster),
  computes barriers, caches everything."""
  
  def __init__(self, model_source, cache_dir="data/cache/neb_cache", device "cuda",
               default_dtype="float64", n_images=5, fmax=0.05, max_steps=300,
               cluster=None, model_filename="model.model"):
               
    """
    model_source : local path to .model file, or Hugging Face repo ID
    cluster      : None -> full periodic NEB (reference mode)
                   dict(R_active=5.0, R_shell=7.0) -> frozen-shell cluster
    """
    self.device, self.dtype = device, default_dtype
    self.n_images, self.fmax, self.max_steps = n_images, fmax, max_steps
    self.cluster = cluster
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Resolve & load model
    p = Path(model_source)
    if p.exists():
      self.model_path = p
    else:
      from huggingface_hub import hf_hub_download
      self.model_path = Path(hf_hub_download(
        repo_id=model_source, filename=model_filename,
        cache_dir=cache_dir / "hf"))
    self.model_id = self.model_path.name