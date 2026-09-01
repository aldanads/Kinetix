# kinetix/calculators/base.py
"""Abstract interface for activation-energy providers in Kinetix.

Contract
--------
"get_barrier" returns the TOTAL effective barrier (eV) for the transition
in the CURRENT local environment, i.e. it REPLACES
"Act_E_dict[event] + energy_change".  Electric-field and temperature
corrections remain the responsibility of "Site.transition_rates".
"""

from abc import ABC, abstractmethod
from typing import Optional

class ActivationEnergyCalculator(ABC):
  
  @abstractmethod
  def get_barrier(self, lattice, origin_idx, dest_idx, event_id=None) -> float:
    """Activation energy (eV) for the hop origin -> dest in lattice"""
    
  def uncertainty(self, lattice, origin_idx, dest_idx, event_id=None) -> Optional[float]:
    """Optional prediction uncertainty; None = not available."""
    return None
    
  def warmup(self, lattice, transitions) -> None:
    """Optional hook to pre-compute barrier for a list of (origin_idx, dest_idx, event_id). 
    Default: sequential calls."""
    for origin, dest, event_id in transitions:
      self.get_barrier(lattice, origin, dest, event_id)