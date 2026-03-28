# kinetix/configs/defect_config.py
"""Defect configuration dataclasses."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class SiteType(Enum):
  INTERSTITIAL = "interstitial"
  SUBLATTICE = "sublattice"

class EventType(Enum):
  MIGRATION = "migration"
  REACTION = "reaction"
  REDUCTION = "reduction"
  OXIDATION = "oxidation"
    
@dataclass
class DefectConfig:
  """Configuration for a single defect species."""
  name: str
  symbol: str
  charge: int
  site_type: str  # 'interstitial' or sublattice name
  allowed_sublattices: List[str]
  initial_concentration_bulk: float = 0.0
  initial_concentration_GB: float = 0.0
  valid_target_species: List[str] = field(default_factory=list)
  activation_energies_key: str = ""
  enabled_events: List[str] = field(default_factory=list)
  CN_matters: bool = False
  sites_generation_layer: Optional[str] = None
  description: str = ""
    
  # Passivation (for vacancies)
  passivation_level: int = 0
  max_passivation_level: int = 0
  charge_per_passivation: int = 0
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'symbol': self.symbol,
      'charge': self.charge,
      'site_type': self.site_type,
      'allowed_sublattices': self.allowed_sublattices,
      'initial_concentration_bulk': self.initial_concentration_bulk,
      'initial_concentration_GB': self.initial_concentration_GB,
      'valid_target_species': self.valid_target_species,
      'activation_energies_key': self.activation_energies_key,
      'enabled_events': self.enabled_events,
      'CN_matters': self.CN_matters,
      'sites_generation_layer': self.sites_generation_layer,
      'description': self.description,
      'passivation_level': self.passivation_level,
      'max_passivation_level': self.max_passivation_level,
      'charge_per_passivation': self.charge_per_passivation,
    }

@dataclass
class DefectsConfig:
  """Collection of all defect configurations."""
  defects: Dict[str, DefectConfig] = field(default_factory=dict)
    
  def add_defect(self, defect: DefectConfig):
    """Add a defect configuration"""
    self.defects[defect.name] = defect
    
  def to_dict(self) -> Dict[str, Dict[str, Any]]:
    """Convert all defects to dictionary"""
    return {name: defect.to_dict() for name, defect in self.defects.items()}