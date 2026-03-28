# kinetix/configs/reaction_config.py
"""Reaction configuration dataclasses."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ReactionSpecies:
  """A species in a reaction (reactant or product)."""
  symbol: str
  sublattice: str
  site_index: Any = None  # int, 'neighbor', or None
  key: Optional[str] = None
  passivation_increment: int = 0

@dataclass
class ReactionConfig:
  """Configuration for a single reaction."""
  name: str
  type: str  # 'bimolecular_neighbor', 'bimolecular_capture', 'unimolecular_escape'
  reactants: List[ReactionSpecies]
  products: List[ReactionSpecies]
  enabled: bool = True
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'name': self.name,
      'type': self.type,
      'reactants': [
        {
          'symbol': r.symbol,
          'sublattice': r.sublattice,
          'site_index': r.site_index,
          'key': r.key,
          'passivation_increment': r.passivation_increment,
        }
        for r in self.reactants
      ],
      'products': [
        {
          'symbol': p.symbol,
          'sublattice': p.sublattice,
          'site_index': p.site_index,
          'key': p.key,
          'passivation_increment': p.passivation_increment,
        }
        for p in self.products
      ],
      'enabled': self.enabled,
    }

@dataclass
class ReactionsConfig:
  """Collection of all reaction configurations."""
  reactions: Dict[str, ReactionConfig] = field(default_factory=dict)
    
  def add_reaction(self, reaction: ReactionConfig):
    """Add a reaction configuration"""
    self.reactions[reaction.name] = reaction
    
  def to_dict(self) -> Dict[str, Dict[str, Any]]:
    """Convert all reactions to dictionary"""
    return {name: reaction.to_dict() for name, reaction in self.reactions.items()}