# kinetix/configs/reaction_config.py
"""Reaction configuration dataclasses."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

@dataclass
class ReactionSpecies:
  """A species in a reaction (reactant or product)."""
  symbol: str
  sublattice: str
  site_index: Any = None  # int, 'neighbor', or None
  key: Optional[str] = None
  passivation_increment: int = 0
  min_passivation: int = 0
  
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'ReactionSpecies':
    """Create from dictionary (loaded from YAML)"""
    return cls(
      symbol=data.get('symbol'),
      sublattice=data.get('sublattice',''),
      site_index=data.get('site_index'),
      key=data.get('key'),
      passivation_increment=data.get('passivation_increment',0),
      min_passivation=data.get('min_passivation',0),
    )
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for YAML serialization"""
    result = {
      'symbol': self.symbol,
      'sublattice': self.sublattice,
    }
    if self.site_index is not None:
      result['site_index'] = self.site_index
    if self.key is not None:
      result['key'] = self.key
    if self.passivation_increment != 0:
      result['passivation_increment'] = self.passivation_increment
    if self.min_passivation != 0:
      result['min_passivation'] = self.min_passivation
      
    return result
    

@dataclass
class ReactionConfig:
  """Configuration for a single reaction."""
  name: str
  type: str  # 'bimolecular_neighbor', 'bimolecular_capture', 'unimolecular_escape'
  reactants: List[ReactionSpecies]
  products: List[ReactionSpecies]
  enabled: bool = True
  
  @classmethod
  def from_dict(cls, name: str, data: Dict[str, Any]) -> 'ReactionConfig':
    """Create from dictionary (loaded from YAML)"""
    reactants = [ReactionSpecies.from_dict(r) for r in data.get('reactants', [])]
    products = [ReactionSpecies.from_dict(p) for p in data.get('products', [])]
    
    return cls(
      name=data['name'],
      type=data['type'],
      reactants=reactants,
      products=products,
      enabled=data['enabled']
    )
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for backwards compatibility"""
    return {
      'name': self.name,
      'type': self.type,
      'reactants': [r.to_dict() for r in self.reactants],
      'products': [p.to_dict() for p in self.products],
      'enabled': self.enabled,
    }

@dataclass
class ReactionsConfig:
  """Collection of all reaction configurations."""
  reactions: Dict[str, ReactionConfig] = field(default_factory=dict)
  description: str = ""
    
  def add_reaction(self, key: str, reaction: ReactionConfig):
    """Add a reaction configuration"""
    self.reactions[key] = reaction
    
  def to_dict(self) -> Dict[str, Dict[str, Any]]:
    """Convert all reactions to dictionary"""
    return {name: reaction.to_dict() for name, reaction in self.reactions.items()}
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> 'ReactionsConfig':
    """
    Load reaction configurations from YAML file.
        
    Args:
      yaml_path: Path to reactions YAML file
        
    Returns:
      ReactionsConfig with all reactions loaded
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
      raise FileNotFoundError(f'Reaction config file not found: {yaml_path}')
      
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
      
    config = cls(
      description=data.get('metadata',{}).get('description', '')
    )
    
    # Load each reaction from YAML
    reactions_data = data.get('reactions', {})
    for reaction_name, reaction_dict in reactions_data.items():
      reaction = ReactionConfig.from_dict(reaction_name, reaction_dict)
      config.add_reaction(reaction_name, reaction)
      
    return config