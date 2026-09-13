# kinetix/configs/reaction_config.py
"""Reaction configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import yaml

@dataclass
class ReactionSpecies:
  """A species in a reaction (reactant or product).

  Attributes:
    symbol: Chemical symbol of the species.
    sublattice: Sublattice the species occupies.
    site_index: Site selector: an integer index, 'neighbor', or None.
    key: Optional unique key for the species instance.
    passivation_increment: Change in passivation level applied by the event.
    min_passivation: Minimum passivation level required for the event.
  """
  symbol: str
  sublattice: str
  site_index: Any = None  # int, 'neighbor', or None
  key: str | None = None
  passivation_increment: int = 0
  min_passivation: int = 0
  
  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> ReactionSpecies:
    """Create a ReactionSpecies from a dictionary (loaded from YAML).

    Args:
      data: Mapping of species fields; missing keys fall back to defaults.

    Returns:
      The populated ReactionSpecies instance.
    """
    return cls(
      symbol=data.get('symbol'),
      sublattice=data.get('sublattice',''),
      site_index=data.get('site_index'),
      key=data.get('key'),
      passivation_increment=data.get('passivation_increment',0),
      min_passivation=data.get('min_passivation',0),
    )
    
  def to_dict(self) -> dict[str, Any]:
    """Convert the species to a dictionary for YAML serialization.

    Returns:
      Dictionary of fields; optional fields are omitted when unset.
    """
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
  """Configuration for a single reaction.

  Attributes:
    name: Unique identifier of the reaction.
    type: One of 'bimolecular_neighbor', 'bimolecular_capture',
      'unimolecular_escape'.
    reactants: Species consumed by the reaction.
    products: Species produced by the reaction.
    enabled: Whether the reaction is active.
    field_dependent: Whether the reaction rate depends on the E-field.
    field_coupling: Coupling factor between E-field and reaction rate.
    sites_removal_layer: Which layer's sites are removed for this reaction.
  """
  name: str
  type: str  # 'bimolecular_neighbor', 'bimolecular_capture', 'unimolecular_escape'
  reactants: list[ReactionSpecies]
  products: list[ReactionSpecies]
  enabled: bool = True
  field_dependent: bool = True
  field_coupling: float = 1.0
  sites_removal_layer: str = "bottom_layer"
  
  @classmethod
  def from_dict(cls, name: str, data: dict[str, Any]) -> ReactionConfig:
    """Create a ReactionConfig from a dictionary (loaded from YAML).

    Args:
      name: Name of the reaction (unused; the name is taken from
        ``data['name']``).
      data: Mapping of reaction fields; ``reactants`` and ``products``
        entries are converted to ReactionSpecies objects.

    Returns:
      The populated ReactionConfig instance.
    """
    reactants = [ReactionSpecies.from_dict(r) for r in data.get('reactants', [])]
    products = [ReactionSpecies.from_dict(p) for p in data.get('products', [])]
    
    return cls(
      name=data['name'],
      type=data['type'],
      reactants=reactants,
      products=products,
      enabled=data['enabled'],
      field_dependent=data.get('field_dependent'),
      field_coupling=data.get('field_coupling'),
      sites_removal_layer=data.get('sites_removal_layer')
    )
    
  def to_dict(self) -> dict[str, Any]:
    """Convert the reaction to a plain dictionary.

    Returns:
      Dictionary of all reaction fields, with reactants/products serialized
      via their own ``to_dict`` methods.
    """
    return {
      'name': self.name,
      'type': self.type,
      'reactants': [r.to_dict() for r in self.reactants],
      'products': [p.to_dict() for p in self.products],
      'enabled': self.enabled,
      'field_dependent': self.field_dependent,
      'field_coupling': self.field_coupling,
      'sites_removal_layer': self.sites_removal_layer
    }

@dataclass
class ReactionsConfig:
  """Collection of all reaction configurations.

  Attributes:
    reactions: Mapping from reaction name to its configuration.
    description: Free-text description taken from the YAML metadata.
  """
  reactions: dict[str, ReactionConfig] = field(default_factory=dict)
  description: str = ""
    
  def add_reaction(self, key: str, reaction: ReactionConfig) -> None:
    """Register a reaction configuration.

    Args:
      key: Name to register the reaction under.
      reaction: Reaction configuration to add.
    """
    self.reactions[key] = reaction
    
  def to_dict(self) -> dict[str, dict[str, Any]]:
    """Convert all reactions to a dictionary.

    Returns:
      Mapping from reaction name to that reaction's dictionary form.
    """
    return {name: reaction.to_dict() for name, reaction in self.reactions.items()}
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> ReactionsConfig:
    """
    Load reaction configurations from YAML file.

    Args:
      yaml_path: Path to reactions YAML file.

    Returns:
      ReactionsConfig with all reactions loaded.

    Raises:
      FileNotFoundError: If the YAML file does not exist.
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