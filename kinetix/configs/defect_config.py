# kinetix/configs/defect_config.py
"""Defect configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from pathlib import Path
import yaml

class SiteType(Enum):
  """Type of lattice site a defect can occupy."""
  INTERSTITIAL = "interstitial"
  SUBLATTICE = "sublattice"

class EventType(Enum):
  """KMC event types a defect can participate in."""
  MIGRATION = "migration"
  REACTION = "reaction"
  REDUCTION = "reduction"
  OXIDATION = "oxidation"
    
@dataclass
class DefectConfig:
  """Configuration for a single defect species.

  Attributes:
    name: Unique identifier for the defect species.
    symbol: Short symbol used to identify the defect in state keys.
    charge: Charge of the defect in units of the elementary charge.
    site_type: Either 'interstitial' or the name of a sublattice.
    allowed_sublattices: Sublattice names the defect is allowed to occupy.
    physical_element: Element label for atomistic calculators (MACE);
      None means a pseudo-particle such as a vacancy.
    initial_concentration_bulk: Initial bulk occupancy fraction.
    initial_concentration_GB: Initial grain-boundary occupancy fraction.
    valid_target_species: Symbols of species this defect can react with.
    activation_energies_key: Key into the activation-energies JSON file.
    enabled_events: Event types (EventType values) enabled for this defect.
    CN_matters: Whether coordination number affects this defect's rates.
    sites_generation_layer: Layer in which interstitial sites are generated.
    interface_tolerance_generation: Tolerance (nm) for interface site
      generation.
    migrating_attributes: Site attributes tracked for migrating defects.
    field_dependent_generation: Whether generation depends on the E-field.
    electrode_scavenging: Whether the defect is scavenged at electrodes.
    description: Free-text description taken from the YAML file.
  """
  name: str
  symbol: str
  charge: int
  site_type: str  # 'interstitial' or sublattice name
  allowed_sublattices: list[str]
  physical_element: str | None = None  # element label for atomistic calculators (MACE); None = pseudo-particle (e.g. vacancy)
  initial_concentration_bulk: float = 0.0
  initial_concentration_GB: float = 0.0
  valid_target_species: list[str] = field(default_factory=list)
  activation_energies_key: str = ""
  enabled_events: list[str] = field(default_factory=list)
  CN_matters: bool = False
  sites_generation_layer: str | None = None
  interface_tolerance_generation: float | None = 0.0
  migrating_attributes: list[str] = field(default_factory=list)
  field_dependent_generation: bool = False
  electrode_scavenging: bool = False
  description: str = ""
    
  # Passivation (for vacancies) - OPTIONAL: only for defects that can be passivated
  passivation_level: int | None = None
  max_passivation_level: int | None = None
  charge_per_passivation: int | None = None
    
  def to_dict(self) -> dict[str, Any]:
    """Convert the defect configuration to a plain dictionary.

    Returns:
      Dictionary of all scalar and list fields. Passivation fields are
      included only when ``charge_per_passivation`` is set.
    """
    result = {
      'symbol': self.symbol,
      'physical_element': self.physical_element,
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
      'interface_tolerance_generation': self.interface_tolerance_generation,
      'migrating_attributes': self.migrating_attributes,
      'field_dependent_generation': self.field_dependent_generation,
      'electrode_scavenging': self.electrode_scavenging,
      'description': self.description,
    }
    
    if self.charge_per_passivation is not None:
      result['passivation_level'] = self.passivation_level
      result['max_passivation_level'] = self.max_passivation_level
      result['charge_per_passivation'] = self.charge_per_passivation
      
    return result
    
  @classmethod
  def from_dict(cls, name: str, data: dict[str, Any]) -> DefectConfig:
    """Create a DefectConfig from a dictionary (e.g. loaded YAML).

    Args:
      name: Name to assign to the resulting defect.
      data: Mapping of defect fields; missing keys fall back to defaults.

    Returns:
      The populated DefectConfig instance.
    """
    return cls(
      name=name,
      symbol=data.get('symbol', ''),
      physical_element=data.get('physical_element'),
      charge=data.get('charge', 0),
      site_type=data.get('site_type', 'interstitial'),
      allowed_sublattices=data.get('allowed_sublattices', []),
      initial_concentration_bulk=data.get('initial_concentration_bulk', 0.0),
      initial_concentration_GB=data.get('initial_concentration_GB', 0.0),
      valid_target_species=data.get('valid_target_species', []),
      activation_energies_key=data.get('activation_energies_key', ''),
      enabled_events=data.get('enabled_events', []),
      CN_matters=data.get('CN_matters', False),
      sites_generation_layer=data.get('sites_generation_layer'),
      interface_tolerance_generation=data.get('interface_tolerance_generation'),
      migrating_attributes=data.get('migrating_attributes'),
      field_dependent_generation=data.get('field_dependent_generation'),
      electrode_scavenging=data.get('electrode_scavenging'),
      description=data.get('description', ''),
      passivation_level=data.get('passivation_level'),
      max_passivation_level=data.get('max_passivation_level'),
      charge_per_passivation=data.get('charge_per_passivation'),
    )

@dataclass
class DefectsConfig:
  """Collection of all defect configurations.

  Attributes:
    defects: Mapping from defect name to its configuration.
  """
  defects: dict[str, DefectConfig] = field(default_factory=dict)
    
  def add_defect(self, defect: DefectConfig) -> None:
    """Register a defect configuration.

    Args:
      defect: Defect configuration to add, keyed by ``defect.name``.
    """
    self.defects[defect.name] = defect
    
  def to_dict(self) -> dict[str, dict[str, Any]]:
    """Convert all defects to a dictionary.

    Returns:
      Mapping from defect name to that defect's dictionary form.
    """
    return {name: defect.to_dict() for name, defect in self.defects.items()}
    
  @classmethod
  def from_yaml(cls, yaml_path: Path) -> DefectsConfig:
    """
    Load defect configurations from YAML file.

    Args:
      yaml_path: Path to defects_config.yaml.

    Returns:
      DefectsConfig object with all defects loaded.

    Raises:
      FileNotFoundError: If the YAML file does not exist.
    """
    yaml_path = Path(yaml_path)
        
    if not yaml_path.exists():
      raise FileNotFoundError(f"Defect config file not found: {yaml_path}")
        
    with open(yaml_path, 'r') as f:
      data = yaml.safe_load(f)
        
    # Create empty config object
    config = cls()
    
    # Load each defect from YAML
    defects_data = data.get('defects', {})
    for defect_name, defect_dict in defects_data.items():
      defect = DefectConfig.from_dict(defect_name, defect_dict)
      config.add_defect(defect)
        
    return config