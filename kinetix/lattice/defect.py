# kinetix/lattice/defect.py
"""Defect state objects for the Site/Defect composition model.

This module introduces the target data model for the site/defect decoupling
refactor (Phase 1). The classes defined here are NOT yet wired into the
simulation: Site, KMCSimulator and the kMC loop still operate on the
existing defects_config dict flow. These objects are introduced alongside the
old code so later phases can migrate to them incrementally.

Design rules (see the site/defect decoupling investigation report, Part 4):

* Every Site will hold exactly one Defect (never None, never shared between
  sites). Empty sites get a fresh Defect built from EMPTY_DEFECT_CONFIG.
* Defect carries all defect-carried dynamic state (charge, passivation_level,
  events) plus its immutable DefectConfig reference.
* DefectConfig stays config-time-immutable; ``enabled_events`` (here exposed
  as Defect.activities) never changes during a simulation.
* Event replaces the raw ``[rate, destination, event_label, E_act]`` list.
  ``rate`` is site-local (it depends on the site's temperature and electric
  field), which is why Event instances must never be shared between sites.
* EMPTY_DEFECT_CONFIG is a template constant only. It must never be inserted
  into a defects registry, into ``DefectsConfig``, or handed to Site — it
  would otherwise leak into registry-driven code paths (_active_site_types,
  defect_gen, species id generation, MACE species maps, state loading).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

from kinetix.configs.defect_config import DefectConfig

# Event labels: an int is a migration label (index into the lattice's unique
# displacement-vector table); a str is an event name ('generation', 'reduction',
# 'oxidation', or a reaction name from reactions_config).
EventLabel = int | str


@dataclass(slots=True)
class Event:
    """One kMC event owned by a Defect (never shared between sites).

    Attributes:
        label: Migration int label, or an event-name string ('generation',
            'reduction', 'oxidation', or a reaction name).
        destination: Destination site index for migration and neighbour
            reactions; the owning site's index for redox/generation events
            (matches the current site_events convention).
        barrier: Activation energy E_act as registered by available_pathways,
            i.e. BEFORE per-step field/GB correction in transition_rates.
        rate: Computed by Site.transition_rates. 0.0 means 'not yet rated';
            rating either inserts (3-element legacy shape) or overwrites
            position 0 (4-element legacy shape).
        generates: Generation events only: registry key of the DefectConfig
            to instantiate when the event fires. None for all other kinds.
    """

    label: EventLabel
    destination: Hashable
    barrier: float
    rate: float = 0.0
    generates: str | None = None

    @property
    def is_migration(self) -> bool:
        """True if this is a migration event (int label convention)."""
        return isinstance(self.label, int)

    def catalog_tuple(self, owner_idx: Hashable) -> tuple:
        """Adapter for the hot-path kMC catalog format.

        KMCSimulator._kmc_step builds (rate, destination, label, owner)
        tuples to feed the balanced tree; this keeps that representation (and
        therefore its performance characteristics) unchanged during Phase 4.
        """
        return (self.rate, self.destination, self.label, owner_idx)


@dataclass(slots=True)
class Defect:
    """All dynamic, defect-carried state. Exactly one instance per site.

    Attributes:
        config: Shared, config-time-immutable DefectConfig. Never mutated at
            runtime; lookup key into the lattice's defects registry.
        chemical_specie: Symbol of the species currently occupying the site.
        charge: Runtime charge; changes with redox reactions, grain-boundary
            entry/exit and passivation charge transfer.
        passivation_level: Runtime passivation state; changes with
            passivation reactions. Normalized to 0 even when the underlying
            DefectConfig declares passivation_level=None (removes the
            conditional-attribute gap of the current Site model).
        events: This defect's Event objects. Rebuilt per site by
            available_pathways on the dirty-site schedule; per-site mutable,
            hence never shared.
    """

    config: DefectConfig
    chemical_specie: str
    charge: int = 0
    passivation_level: int = 0
    events: list[Event] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Registry key of this defect's configuration."""
        return self.config.name

    @property
    def is_empty(self) -> bool:
        """True for the empty-defect placeholder occupying a vacant site."""
        return self.config.is_empty

    @property
    def sublattice(self) -> str | None:
        """Sublattice/interstitial kind this defect occupies (config)."""
        return self.config.site_type

    @property
    def activities(self) -> tuple[str, ...]:
        """Config-time enabled event kinds; fixed for the defect's lifetime."""
        return tuple(self.config.enabled_events)

    @classmethod
    def from_config(cls,
                    cfg: DefectConfig,
                    *,
                    charge: int | None = None,
                    passivation_level: int | None = None) -> "Defect":
        """Build a runtime Defect from its configuration.

        Args:
            cfg: The (shared, immutable) DefectConfig to reference.
            charge: Runtime charge override; defaults to the configured
                charge (which is itself the hook for GB entry/exit rules).
            passivation_level: Runtime passivation override; defaults to the
                configured initial passivation level, normalized to 0 when
                the config declares None.
        """
        return cls(
            config=cfg,
            chemical_specie=cfg.symbol,
            charge=cfg.charge if charge is None else charge,
            passivation_level=(cfg.passivation_level or 0)
                              if passivation_level is None
                              else passivation_level,
        )


# ---------------------------------------------------------------------------
# Empty-defect template. A config-time constant, NOT a registry entry: it must
# never be added to DefectsConfig or any defects_config dict, because
# registry-driven code paths (active-site classification, generation-site
# scanning, species id generation, MACE species maps, state loading) would
# then treat 'Empty' as a defect species.
# ---------------------------------------------------------------------------
EMPTY_DEFECT_CONFIG = DefectConfig(
    name="Empty",
    symbol="Empty",
    charge=0,
    site_type="Empty",
    allowed_sublattices=[],
    physical_element=None,
    activation_energies_key="Empty",
    enabled_events=("generation",),
    CN_matters=False,
    field_dependent_generation=False,
    electrode_scavenging=False,
    description="Synthetic placeholder for a vacant site; template only, "
                "never registered.",
    is_empty=True,
)


def make_empty_defect() -> Defect:
    """Return a fresh empty Defect.

    A new instance is created per call on purpose: Defect holds per-site
    mutable state (charge, passivation_level, and the events list whose rate
    fields are site-local), so a shared instance would silently couple
    distinct sites. EMPTY_DEFECT_CONFIG itself is safe to share because it is
    config-time immutable.
    """
    return Defect(
        config=EMPTY_DEFECT_CONFIG,
        chemical_specie="Empty",
        charge=0,
        passivation_level=0,
        events=[],
    )
