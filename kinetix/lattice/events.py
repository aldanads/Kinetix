# -*- coding: utf-8 -*-
"""kMC event execution for Kinetix.

Phase 3 of the ``simulator.py`` split: everything that *applies* a chosen kMC
event to the lattice lives here:

  * dispatch (``processes``) and the per-event handlers
    (``_handle_migration_event`` / ``_handle_generation_event`` /
    ``_handle_redox_event`` / ``_handle_reaction_event``),
  * the lookups/predicates the handlers need (``_defect_by_name``,
    ``_find_empty_neighbor``, ``_get_gb_charge_state``, ``_should_scavenge``,
    ``_is_at_top_electrode``, ``_get_mobile_sites``),
  * the site-state mutators that keep the dirty-site bookkeeping consistent
    (``_install_defect_site``, ``_introduce_specie_site``,
    ``_track_occupancy_update``, ``_remove_species_at_site``),
  * the affected-state refresh (``update_sites_topology``,
    ``_update_rates_lazily``).

The handler is stateless: every read/write of simulation state goes through
``self.simulator`` (the ``KMCSimulator``/``simulator``), so MPI rank
ownership, pickles and the golden trace all observe the pre-split state.
``_is_active_site`` stays on the system (lattice construction uses it too) and
is called as ``self.simulator._is_active_site(...)``.

``KMCSimulator`` keeps thin delegates for the names used outside this
module: ``processes`` (superbasin.py calls ``simulator.processes``),
``update_sites_topology`` and ``_introduce_specie_site`` (state_loader.py and
the deposition paths), ``_update_rates_lazily`` (the kMC loop, the golden trace
and the kMC-loop tests) and ``_get_mobile_sites`` (lattice initialisation).
``_kmc_step`` deliberately calls ``self.processes(...)`` - through the delegate
- because the golden trace wraps the *instance* attribute to observe the event
catalog.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kinetix.lattice.simulator import KMCSimulator


class EventHandler:
    """Applies kMC events and refreshes the affected lattice state.

    Args:
        simulator: The ``KMCSimulator``/``simulator`` that owns the lattice.
            All reads/writes of simulation state go through this reference;
            the handler itself holds no simulation state.
    """

    def __init__(self, simulator: KMCSimulator) -> None:
        self.simulator = simulator

    # =========================================================================
    # Event dispatch
    # =========================================================================

    def processes(self, chosen_event) -> None:

        # Execute kMC event and update system state

        sites_needing_support_update = set()
        sites_needing_event_update = {chosen_event[-1]}

        if isinstance(chosen_event[2], int): # Migration event
          self._handle_migration_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )
        elif chosen_event[2] == 'generation':
          self._handle_generation_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )
        elif chosen_event[2] in ['reduction','oxidation']:
          self._handle_redox_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )
        elif any(chosen_event[2] == reaction['name'] for reaction in self.simulator.reactions_config.values()):

          self._handle_reaction_event(
            chosen_event,
            sites_needing_support_update,
            sites_needing_event_update
          )

        self.update_sites_topology(sites_needing_support_update, sites_needing_event_update)
        all_affected_sites = (
          sites_needing_support_update |
          sites_needing_event_update |
          set(self.simulator.generation_sites)
        )
        self.simulator._dirty_sites.update(all_affected_sites)


    # =========================================================================
    # Event handlers
    # =========================================================================

    def _handle_migration_event(self, chosen_event, support_update_sites, event_update_sites) -> None:
        """Handle migration events with object-transfer semantics."""
        source_idx = chosen_event[-1]
        dest_idx = chosen_event[1]
        dest_site = self.simulator.grid_crystal[dest_idx]
        source_site = self.simulator.grid_crystal[source_idx]

        # Check for removal at electrode
        if (self.simulator.allow_specie_removal and dest_site.is_at_top_interface):
            should_scavenge, use_mass_conservation = self._should_scavenge(source_site)

            if should_scavenge:
              if use_mass_conservation:
                defect_name = source_site._get_current_defect_name()
                self.simulator.scavenged_ions[defect_name] = self.simulator.scavenged_ions.get(defect_name,0) + 1

              self._remove_species_at_site(source_idx, support_update_sites,
                                           event_update_sites)

              if self.simulator.poisson_config is not None and self.simulator.poisson_config.solve_Poisson:
                event_update_sites.update(self._get_mobile_sites(self.simulator.active_event_sites))
              return

        # Get source defect and resolve config
        defect_name = source_site._get_current_defect_name()
        chemical_specie = source_site.defect.chemical_specie
        migrating_charge = source_site.defect.charge
        defect = source_site.defect

        # Apply GB charge state modification
        gb_charge = self._get_gb_charge_state(defect_name, dest_site.position, event_type='migration')
        if gb_charge is not None:
          migrating_charge = gb_charge
          defect.charge = gb_charge  # state lives on the Defect: write it through

        # Object transfer: the Defect hops to dest, source becomes empty.
        # The dirty-site bookkeeping matches the legacy
        # _introduce_specie_site / _remove_species_at_site pair exactly.
        self._install_defect_site(dest_idx, support_update_sites,
                                  event_update_sites, defect)
        self._remove_species_at_site(source_idx, support_update_sites,
                                     event_update_sites)

        # Update Poisson-relevant sites
        if self.simulator.poisson_config is not None and self.simulator.poisson_config.solve_Poisson:
          event_update_sites.update(self._get_mobile_sites(self.simulator.active_event_sites))

        # Handle cluster updates for neutral metal atoms
        if chemical_specie in self.simulator.METAL_SPECIES and migrating_charge == 0:
          self.simulator._remove_metal_atom_from_clusters(source_idx)
          self.simulator._add_metal_atom_to_clusters(dest_idx)

    def _should_scavenge(self, site) -> tuple[bool, bool]:
        """
        Check if a defect should be scavenged at the top electrode.

        Scavenging occurs when:
        1. The defect has electrode_scavenging enabled in its config
        2. The ion is electrostatically driven toward the electrode (q*V < 0)

        Returns:
        --------
        tuple : (should_scavenge: bool, use_mass_conservation: bool)
            - should_scavenge: True if the ion is driven toward the electrode (q*V < 0)
            - use_mass_conservation: True if the scavenged count should be tracked
        """
        defect_name = site._get_current_defect_name()
        if defect_name not in self.simulator.defects_config:
          return False, False

        # If the key doesn't exist of is False, no scavenging
        scavenging_cfg = self.simulator.defects_config[defect_name].get('electrode_scavenging')
        if not scavenging_cfg:
          return False, False

        # Determine if using mass conservation
        # Dict form: {mass_conservation: true/false}
        # Bool form: True (no mass conservation)
        if isinstance(scavenging_cfg, dict):
          use_mass_conservation = scavenging_cfg.get('mass_conservation', False)
        else:
          use_mass_conservation = False

        # Electrostatic driving force
        should_scavenge = (site.defect.charge * self.simulator.V) < 0

        return should_scavenge, use_mass_conservation

    def _handle_generation_event(self, chosen_event, support_update_sites, event_update_sites) -> None:
        """ Handle defect generation events """
        dest_idx = chosen_event[1]
        dest_site = self.simulator.grid_crystal[dest_idx]
        defect_name = dest_site._get_current_defect_name()
        chemical_specie = self.simulator.defects_config[defect_name]['symbol']


        # Apply GB charge state modification
        dest_pos = self.simulator.grid_crystal[dest_idx].position
        gb_charge = self._get_gb_charge_state(defect_name, dest_pos, event_type='generation')

        if gb_charge is not None:
          generated_charge = gb_charge
        else:
          generated_charge = self.simulator.defects_config[defect_name]['charge']

        electrode_scavenging = self.simulator.defects_config[defect_name].get('electrode_scavenging')
        if electrode_scavenging:
          if electrode_scavenging.get('mass_conservation'):
            self.simulator.scavenged_ions[defect_name] -= 1

        self._introduce_specie_site(dest_idx, support_update_sites, event_update_sites, chemical_specie, generated_charge)

    def _handle_redox_event(self, chosen_event, support_update_sites, event_update_sites) -> None:
        """Handle redox events with multi-species support."""
        site_idx = chosen_event[1]
        site = self.simulator.grid_crystal[site_idx]

        if chosen_event[2] == 'reduction':
          site.defect.charge -= 1
          event_update_sites.add(site_idx)
          self.simulator._add_metal_atom_to_clusters(site_idx)

        elif chosen_event[2] == 'oxidation':
          if site.is_at_top_interface and self.simulator.V < 0:
            self._remove_species_at_site(site_idx, support_update_sites, event_update_sites)
          else:
            site.defect.charge += 1
            event_update_sites.add(site_idx)
          self.simulator._remove_metal_atom_from_clusters(site_idx)

    def _handle_reaction_event(self, chosen_event, support_update_sites, event_update_sites) -> None:
        """
        Handler for reaction events
        Reads products definitions from reactions_config to update site states
        """
        reaction_name_chosen = chosen_event[2]
        source_idx = chosen_event[-1]
        dest_idx = chosen_event[1]
        sites_involved = [source_idx,dest_idx]

        # Reaction definitions
        for reaction_name, reaction in self.simulator.reactions_config.items():
          if reaction['name'] == reaction_name_chosen:
            products = reaction['products']

        # Track sites that need kMC update
        for i, product in enumerate(products):
          # Determine which site this product applies to
          site_index = product.get('site_index',i)

          # Handle unimolecular reactions (only 1 site involved)
          if site_index == 'neighbor':
            # Spawn product in a random empty neighbor
            # Useful for depassivation where H escapes to void
            origin_site = self.simulator.grid_crystal[source_idx]
            target_idx = self._find_empty_neighbor(origin_site,product)
            if target_idx is None:
              # This should never happen if registration logic is correct.
              # Raising an error: YAML/Logic mismatches?.
              raise ValueError(f"Reaction {reaction_name_chosen} failed: No valid neighbor found during execution. YAML/Logic mismatches?")

          else:
            target_idx = sites_involved[site_index]

          site = self.simulator.grid_crystal[target_idx]

          if product['symbol'] != 'Empty':
            defect = self._defect_by_name(product['symbol'])
            species_changed = (site.defect.chemical_specie != product['symbol'])

            if species_changed:
              if 'charge' in defect:
                ion_charge = defect['charge']

            else:
              ion_charge = site.defect.charge

              # Introduce species
            self._introduce_specie_site(
              target_idx,
              support_update_sites,
              event_update_sites,
              product['symbol'],
              ion_charge
            )

            # Handle passivation increment
            if 'passivation_increment' in product:
              site.defect.passivation_level += product['passivation_increment']

              # Handle charge variation
              if 'charge_per_passivation' in defect:
                site.defect.charge += defect['charge_per_passivation'] * product['passivation_increment']


          else:
            self._remove_species_at_site(
              target_idx,
              support_update_sites,
              event_update_sites
            )


    # =========================================================================
    # Lookups & predicates
    # =========================================================================

    def _defect_by_name(self, symbol) -> dict | None:

        for defect in self.simulator.defects_config.values():
          if defect['symbol'] == symbol:
            return defect

    def _find_empty_neighbor(self, site, product) -> tuple | None:
        """
        Find a random empty neighbor for escape. E.g.: H escaping a V_O

        Args:
          site: The V_O site where depassivation occurs.
          rng: Random number generator (for reproducibility).

        Returns:
          int: Index of selected empty neighbor, or None if no space available.
        """
        empty_neighbors = []
        defect = self._defect_by_name(product['symbol'])

        # 1. Collect all valid empty interstitial neighbors
        for neighbor_idx in site.nearest_neighbors_idx:
          neighbor = self.simulator.grid_crystal[neighbor_idx]
          if neighbor.site_type == product['sublattice'] and neighbor.defect.chemical_specie in defect["valid_target_species"]:
            empty_neighbors.append(neighbor_idx)

        # 2. Return None if no space available (reaction blocked)
        if not empty_neighbors:
          return None

        # 3. Randomly select one neighbor (equal probability)
        return tuple(self.simulator.rng.choice(empty_neighbors))

    def _is_at_top_electrode(self, site_idx) -> bool:
        """ Check if site is at top electrode """
        return self.simulator.grid_crystal[site_idx].is_at_bottom_interface

    def _get_mobile_sites(self, site_indices) -> list:
        """Filter only sites that can have mobile defects"""
        return [idx for idx in site_indices if self.simulator._is_active_site(self.simulator.grid_crystal[idx].site_type)]

    def _get_gb_charge_state(self, defect_name, site_position, event_type='migration') -> int | None:
        """
        Get charge state for a defect based on GB region.

        Parameters:
        -----------
        defect_name : str
            Name of defect (e.g., 'hydrogen_interstitial')
        site_position : tuple
            Site coordinates (x, y, z)
        event_type : str
            'migration', 'reaction', or 'generation'

        Returns:
        --------
        int : Charge state (may differ from defects_config based on GB region)
        """
        # Default: no GB modification

        if not self.simulator.gb_model:
          return None

        gb_config = self.simulator.gb_model.gb_configurations[0]
        event_entries = gb_config['event_modifications'].get(event_type)
        if event_entries is None:
          return None

        # Backward compatibility
        if isinstance(event_entries, dict):
          event_entries = [event_entries]

        # Find the entry that applies to this case
        for entry in event_entries:
        # Check if this defect is affected
          affected_defects = entry.get('affected_defects',[])
          if defect_name and defect_name not in affected_defects:
            continue

          charge_state = entry.get('charge_state', {})
          if not charge_state:
            return None # Affected by GB but has no charge modifications

          # Get site region and return charge state
          site_gb_region = self.simulator.gb_model.get_site_gb_region(site_position)
          return charge_state.get(site_gb_region, None)


    # =========================================================================
    # Site state mutators (dirty-site bookkeeping)
    # =========================================================================

    def _install_defect_site(self, idx, support_update_sites, event_update_sites, defect) -> None:
        """Install an existing Defect at ``idx`` and track affected sites.

        Phase 5 object-transfer counterpart of ``_introduce_specie_site``:
        the Defect object is moved by reference (it carries chemical_specie,
        charge, passivation_level and events), so no state is rebuilt. The
        dirty-site bookkeeping is identical to ``_introduce_specie_site``.
        """
        self.simulator.grid_crystal[idx].install_defect(defect)
        self._track_occupancy_update(idx, support_update_sites, event_update_sites)

    def _introduce_specie_site(self, idx, support_update_sites, event_update_sites, chemical_specie, ion_charge=None) -> None:
        """Introduce species at site and track affected sites."""
        site = self.simulator.grid_crystal[idx]
        site.introduce_specie(chemical_specie, ion_charge)

        self._track_occupancy_update(idx, support_update_sites, event_update_sites)

    def _track_occupancy_update(self, idx, support_update_sites, event_update_sites) -> None:
        """Track the sites affected by a newly occupied site at ``idx``.

        Shared by ``_introduce_specie_site`` and ``_install_defect_site`` so
        both entry points produce bit-identical dirty-site sets.
        """
        site = self.simulator.grid_crystal[idx]

        # Track sites occupied
        if idx not in self.simulator.active_event_sites:
          self.simulator.active_event_sites.append(idx)

        event_update_sites.add(idx)
        support_update_sites.update(site.nearest_neighbors_idx)
        support_update_sites.add(idx)
        for affected_site_idx in support_update_sites:
            affected_site = self.simulator.grid_crystal[affected_site_idx]
            # Add sites that support the affected site
            for supporting_site_idx in affected_site.supp_by:
              if(isinstance(supporting_site_idx, tuple) and
                 self.simulator.grid_crystal[supporting_site_idx].defect.chemical_specie != self.simulator.affected_site):
                 event_update_sites.add(supporting_site_idx)

            # Add the affected site itself if occupied
            if affected_site.defect.chemical_specie != self.simulator.affected_site:
                event_update_sites.add(affected_site_idx)

    def _remove_species_at_site(self, idx, support_update_sites, event_update_sites) -> None:
        """Remove species from site and track affected sites.

        Phase 5: the legacy ``attributes_to_reset`` setattr loop is gone -
        clear_defect() installs a fresh empty Defect, which resets charge,
        passivation_level and site_events in a single step.
        """
        site = self.simulator.grid_crystal[idx]
        site.remove_specie(self.simulator.affected_site)

        if idx in self.simulator.active_event_sites:
          self.simulator.active_event_sites.remove(idx)

        event_update_sites.discard(idx)
        support_update_sites.update(site.nearest_neighbors_idx)
        support_update_sites.add(idx)

        # Include in update_specie_events all the particles that can migrate
        # to the sites in update_supp_av --> It might change the available migrations
        # or the activation energy
        for affected_site_idx in support_update_sites:
            affected_site = self.simulator.grid_crystal[affected_site_idx]

            for supporting_site_idx in affected_site.supp_by:
              if(isinstance(supporting_site_idx, tuple) and
                   self.simulator.grid_crystal[supporting_site_idx].defect.chemical_specie != self.simulator.affected_site):
                   event_update_sites.add(supporting_site_idx)

            # Add the affected site itself if occupied
            if affected_site.defect.chemical_specie != self.simulator.affected_site:
                event_update_sites.add(affected_site_idx)


    # =========================================================================
    # Affected-state refresh
    # =========================================================================

    def update_sites_topology(self, support_update_sites, event_update_sites) -> None:
        """ Update the sites """

        # Update support relationship only for relevant sites
        if support_update_sites:
            reactive_support_sites = self._get_mobile_sites(support_update_sites)
            # There are new sites supported by the new species
            # For loop over neighbors
            for idx in reactive_support_sites:
                self.simulator.grid_crystal[idx].supported_by(
                  self.simulator.grid_crystal, self.simulator.wulff_facets, self.simulator.dir_edge_facets,
                  idx
                )

        # Update generation sites
        #self.simulator.generation_sites = [] # Reset
        for defect_name, defect in self.simulator.defects_config.items():
          if 'generation' in defect['enabled_events']:
            generation_sites = self.simulator.available_generation_sites(support_update_sites,defect_name, defect)
            self.simulator.generation_sites.extend(generation_sites)

        # Update event pathways for mobile sites
        if event_update_sites:
            # Sites are not available because a particle has migrated there
            reactive_event_sites = self._get_mobile_sites(event_update_sites)
            for idx in reactive_event_sites:
                self.simulator.grid_crystal[idx].available_pathways(
                  self.simulator.grid_crystal,idx,self.simulator.facets_type
                )

    def _update_rates_lazily(self, E_field_dict, T_field_dict) -> None:
        """
        Update transition rates based on electric field and temperature.

        Only rank 0 updates rates (runs kMC),
        non-root ranks skip this (they only solve Poisson/heat equations).

        Parameters:
        -----------
        E_field : dict
            Dictionary mapping position tuples to electric field vectors.
            Format: {(x, y, z): (Ex, Ey, Ez)} in V/m
            If None, zero field is assumed
        T_field : dict, optional
            Dictionary mapping position tuples to temperature values.
            Format: {(x, y, z): T} in Kelvin
            If None, ambient temperature is used.
        """
        # === Only rank 0 needs to update transition rates (runs kMC) ===
        if self.simulator.rank != 0:
          return

        if self.simulator._fields_changed:
          # All active + generation sites need update
          sites_to_update = set(self.simulator.active_event_sites) | set(self.simulator.generation_sites)
          self.simulator._fields_changed = False
        else:
          sites_to_update = self.simulator._dirty_sites

        if not sites_to_update:
         return  # Nothing to update


        # === Update transitions rates for all relevant sites ===
        for site_idx in sites_to_update:
          # Create lookup keys from site position
          pos_key = tuple(np.round(self.simulator.grid_crystal[site_idx].position, 6))

          # Get electric field (default: zero vector)
          E_site = E_field_dict.get(pos_key, np.array([0.0, 0.0, 0.0]))

          # Get temperature (default: ambient)
          T_site = T_field_dict.get(pos_key, self.simulator.temperature)

          self.simulator.grid_crystal[site_idx].transition_rates(
              E_site_field=E_site,
              T=T_site,
              migration_pathways = self.simulator.migration_pathways,
              clusters = self.simulator.clusters,
              atom_to_cluster = self.simulator.atom_to_cluster
            )
        self.simulator._dirty_sites.clear()
