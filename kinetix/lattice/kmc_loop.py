# -*- coding: utf-8 -*-
"""kMC (BKL) step orchestration for Kinetix.

Phase 4 of the ``simulator.py`` split: the BKL algorithm and the superbasin
policy that drives it:

  * ``step_kmc`` - the public per-step entry point,
  * ``_kmc_step`` - TR-catalog build (balanced tree), BKL time advance and
    event execution through ``simulator.processes`` (the EventHandler delegate),
  * ``_search_superbasin`` / ``update_superbasin`` - superbasin creation and
    invalidation,
  * ``should_activate_superbasin`` + ``is_filament_percolating`` +
    ``_check_event_based_superbasin`` / ``_check_time_based_superbasin`` /
    ``_slow_timesteps`` - the activation policy those searches use.

The loop holds no simulation state: every read/write goes through
``self.simulator`` (``time``, ``rank``/``mpi_ctx``, ``superbasin_dict``,
``events_tracking``, ...), so MPI rank ownership, pickles and the golden trace
observe the pre-split state. Field solving stays out of the loop: ``step_kmc``
reaches it via ``simulator.solver_coordinator._evaluate_fields_for_kmc()`` /
``simulator.solver_coordinator.get_timestep_limit()``.

MPI NOTE (moved verbatim): ``step_kmc`` keeps its historical ``rank == 0``
guard and the ``mpi_ctx.bcast(payload, root=0)`` of ``simulator.time``. In serial
runs ``simulator.mpi_ctx`` is ``None`` (set in ``initialization.py``), so the
broadcast is skipped and ``rank`` is 0; in MPI runs the loop executes on rank 0
only and *time* is the only synchronised value. This extraction added and
removed no MPI logic - it only reads those two fields from the system.

Global delegate cleanup: the loop is self-contained. Only ``step_kmc`` - the
core public API called by cli.py and the golden trace - survives as a thin
facade on ``KMCSimulator``; the other eight extracted names are reached
directly as ``simulator.kmc_loop.<name>``. ``_kmc_step`` calls
``self.simulator.processes(...)`` - the *retained* EventHandler delegate - on
purpose: the golden trace wraps the instance attribute ``crystal.processes``
(``tests/test_golden_trace.py``) to observe the executed event catalog.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from kinetix.utils.balanced_tree import build_tree, search_value, update_data
from kinetix.utils.superbasin import Superbasin

if TYPE_CHECKING:
    from kinetix.lattice.simulator import KMCSimulator

logger = logging.getLogger(__name__)


class KMCLoop:
    """BKL kMC step orchestration for KMCSimulator.

    The loop is self-contained: ``KMCSimulator`` keeps only the ``step_kmc``
    facade plus a lazy ``kmc_loop`` property; the other eight methods are
    reached as ``simulator.kmc_loop.<name>``.

    The loop runs on rank 0 (serial on every rank when ``mpi_ctx`` is None);
    field solving - the remaining MPI concern - is orchestrated by the
    SolverCoordinator and reached as ``simulator.solver_coordinator.<name>``.

    Args:
        simulator: The ``KMCSimulator``/``simulator`` the loop advances.
            All reads/writes of simulation state go through this reference;
            the loop itself holds no simulation state.
    """

    def __init__(self, simulator: KMCSimulator) -> None:
        self.simulator = simulator

    # =========================================================================
    # BKL step orchestration
    # =========================================================================

    def step_kmc(self, rng) -> None:
        """
        Execute one kMC step and synchronize time across all ranks.

        Handles MPI internally: only rank 0 executes kMC (owns the state),
        then broadcasts the updated time and event info to all ranks.
        Event tracking is managed internally (rank 0 only).

        Parameters:
        -----------
        rng : numpy.random.Generator
            Random number generator for kMC stochastic selection

        Returns:
        --------
        Update the system
        """

        E_field_dict, T_field_dict = self.simulator.solver_coordinator._evaluate_fields_for_kmc()

        # === Step 1: Rank 0 executes kMC, others prepare to receive ===
        if self.simulator.rank == 0:
          # Execute kMC step (modifies self internally)
          kmc_time_step, chosen_event = self._kmc_step(rng, E_field_dict, T_field_dict)

          if chosen_event is not None:
            self.simulator.events_tracking[chosen_event[2]] += 1

          # Check for superbasin (rank 0 only)
          self._search_superbasin(kmc_time_step)

          # Package for broadcasting
          payload = self.simulator.time

        else:
          # Non-root ranks: prepare to receive
          payload = None

        # === Step 2: Broadcast to all ranks
        if self.simulator.mpi_ctx is not None:
          payload = self.simulator.mpi_ctx.bcast(payload, root=0)

        # === Step 3: Unpack and update
        self.simulator.time = payload

    def _kmc_step(self, rng, E_field_dict, T_field_dict) -> tuple:
        """
        Internal kMC step logic (rank 0 only).

        Modifies self in-place (active_event_sites, superbasin_dict, time, etc.).
        Returns only the new values (time step and event info).

        Parameters:
        -----------
        rng : numpy.random.Generator
            Random number generator for stochastic selection

        Returns:
        --------
        time_step : float
            Time elapsed during this step
        chosen_event : tuple or None
            Event details if an event occurred, None otherwise
        """
        if self.simulator._fields_changed or self.simulator._dirty_sites:
          self.simulator.event_handler._update_rates_lazily(E_field_dict, T_field_dict)

        grid_crystal = self.simulator.grid_crystal
        superbasin_dict = self.simulator.superbasin_dict

        # =============================================================================
        # TR_catalog stores:
        #   - TR_catalog[0] = Transition Rate (TR)
        #   - TR_catalog[1] = Arrival site
        #   - TR_catalog[2] = Event label (Migration, desorption, etc.)
        #   - TR_catalog[3] = Starting site
        # =============================================================================

        # --- Build TR catalog ---
        TR_catalog = []
        for idx in self.simulator.active_event_sites + self.simulator.generation_sites:
          if idx not in superbasin_dict:
            TR_catalog.extend([
              event.catalog_tuple(idx)
              for event in grid_crystal[idx].defect.events
            ])
          else:
            # The superbasin keeps its own internal record for the aggregated
            # absorbing-state events: (rate, dest, label, E_act, origin).
            TR_catalog.extend([
              (item[0], item[1], item[2], idx)
              for item in superbasin_dict[idx].site_events_absorbing
            ])


        # Handle case: No events possible
        if not TR_catalog:
          timestep_limit = self.simulator.solver_coordinator.get_timestep_limit()
          self.simulator.track_time(timestep_limit)
          return timestep_limit, None

        # --- Build balanced tree structure ---
        # Each node is the sum of their children, starting from the leaf
        TR_tree = build_tree(TR_catalog)
        sumTR = update_data(TR_tree)

        # --- Handle case: No valid transitions ---
        if sumTR is None or sumTR == 0:
          timestep_limit = self.simulator.solver_coordinator.get_timestep_limit()
          self.simulator.track_time(timestep_limit)
          return timestep_limit, None

        # --- Handle single-node tree case ---
        if type(sumTR) is tuple:
          sumTR = sumTR[0]

        # --- Calculate time step ---
        time_step = -np.log(rng.random()) / sumTR

        # --- Calculate maximum allowed timestep ---
        timestep_limit = self.simulator.solver_coordinator.get_timestep_limit()

        # --- Execute event or advance time ---
        if time_step <= timestep_limit:
          # Event occurs: search tree for selected event
          chosen_event = search_value(TR_tree, sumTR * rng.random())

          # Update system state
          self.simulator.processes(chosen_event)
          self.update_superbasin(chosen_event)
          self.simulator.track_time(time_step)

          return time_step, chosen_event
        else:
          logger.debug('[KMC STEP] No event within time step. Time step: %s, time step limit: %s', time_step, timestep_limit)

          # No event within timestep limit
          self.simulator.track_time(timestep_limit)
          return timestep_limit, None


    # =========================================================================
    # Superbasin search & invalidation
    # =========================================================================

    def _search_superbasin(self, kmc_time_step) -> None:
        """
        Search for and create superbasins based on current system state.

        Called internally after each kMC step (rank 0 only).
        Creates superbasins for sites meeting energy criteria.

        Parameters:
        -----------
        kmc_time_step : float
            Time elapsed during the kMC step (used to check activation criteria)
        """
        # === Early exit: check if superbasin search should be activated ===
        if not self.should_activate_superbasin(kmc_time_step):
          return # No action needed

        # === Get occupied sites (copy to avoid modification during iteration) ===
        # Note: Using slice copy[:] instead of deepcopy for efficiency
        active_event_sites = self.simulator.active_event_sites[:]

        start_time = time.time()

        # === Search for valid superbasin candidates ===
        for idx in active_event_sites:
          for event in self.simulator.grid_crystal[idx].defect.events:
            # Check criteria:
            #   - idx not already in superbasin_dict
            #   - migration event (int label; checked first so the barrier of a
            #     non-migration event is never dereferenced)
            #   - event activation energy <= E_min threshold
            if (idx not in self.simulator.superbasin_dict) and event.is_migration and (event.barrier <= self.simulator.E_min):
              superbasin = Superbasin(idx, self.simulator, self.simulator.E_min, active_event_sites)

              if superbasin.valid:
                self.simulator.superbasin_dict.update({idx: superbasin})

        # === Record elapsed time ===
        end_time = time.time()
        elapsed_time = end_time - start_time

        # === Adaptive threshold: reduce E_min if search takes too long ===
        if elapsed_time > 300 and self.simulator.E_min_lim_superbasin > self.simulator.energy_step:
          self.simulator.E_min -= self.simulator.energy_step

        logger.debug("Elapsed time superbasin: %s seconds", elapsed_time)
        logger.debug("Superbasins generated: %s", len(self.simulator.superbasin_dict))

    def update_superbasin(self, chosen_event) -> None:
        # At every kMC step we have to check if we destroy any superbasin
        # We dismantle the superbasin if the chosen_event affect some of the states
        # that belong to any of the superbasin
        keys_to_delete = [idx for idx, sb in self.simulator.superbasin_dict.items()
                          if chosen_event[1] in sb.superbasin_environment or
                          chosen_event[-1] in sb.superbasin_environment]

        for key in keys_to_delete:
            del self.simulator.superbasin_dict[key]


    # =========================================================================
    # Superbasin activation policy
    # =========================================================================

    def should_activate_superbasin(self, kmc_time_step) -> bool:
        """
        Determine if superbasin approach should be activated based on system state

        Superbasins are only useful when:
          1. A stable, percolating filament exists
          2. The system is trapped in shallow wells

        Returns:
        --------
        bool : True if superbasin should be activated
        """
        if not self.simulator.enabled_superbasin:
          return False

        # Condition 1: Must have a percolating filament
        if not self.is_filament_percolating():
          return False

        # Condition 2: Must be in a trapped regime
        if self.simulator.time_based_superbasin:
          # Memristor switching: time-based superbasin activation
          return self._check_time_based_superbasin(kmc_time_step)
        else:
          # Deposition: event-based superbasin activation
          return self._check_event_based_superbasin()

        return False

    def is_filament_percolating(self) -> bool:
        return any(
          cluster.attached_layer.get('bottom_layer') and
          cluster.attached_layer.get('top_layer')
          for cluster in self.simulator.clusters.values()
        )

    def _check_event_based_superbasin(self) -> bool:
        """
        Check superbasin activation for deposition (based on system changes)
        """
        # Track occupied sites count
        current_occupied = len(self.simulator.active_event_sites)
        self.simulator.superbasin_tracker.append(current_occupied)

        # Keep only recent history
        if len(self.simulator.superbasin_tracker) > self.simulator.n_search_superbasin:
          self.simulator.superbasin_tracker.pop(0)

        # Check if system has been static
        if len(self.simulator.superbasin_tracker) >= self.simulator.n_search_superbasin:
          recent_mean = np.mean(self.simulator.superbasin_tracker[-self.simulator.n_search_superbasin:])
          if abs(recent_mean - current_occupied) < 1e-10: # No change
            self.simulator.nothing_happen_count += 1
          else:
            self.simulator.nothing_happen_count = 0

            # Adjust energy minimum
            if self.simulator.E_min - self.simulator.energy_step > 0:
              self.simulator.E_min -= self.simulator.energy_step
            else:
              self.simulator.E_min = 0


        # Check if superbasin should be activated
        if self.simulator.nothing_happen_count == self.simulator.n_search_superbasin:
          return True

        elif (self.simulator.nothing_happen_count > 0 and self.simulator.nothing_happen_count % self.simulator.n_search_superbasin == 0):
          # Gradually increase E_min back
          if self.simulator.E_min_lim_superbasin >= self.simulator.E_min + self.simulator.energy_step:
            self.simulator.E_min += self.simulator.energy_step
          else:
            self.simulator.E_min = self.simulator.E_min_lim_superbasin
          return True

        return False

    def _check_time_based_superbasin(self, kmc_time_step) -> bool:
        """
        Check superbasin activation for memristor switching (based on time)
        """
        self.simulator.superbasin_tracker.append(kmc_time_step)
        if len(self.simulator.superbasin_tracker) > self.simulator.n_search_superbasin:
          self.simulator.superbasin_tracker.pop(0)

        # Check if current step is slow
        is_slow_step = self._slow_timesteps()

        if is_slow_step:
          self.simulator.nothing_happen_count += 1

          if self.simulator.nothing_happen_count >= self.simulator.n_search_superbasin:
            self.simulator.nothing_happen_count = 0
            return True
        else:
          # Reset counter when we get a larger timestep
          self.simulator.nothing_happen_count = 0

        return False

    def _slow_timesteps(self) -> bool:
        """
        Determine if a timestep is considered "slow" (system barely advances)

        Small timesteps indicate the system is evolving slowly and may be
        stuck in metastable states, making it a good candidate for superbasin.
        """

        # Need to ensure we have enough data points
        if len(self.simulator.superbasin_tracker) < self.simulator.n_search_superbasin:
          return False  # Not enough data yet

        recent_mean_timestep = np.mean(self.simulator.superbasin_tracker[-self.simulator.n_search_superbasin:])

        # Small timestep = slow evolution -> Candidate for superbasin
        # Option 1: Absolute threshold
        if recent_mean_timestep < self.simulator.time_step_limits:
          return True

        # Option 2: Relative threshold (alternative)
        # if recent_mean < 0.1 * self.simulator.voltage_update_time:  # 10% of voltage update interval
        #     return True
        return False
