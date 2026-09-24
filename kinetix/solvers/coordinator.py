# -*- coding: utf-8 -*-
"""Field-solver orchestration for Kinetix.

Phase 2 of the ``simulator.py`` split: everything that *orchestrates* the
Poisson/Heat FEM solvers around the kMC system state lives here:

  * when to solve (``should_solve_fields_now`` / ``get_timestep_limit``),
  * how to build solver inputs (``get_evaluation_points`` + charge/generation
    extraction, ``prepare_clusters_for_bcs``, ``save_electric_bias``),
  * how to consume solver outputs for kMC rate updates
    (``_evaluate_fields_for_kmc``).

The FEM solvers themselves stay in this package (Linux/dolfinx-only);
``cli.py`` still constructs them and attaches them to the system state as
``simulator._poisson_solver`` / ``simulator._heat_solver``. The
coordinator reads them dynamically from the system, so the cli teardown
(``del simulator._poisson_solver``) keeps working before pickling.

All mutable state (``time``, ``last_field_solve_time``, ``V``) stays on the
system — the coordinator holds no simulation state of its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    from kinetix.lattice.simulator import KMCSimulator


class SolverCoordinator:
    """Orchestrates Poisson/Heat field solving around the kMC system state.

    The coordinator is self-contained: no extracted method survives on
    ``KMCSimulator``. Callers reach it directly as
    ``simulator.solver_coordinator.<name>`` (cli.py, the kMC loop, tests).

    Args:
        simulator: The ``KMCSimulator``/``simulator`` whose fields are
            solved. All reads/writes of simulation state go through this
            reference; the coordinator itself is stateless.
    """

    def __init__(self, simulator: KMCSimulator) -> None:
        self.simulator = simulator

    # =========================================================================
    # Solver inputs
    # =========================================================================

    def save_electric_bias(self, V) -> None:
        """Record the applied bias V on the system (read by physics/scavenging)."""
        self.simulator.V = V

    def get_evaluation_points(self):
        """Get evaluation points for electric field calculation.

        Handles MPI broadcasting internally: only rank=root computes,
        then broadcasts results to all ranks.

        Returns:
            tuple: ``(particle_locations, charges, evaluation_points)``
            * particle_locations : np.ndarray — charged particle positions
              [N, 3] in angstrom
            * charges : np.ndarray — charge values [N] in Coulombs
            * evaluation_points : np.ndarray — combined points for E-field
              evaluation [M, 3] in angstrom (particles + generation sites)
        """
        # === Step 1: Rank 0 computes, others prepare to receive ===
        if self.simulator.rank == 0:
            particle_locations, charges = self._extract_particles_charges()
            gen_site_locations = self._extract_generation_site_location()

            if len(particle_locations) > 0:  # In case there is no particles
                evaluation_points = np.concatenate([particle_locations, gen_site_locations], axis=0)
            else:
                evaluation_points = gen_site_locations

            # Package for broadcasting
            payload = (particle_locations, charges, evaluation_points)
        else:
            payload = None

        # === Step 2: Broadcast to all ranks ===
        if self.simulator.mpi_ctx is not None:
            payload = self.simulator.mpi_ctx.bcast(payload, root=0)

        # === Step 3: Unpack and return ===
        particle_locations, charges, evaluation_points = payload
        return particle_locations, charges, evaluation_points

    def _extract_particles_charges(self):
        """Extract charge locations and magnitudes from simulator."""
        particle_locations = []
        charges = []

        for site in self.simulator.active_event_sites:
            particle_locations.append(self.simulator.grid_crystal[site].position)
            charges.append(self.simulator.grid_crystal[site].defect.charge * constants.e * self.simulator.screening_factor)

        if len(particle_locations) == 0:
            particle_locations = np.empty((0, 3), dtype=np.float64)
            charges = np.empty((0,), dtype=np.float64)
        else:
            particle_locations = np.array(particle_locations, dtype=np.float64)
            charges = np.array(charges, dtype=np.float64)

        return particle_locations, charges

    def _extract_generation_site_location(self):
        """Extract generation site locations from simulator."""
        gen_site_locations = []

        for site in self.simulator.generation_sites:
            gen_site_locations.append(self.simulator.grid_crystal[site].position)

        if len(gen_site_locations) == 0:
            return np.empty((0, 3), dtype=np.float64)
        else:
            return np.array(gen_site_locations, dtype=np.float64)

    def prepare_clusters_for_bcs(self):
        """Prepare clusters for boundary condition calculation.

        Handles MPI internally: only rank 0 prepares clusters (has updated kMC
        state), then broadcasts results to all ranks automatically.

        Returns:
            dict: Dictionary of Cluster objects with BC information prepared
            (same on all ranks).
        """
        clusters = None
        # === Step 1: Rank 0 prepares clusters, others prepare to receive
        if self.simulator.rank == 0:
            clusters = self.simulator.clusters
            for cluster in clusters.values():
                cluster.prepare_cluster_for_bcs(self.simulator.grid_crystal, self.simulator.crystal_size)

            # Package for broadcasting
            payload = clusters
        else:
            payload = None

        # === Step 2: Broadcast to all ranks
        if self.simulator.mpi_ctx is not None:
            clusters = self.simulator.mpi_ctx.bcast(payload, root=0)

        return clusters

    # =========================================================================
    # kMC field evaluation
    # =========================================================================

    def _evaluate_fields_for_kmc(self):
        """Evaluate electric and temperature fields for KMC rate updates.

        Returns:
            tuple: (E_field_dict, T_field_dict) on rank 0, (None, None) on
            other ranks.
        """
        if not (hasattr(self.simulator, '_poisson_solver') and self.simulator._poisson_solver is not None):
            if self.simulator.rank == 0:
                return {}, {}
            else:
                return None, None

        # Step 1: Rank 0 determines evaluation points
        evaluation_points = None
        if self.simulator.rank == 0:
            if self.simulator._fields_changed:
                sites_to_update = set(self.simulator.active_event_sites) | set(self.simulator.generation_sites)
            else:
                sites_to_update = self.simulator._dirty_sites

            if sites_to_update:
                evaluation_points = np.array([
                    self.simulator.grid_crystal[site_idx].position
                    for site_idx in sites_to_update
                ], dtype=np.float64)
            else:
                evaluation_points = np.empty((0, 3), dtype=np.float64)

        # Step 2: Broadcast evaluation points to all ranks (skipped when mpi_ctx is None)
        if self.simulator.mpi_ctx is not None:
            evaluation_points = self.simulator.mpi_ctx.bcast(evaluation_points, root=0)

        # Step 3: All ranks participate in field evaluation
        if len(evaluation_points) > 0:
            E_field_dict = self.simulator._poisson_solver.evaluate_electric_field_at_points(evaluation_points)

            if hasattr(self.simulator, '_heat_solver') and self.simulator._heat_solver is not None:
                T_field_dict = self.simulator._heat_solver.evaluate_temperature_at_points(evaluation_points)
            else:
                T_field_dict = {}
        else:
            E_field_dict = {}
            T_field_dict = {}

        # Step 4: Only rank 0 returns the results
        if self.simulator.rank == 0:
            return E_field_dict, T_field_dict
        else:
            return None, None

    # =========================================================================
    # Field-solve scheduling
    # =========================================================================

    def get_timestep_limit(self):
        """Calculate maximum timestep based on next Poisson solve time."""
        next_field_time = self.simulator.last_field_solve_time + self.simulator.timestep_limits
        timestep_limit = next_field_time - self.simulator.time

        tolerance = 1.e-12 * self.simulator.timestep_limits

        if timestep_limit < tolerance:
            self.simulator.time = next_field_time
            timestep_limit = 0.0

        return next_field_time - self.simulator.time

    def should_solve_fields_now(self, elec_controller, tol=1e-12):
        """Check if field solvers (Poisson, Heat) should be solved at current time.

        Handles voltage update timing and tracks last field solve time.
        Only relevant on rank 0 (but safe to call on all ranks).

        Args:
            elec_controller: Controller with a ``voltage_update_time`` attribute.
            tol: Time tolerance for floating-point comparison (default 1e-12 s).

        Returns:
            tuple: ``(should_solve, is_snapshot)``.
        """
        # Initialize last_poisson_time if not set
        if not hasattr(self.simulator, 'last_field_solve_time'):
            self.simulator.last_field_solve_time = -float('inf')

        # Calculate next scheduled solve time
        next_solve_time = self.simulator.last_field_solve_time + elec_controller.voltage_update_time

        # Check if current time has reached next solve time
        if self.simulator.time >= next_solve_time - tol:
            # Update last_poisson_time for next iteration
            if self.simulator.last_field_solve_time == -float('inf'):
                self.simulator.last_field_solve_time = self.simulator.time
            else:
                self.simulator.last_field_solve_time = next_solve_time
                self.simulator.time = next_solve_time

            return True, True  # should_solve=True, is_snapshot=True
        return False, False  # should_solve=False, is_snapshot=False