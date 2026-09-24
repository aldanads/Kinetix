# -*- coding: utf-8 -*-
"""Lattice construction and initialization for Kinetix.

Phase 5 of the ``crystal.py`` split: everything that BUILDS or LOADS the
lattice grid during initialization:

  * the structure/MP model (``lattice_model`` + pymatgen helpers + MP cache),
  * grid assembly (``crystal_grid``: the pickled-grid load path AND the
    from-scratch build path - site construction, interface flags, the live
    ``defects_config`` binding),
  * migration pathways (``_initialize_migration_pathways`` +
    ``_validate_migration_network``) and the neighbor machinery (k-d tree,
    periodic images, percolation radius search, sequential/parallel neighbor
    analysis, missing-neighbor repair),
  * interstitial generation (Voronoi/refine/validate, OVITO dump),
  * coordinates, Wulff shape/edges, and cluster-tracking init.

The builder holds NO simulation state: every read/write goes through
``self.system`` (``grid_crystal``, ``structure``, ``Act_E_dict``,
``coord_cache``, ``rank``/``mpi_ctx``, ...), so pickles, MPI rank ownership
and the golden trace observe the pre-split state. The lazy
``Crystal_Lattice.lattice_builder`` property (local import, same pattern as
``solver_coordinator``/``kmc_loop``) instantiates one builder per system.

Construction-time collaborators that STAY on ``Crystal_Lattice``:
``_is_active_site`` (Phase 3 rule - lattice construction uses it, so the
builder calls ``self.system._is_active_site``) and ``_minimum_image_vector``
(runtime callers in the MACE NEB calculator and active learning, so the
builder calls ``self.system._minimum_image_vector``).

MPI NOTE (moved verbatim): construction is NOT entirely rank-free -
``_save_mp_cache`` and ``lattice_model`` guard writes with ``rank == 0``,
``lattice_model`` broadcasts the fetched structure with
``mpi_ctx.bcast(..., root=0)``, and ``_initialize_migration_pathways`` /
``_generate_interstitial_sites`` carry their own ``rank == 0`` guards. All
are now read as ``self.system.rank`` / ``self.system.mpi_ctx``: no MPI logic
was added or removed, and the builder holds no rank/mpi state. Serial runs
pass ``mpi_ctx=None`` (``initialization.py``), so every guard is trivially
rank 0 and the broadcast is skipped.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from pymatgen.analysis.defects.generators import (
    ChargeInterstitialGenerator,
    VoronoiInterstitialGenerator,
)
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Element
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from kinetix.lattice.cluster import Cluster
from kinetix.lattice.grain_boundary import GrainBoundary
from kinetix.lattice.site import Site

if TYPE_CHECKING:
    from kinetix.lattice.crystal import Crystal_Lattice

logger = logging.getLogger(__name__)


class LatticeBuilder:
    """Lattice construction and initialization for Crystal_Lattice.

    Crystal_Lattice keeps thin delegates with the original method names so
    external callers (initialization.py, cli.py, tests, metadata.py) are
    unchanged. The builder runs during initialization: the grid fast-path
    (pickled grid) and the from-scratch build path both live here.

    Args:
        system: The ``Crystal_Lattice``/``System_state`` whose lattice is
            built. All reads/writes of simulation state go through this
            reference; the builder itself holds no simulation state.
    """

    def __init__(self, system: Crystal_Lattice) -> None:
        self.system = system

    # =========================================================================
    # Structure & Materials-Project model
    # =========================================================================

    def _load_mp_cache(self, key:str) -> Dict[str, Any]:
        """Load material data from local cache file."""
        if self.system.cache_dir:
          cache_path = self.system.cache_dir / f'{key}.json'
          if cache_path.exists():
            with open(cache_path, 'r') as f:
              return json.load(f)
        return None

    def _save_mp_cache(self, key:str, data: Dict[str,Any]):
        """Save material data to local cache file (rank 0 only)."""
        if self.system.rank == 0 and self.system.cache_dir:
          self.system.cache_dir.mkdir(parents=True, exist_ok=True)
          cache_path = self.system.cache_dir / f'{key}.json'
          with open(cache_path,'w') as f:
            json.dump(data,f,indent=2)

    def lattice_model(self, api_key, mode, affected_site=None, miller_indices=(0, 0, 1)):
        """
        Generate a PRISTINE lattice model. Defects are added later in crystal_grid.
        """

        # === Try cache firtst ===
        cache_key = f'structure_{self.system.id_material}'
        structure_dict = self._load_mp_cache(cache_key)

        if structure_dict is None:
          # === Cache miss: fetch from API (rank 0)
          if self.system.rank == 0:
            try:
              with MPRester(api_key) as mpr:
                structure = mpr.get_structure_by_material_id(self.system.id_material)
              structure_dict = structure.as_dict()
              self._save_mp_cache(cache_key, structure_dict)
            except Exception as e:
              logger.warning('API fetch failed: %s', e)
              structure_dict = {'error': str(e)}
          else:
            structure_dict = None

          # Broadcast (skipped when mpi_ctx is None -> truly serial run)
          if self.system.mpi_ctx is not None:
            structure_dict = self.system.mpi_ctx.bcast(structure_dict, root=0)

          if 'error' in structure_dict:
            raise RuntimeError(f"Failed to fetch structure: {structure_dict['error']}")

        structure = Structure.from_dict(structure_dict)

        # Always use pristine structure -> Defects insertion in grid_crystal
        sga = SpacegroupAnalyzer(structure)
        structure_conv = sga.get_conventional_standard_structure()

        self.system.miller_indices = miller_indices
        structure_oriented = self._apply_miller_orientation(structure_conv, miller_indices)

        self.system.structure_basic = structure_oriented
        self.system.lattice_constants = tuple(np.array(structure_oriented.lattice.abc) / 10) # nm

        # Set chemical specie notation
        if mode == 'vacancy' and affected_site:
          self.system.chemical_specie = f"V_{affected_site}"
        elif mode == 'interstitial':
          # Use first interstitial symbol for naming
          interstitials = [
            cfg["symbol"] for cfg in self.system.defects_config.values()
            if cfg["site_type"] == "interstitial"
          ]
          self.system.chemical_specie = interstitials[0] if interstitials else "interstitial"
        else:
          self.system.chemical_specie = structure_oriented.composition.reduced_formula

        # Create full supercell
        self.system.structure = self._create_supercell(structure_oriented)
        self.system.crystal_size = self.system.structure.lattice.abc

        self._compute_basis_vectors()

    def _is_inside_supercell(self, cart_pos, supercell_lattice):
        """
        Check if a Cartesian position is inside the supercell defined by the lattice.

        Parameters:
            cart_pos (array-like): Cartesian coordinates of the position
            supercell_lattice (Lattice): Pymatgen Lattice object of the supercell
        """
        # Convert Cartesian to fractional coordinates
        frac_pos = supercell_lattice.get_fractional_coords(cart_pos)
        tol = 1e-8
        # Check if all fractional coordinates are within [0, 1)
        return np.all((frac_pos >= - tol) & (frac_pos < 1 + tol))

    def _apply_miller_orientation(self, structure, miller_indices):
        """
        Orient structure so that the specified Miller direction aligns with the z-axis.

        This method:
        1. Converts Miller indices to Cartesian coordinates
        2. Creates a rotation matrix to align this direction with z-axis
        3. Chooses an appropriate in-plane orientation for x and y
        4. Applies the rotation to the structure

        Parameters:
            structure: Input pymatgen Structure
            miller_indices (tuple): (h, k, l) Miller indices

        Returns:
            Structure: Rotated structure with [hkl] along z-axis
        """

        h, k, l = miller_indices

        # Handle special case: (0,0,0) or default to no rotation
        if h == 0 and k == 0 and l == 0:
            return structure.copy()

        # Convert Miller indices to Cartesian direction in the original lattice
        # [hkl] in fractional coordinates -> Cartesian
        miller_direction = structure.lattice.get_cartesian_coords([h, k, l])
        miller_direction = miller_direction / np.linalg.norm(miller_direction)

        # Target: align miller_direction with z-axis [0, 0, 1]
        z_axis = np.array([0, 0, 1])

        # Create rotation matrix using Rodrigues' rotation formula
        # We need to rotate miller_direction onto z_axis
        rotation_matrix = self._get_rotation_matrix(miller_direction, z_axis)

        # Apply rotation to structure
        symm_op = SymmOp.from_rotation_and_translation(rotation_matrix, [0, 0, 0])
        structure_rotated = structure.copy()
        structure_rotated.apply_operation(symm_op)

        return structure_rotated

    def _get_rotation_matrix(self, vec1, vec2):
        """
        Calculate rotation matrix that rotates vec1 to align with vec2.

        Uses Rodrigues' rotation formula. Handles the special case where
        vectors are parallel or anti-parallel.

        Parameters:
            vec1 (array): Starting vector (will be rotated)
            vec2 (array): Target vector (destination)

        Returns:
            np.ndarray: 3x3 rotation matrix
        """

        # Normalize vectors
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)

        # Check if vectors are already aligned
        if np.allclose(v1, v2):
            return np.eye(3)

        # Check if vectors are opposite (anti-parallel)
        if np.allclose(v1, -v2):
            # Rotate 180� around any perpendicular axis
            # Find a perpendicular vector
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            perp = np.cross(v1, perp)
            perp = perp / np.linalg.norm(perp)
            # 180� rotation around perp
            return 2 * np.outer(perp, perp) - np.eye(3)

        # Rodrigues' rotation formula
        # Rotation axis (perpendicular to both vectors)
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)

        # Rotation angle
        cos_angle = np.dot(v1, v2)
        sin_angle = np.linalg.norm(np.cross(v1, v2))

        # Rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

        return R

    def _create_supercell(self, unit_cell):
        """
        Create a full supercell of the pristine lattice.

        Parameters:
            unit_cell (Structure): Oriented unit cell from Materials Project

        Returns:
            Structure: Supercell matching target dimensions in self.system.crystal_size
        """
        lattice_params = np.array(unit_cell.lattice.abc)
        target_dims = np.array(self.system.crystal_size)
        repetitions = np.ceil(target_dims / lattice_params).astype(int)
        repetitions = np.maximum(repetitions, 1)

        scaling_matrix = np.diag(repetitions)
        return unit_cell * scaling_matrix

    def _compute_basis_vectors(self):
        """
        Compute basis vectors for KMC grid based on minimum fractional coordinate spacing.

        Sets self.system.basis_vectors to lattice vectors scaled by minimum atomic spacing.
        This creates a grid where integer multiples correspond to atomic positions.
        """
        # Find minimum non-zero fractional coordinate spacing
        frac_coords = np.array([site.frac_coords for site in self.system.structure_basic])

        # Get unique sorted fractional coordinates for each direction
        min_spacing = []

        for dim in range(3):
            coords = np.sort(np.unique(np.round(frac_coords[:, dim], decimals=10)))
            coords = coords[coords > 1e-10]  # Remove zeros

            if len(coords) > 1:
                # Minimum spacing between adjacent positions
                spacings = np.diff(coords)
                min_spacing.append(np.min(spacings[spacings > 1e-10]))
            elif len(coords) == 1:
                # Single position means spacing is the coordinate itself
                min_spacing.append(coords[0])
            else:
                # Fallback: use 1.0 (full lattice vector)
                min_spacing.append(1.0)

        # Use minimum spacing across all dimensions for uniform scaling
        min_non_zero_element = min(min_spacing)

        # Scale lattice vectors by minimum spacing
        # This gives basis vectors where integer steps land on atomic sites
        self.system.basis_vectors = np.array(self.system.structure_basic.lattice.matrix) * min_non_zero_element


    # =========================================================================
    # Migration pathways & network validation
    # =========================================================================

    def _initialize_migration_pathways(self, radius_neighbors, reset_energies=False):
        """Initialize migration pathways from the COMPLETE grid_crystal."""
        self.system.event_labels = {}
        self.system.migration_pathways = {}
        i = 0

        # Brute-force neighbor search (O(N2)), but only done during initialization
        for site_idx in self.system.grid_crystal.keys():
          site_pos = self.system.grid_crystal[site_idx].position

          # Get neighbors using k-d tree
          neighbor_site_indices = self._get_neighbors_for_site(site_idx, radius_neighbors)

          # Process each neighbor
          for neighbor_idx in neighbor_site_indices:
            if neighbor_idx == site_idx:
              continue

            neighbor_pos = self.system.grid_crystal[neighbor_idx].position

            vector = self.system._minimum_image_vector(np.array(neighbor_pos) - np.array(site_pos))
            dist = np.linalg.norm(vector)
            if dist < 1e-10:
              continue
            # Create migration key
            migration_vector_key = tuple(np.round(vector, decimals=6))

            if migration_vector_key not in self.system.event_labels:
              self.system.event_labels[migration_vector_key] = i
              self.system.migration_pathways[i] = {
                'direction': vector / np.linalg.norm(vector),
                'distance': dist
              }
              i += 1

        self.system.num_event = len(self.system.event_labels) + 2

        if self.system.rank == 0:
          self._validate_migration_network(radius_neighbors)

          # NEW: Percolation sanity check
          percolated, n_connected = self._check_percolation_at_radius(
            radius_neighbors, site_type="interstitial"
          )
          if percolated:
            logger.info("Percolation check: PASSED (%d connected sites)", n_connected)
          else:
            logger.warning(
              "Percolation check: FAILED at radius=%.2f Å. "
              "No connected path from bottom to top electrode. "
              "Filament formation will not be possible.",
              radius_neighbors
            )

        # Electric field-dependent barriers
        if (self.system.poisson_config is not None and
            self.system.poisson_config.solve_Poisson):

            for name in self.system.defects_config.keys():
              Act_E_mig = {}
              for key, migration_vector in self.system.migration_pathways.items():
                z_component = migration_vector['direction'][2]
                if np.isclose(z_component, 0.0, atol=1e-9):
                  Act_E_mig[key] = self.system.Act_E_dict[name].get('E_mig_plane')
                elif z_component > 0:
                  Act_E_mig[key] = self.system.Act_E_dict[name].get('E_mig_upward')
                else:
                  Act_E_mig[key] = self.system.Act_E_dict[name].get('E_mig_downward')
              self.system.Act_E_dict[name]['E_mig'] = Act_E_mig

            for site in self.system.grid_crystal.values():
              if self.system._is_active_site(site.site_type):
                site.Act_E_dict = self._efficient_act_e_copy(self.system.Act_E_dict)
              else:
                site.Act_E_dict = {}

            if reset_energies:
              for site in self.system.grid_crystal.values():
                site.defect.events = [] # Clear old events

    def _validate_migration_network(self, radius=None):
        """
        Validate the interstitial migration network and print key statistics.

        Args:
            radius (float, optional): If provided, validates against expected radius
        """
        # Collect migration data
        migration_distances = []
        neighbor_counts = []
        interstitial_positions = []

        for site_idx, site in self.system.grid_crystal.items():
          if site.site_type == 'interstitial':
            interstitial_neighbors = 0
            interstitial_positions.append(site.position)
            neighbor_site_indices = self._get_neighbors_for_site(site_idx, radius)

            for neighbor_idx in neighbor_site_indices:
            #for neighbor_idx in site.nearest_neighbors_idx:
              neighbor = self.system.grid_crystal[neighbor_idx]
              if neighbor.site_type == "interstitial":
                dist = np.linalg.norm(
                  np.array(site.position) - np.array(neighbor.position)
                )
                if dist < max(self.system.crystal_size) * 0.8:
                  migration_distances.append(dist)

                interstitial_neighbors += 1

            neighbor_counts.append(interstitial_neighbors)

        if not migration_distances:
          logger.warning("No interstitial-interstitial migration pathways found!")
          return

        # Basic distance statistics
        min_dist = min(migration_distances)
        max_dist = max(migration_distances)
        avg_dist = np.mean(migration_distances)
        std_dist = np.std(migration_distances)

        # Neighbor statistics
        min_neighbors = min(neighbor_counts)
        max_neighbors = max(neighbor_counts)
        avg_neighbors = np.mean(neighbor_counts)


        logger.debug("Migration Network Validation: %d pathways, dist min %.3f / max %.3f / avg %.3f +- %.3f angstroms",
                len(migration_distances), min_dist, max_dist, avg_dist, std_dist)
        logger.debug("Neighbor stats: min %d / max %d / avg %.1f per site",
                min_neighbors, max_neighbors, avg_neighbors)

        # Validation warnings
        if radius is not None:
          if max_dist > radius * 1.05:
            logger.warning("Max distance (%.3f angstroms) exceeds search radius (%.1f angstroms)",
                max_dist, radius)

        if avg_dist > 4:
          logger.warning("Average migration distance (%.3f angstroms) seems high for direct hopping", avg_dist)

        if avg_neighbors < 4:
          logger.warning("Low connectivity (%.1f neighbors/site) may limit filament formation", avg_neighbors)

        if avg_neighbors > 15:
          logger.warning("High connectivity (%.1f neighbors/site) may include unrealistic pathways", avg_neighbors)

        if min_dist < 1.5:
          positions = np.array(interstitial_positions)
          from scipy.spatial.distance import pdist
          distances = pdist(positions)
          min_dist_inters = np.min(distances)

          if min_dist_inters < 1.5:
            logger.warning("Very close interstitial sites detected (min distance %.3f angstroms)", min_dist_inters)
            # Find problematic pairs
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(positions, positions)
            np.fill_diagonal(dist_matrix, np.inf)
            close_pairs = np.where(dist_matrix < 1.5)

            for i, j in zip(close_pairs[0][:5], close_pairs[1][:5]):  # Show first 5
              logger.debug("   Sites %d and %d: %.3f angstroms apart", i, j, dist_matrix[i,j])


    # =========================================================================
    # Neighbor search & radius analysis
    # =========================================================================

    def _build_kdtree(self):
        """Build and store k-d tree for reuse."""
        from scipy.spatial import cKDTree
        positions = np.array([site.position for site in self.system.grid_crystal.values()])
        self.system._kdtree_positions = positions
        self.system._kdtree_indices = list(self.system.grid_crystal.keys())
        self.system._kdtree = cKDTree(positions)

    def _get_neighbors_for_site(self,site_idx,radius):
        """Get neighbors for a specific site using stored k-d tree."""
        site_pos = self.system.grid_crystal[site_idx].position
        all_neighbor_indices = set() # Use set to automatically deduplicate

        # 1. Generate all periodic image positions to query
        query_positions = self._generate_periodic_images(site_pos, radius)

        # 2. Query k-d tree for each image position
        for query_pos in query_positions:
          neighbor_array_indices = self.system._kdtree.query_ball_point(query_pos,radius)
          #Convert to site indices
          for i in neighbor_array_indices:
            neighbor_idx = self.system._kdtree_indices[i]
            if neighbor_idx != site_idx:
              all_neighbor_indices.add(neighbor_idx)

        return list(all_neighbor_indices)

    def _generate_periodic_images(self,site_pos,radius):
        """
        Generate query positions including periodic images for LATERAL boundaries
        only (x, y). Top and bottom (z) are electrodes with open boundaries.

        Uses fractional coordinates for boundary detection and lattice vectors
        for PBC translations, so it is correct for any cell shape (including
        monoclinic), not just orthogonal cells.

        Returns:
            List of positions to query (original + lateral periodic images).
        """
        site_pos = np.array(site_pos, dtype=float)
        lattice = self.system.structure.lattice

        # Lateral lattice vectors (a, b). The c-direction is open (electrodes).
        a_vec = lattice.matrix[0]  # First lattice vector
        b_vec = lattice.matrix[1]  # Second lattice vector

        # Fractional position for boundary detection
        frac = lattice.get_fractional_coords(site_pos)

        # Fractional threshold: approximate conversion of the Cartesian radius
        # into fractional units along each lateral direction.
        frac_thr_a = radius / np.linalg.norm(a_vec)
        frac_thr_b = radius / np.linalg.norm(b_vec)

        # Which lateral boundaries is this site close to?
        near_lower_a = frac[0] < frac_thr_a
        near_upper_a = frac[0] > 1.0 - frac_thr_a
        near_lower_b = frac[1] < frac_thr_b
        near_upper_b = frac[1] > 1.0 - frac_thr_b

        # Build the list of shifts for each lateral direction.
        # np.zeros(3) = "no shift in this direction" (keeps the original position).

        a_shifts = [np.zeros(3)]
        if near_lower_a:
            a_shifts.append(a_vec)  # Shift by +a_vec
        if near_upper_a:
            a_shifts.append(-a_vec)  # Shift by -a_vec

        b_shifts = [np.zeros(3)]
        if near_lower_b:
            b_shifts.append(b_vec)  # Shift by +b_vec
        if near_upper_b:
            b_shifts.append(-b_vec)  # Shift by -b_vec

        # All combinations of a- and b-shifts. This automatically produces:
        #   - the original position       (0, 0)
        #   - single-axis images          (±a, 0), (0, ±b)
        #   - corner/diagonal images      (±a, ±b)
        query_positions = [site_pos + sa + sb
                          for sa, sb in product(a_shifts, b_shifts)]

        return query_positions

    def _check_percolation_at_radius(self, radius, site_type="interstitial"):
        """Check whether `site_type` sites percolate bottom<->top at a radius.

        The bottom and top electrode bands are taken as the OUTERMOST ROW of
        the sublattice on each side (at least one full row of `site_type`
        sites is included in each band), so the check works even when the
        sublattice does not reach the cell faces - e.g. the interstitial
        sublattice of a thin film. A DFS from all bottom-row sites then
        follows same-`site_type` neighbors found with the KD-tree
        (`_get_neighbors_for_site`), so no brute-force O(N^2) search is
        needed.

        Parameters
        ----------
        radius : float
            Neighbor search radius (Å).
        site_type : str
            Site type to analyze ("interstitial", "O", ...).

        Returns
        -------
        (bool, int)
            (whether a bottom-row site connects to a top-row site, number of
            sites visited by the DFS starting from all bottom-row sites).
        """
        # Lazily build the k-d tree if it is not already available (e.g. when
        # these helpers are called directly on a loaded grid).
        if getattr(self.system, "_kdtree", None) is None:
          self._build_kdtree()

        lattice = self.system.structure.lattice

        # Fractional z of every site of the requested type
        frac_z = {}
        for idx, site in self.system.grid_crystal.items():
          if site.site_type != site_type:
            continue
          frac_z[idx] = lattice.get_fractional_coords(site.position)[2]

        if len(frac_z) < 2:
          return False, 0

        # Electrode bands from the actual rows of the sublattice, so that at
        # least one full row of sites is always included on each side. Each
        # band edge is the midpoint between the outermost row and its neighbor.
        z_planes = sorted({round(z, 4) for z in frac_z.values()})
        if len(z_planes) < 2:
          return False, 0

        bottom_edge = z_planes[0] + (z_planes[1] - z_planes[0]) / 2.0
        top_edge = z_planes[-1] - (z_planes[-1] - z_planes[-2]) / 2.0

        bottom_sites = {idx for idx, z in frac_z.items() if z <= bottom_edge}
        top_sites = {idx for idx, z in frac_z.items() if z >= top_edge}

        if not bottom_sites or not top_sites:
          return False, 0

        # DFS from all bottom sites, following only same-site_type neighbors
        visited = set()
        stack = list(bottom_sites)

        while stack:
          current = stack.pop()
          if current in visited:
            continue
          visited.add(current)

          if current in top_sites:
            return True, len(visited)

          # Use KD-tree for efficient neighbor search
          neighbor_indices = self._get_neighbors_for_site(current, radius)

          for neighbor_idx in neighbor_indices:
            if neighbor_idx in visited:
              continue
            neighbor = self.system.grid_crystal[neighbor_idx]
            if neighbor.site_type != site_type:
              continue
            stack.append(neighbor_idx)

        return False, len(visited)

    def find_optimal_radius(self, site_type="interstitial",
                              min_radius=1.5, max_radius=6.0, step=0.25,
                              safety_margin=0.5):
        """Find the minimum radius_neighbors that gives percolation.

        Uses binary search with the existing KD-tree for efficient neighbor
        lookup (see `_check_percolation_at_radius`).

        Parameters
        ----------
        site_type : str
            Site type to check percolation for ("interstitial" or "O").
        min_radius : float
            Lower bound for the search (Å).
        max_radius : float
            Upper bound for the search (Å).
        step : float
            Precision of the search (Å).
        safety_margin : float
            Extra margin added to the percolation threshold (Å).

        Returns
        -------
        float
            Optimal radius_neighbors value.
        """
        # First check if percolation is even possible at max_radius
        percolated, _ = self._check_percolation_at_radius(max_radius, site_type)
        if not percolated:
          logger.warning(
            "No percolation even at max_radius=%.1f Å for site_type=%s. "
            "Grid may be too sparse. Using max_radius.",
            max_radius, site_type
          )
          return max_radius

        # Binary search for the minimum percolation radius
        lo, hi = min_radius, max_radius
        while hi - lo > step:
          mid = (lo + hi) / 2.0
          percolated, _ = self._check_percolation_at_radius(mid, site_type)
          if percolated:
            hi = mid
          else:
            lo = mid

        # Add safety margin to ensure robust connectivity
        optimal_radius = hi + safety_margin

        logger.info(
          "Optimal radius_neighbors: %.2f Å (%s, "
          "percolation threshold at %.2f Å, +%.2f Å margin)",
          optimal_radius, site_type, hi, safety_margin
        )

        return optimal_radius

    def diagnose_steep_down(self, site_idx, radius_neighbors):
        site = self.system.grid_crystal[site_idx]
        pos = np.array(site.position)
        logger.debug("=== Site %s at %s ===", site_idx, site.position)

        # Find ALL sites within radius (raw KDTree, no PBC filtering)
        neighbor_indices = self._get_neighbors_for_site(site_idx, radius_neighbors)

        steep_up, steep_down, shallow = [], [], []
        for n_idx in neighbor_indices:
            if n_idx == site_idx:
                continue
            npos = np.array(self.system.grid_crystal[n_idx].position)
            vec = npos - pos
            # Apply minimum-image on x,y
            for d in range(2):
                L = self.system.crystal_size[d]
                if vec[d] >  L/2: vec[d] -= L
                if vec[d] < -L/2: vec[d] += L
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
              continue
            unit = vec / dist
            z = unit[2]
            if z >  0.5: steep_up.append((n_idx, unit, dist))
            elif z < -0.5: steep_down.append((n_idx, unit, dist))
            else: shallow.append((n_idx, unit, dist))

        logger.debug("Steep UP neighbors   (z>+0.5): %d", len(steep_up))
        for idx,u,d in steep_up:   logger.debug("   idx=%s dir=%s dist=%.3f specie=%s", idx, np.round(u,3), d, self.system.grid_crystal[idx].defect.chemical_specie)
        logger.debug("Steep DOWN neighbors (z<-0.5): %d", len(steep_down))
        for idx,u,d in steep_down: logger.debug("   idx=%s dir=%s dist=%.3f specie=%s", idx, np.round(u,3), d, self.system.grid_crystal[idx].defect.chemical_specie)
        logger.debug("Shallow neighbors: %d", len(shallow))

    def diagnose_interstitial_presence(self, site_idx, radius_neighbors, z_window=3.0):
        """Check whether interstitial sites exist above/below the corner site,
        using the KDTree for efficient spatial filtering."""
        site = self.system.grid_crystal[site_idx]
        pos = np.array(site.position)
        logger.debug("=== Interstitial inventory near %s at %s ===", site_idx, np.round(pos,3))
        logger.debug("radius_neighbors = %s", radius_neighbors)

        # Use KDTree to get candidates within radius (efficient)
        candidate_indices = self.system._kdtree.query_ball_point(pos, radius_neighbors)

        above, below = [], []
        for i in candidate_indices:
            idx = self.system._kdtree_indices[i]
            if idx == site_idx:
                continue
            s = self.system.grid_crystal[idx]
            if s.site_type != 'interstitial':
                continue

            vec = np.array(s.position) - pos
            # Minimum-image wrap on x,y
            for d in range(2):
                L = self.system.crystal_size[d]
                if vec[d] >  L/2: vec[d] -= L
                if vec[d] < -L/2: vec[d] += L

            dz = vec[2]
            lat = np.linalg.norm(vec[:2])

            if 0 < dz < z_window:
                above.append((idx, lat, dz, s.defect.chemical_specie))
            elif -z_window < dz < 0:
                below.append((idx, lat, dz, s.defect.chemical_specie))

        logger.debug("Interstitial sites ABOVE (0 < dz < %s): %d", z_window, len(above))
        for idx, lat, dz, sp in sorted(above, key=lambda x: x[2]):
            logger.debug("   idx=%s lat_dist=%.3f dz=+%.3f specie=%s", idx, lat, dz, sp)
        logger.debug("Interstitial sites BELOW (-%s < dz < 0): %d", z_window, len(below))
        for idx, lat, dz, sp in sorted(below, key=lambda x: -x[2]):
            logger.debug("   idx=%s lat_dist=%.3f dz=%.3f specie=%s", idx, lat, dz, sp)


    # =========================================================================
    # Grid assembly
    # =========================================================================

    def crystal_grid(self,grid_crystal,radius_neighbors,mode,affected_site,api_key):

        self.system.coord_cache = {}

        # Loading existing grid
        if grid_crystal is not None:
          self.system.grid_crystal = grid_crystal
          # Legacy grids were pickled before Site.idx existed; backfill the
          # index key onto every loaded site that does not carry one yet.
          for idx, site in self.system.grid_crystal.items():
            if getattr(site, 'idx', None) is None:
              site.idx = idx
          # Live-config binding (Phase 6).  Loaded sites otherwise run on the
          # ``defects_config`` copy stored inside the grid pickle, which can go
          # stale relative to the preset's YAML while every lattice-level lookup
          # (defect_gen, generation sites, ...) already uses the live registry.
          # Binding one shared reference keeps a single source of truth for the
          # site-level lookups too (allowed_sublattices, valid_target_species,
          # CN_matters, symbols) at the cost of one pointer store per site.
          if self.system.defects_config:
            for site in self.system.grid_crystal.values():
              site.defects_config = self.system.defects_config
          self._compute_interface_flags()
          # Initialize pathways for loaded grids too
          self._build_kdtree()

          if mode == "interstitial":
            optimal_radius = self.find_optimal_radius(
                site_type="interstitial",
                min_radius=1.5,
                max_radius=radius_neighbors,
                safety_margin=0.5
            )
            radius_neighbors = optimal_radius

          self._initialize_migration_pathways(radius_neighbors, reset_energies=True)
        else:

          # MPI-agnostic: this method is a pure grid builder/loader. It knows
          # nothing about ranks - each process that calls it builds/loads its
          # own grid locally.



          logger.info('Initializing grid_crystal with %d host sites', len(self.system.structure))
          total_start_time = time.perf_counter()

          # --- STEP 1: Build host lattice with REAL chemical species ---
          start_time = time.perf_counter()
          self.system.grid_crystal = {}
          for site in self.system.structure:
            idx = self.get_idx_coords(site.coords, self.system.basis_vectors)
            site_type = site.specie.symbol
            is_active = self.system._is_active_site(site_type)

            self.system.grid_crystal[idx] = Site(
              chemical_specie=site_type,
              position=tuple(site.coords),
              site_type=site_type,
              Act_E_dict=self._efficient_act_e_copy(self.system.Act_E_dict) if is_active else {}, # Its own copy
              defects_config = self.system.defects_config,
              reactions_config = self.system.reactions_config,
              is_active_site=is_active,
              idx=idx
            )

          logger.info("Step 1 (Build host lattice): %.4f seconds", time.perf_counter() - start_time)

          # --- STEP 2: Handle boundary sites (if needed) ---
          start_time = time.perf_counter()
          self._build_kdtree()
          self._handle_missing_neighbors(radius_neighbors, affected_site)
          logger.info("Step 2 (Boundary sites): %.4f seconds", time.perf_counter() - start_time)

          # --- STEP 3: Add interstitial/hollow sites ---
          start_time = time.perf_counter()
          interstitial_count = 0
          if mode == "interstitial":
            for pos in self._generate_interstitial_sites(api_key=None):
              idx = self.get_idx_coords(pos, self.system.basis_vectors)
              if idx not in self.system.grid_crystal:
                self.system.grid_crystal[idx] = Site(
                  chemical_specie=affected_site,
                  position=tuple(pos),
                  site_type="interstitial",
                  Act_E_dict=self._efficient_act_e_copy(self.system.Act_E_dict),
                  defects_config = self.system.defects_config,
                  reactions_config = self.system.reactions_config,
                  is_active_site=True, # Interstitials are always active
                  idx=idx,
                )
                interstitial_count += 1

          logger.info("Step 3 (Interstitial sites): %.4f seconds", time.perf_counter() - start_time)
          logger.info("Total sites created: %d (%d host + %d interstitial)",
                      len(self.system.grid_crystal), len(self.system.structure), interstitial_count)

          # === STEP 3.5: Set interface flags ===
          self._compute_interface_flags()

          # --- STEP 3.75: Automated radius_neighbors selection (creation only) ---
          # Rebuild the KD-tree with the interstitial sites added in Step 3,
          # then find the minimum radius that percolates bottom<->top. This
          # only runs during grid CREATION (loading a cached grid keeps the
          # preset radius). The preset radius_neighbors is the search upper
          # bound and the returned optimal value overrides it below.
          self._build_kdtree()
          if mode == "interstitial":
            optimal_radius = self.find_optimal_radius(
              site_type="interstitial",
              min_radius=1.5,
              max_radius=radius_neighbors,  # Use preset value as upper bound
              safety_margin=0.5
            )
            # Override the preset radius with the optimal one
            radius_neighbors = optimal_radius

          # --- STEP 4: Initialize migration pathways from grid ---
          start_time = time.perf_counter()
          self._build_kdtree()
          self._initialize_migration_pathways(radius_neighbors, reset_energies=False)

          logger.info("Step 4 (Migration pathways): %.4f seconds", time.perf_counter() - start_time)

          # --- STEP 5: Neighbor analysis (uses FULL grid) ---
          start_time = time.perf_counter()
          self._sequencial_neighbors_analysis()
          logger.info("Step 5 (Neighbor analysis): %.4f seconds", time.perf_counter() - start_time)
          logger.info("TOTAL INITIALIZATION TIME: %.4f seconds", time.perf_counter() - total_start_time)

        # --- STEP 6: Grain Boundaries (if applicable) ---
        start_time = time.perf_counter()
        if hasattr(self.system, 'gb_configurations'):
          self.system.gb_model = GrainBoundary(self.system.crystal_size,self.system.gb_configurations)

          mig_paths = self.system.migration_pathways
          defects_cfg = self.system.defects_config
          reactions_cfg = self.system.reactions_config

          sites_list = list(self.system.grid_crystal.values())

          for i, site in enumerate(sites_list):
            self.system.gb_model.modify_act_energy_GB(site, mig_paths, defects_cfg, reactions_cfg)

          logger.info("Step 6 (Grain boundaries): %.4f seconds", time.perf_counter() - start_time)


        logger.info('Finished grid initialization')


    # =========================================================================
    # Site initialization helpers
    # =========================================================================

    def _efficient_act_e_copy(self, base_dict):
        """
        Memory-efficient copy of Act_E_dict.
        - Copies inner dicts (so per-site energy mods don't leak)
        - Shares CN lists (they are only READ, never mutated per-site)
        """
        site_dict = {}
        for defect_name, energies in base_dict.items():
          # Shallow copy of inner dict: creates new dict, shares list/float refs
          site_dict[defect_name] = energies.copy()

          # Keep CN list reference (safe because they're never mutated)
          if 'CN_clustering_energy' in energies:
            site_dict[defect_name]['CN_clustering_energy'] = energies['CN_clustering_energy']
          if 'CN_redox_energy' in energies:
            site_dict[defect_name]['CN_redox_energy'] = energies['CN_redox_energy']

        return site_dict

    def _get_applicable_defects_for_site(self,site_type):
        """
        Determine which defect configurations apply to this site.
        Called once during site initialization.
        """
        applicable_defects = []
        # Check all defect configurations
        for defect_name, cfg in self.system.defects_config.items():
          allowed_sublattices = cfg.get("allowed_sublattices",[])
          if site_type in allowed_sublattices:
            applicable_defects.append(defect_name)

        return applicable_defects

    def _compute_interface_flags(self):
        """
        Compute interface flags for all sites based on per-site_type z-ranges
        Called after grid is built of loaded
        """
        for defect_name, defect_cfg in self.system.defects_config.items():
          site_type_defect = defect_cfg['site_type']
          type_sites = [s for s in self.system.grid_crystal.values() if s.site_type == site_type_defect]

          z_positions = sorted(set(round(s.position[2], 4) for s in type_sites))
          bottom_z = z_positions[0]
          top_z = z_positions[-1]

          for site in type_sites:
            site.set_interface_flags(bottom_z, top_z)

    def _generate_interstitial_sites(self,api_key=None):
        """
        Generate interstitial sites using the actual species being simulated.
        """

        # Get the interstitial species from defect_config
        interstitial_species = None
        for name, cfg in self.system.defects_config.items():
          if cfg.get('site_type') == 'interstitial':
            interstitial_species = cfg.get("base_element", cfg["symbol"].split("_")[0])
            break

        if interstitial_species is None:
          raise ValueError("No interstitial species found in defects_config")


        # Default minimum distance from atoms, adjust if you find sites too close/far from atoms
        MIN_DISTANCE_FROM_ATOMS = self.system.interstitial_generation['min_distance']  # Angstroms


        # =========================================================================
        # METHOD 1: Try MP charge density (keep for future, will likely fail)
        # =========================================================================
        base_positions_unit_cell = []

        if api_key:
          try:
            with MPRester(api_key) as mpr:
              chgcar = mpr.get_charge_density_from_material_id(self.system.id_material)

              if chgcar is not None:
                logger.info("Charge density retrieved (grid: %s)", chgcar.data.shape)
                cig = ChargeInterstitialGenerator()
                defects = cig.generate(chgcar, insert_species=[interstitial_species])

                defect_struct = next(defects).defect_structure
                # Find the interstitial atom
                for site in reversed(defect_struct):
                  if site.specie.symbol == interstitial_species:
                    base_positions_unit_cell.append(site.coords)
                    break
              else:
                logger.info(" MP returned None (data not available)")


          except Exception as e:
            logger.warning('Warning: MP method failed: %s: %s', type(e).__name__, e)

        # =========================================================================
        # METHOD 2: Voronoi tessellation (PRIMARY - reliable)
        # =========================================================================
        if not base_positions_unit_cell:
          logger.info("\n Using Voronoi tesellation")
          base_positions_unit_cell = self._find_interstitials_voronoi(
            interstitial_species,
            min_distance=MIN_DISTANCE_FROM_ATOMS
          )

        if self.system.calculator_config and self.system.calculator_config.interstitial_refinement.enabled:
          base_positions_unit_cell = self._refine_interstitial_positions(base_positions_unit_cell, interstitial_species)

        # Validate interstitial spacing in the unit cell
        if self.system.rank == 0:
          self._validate_interstitial_positions(base_positions_unit_cell, self.system.structure_basic)

          if self.system.interstitial_generation['create_interstitial_xyz_file']:
            self.create_ovito_xyz_file(interstitial_species, base_positions_unit_cell)


        # =========================================================================
        # Replicate in supercell
        # =========================================================================
        unit_cell_lattice = self.system.structure_basic.lattice
        supercell_interstitials = []
        repetitions = np.ceil(np.array(self.system.crystal_size) / np.array(unit_cell_lattice.abc)).astype(int)

        for cart_pos in base_positions_unit_cell:
          for i in range(repetitions[0]):
            for j in range(repetitions[1]):
              for k in range(repetitions[2]):
                offset = (i * unit_cell_lattice.matrix[0] +
                          j * unit_cell_lattice.matrix[1] +
                          k * unit_cell_lattice.matrix[2])
                new_pos = cart_pos + offset

                if self._is_inside_supercell(new_pos, self.system.structure.lattice):
                    supercell_interstitials.append(new_pos)

        # Remove duplicates
        unique_positions = []
        tol = 0.1
        for pos in supercell_interstitials:
          if not any(np.linalg.norm(np.array(pos) - np.array(existing)) < tol
                     for existing in unique_positions):
            unique_positions.append(pos)

        return unique_positions

    def _find_interstitials_voronoi(self, interstitial_species, min_distance=0.3):
        """
        Find interstitial sites using using pymatgen's VoronoiInterstitialGenerator.
        Finds Voronoi vertices and filters by minimum distance

        Args:
          interstitial_species: Element symbol (e.g., 'H', 'Ag')
          min_distance: Minimum distance from existing atoms (angstroms)
                       Default 1.2 angstroms

        Returns:
          List of cartesian coordinates for interstitial sites
        """

        structure = self.system.structure_basic
        interstitial_positions = []
        # Use InterstitialGenerator

        gen = VoronoiInterstitialGenerator(min_dist=min_distance, clustering_tol=self.system.interstitial_generation['clustering_tol'] )

        sga = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
        symm_ops = sga.get_symmetry_operations()

        # Iterate over the generator
        for interstitial in gen.generate(structure, insert_species=[interstitial_species]):
          unique_frac_coords = interstitial.site.frac_coords

          # Generate all symmetry-equivalent positions
          equiv_positions = set()
          for symm_op in symm_ops:
            new_frac = symm_op.operate(unique_frac_coords)
            new_frac = tuple(np.round(np.mod(new_frac,1.0), 6))
            equiv_positions.add(new_frac)

          for frac_pos in equiv_positions:
            cart_pos = structure.lattice.get_cartesian_coords(frac_pos)

            # Duplicate check (VoronoiGen does clustering, we add this for safety)
            is_duplicate = any(
              np.linalg.norm(cart_pos - existing) < 0.1
              for existing in interstitial_positions
            )

            if not is_duplicate:
              interstitial_positions.append(cart_pos)

        return interstitial_positions

    def _refine_interstitial_positions(self, voronoi_positions, interstitial_species):
        """Refine Voronoi positions to true energy minima using MACE.

        Only runs on the unit-cell positions (typically 4 sites), not the
        full supercell. The refined positions are then replicated normally.
        """
        from kinetix.calculators.mace_neb import KinetixMACEAdapter
        cfg = self.system.calculator_config
        adapter = KinetixMACEAdapter(
          model_source=cfg.model,
          kx=self.system,
          cache_dir=cfg.cache_dir,
          model_filename=cfg.model_filename,
          device=cfg.device,
          default_dtype=cfg.default_dtype,
          n_images=cfg.n_images,
          fmax=cfg.fmax,
          max_steps=cfg.max_steps,
          cluster=cfg.cluster # dict: {"R_active": 5.0, "R_shell": 7.0}
          )

        refined_positions = []
        unit_cell_lattice = self.system.structure_basic.lattice
        supercell_lattice = self.system.structure.lattice

        # Get the Cartesian center of the unit cell and the supercell
        unit_center_cart = unit_cell_lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        super_center_cart = supercell_lattice.get_cartesian_coords([0.5, 0.5, 0.5])

        # Find the center of the supercell in fractional coords
        # This is where we'll place the interstitial for refinement
        for i, pos in enumerate(voronoi_positions):
          # Shift the trial position so it sits in the middle of the supercell
          supercell_pos = super_center_cart + (pos - unit_center_cart)

          # Build a small temporary structure around this Voronoi site
          # Place O_i at pos, relax, get the true position
          refined_pos, disp, energy = adapter.refine_interstitial_site(self.system.grid_crystal,
            supercell_pos, element=interstitial_species
          )

          logger.debug(f"Voronoi site {i}: displacement = {disp:.3f} angstroms (from ({supercell_pos}) to ({refined_pos})), "
                      f"energy = {energy:.3f} eV")

          # --- Map Back to Unit Cell ---
          # Fractional coordinates are invariant to the size of the cell for periodic wrapping.
          # Getting the fractional coords of the relaxed position relative to the supercell
          refined_frac = supercell_lattice.get_fractional_coords(refined_pos)

          # Folding it back to [0, 1)] to obtain the exact unit cell fractional coordinates
          refined_frac_wrapped = np.mod(refined_frac, 1.0)

          # Convert back to unit-cell Cartesian for the replicator
          refined_unit_cart = unit_cell_lattice.get_cartesian_coords(refined_frac_wrapped)

          refined_positions.append(refined_unit_cart)
        return refined_positions

    def _cluster_and_average(self, positions, threshold=0.7):
        """Cluster positions within threshold distance and average each cluster.

        Parameters
        ----------
        positions : list of array-like
            Positions to cluster (e.g., refined interstitial sites)
        threshold : float
            Maximum distance (Å) to consider two sites as the same cluster

        Returns
        -------
        list of np.ndarray
            Centroid positions for each cluster
        """
        if not positions:
          return []

        # Start with each position as its own cluster
        # Each cluster is a list of original Cartesian positions
        clusters = [[np.array(pos)] for pos in positions]

        while True:
          if len(clusters) <= 1:
            break

          # 1. Find the closest pair of cluster centroids
          min_dist = np.inf
          merge_i, merge_j = -1, -1

          for i in range(len(clusters)):
            c_i = np.mean(clusters[i], axis=0)
            for j in range(i + 1, len(clusters)):
              c_j = np.mean(clusters[j], axis=0)
              dist = np.linalg.norm(c_i - c_j)
              if dist < min_dist:
                min_dist = dist
                merge_i, merge_j = i, j

          # 2. If the closest pair is within the threshold, merge them
          if min_dist < threshold:
            logger.debug(f"Merging cluster {merge_i} and {merge_j} "
                         f"(distance {min_dist:.3f} angstroms < {threshold} angstroms)")

            # Combine the atoms from both clusters
            clusters[merge_i].extend(clusters[merge_j])

            # Remove the merged cluster from the list
            del clusters[merge_j]
          else:
            # No more merges possible; the closest pair is to far apart
            break

        # 3. Calculate final centroids for the remaining clusters
        final_positions = [np.mean(c, axis=0) for c in clusters]

        logger.debug(f"Clustered {len(positions)} refined sites into "
                     f"{len(final_positions)} unique interstitial pockets.")

        return final_positions

    def _validate_interstitial_positions(self, positions, structure):
        from scipy.spatial.distance import cdist, pdist

        # Get all atomic positions
        atom_positions = np.array([site.coords for site in structure])
        interstitial_positions = np.array(positions)

        #Calculate minimum distances to atoms
        distances = cdist(interstitial_positions, atom_positions)
        min_distances = np.min(distances,axis=1)

        logger.info("Interstitial-Host spacing: min %.3f, max %.3f, avg %.3f angstroms",
              np.min(min_distances), np.max(min_distances), np.mean(min_distances))

        assert np.min(min_distances) >= 0.4

        # 2. Calculate Interstitial-Interstitial distances (accounting for PBC)
        # Replicate the unit cell positions into a 3x3x3 grid of cells
        lattice = structure.lattice
        replicated_positions = []
        for i in [-1, 0, 1]:
          for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
              offset = i * lattice.matrix[0] + j * lattice.matrix[1] + k * lattice.matrix[2]
              replicated_positions.extend(interstitial_positions + offset)

        replicated_positions = np.array(replicated_positions)

        min_inter_distances = []
        for site in interstitial_positions:
            # Distance from this central site to ALL 27 replicated cells
            dists = np.linalg.norm(replicated_positions - site, axis=1)

            # Filter out the site itself (distance ~ 0.0)
            neighbor_dists = dists[dists > 0.1]

            if len(neighbor_dists) > 0:
                min_inter_distances.append(np.min(neighbor_dists))

        if min_inter_distances:
          logger.info("Interstitial-Interstitial spacing: min %.3f, max %.3f, avg %.3f angstroms",
                  np.min(min_inter_distances), np.max(min_inter_distances), np.mean(min_inter_distances))
        else:
          logger.warning("Could not compute interstitial-interstitial distances.")

    def create_ovito_xyz_file(self,interstitial_species,base_positions_unit_cell, filename="interstitials.xyz"):
        """Create XYZ file for OVITO visualization."""
        # Get host atom positions
        host_atoms = []
        for site in self.system.structure_basic:
            host_atoms.append({
                'element': site.specie.symbol,
                'x': site.coords[0],
                'y': site.coords[1],
                'z': site.coords[2]
            })

        # Get interstitial positions
        interstitials = []
        for pos in base_positions_unit_cell:
            interstitials.append({
                'element': interstitial_species + "_i",  # or use your actual interstitial species like 'Ag'
                'x': pos[0],
                'y': pos[1],
                'z': pos[2]
            })

        # Write XYZ file
        total_atoms = len(host_atoms) + len(interstitials)
        with open(filename, 'w') as f:
            f.write(f"{total_atoms}\n")
            f.write("Generated by KMC simulator\n")

            # Write host atoms
            for atom in host_atoms:
                f.write(f"{atom['element']} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

            # Write interstitials
            for interstitial in interstitials:
                f.write(f"{interstitial['element']} {interstitial['x']:.6f} {interstitial['y']:.6f} {interstitial['z']:.6f}\n")

        logger.info("XYZ file saved: %s", filename)

    def _handle_missing_neighbors(self,radius_neighbors, affected_site):
        """
        Identify and add missing neighbor sites to the grid_crystal.
        Some neighbor sites weren't initially created during grid initialization using structure.
        This method finds these missing sites and adds them to ensure complete neighbor analysis.

        Args:
        radius_neighbors (float): Neighbor search radius
        affected_site: Site type for newly created sites
        """
        tol = 1e-6
        domain_height = tol
        lattice = self.system.structure.lattice

        for site in self.system.structure:
            # Neighbors for each idx in grid_crystal
            neighbors = self.system.structure.get_neighbors(site,radius_neighbors)

            # Some sites are not created with the dictionary comprenhension
            # If the sites have neighbors that are within the crystal dimension range
            # but not included, we included
            for neigh in neighbors:
              pos = neigh.coords
              idx = self.get_idx_coords(pos,self.system.basis_vectors)

              frac = lattice.get_fractional_coords(pos)
              # (1) Vertical membership: within the home cell along the film-normal
              #     direction.
              if not (-tol <= frac[2] <= 1 + tol):
                continue  # Skip if outside the z-range of the unit cell

              if pos[2] > domain_height:
                domain_height = pos[2]

              if idx not in self.system.grid_crystal:
                # (2) Home-cell test in the periodic xy plane: fractional x,y in
                #     [0,1) means this is NOT a wrapped periodic image.
                is_home_xy = ((-tol <= frac[0] <= 1 + tol) and
                              (-tol <= frac[1] <= 1 + tol))

                # If not in the boundary region, where we should apply periodic boundary conditions
                if is_home_xy:
                  site_type = neigh.specie.symbol
                  is_active = self.system._is_active_site(site_type)

                  self.system.grid_crystal[idx] = Site(
                    chemical_specie = site_type,
                    position = tuple(pos),
                    site_type = site_type,
                    Act_E_dict = self._efficient_act_e_copy(self.system.Act_E_dict) if is_active else {},
                    defects_config = self.system.defects_config,
                    reactions_config = self.system.reactions_config,
                    is_active_site=is_active,
                    idx=idx
                  )

        self.system.domain_height = domain_height


    # =========================================================================
    # Neighbor analysis
    # =========================================================================

    def _parallel_neighbors_analysis(self, num_cores=4):
        """Parallel neighbor analysis using grid_crystal."""
        import concurrent.futures
        import math

        try:
          from itertools import batched
        except ImportError: # Python <3.12
          import itertools
          def batched(iterable,n):
            it = iter(iterable)
            while True:
              batch = list(itertools.islice(it,n))
              if not batch:
                break
              yield batch

        grid_keys = list(self.system.grid_crystal.keys())
        batch_size = math.ceil(len(grid_keys) / num_cores)
        batches = list(batched(grid_keys, batch_size))

        # Shared data (immutable)
        shared_data = {
          'crystal_size': self.system.crystal_size,
          'event_labels': self.system.event_labels,
          'radius_neighbors': self.system.radius_neighbors
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
          futures = [
            executor.submit(
              self._process_batch_sites_worker,
              batch,
              self.system.grid_crystal,
              shared_data
            )
            for batch in batches
          ]
          concurrent.futures.wait(futures)

    def _sequencial_neighbors_analysis(self):
        """Sequential neighbor analysis using FULL grid_crystal."""

        for site_idx in self.system.grid_crystal.keys():
          site = self.system.grid_crystal[site_idx]

          # Get neighbors using k-d tree
          neighbor_site_indices = self._get_neighbors_for_site(site_idx, self.system.radius_neighbors)

          if site_idx in neighbor_site_indices:
            neighbor_site_indices.remove(site_idx)

          site.neighbors_analysis(
            self.system.grid_crystal,
            neighbor_site_indices,
            self.system.crystal_size,
            self.system.event_labels,
            site_idx
          )

    def get_num_cores(self,local_max_cores=6):
        # Get number of cores in SLURM, PBS or local machine

        # HPC: SLURM
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            try:
                cores = int(os.environ['SLURM_CPUS_PER_TASK'])
                return max(1, cores - 1)  # Reserve 1 for system/OpenMP
            except (ValueError, TypeError):
                pass

        # HPC: PBS
        if 'PBS_NUM_PPN' in os.environ:
            try:
                cores = int(os.environ['PBS_NUM_PPN'])
                # Reserve 1 core for OpenMP/system processes
                return max(1, cores - 1)
            except ValueError:
                pass

        # Fallback to OMP_NUM_THREADS if set
        if 'OMP_NUM_THREADS' in os.environ:
            try:
                cores = int(os.environ['OMP_NUM_THREADS'])
                return max(1, cores - 1)
            except ValueError:
                pass

        # Local machine:
        try:
            import psutil
            cores = psutil.cpu_count(logical=True)
            if cores is None:
                cores = os.cpu_count() or 1

        except ImportError:
            cores = os.cpu_count() or 1
        return min(cores, local_max_cores)

    def _process_batch_sites_worker(self, batch_keys, grid_crystal, shared_data):
        """
        Worker using FULL grid for neighbor search.

        Args:
              batch_keys: List of site indices to process
              batch_sites: Dict of sites to modify (copies)
              full_grid: Complete grid for neighbor lookups (read-only)
              shared_data: Dict with structure, etc.

          Returns:
              Dict of {idx: modified_site} for sites in this batch
        """
        crystal_size = shared_data['crystal_size']
        event_labels = shared_data['event_labels']
        radius_neighbors = shared_data['radius_neighbors']

        for idx in batch_keys:
          site = grid_crystal[idx]

          # Find neighbors within radius
          site_pos = np.array(site.position)
          neighbors_idx = []
          neighbors_positions = []

          for neigh_idx, neigh_site in grid_crystal.items():
            if neigh_idx == idx:
              continue
            dist = np.linalg.norm(np.array(neigh_site.position) - site_pos)
            if dist <= radius_neighbors:
              neighbors_idx.append(neigh_idx)
              neighbors_positions.append(neigh_site.position)

          # Modify original object in-place
          site.neighbors_analysis(
            grid_crystal,
            neighbors_idx,
            neighbors_positions,
            crystal_size,
            event_labels,
            idx
          )


    # =========================================================================
    # Coordinates, Wulff shape & edges
    # =========================================================================

    def get_idx_coords(self, coords,basis_vectors):
        # Check if the coordinates are already in the cache
        coords_tuple = tuple(coords)
        if coords_tuple not in self.system.coord_cache:
            # Calculate and cache the rounded coordinates
            # np.linalg.solve --> To obtain the linear combination of the basis vector for the site coordinate
            idx_coords = np.linalg.solve(basis_vectors.transpose(), coords)
            idx_coords = tuple(np.round(idx_coords).astype(int))
            self.system.coord_cache[coords_tuple] = idx_coords
        return self.system.coord_cache[coords_tuple]

    def Wulff_Shape(self,api_key):

        with MPRester(api_key=api_key) as mpr:

            # Check what attributes are available in mpr
            # available_attributes = [attr for attr in dir(mpr) if not attr.startswith('_')]
            # print(f"Available attributes in MPRester: {available_attributes}")
            # surface_properties_doc = mpr.surface_properties.search(material_ids=[self.system.id_material])

            surface_properties_doc = mpr.materials.surface_properties.search(
                material_ids=self.system.id_material
                )

        miller_indices = []
        surface_energies = []
        for surface in surface_properties_doc[0].surfaces:
            miller_indices.append(tuple(surface.miller_index))
            surface_energies.append(surface.surface_energy)

        self.system.wulff_shape = WulffShape(self.system.structure_basic.lattice, miller_indices,surface_energies)

        # We can show the Wulff Shape with the following:
        #self.system.wulff_shape.show()

        self.system.wulff_facets = [] # Miller index and normal vector
        for facet in self.system.wulff_shape.facets:
            self.system.wulff_facets.append([facet.miller,facet.normal])

        # I can still eliminate the parallel normal vectors
        self.system.wulff_facets = sorted(self.system.wulff_facets,key = lambda x:x[0][0])

    def create_edges(self,facets_type):

        """
        To obtain the relation between the migration, the edges and the facets we first need:
            1. Calculate the edges
            2. Calculate the facets parallel to each migration
            3. Calculate the edges parallel to each migration
            4. Associate each edge with a facet type
            5. Relation between migration direction, edges and facet type

            Create a dictionary:
                dir_edge_facets[migration_label] = aux_edge_facet
                aux_edge_facet - list of lists:
                    aux_edge_facet[0][0] = (4, 11) migrations defining the edge
                    aux_edge_facet[0][1] = (1,0,0) facets

        """
        lattice = self.system.structure.lattice
        for idx,site in self.system.grid_crystal.items():
          frac = lattice.get_fractional_coords(site.position)
          if ((0.45 < frac[0] < 0.55)
              and (0.45 < frac[1] < 0.55)
              and frac[2] < 0.2):
              break            # Introduce specie in the site

        # Obtain the different edge in the plane
        # Neighbors only in plane
        neighbors = [[self.system.grid_crystal[neigh[0]].position,neigh[1]] for neigh in self.system.grid_crystal[idx].migration_paths['Plane']]
        # Minimum distance between neighbors
        min_dist = np.linalg.norm(np.array(self.system.grid_crystal[idx].position) - np.array(np.array(neighbors[4][0])))
        edges = {}

        for neighbor in neighbors:
            for j in range(len(neighbors)):
                if (math.isclose(np.linalg.norm(np.array(neighbor[0]) - np.array(np.array(neighbors[j][0]))), min_dist)) and ((neighbors[j][1],neighbor[1]) not in
                                                                                                                              edges):
                    edges[(neighbor[1],neighbors[j][1])] = np.array(neighbor[0]) - np.array(np.array(neighbors[j][0]))

        # Calculate the facets that are parallel to each migration
        mig_directions = {neigh[1]:np.array(self.system.grid_crystal[neigh[0]].position) - np.array(self.system.grid_crystal[idx].position) for neigh in self.system.grid_crystal[idx].migration_paths['Plane']}
        mig_parallel_facets = {}
        #Search for the facets that are parallel to the migration direction
        for mig_direct,vector in mig_directions.items():
            facet_list = []
            for facet in self.system.wulff_facets:
                if (facet[1][2] > 0 and facet[1][2] != 1 and # Screen facets that are looking downward or parallel to the x-y plane
                    abs(np.dot(facet[1][:2],vector[:2])) < 1e-12): # Perpendicular between facet normal vector (x-y) and migration direction
                        facet_list.append(facet)

            mig_parallel_facets[mig_direct] = facet_list


        # Calculate the edges parallel to the migration
        parallel_mig_direction_edges = {}
        for mig,vector_dir in mig_directions.items():
            list_edges = []

            for edge,vector in edges.items():
                if np.linalg.norm(np.cross(vector,vector_dir)) < 1e-12: # Migration vector is parallel to the edges
                    list_edges.append(edge)

            parallel_mig_direction_edges[mig] = list_edges


        # Associate edge with facets
        # Edge is defined by two migrations: we sum those vectors to obtain a vector that should be parallel to the facet normal vector
        self.system.dir_edge_facets = {}
        for mig,edges_2 in parallel_mig_direction_edges.items():
            aux_edge_facet = []
            for edge in edges_2:
                v1 = mig_directions[edge[0]] + mig_directions[edge[1]]

                for facet in mig_parallel_facets[mig]:
                    if np.dot(v1,facet[1]) > 0: # Pointing in the same direction
                       aux_edge_facet.append([edge,facet[0]])

            self.system.dir_edge_facets[mig] = aux_edge_facet

        self.system.dir_edge_facets = {
            key: [sublist for sublist in value if sublist[1] in facets_type]
            for key, value in self.system.dir_edge_facets.items()
        }


    # =========================================================================
    # Cluster tracking init
    # =========================================================================

    def _initialize_cluster_tracking(self):
        """ Call once at simulation start """
        self.system.atom_to_cluster = {} # Mapping from site_id to cluster_id
        self.system.clusters = {}
        self.system.next_cluster_id = 0
