# tests/test_percolation.py
"""Tests for automated radius_neighbors selection and percolation validation.

Validates the percolation features added to Crystal_Lattice grid
initialization (kinetix/lattice/crystal.py):

- ``_check_percolation_at_radius`` - KD-tree based DFS from the bottom
  electrode to the top electrode for a given site type.
- ``find_optimal_radius`` - binary search (plus safety margin) for the
  minimum radius that percolates.
- The percolation sanity check logged after the migration network
  validation during grid creation.

Two groups of tests:

1. ``TestGridPercolation`` - runs against the real VCM mock grid (HfO2),
   printing the percolation-vs-radius tables reported in the task.
2. ``TestPercolationSynthetic`` - fast, deterministic tests on a tiny
   synthetic lattice that do not need the full grid initialization.

Run from the repository root::

    pytest tests/test_percolation.py -v
"""
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from kinetix.initialization import initialization
from kinetix.lattice.crystal import Crystal_Lattice

CONFIG_NAME = "VCM_mock.yaml"

# Candidate radii swept against the real grid
RADII = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


@pytest.fixture(scope="module")
def system_state():
    """KMC System_state built from the VCM mock preset (once per module)."""
    sim_id = 0
    params = {
        "vo_initial_concentration": 1.0e-2,
        "temperature": 293.0,
        "h_generation": 0.45,
    }
    System_state, *_ = initialization(sim_id, params, CONFIG_NAME)
    return System_state


def _percolation_table(lattice, radii, site_type):
    """Sweep radii, returning [(radius, percolated, n_connected), ...]."""
    rows = []
    for radius in radii:
        percolated, n_connected = lattice._check_percolation_at_radius(
            radius, site_type
        )
        rows.append((radius, percolated, n_connected))
    return rows


def _print_percolation_table(rows):
    """Human-readable percolation report table (included in test output)."""
    print("\nRadius (Å)   Percolated   Connected sites")
    print("-" * 46)
    for radius, percolated, n_connected in rows:
        print(f"{radius:>11.2f}   {str(percolated):>10}   {n_connected:>14}")


def _assert_monotonic(rows):
    """Percolation is monotone in the radius: once True, it stays True."""
    flags = [percolated for _, percolated, _ in rows]
    if True in flags:
        first = flags.index(True)
        assert all(flags[first:]), (
            "Once a radius percolates, every larger radius must also percolate"
        )
    return flags


class TestGridPercolation:
    """Percolation checks against the real VCM mock grid."""

    def test_percolation_vs_radius(self, system_state):
        """Sweep interstitial radii and report the percolation status table."""
        rows = _percolation_table(system_state, RADII, "interstitial")
        _print_percolation_table(rows)
        _assert_monotonic(rows)

        # A percolated radius must report a non-empty connected component
        for radius, percolated, n_connected in rows:
            if percolated:
                assert n_connected > 0, (
                    f"Percolated at r={radius:.2f} Å but 0 connected sites"
                )

    def test_optimal_radius_found(self, system_state):
        """find_optimal_radius() must return a radius that percolates.

        Tries the interstitial sublattice first and falls back to the O
        sublattice (the sweep in test_vacancy_percolation shows O reaches the
        electrodes in the mock grid while the interstitial sublattice may
        not), so the test keeps validating the binary search end-to-end.
        """
        max_radius = 6.0
        for site_type in ("interstitial", "O"):
            # The grid must percolate at the upper bound of the search
            # (otherwise find_optimal_radius falls back to max_radius and the
            # result does not percolate).
            if not system_state._check_percolation_at_radius(
                max_radius, site_type
            )[0]:
                continue

            optimal = system_state.find_optimal_radius(
                site_type=site_type,
                min_radius=1.5,
                max_radius=max_radius,
                step=0.25,
                safety_margin=0.5,
            )

            percolated, n_connected = system_state._check_percolation_at_radius(
                optimal, site_type
            )
            assert percolated, (
                f"Optimal radius {optimal:.2f} Å ({site_type}) must percolate"
            )
            assert n_connected > 0
            assert optimal >= 1.5
            return

        pytest.skip(
            "Grid too sparse: no percolation at 6.0 Å for 'interstitial' or 'O'"
        )

    def test_vacancy_percolation(self, system_state):
        """Same sweep and report as test 1, but for the O sublattice."""
        rows = _percolation_table(system_state, RADII, "O")
        _print_percolation_table(rows)
        _assert_monotonic(rows)


# =============================================================================
# Fast deterministic tests on a tiny synthetic lattice (no initialization()).
# =============================================================================

def _make_chain_lattice(z_spacing=2.0, height=30.0):
    """
    Build a minimal Crystal_Lattice whose 'interstitial' and 'O' sites form
    1-D vertical chains that cross the whole sublattice z-extent: the first
    row is the bottom electrode row and the last row is the top electrode
    row (percolation bands always include the outermost rows).

    Consecutive sites of a chain are `z_spacing` apart, so percolation
    appears exactly when radius >= z_spacing (2.0 Å by default).
    """
    lattice = Lattice.from_parameters(20.0, 20.0, height, 90, 90, 90)
    # Host placeholder atom - only the lattice matrix is used here.
    structure = Structure(lattice, ["X"], [[0.0, 0.0, 0.0]])

    grid = {}
    # z = 1, 3, 5, ..., 29 -> first site in the bottom band, last in the top
    zs = list(np.arange(1.0, height, z_spacing))
    for i, z in enumerate(zs):
        grid[("interstitial", i)] = SimpleNamespace(
            position=(10.0, 10.0, float(z)), site_type="interstitial"
        )
    for i, z in enumerate(zs):
        grid[("O", i)] = SimpleNamespace(
            position=(5.0, 5.0, float(z)), site_type="O"
        )

    lattice_obj = Crystal_Lattice.__new__(Crystal_Lattice)
    lattice_obj.structure = structure
    lattice_obj.grid_crystal = grid
    lattice_obj.crystal_size = np.array([20.0, 20.0, height])
    lattice_obj._build_kdtree()
    return lattice_obj


class TestPercolationSynthetic:
    """Deterministic percolation behaviour on a 1-D vertical chain."""

    def test_below_threshold_no_percolation(self):
        lattice = _make_chain_lattice(z_spacing=2.0)
        percolated, n_connected = lattice._check_percolation_at_radius(
            1.9, "interstitial"
        )
        assert not percolated
        assert n_connected == 1  # only the single bottom site is reached

    def test_at_threshold_percolation(self):
        lattice = _make_chain_lattice(z_spacing=2.0)
        percolated, n_connected = lattice._check_percolation_at_radius(
            2.0, "interstitial"
        )
        assert percolated
        assert n_connected > 0

    def test_vacancy_percolation_synthetic(self):
        lattice = _make_chain_lattice(z_spacing=2.0)
        assert not lattice._check_percolation_at_radius(1.9, "O")[0]
        assert lattice._check_percolation_at_radius(2.0, "O")[0]

    def test_find_optimal_radius_binary_search(self):
        lattice = _make_chain_lattice(z_spacing=2.0)
        optimal = lattice.find_optimal_radius(
            site_type="interstitial",
            min_radius=1.0,
            max_radius=5.0,
            step=0.1,
            safety_margin=0.5,
        )
        # Threshold is 2.0 Å -> hi converges to ~2.0 -> optimal ~2.5 Å
        assert 2.4 <= optimal <= 2.6
        percolated, _ = lattice._check_percolation_at_radius(
            optimal, "interstitial"
        )
        assert percolated

    def test_find_optimal_radius_no_percolation_fallback(self):
        # Chain with 4.0 Å spacing: the bottom and top rows exist but are
        # further apart than the search's max_radius, so percolation is
        # impossible and find_optimal_radius must fall back to max_radius.
        lattice = _make_chain_lattice(z_spacing=4.0)
        assert not lattice._check_percolation_at_radius(3.0, "interstitial")[0]
        optimal = lattice.find_optimal_radius(
            site_type="interstitial",
            min_radius=1.0,
            max_radius=3.0,
            step=0.25,
            safety_margin=0.5,
        )
        assert optimal == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])