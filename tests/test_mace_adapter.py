# tests/test_mace_adapter.py
"""Pytest integration tests for the MACE CI-NEB adapter (KinetixMACEAdapter).

Run from the repository root::

    pytest tests/test_mace_adapter.py -v

Skips cleanly when the MACE model file or the optional mace-torch stack
(torch + mace) is unavailable.
"""
import time
from pathlib import Path

import numpy as np
import pytest

# Repository root (parent of tests/); the model lives under data/cache/...
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Module-level import guards ---------------------------------------------
# Import the adapter from its submodule (kinetix/calculators/__init__.py is an
# empty marker package and intentionally does not pull the mace_neb dependency
# chain). If any import-time dependency (e.g. ASE) is missing, skip the whole
# module cleanly instead of failing collection.
try:
    from kinetix.initialization import initialization
    from kinetix.calculators.mace_neb import KinetixMACEAdapter
except ImportError as exc:  # pragma: no cover - only hit when deps are absent
    pytest.skip(f"MACE adapter imports unavailable: {exc}",
                allow_module_level=True)

# --- Constants ---------------------------------------------------------------
CONFIG_NAME = "VCM_mock.yaml"
MODEL_PATH = (REPO_ROOT / "data" / "cache" / "neb_cache"
              / "HfO2_mh1_F_LONG_cpu.model")
R_ACTIVE = 5.0
R_SHELL = 7.0
BARRIER_BOUNDS = (0.05, 3.0)  # eV
CACHE_MAX_S = 0.1             # seconds budget for a cached barrier lookup


@pytest.fixture(scope="module")
def system_state():
    """KMC System_state built from the VCM mock preset (once per session)."""
    sim_id = 0
    params = {
        "vo_initial_concentration": 1.0e-2,
        "temperature": 293.0,
        "h_generation": 0.45,
    }
    System_state, *_ = initialization(sim_id, params, CONFIG_NAME)
    return System_state


@pytest.fixture(scope="module")
def mace_adapter(system_state):
    """CPU-bound KinetixMACEAdapter, or a clean skip if MACE is unavailable."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"MACE model not found: {MODEL_PATH}")

    try:
        adapter = KinetixMACEAdapter(
            str(MODEL_PATH),
            kx=system_state,
            device="cpu",
            cluster={"R_active": R_ACTIVE, "R_shell": R_SHELL},
            n_images=5,
            fmax=0.05,
        )
    except ImportError as exc:
        pytest.skip(f"MACE dependencies unavailable: {exc}")

    return adapter


@pytest.fixture(scope="module")
def oi_hop(system_state):
    """(origin_idx, dest_idx) for an oxygen interstitial hop well inside the
    domain (>= R_SHELL from the z boundaries). Built once per session."""
    Lz = system_state.crystal_size[2]
    center = np.array(system_state.crystal_size) / 2.0

    # Empty interstitial closest to the geometric center, away from the
    # z boundaries, so the cluster cut sees bulk-like surroundings.
    best, origin_idx = np.inf, None
    for idx, site in system_state.grid_crystal.items():
        if site.site_type != "interstitial" or site.chemical_specie != "Empty":
            continue
        if not (R_SHELL <= site.position[2] <= Lz - R_SHELL):
            continue
        d = np.linalg.norm(np.array(site.position) - center)
        if d < best:
            best, origin_idx = d, idx

    if origin_idx is None:
        pytest.skip("No bulk-like empty interstitial found in the mock grid")

    # Introduce an oxygen interstitial at the origin site.
    cfg = system_state.defects_config["oxygen_interstitial"]
    support_update_sites = set()
    event_update_sites = set()
    system_state._introduce_specie_site(
        origin_idx, support_update_sites, event_update_sites,
        cfg["symbol"], cfg["charge"],
    )

    # Destination: a neighboring empty interstitial.
    origin = system_state.grid_crystal[origin_idx]
    dest_idx = next(
        n for n in origin.nearest_neighbors_idx
        if system_state.grid_crystal[n].site_type == "interstitial"
        and system_state.grid_crystal[n].chemical_specie == "Empty"
    )

    system_state.update_sites_topology(support_update_sites,
                                       event_update_sites)
    return origin_idx, dest_idx


# =============================================================================
# Geometry tests: pure adapter bookkeeping, no NEB run required.
# =============================================================================
class TestMACEAdapterGeometry:
    """Candidate-cluster search and IS/FS pair construction."""

    def test_candidate_keys_exact(self, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        # Hop center: midpoint between origin and destination.
        origin_pos = np.array(grid[origin_idx].position, float)
        dest_pos = np.array(grid[dest_idx].position, float)
        center = 0.5 * (origin_pos + dest_pos)

        # Brute-force expected set: every grid site whose minimum-image
        # distance to the center is <= R_SHELL.
        expected = {
            k for k, s in grid.items()
            if np.linalg.norm(
                mace_adapter.kx._minimum_image_vector(
                    np.array(s.position, float) - center)
            ) <= R_SHELL
        }

        found = set(mace_adapter._candidate_keys(center))
        assert expected == found, "candidate search mismatch"
        print(f"found {len(found)} candidate sites within "
              f"R_shell={R_SHELL} Angstrom")

    def test_build_pair_geometry(self, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        start, end, frozen = mace_adapter.build_pair(
            grid, origin_idx, dest_idx)

        assert len(start) == len(end)
        assert sorted(set(start.symbols)) == ["Hf", "O"]
        assert np.allclose(start.positions[-1], grid[origin_idx].position)
        assert np.allclose(end.positions[-1], grid[dest_idx].position)
        assert len(start) - 1 not in frozen, "moving atom must never be frozen"
        assert len(frozen) > 0, "expected a frozen shell in cluster mode"
        print(f"cluster: {len(start)} atoms "
              f"({len(frozen)} frozen in the R_Active={R_ACTIVE} shell)")


# =============================================================================
# Barrier tests: actual CI-NEB runs (need torch + mace-torch installed).
# =============================================================================
class TestMACEAdapterBarrier:
    """Barrier sanity/convergence and SQLite cache-hit performance.

    These tests actually run CI-NEB and therefore additionally need the
    optional mace-torch stack (torch + mace); without it they skip cleanly.
    """

    @pytest.fixture(scope="module")
    def requires_mace(self):
        """Skip the barrier tests cleanly when torch/mace-torch are missing."""
        try:
            import mace  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"mace-torch (torch + mace) not installed: {exc}")

    def test_barrier_sane_and_converged(self, requires_mace, mace_adapter,
                                        oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        result = mace_adapter.get_barrier(grid, origin_idx, dest_idx,
                                          full_output=True)
        assert result["converged"] is True
        assert (BARRIER_BOUNDS[0] < result["barrier"]
                < BARRIER_BOUNDS[1])
        print(f"O_i hop barrier: {result['barrier']:.3f} eV "
              f"(converged={result['converged']})")

    def test_cache_hit_fast(self, requires_mace, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        # First call populates the SQLite cache.
        mace_adapter.get_barrier(grid, origin_idx, dest_idx)

        t0 = time.perf_counter()
        mace_adapter.get_barrier(grid, origin_idx, dest_idx)
        dt = time.perf_counter() - t0
        assert dt < CACHE_MAX_S, f"cache hit too slow: {dt:.4f}s"
        print(f"cache hit time: {dt:.4f} s")