# tests/test_mace_adapter.py
"""Pytest integration tests for the MACE CI-NEB adapter (KinetixMACEAdapter).

Run from the repository root::

    pytest tests/test_mace_adapter.py -v

Skips cleanly when the MACE model file or the optional mace-torch stack
(torch + mace) is unavailable.
"""
import csv
import os
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
    lattice = system_state.structure.lattice
    # Geometric center of the supercell (orientation-independent).
    center = lattice.get_cartesian_coords([0.5, 0.5, 0.5])

    # Fractional threshold: convert R_SHELL (Å) to fractional z units.
    # Uses the z-component of the c-vector (the film-normal direction).
    z_height = abs(lattice.matrix[2][2])
    frac_thr = R_SHELL / z_height

    # Empty interstitial closest to the geometric center, away from the
    # z boundaries, so the cluster cut sees bulk-like surroundings.
    best, origin_idx = np.inf, None
    for idx, site in system_state.grid_crystal.items():
        if site.site_type != "interstitial" or site.chemical_specie != "Empty":
            continue

        # Orientation-independent z-boundary check (replaces Cartesian z
        # vs crystal_size[2]). Site must be >= R_SHELL from top and bottom.
        frac = lattice.get_fractional_coords(site.position)
        if not (frac_thr <= frac[2] <= 1.0 - frac_thr):
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
    dest_idx = None
    for n in origin.nearest_neighbors_idx:
        neighbor = system_state.grid_crystal[n]
        if neighbor.site_type == "interstitial" and neighbor.chemical_specie == "Empty":
            dest_idx = n
            break

    # Fallback: search the whole grid for the nearest empty interstitial
    if dest_idx is None:
        origin_pos = np.array(origin.position)
        best_dist = np.inf
        for idx, neighbor in system_state.grid_crystal.items():
            if idx == origin_idx:
                continue
            if neighbor.site_type != "interstitial":
                continue
            if neighbor.chemical_specie != "Empty":
                continue
            # Use minimum image distance
            v = system_state._minimum_image_vector(np.array(neighbor.position) - origin_pos)
            dist = np.linalg.norm(v)
            if dist < best_dist:
                best_dist = dist
                dest_idx = idx

        if dest_idx is None:
            pytest.skip("No neighboring empty interstitial found for the hop test")

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

        """
        from ase.io import write
        write("IS.extxyz", start)
        write("FS.extxyz", end)
        """    

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
                                          use_cache=True, full_output=True)

        print(f"Profile (eV, rel to IS): {result['profile']}")   # ← ADD THIS
        print(f"Max along band: {max(result['profile']):.4f} eV")

        assert result["converged"] is True
        assert (BARRIER_BOUNDS[0] < result["barrier"]
                < BARRIER_BOUNDS[1])
        print(f"O_i hop barrier: {result['barrier']:.3f} eV "
              f"(converged={result['converged']})")

    def test_cache_hit_fast(self, requires_mace, mace_adapter, oi_hop):
        grid = mace_adapter.kx.grid_crystal
        origin_idx, dest_idx = oi_hop

        # First call populates the SQLite cache.
        mace_adapter.get_barrier(grid, origin_idx, dest_idx,
                                 use_cache=True)

        t0 = time.perf_counter()
        mace_adapter.get_barrier(grid, origin_idx, dest_idx, use_cache=True)
        dt = time.perf_counter() - t0
        assert dt < CACHE_MAX_S, f"cache hit too slow: {dt:.4f}s"
        print(f"cache hit time: {dt:.4f} s")

    # def


# =============================================================================
# Comprehensive pathway tests: barriers for ALL migration pathways from
# representative bulk-like sites (SLOW: many CI-NEB calculations, hours).
# Run with: pytest tests/test_mace_adapter.py -k AllPathways -v
# Skip with: pytest tests/test_mace_adapter.py -m "not slow"
# =============================================================================
def _representative_site(system_state, wanted_types, wanted_specie=None):
    """Site of one of wanted_types closest to the supercell center, away from
    the z boundaries (>= R_SHELL fractional check, as in oi_hop)."""
    lattice = system_state.structure.lattice
    center = lattice.get_cartesian_coords([0.5, 0.5, 0.5])
    z_height = abs(lattice.matrix[2][2])
    frac_thr = R_SHELL / z_height

    candidates = []
    for idx, site in system_state.grid_crystal.items():
        if site.site_type not in wanted_types:
            continue
        if wanted_specie is not None and site.chemical_specie != wanted_specie:
            continue
        frac = lattice.get_fractional_coords(site.position)
        if not (frac_thr <= frac[2] <= 1.0 - frac_thr):
            continue
        d = np.linalg.norm(np.array(site.position, float) - center)
        candidates.append((d, idx))

    if not candidates:
        pytest.skip(f"No bulk-like {wanted_types} site found in the mock grid")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@pytest.mark.slow
class TestMACEAdapterAllPathways:
    """Comprehensive barrier calculations for ALL pathways from representative
    bulk-like sites. Each hop runs a full CI-NEB relaxation, so this class
    takes hours; deselect during rapid development with `pytest -m "not slow"`.
    """

    @pytest.fixture(scope="class")
    def representative_interstitial(self, system_state):
        """Bulk-like EMPTY interstitial closest to the supercell center."""
        return _representative_site(system_state, ("interstitial",), "Empty")

    @pytest.fixture(scope="class")
    def representative_vacancy(self, system_state):
        """Host oxygen site closest to the supercell center."""
        return _representative_site(system_state, ("O",), "O")

    @staticmethod
    def _calculate_all_pathways(system_state, mace_adapter, origin_idx,
                                dest_site_types, csv_path):
        """Barriers for all eligible neighbors of origin_idx (defect already
        introduced by the caller). Prints results and saves them to csv_path."""
        grid = system_state.grid_crystal
        origin_site = grid[origin_idx]
        origin_pos = np.array(origin_site.position, float)

        results = []
        for neighbor_idx in origin_site.nearest_neighbors_idx:
            neighbor_site = grid[neighbor_idx]
            if neighbor_site.site_type not in dest_site_types:
                continue

            # Distance under the minimum image convention.
            dest_pos = np.array(neighbor_site.position, float)
            v = system_state._minimum_image_vector(dest_pos - origin_pos)
            distance = np.linalg.norm(v)

            try:
                barrier = mace_adapter.get_barrier(
                    grid, origin_idx, neighbor_idx, use_cache=True)
            except Exception as exc:  # non-converging NEB must not kill sweep
                barrier = None
                print(f"FAILED barrier for hop {origin_idx} -> {neighbor_idx}: {exc}")

            specie = neighbor_site.chemical_specie
            results.append({
                "origin_idx": origin_idx,
                "dest_idx": neighbor_idx,
                "origin_pos": " ".join(f"{c:.4f}" for c in origin_pos),
                "dest_pos": " ".join(f"{c:.4f}" for c in dest_pos),
                "distance": f"{distance:.4f}",
                "barrier": f"{barrier:.4f}" if barrier is not None else "",
                "chemical_specie": specie,
            })

            barrier_str = f"{barrier:.3f} eV" if barrier is not None else "FAILED"
            print(f"Hop {origin_idx} -> {neighbor_idx}: "
                  f"origin_pos={origin_pos} final_pos={dest_pos} "
                  f"distance={distance:.3f} Ang barrier={barrier_str} "
                  f"specie={specie}")

        if results:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to {csv_path}")

        n_ok = sum(1 for r in results if r["barrier"])
        print(f"Pathway sweep from site {origin_idx}: {len(results)} pathways, "
              f"{n_ok} barriers OK, {len(results) - n_ok} failed")
        return results

    @pytest.mark.slow
    def test_all_interstitial_pathways(self, system_state, mace_adapter,
                                       representative_interstitial):
        """Barriers for ALL interstitial hops from a representative O_i site."""
        origin_idx = representative_interstitial

        # Introduce an oxygen interstitial at the representative site.
        cfg = system_state.defects_config["oxygen_interstitial"]
        support_update_sites = set()
        event_update_sites = set()
        system_state._introduce_specie_site(
            origin_idx, support_update_sites, event_update_sites,
            cfg["symbol"], cfg["charge"],
        )
        system_state.update_sites_topology(support_update_sites,
                                           event_update_sites)

        csv_path = REPO_ROOT / "test_output" / "interstitial_pathways.csv"
        results = self._calculate_all_pathways(
            system_state, mace_adapter, origin_idx, ("interstitial",),
            str(csv_path))

        assert len(results) > 0, "No interstitial neighbor pathways found"

    @pytest.mark.slow
    def test_all_vacancy_pathways(self, system_state, mace_adapter,
                                  representative_vacancy):
        """Barriers for ALL oxygen hops from a representative V_O site.

        V_O is represented by the host oxygen site becoming an Empty space;
        _introduce_specie_site with the oxygen_vacancy config marks it V_O and
        the MACE adapter then builds the IS/FS pair with the oxygen REMOVED
        (build_pair skips the vacant site's atom), so no special handling is
        needed here beyond the standard introduction.
        """
        origin_idx = representative_vacancy

        # Introduce an oxygen vacancy at the representative host site.
        cfg = system_state.defects_config["oxygen_vacancy"]
        support_update_sites = set()
        event_update_sites = set()
        system_state._introduce_specie_site(
            origin_idx, support_update_sites, event_update_sites,
            cfg["symbol"], cfg["charge"],
        )
        system_state.update_sites_topology(support_update_sites,
                                           event_update_sites)

        csv_path = REPO_ROOT / "test_output" / "vacancy_pathways.csv"
        results = self._calculate_all_pathways(
            system_state, mace_adapter, origin_idx, ("O",), str(csv_path))

        assert len(results) > 0, "No oxygen neighbor pathways found"