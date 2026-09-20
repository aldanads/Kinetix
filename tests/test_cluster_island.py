# tests/test_cluster_island.py
"""
Behavioral spec for morphology detection: metal clusters (kinetix/lattice/cluster.py,
Crystal_Lattice._dfs_find_components in crystal.py) and islands/terraces
(kinetix/lattice/island.py, Crystal_Lattice.detect_islands/build_island/islands_analysis).

These algorithms will move during the crystal.py split, so they are pinned here
using small mock grids with known connectivity (tuple site indices, explicit
nearest_neighbors_idx / supp_by / migration_paths).

The surviving Crystal_Lattice graph-traversal methods (_dfs_find_components,
detect_islands, build_island, islands_analysis) only touch ``self.grid_crystal``
plus a handful of scalar attributes, so they are exercised through unbound calls
on a lightweight stand-in object - no Crystal_Lattice constructor, no physics.

PRODUCTION QUIRKS (documented):
  1. Cluster.update_electrode_contact uses if 'bottom_layer' in supp_by / elif
     'top_layer' - a site supported by BOTH layers counts only as bottom.
  2. Cluster._slice_cluster silently DROPS a slice if it overlaps an already
     accepted slice (overlap -> skip), instead of merging.
  3. island.py::_build_cluster_with_slices evaluates self.slice_list[i+1] and
     [i+2] while scanning for the merge layer -> IndexError when a single-slice
     layer sits within the last two layers (instead of falling through to the
     System_state.layers fallback).

REMOVED TEST SUITES: TestDfsExplore and TestMetalClustersAnalysis were deleted
because Crystal_Lattice._dfs_explore and Crystal_Lattice.metal_clusters_analysis
were removed as dead code in commit 8630510 ("fix bugs in cluster logic"); the
tests exercised only the deleted methods and failed with AttributeError.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from kinetix.lattice.cluster import Cluster
from kinetix.lattice.crystal import Crystal_Lattice
from kinetix.lattice.island import Island


# =============================================================================
# Mock helpers - mock grids with known structure
# =============================================================================

def make_site(
    position,
    specie="Ag",
    ion_charge=0,
    neighbors=(),
    supp_by=(),
    plane=(),
    up=(),
    down=(),
):
    """A minimal Site stand-in with only the attributes the morphology
    algorithms read."""
    return SimpleNamespace(
        position=position,
        chemical_specie=specie,
        ion_charge=ion_charge,
        nearest_neighbors_idx=list(neighbors),
        supp_by=list(supp_by),
        migration_paths={
            "Plane": [(p, None) for p in plane],
            "Up": [(u, None) for u in up],
            "Down": [(d, None) for d in down],
        },
        in_cluster_with_electrode={"bottom_layer": False, "top_layer": False},
    )


def make_grid_5x5(occupied, base_specie="Ag", empty_specie="Empty", periodic=False):
    """5x5 single-layer grid (indices (x, y), all at z=0).

    ``occupied`` is an iterable of (x, y) sites that carry ``base_specie``;
    every other site is 'Empty'. Nearest neighbors are +/-x, +/-y without
    wrapping unless ``periodic`` (boundary sites then have fewer neighbors -
    the boundary edge case).
    """
    coords = [(x, y) for x in range(5) for y in range(5)]

    def nbrs(c):
        x, y = c
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if periodic:
            cand = [(a % 5, b % 5) for a, b in cand]
        return [n for n in cand if n in coords]

    occupied = set(occupied)

    return {
        c: make_site(
            position=(c[0], c[1], 0),
            specie=base_specie if c in occupied else empty_specie,
            neighbors=nbrs(c),
            # detect_islands/build_island traverse migration_paths (not
            # nearest_neighbors_idx), so Plane links are populated as well.
            plane=nbrs(c),
        )
        for c in coords
    }


def call(method, fake_self, *args, **kwargs):
    """Call a Crystal_Lattice method through a stand-in self (unbound call)."""
    return method(fake_self, *args, **kwargs)


def make_lattice(grid):
    """A Crystal_Lattice without __init__ (no grid rebuild, no physics).

    Using a real instance (rather than SimpleNamespace) matters for the
    recursive traversals - detect_islands re-enters itself through
    ``self.detect_islands``, which only resolves on a Crystal_Lattice instance.
    """
    lattice = Crystal_Lattice.__new__(Crystal_Lattice)
    lattice.grid_crystal = grid
    return lattice

class TestDfsFindComponents:
    """Connected same-species neutral sites group into components."""

    def _grid(self):
        # L-shaped metal trimer a-b-c plus isolated d.
        grid = {
            "a": make_site(position=(0, 0, 0), neighbors=["b", "c"]),
            "b": make_site(position=(1, 0, 0), neighbors=["a"]),
            "c": make_site(position=(0, 1, 0), neighbors=["a"]),
            "d": make_site(position=(5, 5, 0), neighbors=[]),
        }
        return grid, ["a", "b", "c", "d"]

    def test_connected_sites_form_one_component(self):
        grid, atoms = self._grid()
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        # Two components: the L-trimer and the isolated 'd'.
        assert len(components) == 2
        assert set(components[0]) == {"a", "b", "c"}

    def test_isolated_site_is_its_own_component(self):
        grid, atoms = self._grid()
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert ["d"] in components

    def test_cluster_size_distribution(self):
        grid, atoms = self._grid()
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert Counter(len(c) for c in components) == Counter({3: 1, 1: 1})

    def test_charged_neighbor_breaks_connectivity(self):
        # DFS only follows neighbors with ion_charge == 0, so the charged 'b'
        # is not reachable from 'a'. Note the filter is applied to TRAVERSED
        # neighbors only: 'b' itself is added whenever it seeds a component,
        # which is why {'b', 'c'} stay connected when 'b' is the seed.
        grid = {
            "a": make_site(position=(0, 0, 0), neighbors=["b"]),
            "b": make_site(position=(1, 0, 0), neighbors=["a", "c"], ion_charge=1),
            "c": make_site(position=(2, 0, 0), neighbors=["b"]),
        }
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), ["a", "b", "c"], grid
        )
        assert sorted(map(sorted, components)) == [["a"], ["b", "c"]]

    def test_empty_atom_list(self):
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), [], {}
        )
        assert components == []

    def test_5x5_grid_boundary_site_is_isolated(self):
        # Non-periodic grid: a lone metal at the corner stays a single-site
        # component even though it has Empty neighbors.
        grid = make_grid_5x5(occupied=[(0, 0)])
        atoms = [c for c, s in grid.items() if s.chemical_specie == "Ag"]
        assert atoms == [(0, 0)]
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert components == [[(0, 0)]]

    def test_5x5_grid_l_shaped_trimer(self):
        # The canonical example: (1,1), (1,2), (2,1) occupied -> one 3-site
        # cluster; everything else Empty.
        grid = make_grid_5x5(occupied=[(1, 1), (1, 2), (2, 1)])
        atoms = [c for c, s in grid.items() if s.chemical_specie == "Ag"]
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert len(components) == 1
        assert set(components[0]) == {(1, 1), (1, 2), (2, 1)}

    def test_5x5_grid_periodic_wrap_merges_corners(self):
        # Periodic grid: corners (0,0) and (4,0) become neighbors.
        grid = make_grid_5x5(occupied=[(0, 0), (4, 0)], periodic=True)
        atoms = [c for c, s in grid.items() if s.chemical_specie == "Ag"]
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert len(components) == 1 and len(components[0]) == 2


# =============================================================================
# Cluster class (cluster.py)
# =============================================================================

class TestClusterClass:
    """Cluster bookkeeping: size, electrode contact, internal atoms, distance."""

    def _grid(self):
        return {
            "s1": make_site(position=(0, 0, 0), supp_by=["bottom_layer"],
                            neighbors=["s2", "s3"]),
            "s2": make_site(position=(0, 0, 5), supp_by=["top_layer"],
                            neighbors=["s1", "s4"]),
            "s3": make_site(position=(0, 1, 0), supp_by=[], neighbors=["s1"]),
            "s4": make_site(position=(0, 0, 9), supp_by=[], neighbors=["s2"]),
        }

    def _cluster(self, grid):
        return Cluster(["s1", "s2", "s3"],
                       [grid["s1"].position, grid["s2"].position, grid["s3"].position],
                       {"bottom_layer": False, "top_layer": False},
                       {"conductive_filament": 1e6})

    def test_constructor_records_size_and_ids(self):
        cluster = self._cluster(self._grid())
        assert cluster.atoms_id == {"s1", "s2", "s3"}
        assert cluster.size == 3
        assert cluster.conductivity["conductive_filament"] == 1e6

    def test_update_electrode_contact_flags_and_propagation(self):
        grid = self._grid()
        cluster = self._cluster(grid)
        cluster.update_electrode_contact(grid)
        assert cluster.attached_layer == {"bottom_layer": True, "top_layer": True}
        assert cluster.interface_sites_bottom == {"s1"}
        assert cluster.interface_sites_top == {"s2"}
        # Flag propagation reaches every atom in the cluster.
        for site_id in cluster.atoms_id:
            assert grid[site_id].in_cluster_with_electrode == {
                "bottom_layer": True, "top_layer": True
            }

    def test_update_electrode_contact_site_on_both_layers_counts_as_bottom(self):
        # Quirk 2: if/elif ordering -> 'bottom_layer' wins for dual-supported sites.
        grid = {
            "s1": make_site(position=(0, 0, 0),
                            supp_by=["bottom_layer", "top_layer"]),
        }
        cluster = Cluster(["s1"], [grid["s1"].position],
                          {"bottom_layer": False, "top_layer": False},
                          {"conductive_filament": 1e6})
        cluster.update_electrode_contact(grid)
        assert cluster.attached_layer == {"bottom_layer": True, "top_layer": False}

    def test_identify_internal_atoms(self):
        # s1's neighbors (s2, s3) are all in the cluster -> internal. s3's only
        # neighbor (s1) is also in-cluster -> internal. s2 additionally touches
        # s4 (outside the cluster) -> surface atom.
        grid = self._grid()
        cluster = self._cluster(grid)
        cluster._identify_internal_atoms(grid)
        assert cluster.internal_sites == {"s1", "s3"}
        # Order follows self.atoms_id (a set) -> compare as a set.
        assert set(cluster.internal_atom_positions) == {
            grid["s1"].position, grid["s3"].position
        }

    def test_distance_to_electrode_detached_cluster(self):
        grid = self._grid()
        cluster = self._cluster(grid)
        cluster._get_distance_to_electrode((10, 10, 10))
        assert cluster.distance_electrode == 10

    def test_distance_to_electrode_top_attached(self):
        grid = self._grid()
        cluster = self._cluster(grid)
        cluster.attached_layer = {"bottom_layer": False, "top_layer": True}
        cluster._get_distance_to_electrode((10, 10, 10))
        # min(z) of the cluster is 0 -> distance to the top electrode is 0.
        assert cluster.distance_electrode == 0

    def test_distance_to_electrode_bottom_attached(self):
        grid = self._grid()
        cluster = self._cluster(grid)
        cluster.attached_layer = {"bottom_layer": True, "top_layer": False}
        cluster._get_distance_to_electrode((10, 10, 10))
        # max(z) of the cluster is 5 -> distance to the bottom electrode is 5.
        assert cluster.distance_electrode == 5


class TestClusterSlicing:
    """_slice_cluster: Plane-connected components within atoms_id, seeded by
    descending z. Note: _build_slice follows 'Plane' links regardless of z, so
    a slice is a Plane-connected component, not strictly a z-layer.

    Precondition: _slice_cluster reads self.internal_sites, which is only
    populated by _identify_internal_atoms (production order in
    prepare_cluster_for_bcs). The helper below enforces that order.
    """

    def _grid(self, plane_ab=False):
        grid = {
            "a": make_site(position=(0, 0, 0), neighbors=["b"]),
            "b": make_site(position=(0, 0, 2), neighbors=["a"]),
            "c": make_site(position=(0, 0, 4), neighbors=[]),
            "off": make_site(position=(1, 0, 0), plane=["a"], specie="Ag"),
        }
        if plane_ab:
            grid["a"].migration_paths["Plane"] = [("b", None)]
            grid["b"].migration_paths["Plane"] = [("a", None)]
        cluster = Cluster(["a", "b", "c"],
                          [grid["a"].position, grid["b"].position, grid["c"].position],
                          {"bottom_layer": True, "top_layer": True},
                          {"conductive_filament": 1e6})
        return grid, cluster

    def _sliced(self, plane_ab=False):
        """Cluster with internal atoms identified, then sliced."""
        grid, cluster = self._grid(plane_ab=plane_ab)
        cluster._identify_internal_atoms(grid)
        cluster._slice_cluster(grid)
        return grid, cluster

    def test_unconnected_sites_get_one_slice_each(self):
        grid, cluster = self._sliced()
        assert len(cluster.slice_list) == 3
        # Slices are seeded by descending z: c (4), b (2), a (0).
        assert cluster.slice_list[0] == ["c"]
        assert cluster.slice_list[1] == ["b"]
        assert cluster.slice_list[2] == ["a"]

    def test_plane_connected_sites_share_a_slice(self):
        grid, cluster = self._sliced(plane_ab=True)
        # a-b are Plane-linked -> they form ONE slice (seeded from the highest-z
        # member, b). 'c' has no Plane links, so it stays its own slice.
        assert len(cluster.slice_list) == 2
        assert {"a", "b"} in [set(sl) for sl in cluster.slice_list]
        assert ["c"] in cluster.slice_list

    def test_slice_ignores_sites_outside_the_cluster(self):
        grid, cluster = self._sliced()
        all_sliced = {s for sl in cluster.slice_list for s in sl}
        assert "off" not in all_sliced

    def test_overlapping_slice_is_dropped_not_merged(self):
        # Quirk 3: c->a and b->a Plane links. Seed from c builds {c, a};
        # the seed from b would build {b, a} which overlaps total_visited ->
        # b is silently dropped from ALL slices.
        grid, cluster = self._grid()
        grid["c"].migration_paths["Plane"] = [("a", None)]
        grid["b"].migration_paths["Plane"] = [("a", None)]
        cluster._identify_internal_atoms(grid)
        cluster._slice_cluster(grid)
        all_sliced = {s for sl in cluster.slice_list for s in sl}
        assert all_sliced == {"c", "a"}
        assert "b" in cluster.atoms_id  # ...yet b belongs to the cluster

    def test_slice_internal_positions_recorded(self):
        grid, cluster = self._grid()
        cluster._identify_internal_atoms(grid)
        cluster._slice_cluster(grid)
        # All three sites have fully-in-cluster neighbor sets -> internal.
        assert sum(len(p) for p in cluster.slice_internal_positions_per_slice) == 3


class TestClusterResistance:
    """Bridging cluster: per-layer resistances and voltage partition.

    Preconditions encoded here (production order in prepare_cluster_for_bcs):
      update_electrode_contact -> _identify_internal_atoms -> _slice_cluster ->
      _cluster_resistance. interface_sites_top/bottom only exist after
      update_electrode_contact, and _cluster_resistance reads them.

    The cluster is a single-site-per-layer column (t, m, b at z = 4, 2, 0) so
    the cached layer thickness (|z(slice0) - z(slice1)|) is 2 Angstroms.
    """

    def _setup(self):
        grid = {
            "t": make_site(position=(0, 0, 4), supp_by=["top_layer"]),
            "m": make_site(position=(0, 0, 2), supp_by=[]),
            "b": make_site(position=(0, 0, 0), supp_by=["bottom_layer"]),
            # Geometry padding outside the cluster: gives the grid an x and y
            # extent > 0 for the area-per-site geometry cache.
            "dx": make_site(position=(1, 0, 0), specie="Hf"),
            "dy": make_site(position=(0, 1, 0), specie="Hf"),
        }
        cluster = Cluster(["t", "m", "b"],
                          [grid[s].position for s in ("t", "m", "b")],
                          {"bottom_layer": False, "top_layer": False},
                          {"conductive_filament": 1e6,
                           "interface_top": 1e5,
                           "interface_bottom": None})
        cluster.update_electrode_contact(grid)
        cluster.prepare_cluster_for_bcs(grid, (10, 10, 4))
        return grid, cluster

    def test_resistances_per_layer(self):
        _, cluster = self._setup()
        # 3 singleton slices, one per z level (distinct z -> deterministic
        # order). The top slice carries interface_top (1e5); the middle and
        # bottom slices fall back to the bulk filament sigma (1e6) because
        # interface_bottom is None. Geometry is identical for every layer, so
        # the resistance ratios follow the sigma ratios:
        assert len(cluster.layers_resistance) == 3
        assert cluster.layers_resistance[0] == pytest.approx(
            10 * cluster.layers_resistance[1]
        )
        assert cluster.layers_resistance[1] == pytest.approx(
            cluster.layers_resistance[2]
        )
        assert cluster.total_resistance == pytest.approx(sum(cluster.layers_resistance))

    def test_voltage_partition_across_layers(self):
        _, cluster = self._setup()
        potentials = cluster.voltage_across_cluster(V_top=1.0, V_bottom=0.0)
        assert len(potentials) == 3
        assert potentials[0] == pytest.approx(1.0)
        # potentials[i] = V_top - I * sum(R[:i]) -> strictly decreasing.
        current = 1.0 / cluster.total_resistance
        expected = [
            1.0 - current * sum(cluster.layers_resistance[:i]) for i in range(3)
        ]
        assert potentials == pytest.approx(expected)
        assert all(p0 > p1 for p0, p1 in zip(potentials, potentials[1:]))


# =============================================================================
# Island detection - Crystal_Lattice.detect_islands (unbound call)
# =============================================================================

class TestDetectIslands:
    """detect_islands: plane-connected same-specie sites form one island."""

    def _grid(self):
        return make_grid_5x5({(1, 1), (1, 2), (2, 1)})

    def test_connected_sites_form_one_island(self):
        """The 3-site L cluster from the task example is one island slice."""
        grid = self._grid()
        visited, island_slice = call(
            Crystal_Lattice.detect_islands,
            make_lattice(grid),
            (1, 1), set(), set(), "Ag",
        )
        assert island_slice == {(1, 1), (1, 2), (2, 1)}
        assert visited == island_slice

    def test_species_filter_excludes_other_species(self):
        """An 'Empty' start site never yields an 'Ag' island."""
        grid = self._grid()
        visited, island_slice = call(
            Crystal_Lattice.detect_islands,
            make_lattice(grid),
            (0, 0), set(), set(), "Ag",
        )
        assert island_slice == set()

    def test_empty_specie_forms_its_own_island(self):
        """The complement (22 Empty sites) is a single connected island."""
        grid = self._grid()
        visited, island_slice = call(
            Crystal_Lattice.detect_islands,
            make_lattice(grid),
            (0, 0), set(), set(), "Empty",
        )
        assert len(island_slice) == 22
        assert all(grid[s].chemical_specie == "Empty" for s in island_slice)

    def test_previsited_start_yields_nothing(self):
        """A start site that is already visited is not re-added."""
        grid = self._grid()
        visited, island_slice = call(
            Crystal_Lattice.detect_islands,
            make_lattice(grid),
            (1, 1), {(1, 1)}, set(), "Ag",
        )
        assert island_slice == set()
        assert visited == {(1, 1)}


# =============================================================================
# Island building - Crystal_Lattice.build_island (unbound call)
# =============================================================================

class TestBuildIsland:
    """build_island: Up + Plane + Down connectivity merges a 3D island."""

    def _vertical_chain_grid(self, top_specie="Ag"):
        return {
            "a": make_site(position=(0, 0, 0), specie="Ag", plane=["b"], up=["c"]),
            "b": make_site(position=(2, 0, 0), specie="Ag", plane=["a"]),
            "c": make_site(position=(0, 0, 2), specie=top_specie, down=["a"]),
        }

    def test_vertical_and_plane_links_merge_into_one_island(self):
        grid = self._vertical_chain_grid()
        visited, island_sites = call(
            Crystal_Lattice.build_island,
            make_lattice(grid),
            set(), set(), "a", "Ag",
        )
        assert island_sites == {"a", "b", "c"}
        assert visited == island_sites

    def test_species_filter_stops_vertical_growth(self):
        """A differently-specied site above is not absorbed."""
        grid = self._vertical_chain_grid(top_specie="O")
        visited, island_sites = call(
            Crystal_Lattice.build_island,
            make_lattice(grid),
            set(), set(), "a", "Ag",
        )
        assert island_sites == {"a", "b"}

    def test_island_does_not_revisit_sites(self):
        """Diamond topology (two paths to the same site) is handled once."""
        grid = {
            "a": make_site(position=(0, 0, 0), specie="Ag", plane=["b", "d"]),
            "b": make_site(position=(2, 0, 0), specie="Ag", plane=["a"], up=["c"]),
            "d": make_site(position=(0, 2, 0), specie="Ag", plane=["a"], up=["c"]),
            "c": make_site(position=(0, 0, 2), specie="Ag", down=["b", "d"]),
        }
        visited, island_sites = call(
            Crystal_Lattice.build_island,
            make_lattice(grid),
            set(), set(), "a", "Ag",
        )
        assert island_sites == {"a", "b", "c", "d"}


# =============================================================================
# Island layers & terraces - Island._layers_calculation / _island_terrace
# =============================================================================

def make_island_system():
    """A 6-layer mock system: 30 sites (5x5 base at z=0 + 5 at z=2),
    basis z-vector (0,0,1) -> z_step = 2, z_steps = 6, so
    sites_per_layer = 30/6 = 5.0 and area_per_site = 100/5 = 20.0."""
    grid = {}
    for x in range(0, 10, 2):
        for y in range(0, 10, 2):
            grid[(x, y, 0)] = make_site(position=(x, y, 0), specie="Ag")
    for x in range(0, 10, 2):
        grid[(x, 0, 2)] = make_site(position=(x, 0, 2), specie="Ag")
    system = SimpleNamespace(
        grid_crystal=grid,
        basis_vectors=[(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 1)],
        crystal_size=(10, 10, 10),
        chemical_specie="Ag",
        idx_to_cart=lambda idx: grid[idx].position,
    )
    return grid, system


class TestIslandLayersTerraces:
    """Per-layer occupancy counts and terrace areas of an island."""

    def test_layers_count_occupied_sites_per_z(self):
        grid, system = make_island_system()
        island = Island(1, 2.0, {(0, 0, 2), (2, 0, 2), (0, 0, 0), (2, 0, 0)})
        layers = island._layers_calculation(system)
        assert layers == [2, 2, 0, 0, 0, 0]

    def test_layers_ignore_empty_sites(self):
        grid, system = make_island_system()
        island = Island(1, 2.0, {(0, 0, 2), (2, 0, 2), (4, 0, 0)})
        grid[(0, 0, 2)].chemical_specie = "Empty"
        layers = island._layers_calculation(system)
        assert layers == [1, 1, 0, 0, 0, 0]

    def test_flat_island_single_terrace(self):
        """A flat 3-site island at z=0 (layers [3,0,0,0,0,0]).

        terraces[0] = unoccupied area of layer 0 = (5-3)*20 = 40.
        terraces[1] = step down from layer 0 to layer 1 = (3-0)*20 = 60.
        """
        grid, system = make_island_system()
        island = Island(0, 0.0, {(0, 0, 0), (2, 0, 0), (4, 0, 0)})
        layers = island._layers_calculation(system)
        terraces = island._island_terrace(system, layers)
        assert terraces == [40.0, 60.0, 0.0, 0.0, 0.0, 0.0]

    def test_stepped_island_two_terrace_levels(self):
        """2 sites at z=0 + 1 at z=2 -> layers [2,1,0,0,0,0] ->
        terraces [(5-2)*20, (2-1)*20, (1-0)*20, 0, 0, 0]."""
        grid, system = make_island_system()
        island = Island(1, 2.0, {(0, 0, 2), (0, 0, 0), (2, 0, 0)})
        layers = island._layers_calculation(system)
        terraces = island._island_terrace(system, layers)
        assert terraces == [60.0, 20.0, 20.0, 0.0, 0.0, 0.0]

    def test_negative_difference_is_clamped_to_zero(self):
        """An overhanging layer (larger than the one below) adds no terrace."""
        grid, system = make_island_system()
        island = Island(1, 2.0, {(0, 0, 0), (0, 0, 2), (2, 0, 2), (4, 0, 2)})
        layers = island._layers_calculation(system)
        assert layers == [1, 3, 0, 0, 0, 0]
        terraces = island._island_terrace(system, layers)
        assert terraces[0] == pytest.approx(4.0 * 20.0)
        assert terraces[1] == 0.0  # (1-3)*20 < 0 -> clamped




# =============================================================================
# Edge cases required by the morphology spec
# =============================================================================

class TestMorphologyEdgeCases:
    """Single-site clusters, one giant cluster, and empty grids."""

    def test_5x5_grid_all_sites_occupied_is_one_giant_cluster(self):
        grid = make_grid_5x5(occupied=[(x, y) for x in range(5) for y in range(5)])
        atoms = [c for c, s in grid.items() if s.chemical_specie == "Ag"]
        assert len(atoms) == 25
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert len(components) == 1
        assert len(components[0]) == 25

    def test_5x5_grid_all_empty_yields_no_metal_atoms(self):
        grid = make_grid_5x5(occupied=[])
        atoms = [c for c, s in grid.items() if s.chemical_specie == "Ag"]
        components = call(
            Crystal_Lattice._dfs_find_components, SimpleNamespace(), atoms, grid
        )
        assert atoms == []
        assert components == []

    def test_single_site_cluster_object(self):
        grid = {(0, 0, 0): make_site(position=(0, 0, 0), neighbors=[])}
        cluster = Cluster(
            [(0, 0, 0)], [grid[(0, 0, 0)].position],
            {"bottom_layer": False, "top_layer": False},
            {"conductive_filament": 1e6},
        )
        assert cluster.size == 1
        assert cluster.atoms_id == {(0, 0, 0)}
        cluster._identify_internal_atoms(grid)
        # Quirk: a site with NO neighbors satisfies in_cluster == len(neighbors)
        # (0 == 0) and is therefore classified as "internal" (fully coordinated).
        assert cluster.internal_sites == {(0, 0, 0)}

    def test_island_fully_occupied_base_layer_first_terrace_is_unclamped(self):
        """Quirk 4: terraces[0] = (sites_per_layer - layers[0]) * area has no
        clamp, so an over-full base layer yields a NEGATIVE first terrace while
        the per-step entries (layer i-1 vs i) are clamped at zero."""
        grid, system = make_island_system()
        island = Island(0, 0.0, set(grid))  # every grid site belongs to the island
        layers = island._layers_calculation(system)
        assert layers == [25, 5, 0, 0, 0, 0]
        terraces = island._island_terrace(system, layers)
        assert terraces[0] == pytest.approx((5.0 - 25) * 20.0)  # -400, unclamped
        assert terraces[1] == pytest.approx((25 - 5) * 20.0)    # 400
        assert terraces[2] == pytest.approx((5 - 0) * 20.0)     # 100
        assert terraces[3:] == [0, 0, 0]

    def test_empty_island_layers_and_terraces(self):
        grid, system = make_island_system()
        island = Island(0, 0.0, set())
        layers = island._layers_calculation(system)
        assert layers == [0, 0, 0, 0, 0, 0]
        terraces = island._island_terrace(system, layers)
        # Nothing occupied -> the whole base layer is exposed terrace area.
        assert terraces == [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_empty_island_slices_are_all_empty(self):
        grid, system = make_island_system()
        island = Island(0, 0.0, set())
        island._slice_detection(system)
        assert island.slice_list == [[], [], [], [], [], []]
# =============================================================================
# Island slicing - Island._build_slice / _slice_detection
# =============================================================================

class TestIslandSlicing:
    """Plane-connected island sites share a slice; others are separated."""

    def test_build_slice_groups_plane_connected_sites(self):
        grid, system = make_island_system()
        for a, b in [((0, 0, 0), (2, 0, 0)), ((2, 0, 0), (4, 0, 0))]:
            grid[a].migration_paths["Plane"].append((b, None))
            grid[b].migration_paths["Plane"].append((a, None))
        island = Island(0, 0.0, {(0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 8, 0)})
        slice_sites = island._build_slice(system, {(0, 0, 0)}, (0, 0, 0))
        assert slice_sites == {(0, 0, 0), (2, 0, 0), (4, 0, 0)}

    def test_slice_detection_separates_layers(self):
        grid, system = make_island_system()
        for a, b in [((0, 0, 0), (2, 0, 0)), ((0, 0, 2), (2, 0, 2))]:
            grid[a].migration_paths["Plane"].append((b, None))
            grid[b].migration_paths["Plane"].append((a, None))
        island = Island(0, 0.0, {(0, 0, 0), (2, 0, 0), (0, 0, 2), (2, 0, 2)})
        island._slice_detection(system)
        assert len(island.slice_list[0]) == 1  # base pair shares a slice
        assert set(island.slice_list[0][0]) == {(0, 0, 0), (2, 0, 0)}
        assert len(island.slice_list[1]) == 1  # z=2 pair shares a slice
        assert set(island.slice_list[1][0]) == {(0, 0, 2), (2, 0, 2)}
        assert all(len(island.slice_list[i]) == 0 for i in (2, 3, 4, 5))

    def test_slice_detection_splits_disconnected_same_layer_sites(self):
        grid, system = make_island_system()
        island = Island(0, 0.0, {(0, 0, 0), (2, 0, 0)})  # no Plane links
        island._slice_detection(system)
        assert len(island.slice_list[0]) == 2  # two singleton slices
        assert {frozenset(s) for s in island.slice_list[0]} == {
            frozenset({(0, 0, 0)}),
            frozenset({(2, 0, 0)}),
        }
