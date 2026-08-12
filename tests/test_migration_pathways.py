# tests/test_migration_pathways.py
"""
Tests for PBC-aware neighbor finding and migration pathway registration.

Validates:
- Minimum-image vector wrapping (lateral PBC, open z-boundary)
- PBC-aware neighbor finding across lateral boundaries
- Migration pathway key consistency between initialization and site lookup
- Absence of zero-vector migration keys
- Steep-down neighbor presence at edge sites (regression for the radius bug)
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# =============================================================================
# Helper: Minimal mock for Crystal_Lattice methods under test
# =============================================================================

class MinimalLattice:
    """
    Lightweight stand-in for Crystal_Lattice that provides only the
    attributes needed by _minimum_image_vector, _get_neighbors_for_site,
    and _generate_periodic_images.
    """

    def __init__(self, crystal_size, grid_crystal=None):
        self.crystal_size = np.array(crystal_size, dtype=float)
        self.grid_crystal = grid_crystal if grid_crystal is not None else {}
        self._kdtree = None
        self._kdtree_positions = None
        self._kdtree_indices = None
        
    # --- Methods under test (copied from Crystal_Lattice) ---

    def _minimum_image_vector(self, vec):
        vec = np.array(vec, dtype=float)
        for dim in range(2):
            L = self.crystal_size[dim]
            if vec[dim] > L / 2:
                vec[dim] -= L
            elif vec[dim] < -L / 2:
                vec[dim] += L
        return vec

    def _build_kdtree(self):
        from scipy.spatial import cKDTree
        positions = np.array([site.position for site in self.grid_crystal.values()])
        self._kdtree_positions = positions
        self._kdtree_indices = list(self.grid_crystal.keys())
        self._kdtree = cKDTree(positions)

    def _generate_periodic_images(self, site_pos, radius):
        site_pos = np.array(site_pos)
        query_positions = [site_pos]

        for dim in range(2):
            if site_pos[dim] < radius:
                image_pos = site_pos.copy()
                image_pos[dim] += self.crystal_size[dim]
                query_positions.append(image_pos)
            if site_pos[dim] > self.crystal_size[dim] - radius:
                image_pos = site_pos.copy()
                image_pos[dim] -= self.crystal_size[dim]
                query_positions.append(image_pos)

        if len(query_positions) > 1:
            base_positions = query_positions.copy()
            for base_pos in base_positions:
                for dim in range(2):
                    if base_pos[dim] != site_pos[dim]:
                        continue
                    if site_pos[dim] < radius:
                        image_pos = base_pos.copy()
                        image_pos[dim] += self.crystal_size[dim]
                        if not any(np.allclose(image_pos, qp) for qp in query_positions):
                            query_positions.append(image_pos)
                    if site_pos[dim] > self.crystal_size[dim] - radius:
                        image_pos = base_pos.copy()
                        image_pos[dim] -= self.crystal_size[dim]
                        if not any(np.allclose(image_pos, qp) for qp in query_positions):
                            query_positions.append(image_pos)

        return query_positions

    def _get_neighbors_for_site(self, site_idx, radius):
        site_pos = self.grid_crystal[site_idx].position
        all_neighbor_indices = set()

        query_positions = self._generate_periodic_images(site_pos, radius)

        for query_pos in query_positions:
            neighbor_array_indices = self._kdtree.query_ball_point(query_pos, radius)
            for i in neighbor_array_indices:
                neighbor_idx = self._kdtree_indices[i]
                if neighbor_idx != site_idx:
                    all_neighbor_indices.add(neighbor_idx)

        return list(all_neighbor_indices)
        
class MockSite:
  """Minimal site object for grid construction."""
  
  def __init__(self, position, site_type='interstitial', chemical_specie='Empty'):
    self.position = tuple(position)
    self.site_type = site_type
    self.chemical_specie = chemical_specie
          
# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def domain_size():
    """Arbitrary domain size"""
    return np.array([51.238, 51.238, 101.89])
    
    
@pytest.fixture
def small_lattice(domain_size):
    """
    A small synthetic lattice with sites at known positions:
    - One site at the corner (0, L_y, z_mid)
    - One site at the x-edge (0, y_mid, z_mid)
    - One site in the bulk (L_x/2, L_y/2, z_mid)
    - Neighbor sites at known offsets
    """
    Lx, Ly, Lz = domain_size
    z_mid = Lz / 2

    grid = {}

    # Corner site
    grid['corner'] = MockSite(position=(0.0, Ly, z_mid))

    # Edge site (x=0, y in middle)
    grid['edge_x'] = MockSite(position=(0.0, Ly / 2, z_mid))

    # Bulk site
    grid['bulk'] = MockSite(position=(Lx / 2, Ly / 2, z_mid))

    # Neighbors for corner site (across boundary)
    grid['corner_nb_1'] = MockSite(position=(2.377, Ly - 0.158, z_mid + 1.816))
    grid['corner_nb_2'] = MockSite(position=(Lx - 2.377, 0.158, z_mid + 1.816))
    grid['corner_nb_3'] = MockSite(position=(2.377, Ly - 0.158, z_mid - 2.614))
    
    # --- Neighbors for edge_x site (across x-boundary) ---
    # These sit near x=Lx, so PBC wraps them to appear near x=0
    # Steep UP: dist = sqrt(2.0² + 1.5²) = 2.5 Å
    grid['edge_x_nb_up'] = MockSite(position=(Lx - 2.0, Ly / 2, z_mid + 1.5))
    # Steep DOWN: dist = sqrt(2.0² + 2.5²) = 3.2 Å
    grid['edge_x_nb_down'] = MockSite(position=(Lx - 2.0, Ly / 2, z_mid - 2.5))
    # Shallow: dist = sqrt(2.0² + 0.1²) ˜ 2.0 Å
    grid['edge_x_nb_shallow'] = MockSite(position=(Lx - 2.0, Ly / 2 + 0.1, z_mid))
    
    # Neighbors for bulk site
    grid['bulk_nb_1'] = MockSite(position=(Lx / 2 + 2.0, Ly / 2, z_mid + 1.5))
    grid['bulk_nb_2'] = MockSite(position=(Lx / 2 - 2.0, Ly / 2, z_mid - 1.5))
    grid['bulk_nb_3'] = MockSite(position=(Lx / 2, Ly / 2 + 2.0, z_mid))

    lattice = MinimalLattice(crystal_size=domain_size, grid_crystal=grid)
    lattice._build_kdtree()
    return lattice
    
@pytest.fixture
def simple_cubic_lattice():
    """
    A simple cubic lattice for testing basic PBC wrapping.
    3x3x3 grid with spacing 2.0 in a domain of 6x6x6.
    """
    spacing = 2.0
    n = 3
    L = n * spacing  # 6.0

    grid = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                key = (i, j, k)
                pos = (i * spacing, j * spacing, k * spacing)
                grid[key] = MockSite(position=pos)

    lattice = MinimalLattice(crystal_size=(L, L, L), grid_crystal=grid)
    lattice._build_kdtree()
    return lattice
    
# =============================================================================
# Test: Minimum-Image Vector Wrapping
# =============================================================================

class TestMinimumImageVector:
    """Test the minimum-image wrapping logic for lateral PBC."""

    @pytest.fixture
    def lattice(self, domain_size):
        return MinimalLattice(crystal_size=domain_size)
        
    def test_no_wrapping_needed(self, lattice):
        """Vector within half-domain stays unchanged."""
        vec = np.array([2.0, -1.5, 3.0])
        result = lattice._minimum_image_vector(vec)
        np.testing.assert_allclose(result, [2.0, -1.5, 3.0])
        
    def test_wrapping_positive_x(self, lattice):
        """Vector > L_x/2 in x wraps to negative."""
        Lx = lattice.crystal_size[0]
        vec = np.array([Lx - 1.0, 0.0, 0.0])
        result = lattice._minimum_image_vector(vec)
        assert result[0] == pytest.approx(-1.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)
    
    def test_wrapping_negative_x(self, lattice):
        """Vector < -L_x/2 in x wraps to positive."""
        Lx = lattice.crystal_size[0]
        vec = np.array([-(Lx - 1.0), 0.0, 0.0])
        result = lattice._minimum_image_vector(vec)
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_wrapping_positive_y(self, lattice):
        """Vector > L_y/2 in y wraps to negative."""
        Ly = lattice.crystal_size[1]
        vec = np.array([0.0, Ly - 2.0, 0.0])
        result = lattice._minimum_image_vector(vec)
        assert result[1] == pytest.approx(-2.0, abs=1e-10)

    def test_wrapping_negative_y(self, lattice):
        """Vector < -L_y/2 in y wraps to positive."""
        Ly = lattice.crystal_size[1]
        vec = np.array([0.0, -(Ly - 2.0), 0.0])
        result = lattice._minimum_image_vector(vec)
        assert result[1] == pytest.approx(2.0, abs=1e-10)

    def test_z_never_wrapped_positive(self, lattice):
        """Large positive z-component must NOT be wrapped (open boundary)."""
        vec = np.array([0.0, 0.0, 90.0])
        result = lattice._minimum_image_vector(vec)
        assert result[2] == pytest.approx(90.0)

    def test_z_never_wrapped_negative(self, lattice):
        """Large negative z-component must NOT be wrapped (open boundary)."""
        vec = np.array([0.0, 0.0, -90.0])
        result = lattice._minimum_image_vector(vec)
        assert result[2] == pytest.approx(-90.0)

    def test_combined_wrapping(self, lattice):
        """Both x and y wrap simultaneously, z untouched."""
        Lx, Ly = lattice.crystal_size[0], lattice.crystal_size[1]
        vec = np.array([Lx - 1.0, Ly - 2.0, 50.0])
        result = lattice._minimum_image_vector(vec)
        assert result[0] == pytest.approx(-1.0, abs=1e-10)
        assert result[1] == pytest.approx(-2.0, abs=1e-10)
        assert result[2] == pytest.approx(50.0)

    def test_exactly_half_domain_no_wrap(self, lattice):
        """Vector exactly at L/2 should not wrap (boundary convention)."""
        Lx = lattice.crystal_size[0]
        vec = np.array([Lx / 2, 0.0, 0.0])
        result = lattice._minimum_image_vector(vec)
        # At exactly L/2, the convention is to NOT wrap (only > L/2 wraps)
        assert result[0] == pytest.approx(Lx / 2)
        
    @pytest.mark.parametrize("offset", [0.1, 1.0, 2.5, 5.0])
    def test_wrapping_symmetry(self, lattice, offset):
        """Wrapping +offset and -offset should give symmetric results."""
        Lx = lattice.crystal_size[0]
        vec_pos = np.array([Lx - offset, 0.0, 0.0])
        vec_neg = np.array([-(Lx - offset), 0.0, 0.0])

        result_pos = lattice._minimum_image_vector(vec_pos)
        result_neg = lattice._minimum_image_vector(vec_neg)

        assert result_pos[0] == pytest.approx(-offset, abs=1e-10)
        assert result_neg[0] == pytest.approx(offset, abs=1e-10)

    def test_input_not_mutated(self, lattice):
        """Original input array should not be modified."""
        Lx = lattice.crystal_size[0]
        original = np.array([Lx - 1.0, 0.0, 0.0])
        original_copy = original.copy()
        lattice._minimum_image_vector(original)
        np.testing.assert_allclose(original, original_copy)
        
# =============================================================================
# Test: PBC-Aware Neighbor Finding
# =============================================================================

class TestPBCNeighborFinding:
    """Test that _get_neighbors_for_site correctly finds across-boundary neighbors."""

    def test_bulk_site_finds_all_neighbors(self, small_lattice):
        """A bulk site should find all neighbors within radius without PBC issues."""
        radius = 3.0
        neighbors = small_lattice._get_neighbors_for_site('bulk', radius)
        # Bulk site at (Lx/2, Ly/2, z_mid) should find bulk_nb_1, bulk_nb_2, bulk_nb_3
        assert 'bulk_nb_1' in neighbors
        assert 'bulk_nb_2' in neighbors
        assert 'bulk_nb_3' in neighbors
        
    def test_edge_site_finds_across_boundary(self, small_lattice):
        """A site at x close to 0 should find neighbors near x close to L via PBC images."""
        radius = 3.0
        neighbors = small_lattice._get_neighbors_for_site('edge_x', radius)
        # edge_x is at (0, Ly/2, z_mid)
        # corner_nb_1 is at (2.377, Ly-0.158, z_mid+1.816) - might be within radius
        # The key test: neighbors across x-boundary should be found
        assert 'edge_x_nb_up' in neighbors, "Should find steep-up neighbor across x-boundary"
        assert 'edge_x_nb_shallow' in neighbors, "Should find shallow neighbor across x-bound"
        
    def test_corner_site_finds_diagonal_images(self, small_lattice):
        """A corner site (x close to 0, y close to L) should find neighbors across both boundaries."""
        radius = 3.5
        neighbors = small_lattice._get_neighbors_for_site('corner', radius)
        # Corner is at (0, Ly, z_mid)
        # corner_nb_2 is at (Lx-2.377, 0.158, z_mid+1.816) - across both x and y
        # This tests the diagonal PBC image generation
        assert len(neighbors) > 0, "Corner site should have neighbors via PBC"
        
    def test_no_self_in_neighbors(self, small_lattice):
        """A site should never appear in its own neighbor list."""
        radius = 5.0
        for site_idx in small_lattice.grid_crystal:
            neighbors = small_lattice._get_neighbors_for_site(site_idx, radius)
            assert site_idx not in neighbors, f"Site {site_idx} found in its own neighbors"
            
    def test_neighbor_symmetry(self, simple_cubic_lattice):
        """If A is a neighbor of B, then B should be a neighbor of A (with PBC)."""
        radius = 2.5  # spacing is 2.0, so nearest neighbors are within radius
        lattice = simple_cubic_lattice

        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            for nb_idx in neighbors:
                reverse_neighbors = lattice._get_neighbors_for_site(nb_idx, radius)
                assert site_idx in reverse_neighbors, (
                    f"Asymmetric neighbor: {site_idx} sees {nb_idx}, "
                    f"but {nb_idx} does not see {site_idx}"
                )
                
    def test_radius_zero_gives_no_neighbors(self, small_lattice):
        """With radius=0, no neighbors should be found."""
        neighbors = small_lattice._get_neighbors_for_site('bulk', 0.0)
        assert len(neighbors) == 0
        
    def test_very_large_radius_finds_all(self, small_lattice):
        """With a very large radius, all other sites should be found."""
        radius = 200.0  # Larger than any distance in the domain
        neighbors = small_lattice._get_neighbors_for_site('bulk', radius)
        expected_count = len(small_lattice.grid_crystal) - 1  # All except self
        assert len(neighbors) == expected_count
        
        
# =============================================================================
# Test: Periodic Image Generation
# =============================================================================

class TestPeriodicImageGeneration:
    """Test the _generate_periodic_images helper."""

    @pytest.fixture
    def lattice(self, domain_size):
        return MinimalLattice(crystal_size=domain_size)

    def test_bulk_site_single_image(self, lattice):
        """A bulk site far from boundaries should only have its own position."""
        Lx, Ly = lattice.crystal_size[0], lattice.crystal_size[1]
        pos = np.array([Lx / 2, Ly / 2, 50.0])
        radius = 3.0
        images = lattice._generate_periodic_images(pos, radius)
        assert len(images) == 1, "Bulk site should have only 1 query position"
        np.testing.assert_allclose(images[0], pos)

    def test_x_edge_site_generates_image(self, lattice):
        """A site near x=0 should generate an image at x+L_x."""
        Ly = lattice.crystal_size[1]
        pos = np.array([0.5, Ly / 2, 50.0])
        radius = 3.0
        images = lattice._generate_periodic_images(pos, radius)
        assert len(images) >= 2, "x-edge site should have at least 2 query positions"

        # Check that one image is shifted by L_x
        x_values = [img[0] for img in images]
        assert any(abs(x - (0.5 + lattice.crystal_size[0])) < 1e-10 for x in x_values), (
            "Should have an image shifted by +L_x"
        )

    def test_corner_site_generates_four_images(self, lattice):
        """A corner site (x close to 0, y close to L_y) should generate images in x, y, and diagonal."""
        Lx, Ly = lattice.crystal_size[0], lattice.crystal_size[1]
        pos = np.array([0.5, Ly - 0.5, 50.0])
        radius = 3.0
        images = lattice._generate_periodic_images(pos, radius)
        # Should have: original, +x shift, -y shift, and diagonal (+x, -y)
        assert len(images) >= 4, (
            f"Corner site should have at least 4 query positions, got {len(images)}"
        )

    def test_z_never_generates_images(self, lattice):
        """Sites near z-boundaries should NOT generate periodic images in z."""
        Lx, Ly = lattice.crystal_size[0], lattice.crystal_size[1]
        pos = np.array([Lx / 2, Ly / 2, 0.5])  # Near z=0
        radius = 3.0
        images = lattice._generate_periodic_images(pos, radius)
        # All images should have the same z
        for img in images:
            assert img[2] == pytest.approx(0.5), "z should never be wrapped"
            
# =============================================================================
# Test: Migration Pathway Registration Consistency
# =============================================================================

class TestMigrationPathwayConsistency:
    """
    Test that migration_vector_keys computed during initialization
    match those computed during per-site lookup.
    """

    def _compute_wrapped_key(self, lattice, site_idx, neighbor_idx, decimals=6):
        """Compute the wrapped migration vector key (same logic as production code)."""
        site_pos = np.array(lattice.grid_crystal[site_idx].position)
        neighbor_pos = np.array(lattice.grid_crystal[neighbor_idx].position)
        vector = lattice._minimum_image_vector(neighbor_pos - site_pos)
        dist = np.linalg.norm(vector)
        if dist < 1e-10:
            return None
        return tuple(np.round(vector, decimals=decimals))

    def test_no_zero_vector_keys(self, small_lattice):
        """(0, 0, 0) must never appear as a valid migration key."""
        radius = 5.0
        lattice = small_lattice

        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            for nb_idx in neighbors:
                key = self._compute_wrapped_key(lattice, site_idx, nb_idx)
                if key is not None:
                    assert key != (0.0, 0.0, 0.0), (
                        f"Zero-vector key found for {site_idx} -> {nb_idx}"
                    )

    def test_key_consistency_both_directions(self, small_lattice):
        """Key from A->B should be the negation of key from B->A."""
        radius = 5.0
        lattice = small_lattice

        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            for nb_idx in neighbors:
                key_forward = self._compute_wrapped_key(lattice, site_idx, nb_idx)
                key_reverse = self._compute_wrapped_key(lattice, nb_idx, site_idx)

                if key_forward is not None and key_reverse is not None:
                    for i in range(3):
                        assert key_forward[i] == pytest.approx(-key_reverse[i], abs=1e-5), (
                            f"Key asymmetry: {site_idx}->{nb_idx} = {key_forward}, "
                            f"{nb_idx}->{site_idx} = {key_reverse}"
                        )

    def test_key_deterministic(self, small_lattice):
        """Computing the same key twice should give identical results."""
        radius = 5.0
        lattice = small_lattice

        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            for nb_idx in neighbors:
                key1 = self._compute_wrapped_key(lattice, site_idx, nb_idx)
                key2 = self._compute_wrapped_key(lattice, site_idx, nb_idx)
                assert key1 == key2, "Key computation should be deterministic"

    def test_wrapped_distance_less_than_half_domain(self, small_lattice):
        """After wrapping, lateral distance should be <= L/2."""
        radius = 5.0
        lattice = small_lattice

        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            for nb_idx in neighbors:
                key = self._compute_wrapped_key(lattice, site_idx, nb_idx)
                if key is not None:
                    for dim in range(2):
                        L = lattice.crystal_size[dim]
                        assert abs(key[dim]) <= L / 2 + 1e-6, (
                            f"Wrapped component {key[dim]} exceeds L/2={L/2} "
                            f"for {site_idx}->{nb_idx}"
                        )
                        
# =============================================================================
# Test: Steep-Down Neighbor Presence (Regression for radius bug)
# =============================================================================

class TestSteepDownPresence:
    """
    Regression tests ensuring edge/corner interstitial sites have
    steep-down neighbors within the configured radius.
    """

    RADIUS_NEIGHBORS = 3.6  # The corrected radius value

    def _classify_neighbors(self, lattice, site_idx, radius):
        """Classify neighbors into steep_up, steep_down, shallow."""
        pos = np.array(lattice.grid_crystal[site_idx].position)
        neighbors = lattice._get_neighbors_for_site(site_idx, radius)

        steep_up, steep_down, shallow = [], [], []
        for n_idx in neighbors:
            npos = np.array(lattice.grid_crystal[n_idx].position)
            vec = lattice._minimum_image_vector(npos - pos)
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            unit = vec / dist
            z = unit[2]
            if z > 0.5:
                steep_up.append((n_idx, unit, dist))
            elif z < -0.5:
                steep_down.append((n_idx, unit, dist))
            else:
                shallow.append((n_idx, unit, dist))

        return steep_up, steep_down, shallow

    def test_edge_site_has_steep_down_neighbors(self, small_lattice):
        """Edge interstitial sites must have at least one steep-down neighbor."""
        steep_up, steep_down, shallow = self._classify_neighbors(
            small_lattice, 'edge_x', self.RADIUS_NEIGHBORS
        )
        # The edge_x site should have steep-down neighbors within radius 3.6
        # This is the regression test for the original bug
        # Note: with the synthetic grid, this depends on the fixture setup
        # In production, this would be validated against the real PZT grid
        assert len(steep_up) > 0, "Edge site should have steep-up neighbors"
        assert len(steep_down) > 0, "Edge site should have steep-down neighbors (regression)"
        assert len(shallow) > 0, "Edge site should have shallow neighbors"
        # Verify the specific regression: steep-down within radius 3.6
        for n_idx, unit, dist in steep_down:
          assert dist <= self.RADIUS_NEIGHBORS
          assert unit[2] < -0.5, "Steep-down must have z-component < -0.5"

    def test_bulk_site_has_both_steep_directions(self, small_lattice):
        """Bulk sites should have neighbors in both steep-up and steep-down."""
        steep_up, steep_down, shallow = self._classify_neighbors(
            small_lattice, 'bulk', self.RADIUS_NEIGHBORS
        )
        assert len(steep_up) > 0, "Bulk site should have steep-up neighbors"
        assert len(steep_down) > 0, "Bulk site should have steep-down neighbors"

    def test_steep_down_distance_within_radius(self, small_lattice):
        """All steep-down neighbors must be within radius_neighbors."""
        for site_idx in small_lattice.grid_crystal:
            _, steep_down, _ = self._classify_neighbors(
                small_lattice, site_idx, self.RADIUS_NEIGHBORS
            )
            for n_idx, unit, dist in steep_down:
                assert dist <= self.RADIUS_NEIGHBORS, (
                    f"Steep-down neighbor {n_idx} at dist={dist:.3f} "
                    f"exceeds radius={self.RADIUS_NEIGHBORS}"
                )

    @pytest.mark.parametrize("z_threshold", [0.3, 0.5, 0.7])
    def test_classification_threshold_sensitivity(self, small_lattice, z_threshold):
        """Verify classification is consistent for different thresholds."""
        pos = np.array(small_lattice.grid_crystal['bulk'].position)
        neighbors = small_lattice._get_neighbors_for_site('bulk', self.RADIUS_NEIGHBORS)

        for n_idx in neighbors:
            npos = np.array(small_lattice.grid_crystal[n_idx].position)
            vec = small_lattice._minimum_image_vector(npos - pos)
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            z_comp = vec[2] / dist
            # Just verify classification is mutually exclusive
            is_up = z_comp > z_threshold
            is_down = z_comp < -z_threshold
            is_shallow = not is_up and not is_down
            assert sum([is_up, is_down, is_shallow]) == 1, (
                f"Classification not mutually exclusive for z_comp={z_comp}"
            )

# =============================================================================
# Test: Integration with Site.neighbors_analysis key lookup
# =============================================================================

class TestSiteNeighborsAnalysisIntegration:
    """
    Test that the wrapped keys used in neighbors_analysis match
    those registered in event_labels during _initialize_migration_pathways.
    """

    def _build_event_labels(self, lattice, radius, decimals=6):
        """Simulate _initialize_migration_pathways key registration."""
        event_labels = {}
        i = 0
        for site_idx in lattice.grid_crystal:
            neighbors = lattice._get_neighbors_for_site(site_idx, radius)
            site_pos = np.array(lattice.grid_crystal[site_idx].position)

            for nb_idx in neighbors:
                if nb_idx == site_idx:
                    continue
                neighbor_pos = np.array(lattice.grid_crystal[nb_idx].position)
                vector = lattice._minimum_image_vector(neighbor_pos - site_pos)
                dist = np.linalg.norm(vector)
                if dist < 1e-10:
                    continue
                key = tuple(np.round(vector, decimals=decimals))
                if key not in event_labels:
                    event_labels[key] = i
                    i += 1
        return event_labels

    def test_all_neighbor_keys_in_event_labels(self, small_lattice):
        """Every wrapped key from neighbor lookup must exist in event_labels."""
        radius = 3.6
        event_labels = self._build_event_labels(small_lattice, radius)

        for site_idx in small_lattice.grid_crystal:
            neighbors = small_lattice._get_neighbors_for_site(site_idx, radius)
            site_pos = np.array(small_lattice.grid_crystal[site_idx].position)

            for nb_idx in neighbors:
                if nb_idx == site_idx:
                    continue
                neighbor_pos = np.array(small_lattice.grid_crystal[nb_idx].position)
                vector = small_lattice._minimum_image_vector(neighbor_pos - site_pos)
                dist = np.linalg.norm(vector)
                if dist < 1e-10:
                    continue
                key = tuple(np.round(vector, decimals=6))
                assert key in event_labels, (
                    f"Key {key} from {site_idx}->{nb_idx} not found in event_labels. "
                    f"This would cause a KeyError in Site.neighbors_analysis."
                )

    def test_no_keyerror_on_lookup(self, small_lattice):
        """Simulate the exact lookup pattern from Site.neighbors_analysis."""
        radius = 3.6
        event_labels = self._build_event_labels(small_lattice, radius)

        # This simulates what Site.neighbors_analysis does
        for site_idx in small_lattice.grid_crystal:
            neighbors = small_lattice._get_neighbors_for_site(site_idx, radius)
            site_pos = np.array(small_lattice.grid_crystal[site_idx].position)

            for nb_idx in neighbors:
                if nb_idx == site_idx:
                    continue
                neighbor_pos = np.array(small_lattice.grid_crystal[nb_idx].position)

                # Same wrapping logic as Site.neighbors_analysis
                vector = np.array(neighbor_pos) - np.array(site_pos)
                for dim in range(2):
                    L = small_lattice.crystal_size[dim]
                    if vector[dim] > L / 2:
                        vector[dim] -= L
                    elif vector[dim] < -L / 2:
                        vector[dim] += L

                dist = np.linalg.norm(vector)
                if dist < 1e-10:
                    continue

                migration_vector_key = tuple(np.round(vector, decimals=6))

                # This is the line that was raising KeyError
                try:
                    _ = event_labels[migration_vector_key]
                except KeyError:
                    pytest.fail(
                        f"KeyError: {migration_vector_key} for {site_idx}->{nb_idx}. "
                        f"The wrapping in neighbors_analysis does not match "
                        f"_initialize_migration_pathways."
                    )
                
# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


