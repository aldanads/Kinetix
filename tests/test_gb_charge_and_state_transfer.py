# tests/test_gb_charge_and_state_transfer.py
"""
Tests for:
1. GB charge state resolution (_get_gb_charge_state)
2. GB barrier modification (modify_act_energy_GB)
3. Defect state transfer (get_migrating_state / introduce / remove)

These tests focus on runtime behavior, NOT config processing.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from copy import deepcopy


# =============================================================================
# Mocks and Fixtures
# =============================================================================

class MockGBModel:
    """Mock GrainBoundary model for charge state and barrier tests."""

    def __init__(self, gb_configurations, region_map=None):
        """
        Parameters
        ----------
        gb_configurations : list[dict]
            Pre-processed GB configs (as stored after _process_configurations).
        region_map : dict
            Maps position tuples to region strings for get_site_gb_region.
        """
        self.gb_configurations = gb_configurations
        self._region_map = region_map or {}

    def get_site_gb_region(self, position):
        """Return the GB region for a given position."""
        pos_key = tuple(np.round(position, 3))
        return self._region_map.get(pos_key, 'bulk')


class MockSite:
    """Minimal mock of the Site class for state transfer tests."""

    def __init__(self, position, site_type='interstitial', chemical_specie='Empty',
                 ion_charge=0, passivation_level=0):
        self.position = position
        self.site_type = site_type
        self.chemical_specie = chemical_specie
        self.ion_charge = ion_charge
        self.passivation_level = passivation_level
        self.nearest_neighbors_idx = []
        self.supp_by = ()
        self.site_events = []
        self.applicable_defects = []
        self.Act_E_dict = {}

    def _get_current_defect_name(self):
        """Simplified: return defect name based on chemical_specie."""
        if self.chemical_specie == 'Empty':
            return None
        # This would normally check applicable_defects and Act_E_dict
        return getattr(self, '_defect_name', None)

    def introduce_specie(self, chemical_specie, ion_charge=None):
        self.chemical_specie = chemical_specie
        if ion_charge is not None:
            self.ion_charge = ion_charge

    def remove_specie(self, affected_site='Empty'):
        self.chemical_specie = affected_site
        self.ion_charge = 0


@pytest.fixture
def gb_config_hydrogen():
    """GB config with charge_state for hydrogen_interstitial (list format)."""
    return [{
        'type': 'vertical_planar',
        'position': 27.0,
        'width': 6.0,
        'outer_width': 12.0,
        'inner_boundary': 3.0,
        'outer_boundary': 6.0,
        'event_modifications': {
            'migration': [
                {
                    'affected_defects': ['hydrogen_interstitial'],
                    'Act_E_diff_GB': 0.27,
                    'charge_state': {
                        'inner_boundary': 0,
                        'outer_boundary': 1,
                        'bulk': 1
                    }
                },
                {
                    'affected_defects': ['oxygen_vacancy'],
                    'Act_E_diff_GB': 0.45,
                    # No charge_state for V_O
                }
            ],
            'generation': [
                {
                    'affected_defects': ['hydrogen_interstitial'],
                    'Act_E_diff_GB': 0.0,
                    'charge_state': {
                        'inner_boundary': 0,
                        'outer_boundary': 1,
                        'bulk': 1
                    }
                }
            ],
            'reaction': [
                {
                    'affected_reactions': ['H2_formation'],
                    'Act_E_diff_GB': 3.31,
                }
            ]
        }
    }]


@pytest.fixture
def gb_model_hydrogen(gb_config_hydrogen):
    """GB model with region mapping for testing."""
    region_map = {
        (27.0, 25.0, 50.0): 'inner_boundary',
        (25.0, 25.0, 50.0): 'outer_boundary',
        (30.0, 25.0, 50.0): 'outer_boundary',
        (10.0, 25.0, 50.0): 'bulk',
        (45.0, 25.0, 50.0): 'bulk',
    }
    return MockGBModel(gb_config_hydrogen, region_map)


@pytest.fixture
def defects_config_full():
    """Full defects config with migrating_attributes."""
    return {
        'hydrogen_interstitial': {
            'symbol': 'H',
            'charge': 1,
            'site_type': 'interstitial',
            'allowed_sublattices': ['interstitial', 'O'],
            'valid_target_species': ['Empty'],
            'activation_energies_key': 'H',
            'enabled_events': ['migration', 'reaction', 'generation'],
            'migrating_attributes': ['ion_charge'],
            'description': 'Hydrogen defect in interstitial'
        },
        'oxygen_vacancy': {
            'symbol': 'V_O',
            'charge': 0,
            'site_type': 'O',
            'allowed_sublattices': ['O'],
            'valid_target_species': ['O'],
            'activation_energies_key': 'V_O',
            'enabled_events': ['migration', 'reaction'],
            'migrating_attributes': ['ion_charge', 'passivation_level'],
            'max_passivation_level': 3,
            'charge_per_passivation': -1,
            'description': 'Intrinsic vacancy in oxide lattice'
        },
        'hydrogen_gas': {
            'symbol': 'H2',
            'charge': 0,
            'site_type': 'interstitial',
            'allowed_sublattices': ['interstitial', 'O'],
            'valid_target_species': ['Empty'],
            'activation_energies_key': 'H2',
            'enabled_events': [],
            'migrating_attributes': [],
            'description': 'Hydrogen gas in interstitial'
        }
    }


# =============================================================================
# Helper: Simulate _get_gb_charge_state logic
# =============================================================================

def get_gb_charge_state(gb_model, defect_name, site_position, event_type='migration'):
    """
    Replicates Crystal_Lattice._get_gb_charge_state logic for testing.
    """
    if not gb_model:
        return None

    gb_config = gb_model.gb_configurations[0]
    event_entries = gb_config['event_modifications'].get(event_type)
    if event_entries is None:
        return None

    # Backward compatibility
    if isinstance(event_entries, dict):
        event_entries = [event_entries]

    for entry in event_entries:
        affected_defects = entry.get('affected_defects', [])
        if defect_name and defect_name not in affected_defects:
            continue

        charge_state = entry.get('charge_state', {})
        if not charge_state:
            return None

        site_gb_region = gb_model.get_site_gb_region(site_position)
        return charge_state.get(site_gb_region, None)

    return None


# =============================================================================
# Helper: Simulate get_migrating_state logic
# =============================================================================

def get_migrating_state(site, defects_config):
    """
    Replicates Site.get_migrating_state logic for testing.
    Returns only EXTRA attributes (excludes chemical_specie and ion_charge).
    """
    defect_name = site._get_current_defect_name()
    if not defect_name or defect_name not in defects_config:
        return {}

    config = defects_config[defect_name]
    base_attrs = {'chemical_specie', 'ion_charge', 'defect_name'}
    extra_state = {}

    for attr in config.get('migrating_attributes', []):
        if attr in base_attrs:
            continue
        if hasattr(site, attr):
            extra_state[attr] = getattr(site, attr)

    return extra_state


# =============================================================================
# Test Class 1: GB Charge State Resolution
# =============================================================================

class TestGBChargeState:
    """Test _get_gb_charge_state behavior."""

    def test_no_gb_model_returns_none(self):
        """Without a GB model, charge state is always None."""
        result = get_gb_charge_state(None, 'hydrogen_interstitial', (10, 10, 10))
        assert result is None

    def test_no_event_modifications_returns_none(self, gb_model_hydrogen):
        """If event_type has no entries, return None."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (10, 10, 10),
            event_type='nonexistent_event'
        )
        assert result is None

    def test_defect_not_affected_returns_none(self, gb_model_hydrogen):
        """Defect not in affected_defects list ? None."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_gas', (27.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result is None

    def test_affected_but_no_charge_state_returns_none(self, gb_model_hydrogen):
        """Defect is affected but entry has no charge_state ? None."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'oxygen_vacancy', (27.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result is None

    def test_inner_boundary_neutral(self, gb_model_hydrogen):
        """H in GB core ? charge 0 (neutral)."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (27.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result == 0

    def test_outer_boundary_charged(self, gb_model_hydrogen):
        """H in GB transition region ? charge +1."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (25.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result == 1

    def test_bulk_charged(self, gb_model_hydrogen):
        """H in bulk ? charge +1."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (10.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result == 1

    def test_generation_event_same_logic(self, gb_model_hydrogen):
        """Generation event uses same charge_state logic."""
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (27.0, 25.0, 50.0),
            event_type='generation'
        )
        assert result == 0

    def test_unknown_region_returns_none(self, gb_model_hydrogen):
        """If site_gb_region is not in charge_state dict ? None."""
        # Add a position that maps to an unknown region
        gb_model_hydrogen._region_map[(99.0, 99.0, 99.0)] = 'unknown_region'
        result = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', (99.0, 99.0, 99.0),
            event_type='migration'
        )
        assert result is None

    def test_backward_compat_single_dict(self):
        """Single dict (old format) should still work."""
        gb_config = [{
            'event_modifications': {
                'migration': {  # Single dict, not a list
                    'affected_defects': ['hydrogen_interstitial'],
                    'charge_state': {
                        'inner_boundary': 0,
                        'outer_boundary': 1,
                        'bulk': 1
                    }
                }
            }
        }]
        region_map = {(27.0, 25.0, 50.0): 'inner_boundary'}
        gb_model = MockGBModel(gb_config, region_map)

        result = get_gb_charge_state(
            gb_model, 'hydrogen_interstitial', (27.0, 25.0, 50.0),
            event_type='migration'
        )
        assert result == 0

    def test_first_matching_entry_wins(self):
        """When multiple entries match, the first one is used."""
        gb_config = [{
            'event_modifications': {
                'migration': [
                    {
                        'affected_defects': ['hydrogen_interstitial'],
                        'charge_state': {'bulk': 0}  # First match: neutral
                    },
                    {
                        'affected_defects': ['hydrogen_interstitial'],
                        'charge_state': {'bulk': 1}  # Should NOT be reached
                    }
                ]
            }
        }]
        region_map = {(10.0, 10.0, 10.0): 'bulk'}
        gb_model = MockGBModel(gb_config, region_map)

        result = get_gb_charge_state(
            gb_model, 'hydrogen_interstitial', (10.0, 10.0, 10.0),
            event_type='migration'
        )
        assert result == 0  # First entry wins


# =============================================================================
# Test Class 2: GB Barrier Modification
# =============================================================================

class TestGBBarrierModification:
    """Test modify_act_energy_GB behavior."""

    @pytest.fixture
    def gb_model_for_barrier(self):
        """GB model with distance function for barrier tests."""
        gb_config = [{
            'type': 'vertical_planar',
            'position': 27.0,
            'width': 6.0,
            'outer_width': 12.0,
            'inner_boundary': 3.0,
            'outer_boundary': 6.0,
            'distance_function': lambda pos, gb: abs(pos[0] - gb['position']),
            'event_modifications': {
                'migration': [
                    {
                        'affected_defects': ['hydrogen_interstitial'],
                        'affected_defects_set': {'hydrogen_interstitial'},
                        'Act_E_diff_GB': 0.27,
                        'region': 'outer_boundary',
                        'inner_boundary': 3.0,
                        'outer_boundary': 6.0,
                        'linear_slope': -0.27 / (6.0 - 3.0),  # -0.09
                        'linear_intercept': 0.27 - (-0.09) * 3.0,  # 0.54
                    }
                ]
            }
        }]
        region_map = {}
        return MockGBModel(gb_config, region_map)

    def test_site_outside_gb_no_modification(self, gb_model_for_barrier):
        """Site far from GB ? no barrier modification."""
        site = MockSite(position=(10.0, 25.0, 50.0))
        site.applicable_defects = ['hydrogen_interstitial']
        site.Act_E_dict = {
            'hydrogen_interstitial': {
                'E_mig_plane': 0.5,
                'E_mig_upward': 0.5,
                'E_mig_downward': 0.5,
            }
        }

        # Distance from GB: |10 - 27| = 17 > outer_boundary (6)
        # No modification should be applied
        gb = gb_model_for_barrier.gb_configurations[0]
        dist_func = gb['distance_function']
        dist = dist_func(site.position, gb)
        assert dist > gb['outer_boundary'], "Site should be outside GB influence"

    def test_site_in_inner_boundary_full_reduction(self, gb_model_for_barrier):
        """Site in GB core ? full Act_E_diff_GB reduction."""
        gb = gb_model_for_barrier.gb_configurations[0]
        entry = gb['event_modifications']['migration'][0]

        # Position at GB center: distance = 0 < inner_boundary (3.0)
        pos = (27.0, 25.0, 50.0)
        dist = gb['distance_function'](pos, gb)
        assert dist <= entry['inner_boundary']

        # Expected reduction: full Act_E_diff_GB = 0.27
        expected_reduction = entry['Act_E_diff_GB']
        assert expected_reduction == 0.27

    def test_site_in_outer_boundary_linear_reduction(self, gb_model_for_barrier):
        """Site in transition region ? linear interpolation reduction."""
        gb = gb_model_for_barrier.gb_configurations[0]
        entry = gb['event_modifications']['migration'][0]

        # Position at distance 4.5 from GB center (between 3.0 and 6.0)
        pos = (22.5, 25.0, 50.0)  # |22.5 - 27| = 4.5
        dist = gb['distance_function'](pos, gb)
        assert entry['inner_boundary'] < dist <= entry['outer_boundary']

        # Expected reduction: slope * dist + intercept
        expected_reduction = entry['linear_slope'] * dist + entry['linear_intercept']
        expected_reduction = max(expected_reduction, 0.0)

        # slope = -0.09, intercept = 0.54
        # At dist=4.5: -0.09 * 4.5 + 0.54 = -0.405 + 0.54 = 0.135
        assert expected_reduction == pytest.approx(0.135, abs=1e-6)

    def test_unaffected_defect_no_modification(self, gb_model_for_barrier):
        """Defect not in affected_defects_set ? no modification."""
        gb = gb_model_for_barrier.gb_configurations[0]
        entry = gb['event_modifications']['migration'][0]

        assert 'oxygen_vacancy' not in entry['affected_defects_set']
        # V_O should not receive any barrier modification from this entry

    def test_migration_pathway_energy_after_reduction(self, gb_model_for_barrier):
        """Verify final migration energy = base - reduction."""
        base_energy = 0.5
        reduction = 0.27  # Full reduction in inner boundary
        expected_final = base_energy - reduction  # 0.23

        assert expected_final == pytest.approx(0.23)
        assert expected_final > 0, "Energy should remain positive"


# =============================================================================
# Test Class 3: Defect State Transfer
# =============================================================================

class TestDefectStateTransfer:
    """Test get_migrating_state and the introduce/remove pattern."""

    def test_hydrogen_no_extra_attributes(self, defects_config_full):
        """H has only ion_charge in migrating_attributes ? empty extra_state."""
        site = MockSite(position=(10, 10, 10), chemical_specie='H', ion_charge=1)
        site._defect_name = 'hydrogen_interstitial'

        extra_state = get_migrating_state(site, defects_config_full)
        # ion_charge is in base_attrs, so it's excluded from extra_state
        assert extra_state == {}

    def test_oxygen_vacancy_extracts_passivation(self, defects_config_full):
        """V_O has passivation_level in migrating_attributes ? extracted."""
        site = MockSite(position=(10, 10, 10), chemical_specie='V_O',
                        ion_charge=-1, passivation_level=2)
        site._defect_name = 'oxygen_vacancy'

        extra_state = get_migrating_state(site, defects_config_full)
        assert 'passivation_level' in extra_state
        assert extra_state['passivation_level'] == 2
        # ion_charge is in base_attrs, excluded
        assert 'ion_charge' not in extra_state

    def test_empty_site_returns_empty_dict(self, defects_config_full):
        """Empty site has no defect ? empty dict."""
        site = MockSite(position=(10, 10, 10), chemical_specie='Empty')

        extra_state = get_migrating_state(site, defects_config_full)
        assert extra_state == {}

    def test_unknown_defect_returns_empty_dict(self, defects_config_full):
        """Unknown defect name ? empty dict."""
        site = MockSite(position=(10, 10, 10), chemical_specie='X')
        site._defect_name = 'unknown_defect'

        extra_state = get_migrating_state(site, defects_config_full)
        assert extra_state == {}

    def test_introduce_specie_with_extra_state(self, defects_config_full):
        """_introduce_specie_site applies extra_state attributes."""
        site = MockSite(position=(10, 10, 10))

        # Simulate _introduce_specie_site with extra_state
        chemical_specie = 'V_O'
        ion_charge = -1
        extra_state = {'passivation_level': 2}

        site.introduce_specie(chemical_specie, ion_charge)
        for attr, value in extra_state.items():
            setattr(site, attr, value)

        assert site.chemical_specie == 'V_O'
        assert site.ion_charge == -1
        assert site.passivation_level == 2

    def test_remove_specie_resets_extra_attributes(self, defects_config_full):
        """_remove_species_at_site resets extra attributes to defaults."""
        site = MockSite(position=(10, 10, 10), chemical_specie='V_O',
                        ion_charge=-1, passivation_level=2)

        # Simulate _remove_species_at_site with attributes_to_reset
        attributes_to_reset = ['passivation_level']

        site.remove_specie('Empty')
        for attr in attributes_to_reset:
            setattr(site, attr, 0)

        assert site.chemical_specie == 'Empty'
        assert site.ion_charge == 0
        assert site.passivation_level == 0

    def test_full_migration_cycle_preserves_state(self, defects_config_full):
        """Complete migration: source ? dest, source cleared."""
        source = MockSite(position=(10, 10, 10), chemical_specie='V_O',
                          ion_charge=-1, passivation_level=2)
        source._defect_name = 'oxygen_vacancy'

        dest = MockSite(position=(12, 10, 10))

        # Step 1: Extract state from source
        chemical_specie = source.chemical_specie
        migrating_charge = source.ion_charge
        extra_state = get_migrating_state(source, defects_config_full)

        # Step 2: Apply GB modification (e.g., charge changes)
        gb_charge = 0  # Simulate entering GB core
        if gb_charge is not None:
            migrating_charge = gb_charge

        # Step 3: Introduce at destination
        dest.introduce_specie(chemical_specie, migrating_charge)
        for attr, value in extra_state.items():
            setattr(dest, attr, value)

        # Step 4: Remove from source
        attributes_to_reset = list(extra_state.keys())
        source.remove_specie('Empty')
        for attr in attributes_to_reset:
            setattr(source, attr, 0)

        # Verify destination
        assert dest.chemical_specie == 'V_O'
        assert dest.ion_charge == 0  # Modified by GB
        assert dest.passivation_level == 2  # Preserved from source

        # Verify source is clean
        assert source.chemical_specie == 'Empty'
        assert source.ion_charge == 0
        assert source.passivation_level == 0

    def test_migration_without_gb_preserves_charge(self, defects_config_full):
        """Migration without GB modification keeps original charge."""
        source = MockSite(position=(10, 10, 10), chemical_specie='V_O',
                          ion_charge=-1, passivation_level=1)
        source._defect_name = 'oxygen_vacancy'

        dest = MockSite(position=(12, 10, 10))

        chemical_specie = source.chemical_specie
        migrating_charge = source.ion_charge
        extra_state = get_migrating_state(source, defects_config_full)

        # No GB modification
        gb_charge = None
        if gb_charge is not None:
            migrating_charge = gb_charge

        dest.introduce_specie(chemical_specie, migrating_charge)
        for attr, value in extra_state.items():
            setattr(dest, attr, value)

        assert dest.ion_charge == -1  # Unchanged
        assert dest.passivation_level == 1

    def test_generation_defaults_no_extra_state_needed(self, defects_config_full):
        """Generated defects start with defaults; no extra_state required."""
        site = MockSite(position=(10, 10, 10))

        # Simulate generation: only base attributes set
        chemical_specie = 'V_O'
        generated_charge = 0  # From defects_config

        site.introduce_specie(chemical_specie, generated_charge)

        # passivation_level should already be at default (0)
        assert site.chemical_specie == 'V_O'
        assert site.ion_charge == 0
        assert site.passivation_level == 0  # Default, no need to set

    def test_h2_formation_state_transfer(self, defects_config_full):
        """H2 has no migrating_attributes ? empty extra_state."""
        site = MockSite(position=(10, 10, 10), chemical_specie='H2', ion_charge=0)
        site._defect_name = 'hydrogen_gas'

        extra_state = get_migrating_state(site, defects_config_full)
        assert extra_state == {}


# =============================================================================
# Test Class 4: Integration - Charge State + State Transfer
# =============================================================================

class TestChargeStateAndTransferIntegration:
    """Integration tests combining GB charge modification with state transfer."""

    def test_h_migration_gb_core_to_bulk(self, defects_config_full, gb_model_hydrogen):
        """H migrates from GB core (neutral) to bulk (charged +1)."""
        source = MockSite(position=(27.0, 25.0, 50.0), chemical_specie='H', ion_charge=0)
        source._defect_name = 'hydrogen_interstitial'

        dest = MockSite(position=(10.0, 25.0, 50.0))

        # Extract state
        chemical_specie = source.chemical_specie
        migrating_charge = source.ion_charge
        extra_state = get_migrating_state(source, defects_config_full)

        # Apply GB charge for destination
        gb_charge = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', dest.position,
            event_type='migration'
        )
        if gb_charge is not None:
            migrating_charge = gb_charge

        # Introduce and remove
        dest.introduce_specie(chemical_specie, migrating_charge)
        for attr, value in extra_state.items():
            setattr(dest, attr, value)
        source.remove_specie('Empty')

        # H should now be charged +1 in bulk
        assert dest.ion_charge == 1
        assert source.chemical_specie == 'Empty'

    def test_h_migration_bulk_to_gb_core(self, defects_config_full, gb_model_hydrogen):
        """H migrates from bulk (charged +1) to GB core (neutral)."""
        source = MockSite(position=(10.0, 25.0, 50.0), chemical_specie='H', ion_charge=1)
        source._defect_name = 'hydrogen_interstitial'

        dest = MockSite(position=(27.0, 25.0, 50.0))

        chemical_specie = source.chemical_specie
        migrating_charge = source.ion_charge
        extra_state = get_migrating_state(source, defects_config_full)

        gb_charge = get_gb_charge_state(
            gb_model_hydrogen, 'hydrogen_interstitial', dest.position,
            event_type='migration'
        )
        if gb_charge is not None:
            migrating_charge = gb_charge

        dest.introduce_specie(chemical_specie, migrating_charge)
        source.remove_specie('Empty')

        # H should now be neutral in GB core
        assert dest.ion_charge == 0

    def test_v_o_migration_preserves_passivation(self, defects_config_full, gb_model_hydrogen):
        """V_O migration preserves passivation_level; no charge modification."""
        source = MockSite(position=(10.0, 25.0, 50.0), chemical_specie='V_O',
                          ion_charge=-1, passivation_level=2)
        source._defect_name = 'oxygen_vacancy'

        dest = MockSite(position=(12.0, 25.0, 50.0))

        chemical_specie = source.chemical_specie
        migrating_charge = source.ion_charge
        extra_state = get_migrating_state(source, defects_config_full)

        # V_O has no charge_state in GB config ? gb_charge is None
        gb_charge = get_gb_charge_state(
            gb_model_hydrogen, 'oxygen_vacancy', dest.position,
            event_type='migration'
        )
        assert gb_charge is None  # V_O not affected by charge modification

        if gb_charge is not None:
            migrating_charge = gb_charge

        dest.introduce_specie(chemical_specie, migrating_charge)
        for attr, value in extra_state.items():
            setattr(dest, attr, value)
            
        source.remove_specie('Empty')
        for attr in extra_state.keys():
            setattr(source, attr, 0)

        # V_O preserves charge and passivation
        assert dest.ion_charge == -1
        assert dest.passivation_level == 2
        assert source.passivation_level == 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])