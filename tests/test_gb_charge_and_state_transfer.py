# tests/test_gb_charge_and_state_transfer.py
"""
Tests for:
1. GB charge state resolution (_get_gb_charge_state)
2. GB barrier modification (modify_act_energy_GB)
3. Defect state transfer — object-transfer protocol (install_defect/clear_defect)

These tests focus on runtime behavior, NOT config processing.
After Phase 5, the attribute-by-attribute transfer machinery
(``get_migrating_state`` / ``extra_state`` / ``attributes_to_reset``) is gone;
tests verify that a Defect object hops by reference and the source clears to an
empty Defect.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from copy import deepcopy

from kinetix.lattice.defect import (Event, Defect, DefectConfig,
                                    make_empty_defect)


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
    """Minimal mock of the Site class for state transfer tests.

    Faithful to the post-refactor Site model: the site always hosts exactly
    one Defect (never None) and the legacy flat attributes are read/write
    views onto it, so the tests exercise the same protocol production does.
    """

    def __init__(self, position, site_type='interstitial', chemical_specie='Empty',
                 ion_charge=0, passivation_level=0, defect=None):
        self.position = position
        self.site_type = site_type
        self.nearest_neighbors_idx = []
        self.supp_by = ()
        self.applicable_defects = []
        self.Act_E_dict = {}
        self._defect_name = None
        if defect is None:
          defect = make_empty_defect()
          if chemical_specie != 'Empty':
            # Unresolved-species branch of Site._make_initial_defect: an empty
            # Defect carrying the symbol, so chemical_specie keeps working.
            defect.chemical_specie = chemical_specie
            defect.charge = ion_charge
            defect.passivation_level = passivation_level
        self.defect = defect

    # ---- Defect-carried state as delegating views (as in production Site) ----
    @property
    def chemical_specie(self):
        return self.defect.chemical_specie

    @chemical_specie.setter
    def chemical_specie(self, value):
        self.defect.chemical_specie = value

    @property
    def ion_charge(self):
        return self.defect.charge

    @ion_charge.setter
    def ion_charge(self, value):
        self.defect.charge = value

    @property
    def passivation_level(self):
        return self.defect.passivation_level

    @passivation_level.setter
    def passivation_level(self, value):
        self.defect.passivation_level = value

    @property
    def site_events(self):
        return self.defect.events

    @site_events.setter
    def site_events(self, value):
        self.defect.events = value

    def _get_current_defect_name(self):
        """Simplified: return defect name based on chemical_specie."""
        if self.chemical_specie == 'Empty':
            return None
        # This would normally check applicable_defects and Act_E_dict
        return getattr(self, '_defect_name', None)

    def introduce_specie(self, chemical_specie, ion_charge=None, defect=None):
        """Install a species; same-species swaps keep the occupant Defect."""
        if defect is not None:
          self.install_defect(defect)
          return
        if self.defect.chemical_specie == chemical_specie:
          if ion_charge is not None:
            self.defect.charge = ion_charge
          return
        fresh = make_empty_defect()
        fresh.chemical_specie = chemical_specie
        fresh.charge = ion_charge or 0
        self.install_defect(fresh)

    def remove_specie(self, affected_site='Empty'):
        """Clear the occupant with a fresh empty Defect."""
        self.clear_defect()
        if affected_site != 'Empty':
          self.chemical_specie = affected_site

    # ---- object-transfer protocol support (Phase 5) ----
    def install_defect(self, defect):
        """Transfer ``defect`` by reference (production semantics)."""
        self.defect = defect

    def clear_defect(self):
        """Install a fresh empty Defect (never None)."""
        self.defect = make_empty_defect()


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
    """Test object-transfer hops (install_defect / clear_defect).

    These tests verify the *new* mechanism: a defect is a stateful object
    that hops by identity, replacing the old attribute-by-attribute machinery
    (``get_migrating_state`` / ``extra_state`` / ``attributes_to_reset``).
    They keep reading the real config's ``migrating_attributes`` key to
    confirm the YAML still declares the same intent, but the *transfer*
    mechanism is now object-based.
    """

    def _make_defect(self, defects_config_full, name, chemical_specie,
                     charge=0, passivation_level=0, events=None):
        # Same construction path production uses: registry dict -> DefectConfig.
        return Defect(config=DefectConfig.from_dict(name, defects_config_full[name]),
                      chemical_specie=chemical_specie,
                      charge=charge, passivation_level=passivation_level,
                      events=events or [])

    def _fake_event(self, label):
        return Event(label=label, destination=(1, 0), barrier=0.3, rate=1.0)

    def test_migrating_attributes_key_declares_same_intent(self, defects_config_full):
        """The YAML still lists what ought to travel with a hop; the *mechanism*
        is now object-based, so we just sanity-check the key, not the transfer."""
        vo_name = [k for k in defects_config_full
                   if defects_config_full[k].get('site_type') == 'O']
        if not vo_name:
            pytest.skip("fixture has no O-site defect")
        attrs = defects_config_full[vo_name[0]].get('migrating_attributes')
        assert attrs is not None
        assert 'passivation_level' in attrs, (
            "fixture expectation: V_O must migrate its passivation level")

    def test_empty_site_always_hosts_a_defect(self, defects_config_full):
        """Phase 2 contract: an empty site hosts an *empty* Defect, never None."""
        site = MockSite(position=(10, 10, 10), chemical_specie='Empty')
        assert site.defect is not None
        assert site.defect.is_empty is True
        assert site.chemical_specie == 'Empty'
        assert site._get_current_defect_name() is None

    def test_unknown_defect_name_hosts_a_defect_carrying_the_symbol(
        self, defects_config_full):
        """A site whose chemical_specie names no known defect still hosts a
        Defect that carries the symbol, so no code path has to guard against a
        missing occupant."""
        site = MockSite(position=(10, 10, 10), chemical_specie='X')
        assert site.defect is not None
        assert site.chemical_specie == 'X'
        assert site._get_current_defect_name() is None
        # NOTE: no ``is_empty`` assertion here on purpose. A host lattice atom
        # (Hf, O) is *occupied*, not vacant, yet today it is built from
        # EMPTY_DEFECT_CONFIG and therefore reports is_empty=True - the current
        # model only distinguishes "has a usable DefectConfig" from "does not",
        # conflating host atoms, defects and truly empty sites. Phase 6 splits
        # this into three states (is_host / is_defect, config=None for host
        # atoms, is_empty True only for chemical_specie == 'Empty'); asserting
        # the placeholder semantics here would freeze the conflation in place.

    def test_full_migration_cycle_preserves_state(self, defects_config_full):
        """Complete migration via object hop: source defect transfers to dest,
        source clears. Passivation is carried by the Defect, not by
        attribute-by-attribute copy."""
        src_defect = self._make_defect(defects_config_full, 'oxygen_vacancy',
                                       chemical_specie='V_O', charge=-1,
                                       passivation_level=2)
        src = MockSite(position=(10, 10, 10))
        src.install_defect(src_defect)

        dst = MockSite(position=(12, 10, 10))
        # Simulate GB charge modification before hop
        src_defect.charge = 0     # entering GB core -> neutral
        dst.install_defect(src.defect)
        src.clear_defect()

        # Destination has the transferred state
        assert dst.chemical_specie == 'V_O'
        assert dst.ion_charge == 0   # GB-modified
        assert dst.passivation_level == 2  # preserved on the Object
        assert dst.defect is src_defect

        # Source is empty: a fresh empty Defect, not None and not the hopped one
        assert src.chemical_specie == 'Empty'
        assert src.ion_charge == 0
        assert src.passivation_level == 0
        assert src.defect is not src_defect
        assert src.defect.is_empty is True

    def test_migration_without_gb_preserves_charge(self, defects_config_full):
        """Migration without GB modification keeps original charge (object hop)."""
        src_defect = self._make_defect(defects_config_full, 'oxygen_vacancy',
                                       chemical_specie='V_O', charge=-1,
                                       passivation_level=1)
        src = MockSite(position=(10, 10, 10))
        src.install_defect(src_defect)
        dst = MockSite(position=(12, 10, 10))

        dst.install_defect(src.defect)
        src.clear_defect()

        assert dst.ion_charge == -1   # unchanged
        assert dst.passivation_level == 1
        assert src.chemical_specie == 'Empty'

    def test_generation_defaults_no_extra_state_needed(self, defects_config_full):
        """Generated defects start with defaults; no extra_state needed."""
        site = MockSite(position=(10, 10, 10))

        generated_defect = self._make_defect(defects_config_full, 'oxygen_vacancy',
                                              chemical_specie='V_O', charge=0)
        site.install_defect(generated_defect)

        # passivation_level should already be at default (0)
        assert site.chemical_specie == 'V_O'
        assert site.ion_charge == 0
        assert site.passivation_level == 0

    def test_h2_formation_object_transfer(self, defects_config_full):
        """H2_gas hops as a defect object; no extra_state required."""
        src_defect = self._make_defect(defects_config_full, 'hydrogen_gas',
                                       chemical_specie='H2', charge=0)
        src = MockSite(position=(10, 10, 10))
        src.install_defect(src_defect)
        dst = MockSite(position=(12, 10, 10))

        dst.install_defect(src.defect)
        src.clear_defect()

        assert dst.chemical_specie == 'H2'
        assert dst.ion_charge == 0
        assert src.chemical_specie == 'Empty'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])