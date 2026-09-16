"""Behavioral spec for kinetix.utils.superbasin.Superbasin (Epic 3 prep).

The Superbasin class is about to be refactored (E-field modification fix). 
These tests pin the CURRENT behavior — including known quirks
(e.g. the zeroed off-diagonals in ``transition_matrix``) — so any change
from the fix shows up as a diff here.

Pinned API summary:

* ``Superbasin(idx, System_state, E_min, sites_occupied)`` explores
  migration events (``site_events`` tuples ``[rate, dest, event_idx, E_act]``;
  migrations are those with an int ``event_idx``) from ``idx`` over
  ``System_state.grid_crystal``. States whose outgoing migrations are ALL
  above ``E_min`` are *absorbing*; states with at least one migration at or
  below ``E_min`` are *transient*.
* Workflow: ``trans_absorbing_states`` -> ``transition_matrix`` ->
  ``markov_matrix`` -> ``absorption_probability_matrix`` ->
  ``calculate_transition_rates_absorbing_states`` ->
  ``calculate_superbasin_environment``.
* ``valid`` is False when there are no absorbing or no transient states,
  or when the absorption matrix is ill-conditioned. (There is no "empty"
  superbasin construction; the isolated-state case below is the closest
  edge and yields ``valid=False``.)
* ``energy_step``/``time_step_limits`` are driver-level SuperbasinConfig
  parameters consumed in ``crystal.py``, NOT constructor arguments here;
  the class internally derives ``epsilon_min_decrement=0.1`` and
  ``retry_limit=max(round(E_min/0.1), 2)``.

System_state/site fakes implement only the attributes the class touches;
``processes`` records virtual-move calls instead of mutating physics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import constants

from kinetix.utils.superbasin import Superbasin

KB_EV = constants.physical_constants['Boltzmann constant in eV/K'][0]
NU0 = 7e12  # bond vibration frequency used in the EAct formula


class FakeSite:
  """Minimal site: only the attributes Superbasin reads."""

  def __init__(self, site_events=(), chemical_specie='VO', supp_by=()):
    self.site_events = [list(t) for t in site_events]
    self.chemical_specie = chemical_specie
    self.supp_by = set(supp_by)


class FakeSystemState:
  """Minimal System_state with a process-call recorder (no physics)."""

  def __init__(self, grid_crystal, num_event=5, chemical_specie='VO', sites_occupied=()):
    self.grid_crystal = grid_crystal
    self.sites_occupied = list(sites_occupied)
    self.num_event = num_event
    self.chemical_specie = chemical_specie
    self.allow_specie_removal = True
    self.process_calls = []

  def processes(self, transition):
    self.process_calls.append(tuple(transition))


def two_state_system(E_min=0.4):
  """Canonical valid superbasin: transient 0 -(easy 0.3)-> absorbing 1."""
  s0 = FakeSite(site_events=[(1.0, 1, 2, 0.3)])
  s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
  return Superbasin(0, FakeSystemState([s0, s1]), E_min, [])


def asymmetric_system():
  """Transient 0 (easy->1 rate 2.0, hard->2 rate 3.0), transient 1 (easy->0 rate 1.0), absorbing 2."""
  s0 = FakeSite(site_events=[(2.0, 1, 2, 0.3), (3.0, 2, 3, 0.9)])
  s1 = FakeSite(site_events=[(1.0, 0, 4, 0.2)])
  s2 = FakeSite(site_events=[(0.5, 0, 5, 0.9)])
  return Superbasin(0, FakeSystemState([s0, s1, s2]), 0.4, [])


def symmetric_three_state_system():
  """Two transient states with identical rates into each other and the absorber."""
  s0 = FakeSite(site_events=[(1.0, 1, 2, 0.3), (1.0, 2, 2, 0.3)])
  s1 = FakeSite(site_events=[(1.0, 0, 2, 0.3), (1.0, 2, 2, 0.3)])
  s2 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
  return Superbasin(0, FakeSystemState([s0, s1, s2]), 0.4, [])


class TestConstruction:
  def test_basic_attributes_and_classification(self):
    sb = two_state_system()
    assert sb.particle_idx == 0
    assert sb.E_min == 0.4
    assert sb.epsilon_min_decrement == 0.1
    assert sb.retry_limit == max(round(0.4 / 0.1), 2) == 4
    assert sb.valid is True
    assert sb.absorbing_states == [1]
    assert sb.transient_states == [0]
    assert sb.superbasin_idx == [1, 0]  # absorbing first, then transient
    assert sb.transient_states_transitions == [[1.0, 1, 2, 0.3, 0]]
    assert sb.absorbing_states_transitions == []

  def test_allow_specie_removal_restored(self):
    state = FakeSystemState([FakeSite(site_events=[(1.0, 1, 2, 0.3)]),
                             FakeSite(site_events=[(0.5, 0, 3, 0.9)])])
    state.allow_specie_removal = False
    Superbasin(0, state, 0.4, [])
    assert state.allow_specie_removal is False

  def test_virtual_moves_recorded_and_restored(self):
    s0 = FakeSite(site_events=[(1.0, 1, 2, 0.3)])
    s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
    state = FakeSystemState([s0, s1])
    Superbasin(0, state, 0.4, [])
    # BFS virtual move into the next stack site, then restore to start_idx.
    # The restore reuses `last_transition`, which keeps being updated for
    # every migration seen — including the absorbing site's hard one — so
    # the final call carries site 1's (0.5, event 3) transition.
    assert state.process_calls == [(1.0, 1, 2, 0), (0.5, 0, 3, 1)]


class TestEnergyThreshold:
  @pytest.mark.parametrize('e_act,e_min', [(0.4, 0.4), (0.3, 0.4)])
  def test_transition_at_or_below_E_min_is_transient(self, e_act, e_min):
    s0 = FakeSite(site_events=[(1.0, 1, 2, e_act)])
    s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
    sb = Superbasin(0, FakeSystemState([s0, s1]), e_min, [])
    assert sb.valid is True and sb.transient_states == [0]

  def test_all_hard_start_site_yields_no_transient(self):
    s0 = FakeSite(site_events=[(1.0, 1, 2, 0.400001)])
    s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
    sb = Superbasin(0, FakeSystemState([s0, s1]), 0.4, [])
    assert sb.absorbing_states == [0]
    assert sb.transient_states == []
    assert sb.superbasin_idx == [0]
    assert sb.valid is False

  def test_hard_transition_into_absorbing_state_is_collected(self):
    sb = asymmetric_system()
    assert sb.absorbing_states == [2]
    assert sorted(sb.transient_states) == [0, 1]
    assert sb.absorbing_states_transitions == [[3.0, 2, 3, 0.9, 0]]


class TestMatrices:
  def test_transition_matrix_diagonal_tau_and_quirk(self):
    sb = two_state_system()
    A = sb.A_transitions  # state order: superbasin_idx == [1, 0]
    assert A[0, 0] == 0.0  # absorbing diagonal is zero
    assert A[1, 1] == pytest.approx(1.0)  # transient diagonal = sum of outgoing rates
    # Quirk pinned: off-diagonals collapse to 0 (max(-rate, 0) with the
    # (dest, origin)-keyed lookup). If Epic 3 changes this, update here.
    assert A[0, 1] == 0.0 and A[1, 0] == 0.0

  def test_markov_matrix_rows(self):
    sb = two_state_system()
    M = sb.M_Markov
    assert M[0, 0] == 1.0 and M[0, 1] == 0.0  # absorbing row is one-hot
    assert M[1, 0] == pytest.approx(1.0)  # escape probability from transient 0
    assert M[1, 1] == pytest.approx(0.0)


class TestRates:
  def test_canonical_exit_rate_and_EAct(self):
    sb = two_state_system()
    np.testing.assert_allclose(sb.transition_rates, [1.0])
    expected = -KB_EV * 300 * np.log(np.array([1.0]) / NU0)
    np.testing.assert_allclose(sb.EAct, expected)
    assert len(sb.site_events_absorbing) == 1
    rate, absorbing, event_idx, e_act, particle = sb.site_events_absorbing[0]
    assert rate == pytest.approx(1.0)
    assert absorbing == 1 and particle == 0
    assert event_idx == 5 - 2  # num_event - 2
    assert e_act == pytest.approx(expected[0])

  def test_exit_rates_split_between_absorbing_targets(self):
    """Rates aggregated over a 2-transient chain, verified by independent algebra."""
    sb = asymmetric_system()  # superbasin_idx == [2, 0, 1]
    # From markov rows: origin 0 -> [0.6, 0, 0.4]; origin 1 -> [0, 1, 0]
    T = np.array([[0.0, 0.4], [1.0, 0.0]])
    R = np.array([[0.6], [0.0]])
    N = np.linalg.inv(np.eye(2) - T)
    B = N @ R
    fpt = N.sum(axis=1)
    expected = (B / fpt[:, None]).sum(axis=0)
    np.testing.assert_allclose(sb.transition_rates, expected)
    np.testing.assert_allclose(sb.EAct,
                               -KB_EV * 300 * np.log(sb.transition_rates / NU0))
    assert len(sb.site_events_absorbing) == 1

  def test_all_identical_rates_stay_symmetric(self):
    sb = symmetric_three_state_system()
    A = sb.A_transitions
    assert A[1, 1] == pytest.approx(2.0) and A[2, 2] == pytest.approx(2.0)
    M = sb.M_Markov  # order [2, 0, 1]
    np.testing.assert_allclose(M[1], [0.5, 0.0, 0.5])
    np.testing.assert_allclose(M[2], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(sb.transition_rates, [1.0])


class TestInvalidSuperbasins:
  def test_single_state_without_migrations_is_invalid(self):
    state = FakeSystemState([FakeSite()])  # no site_events at all
    sb = Superbasin(0, state, 0.5, [])
    assert sb.valid is False
    assert sb.absorbing_states == []
    assert sb.transient_states == []
    assert sb.superbasin_idx == []

  def test_non_migration_events_are_ignored(self):
    state = FakeSystemState([FakeSite(site_events=[(1.0, 1, 'gen', 0.1)])])
    sb = Superbasin(0, state, 0.5, [])
    assert sb.valid is False

  def test_allow_specie_removal_restored_on_invalid(self):
    state = FakeSystemState([FakeSite()])
    state.allow_specie_removal = False
    Superbasin(0, state, 0.5, [])
    assert state.allow_specie_removal is False


class TestGridCrystalIntegrity:
  def test_matching_specie_occupied_site_untouched(self):
    s0 = FakeSite(site_events=[(1.0, 1, 2, 0.3)])
    s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)])
    state = FakeSystemState([s0, s1], sites_occupied=[0])
    Superbasin(0, state, 0.4, [0])
    assert state.sites_occupied == [0]
    assert len(state.process_calls) == 2  # only the BFS virtual moves

  def test_mismatched_specie_occupied_site_is_reprocessed(self):
    s0 = FakeSite(site_events=[(1.0, 1, 2, 0.3)])
    s1 = FakeSite(site_events=[(0.5, 0, 3, 0.9)], chemical_specie='X')
    state = FakeSystemState([s0, s1], sites_occupied=[1])
    Superbasin(0, state, 0.4, [1])
    assert state.sites_occupied == []  # removed from occupied set
    assert (0, 1, 4, 1) in state.process_calls  # deposition event, num_event-1 = 4

