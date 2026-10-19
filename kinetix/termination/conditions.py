# kinetix/termination/conditions.py
"""Concrete early-stopping conditions (one class per YAML keyword)."""

from __future__ import annotations

from kinetix.termination.base import StoppingCondition


class AllVOPassivated(StoppingCondition):
  """Stop when (a threshold fraction of) the oxygen vacancies are passivated.

  Scans ``simulator.active_event_sites`` -- the occupied-site registry -- and
  counts entries whose defect carries the ``V_O`` chemical symbol
  (``V_O_0`` = ``passivation_level == 0``, ``V_O_1`` = ``passivation_level
  >= 1``). Stops when the passivated fraction reaches ``threshold``
  (default 1.0, i.e. ``n_V_O_0 == 0``).

  Completeness (every V_O is listed, so the O(N_active) scan misses none):

  * PZT ``oxygen_vacancy`` declares non-empty ``enabled_events``
    (``["migration", "reaction"]``), so the init register (``simulator.py``:
    occupied + enabled -> appended) always lists concentration-seeded V_O.
  * Every later occupancy change funnels through ``_track_occupancy_update``
    (introduce/install appends) and ``_remove_species_at_site`` (removal
    discards), so the list stays exactly the set of occupied sites:
    migration hops, generation, reactions, passivation (product keeps the
    ``V_O`` symbol) and depassivation all keep the V_O registered.
  * Non-V_O occupants sharing the list (e.g. H, H2) are skipped by the
    symbol check; a V_O with an empty event catalog would still be counted,
    because only specie/level are read, never ``defect.events``.
  """

  def __init__(self, threshold: float = 1.0, specie: str = "V_O") -> None:
    """Args:
      threshold: Passivated fraction in (0, 1] that triggers the stop.
      specie: Chemical symbol counted as an oxygen vacancy.
    """
    if not 0.0 < threshold <= 1.0:
      raise ValueError(
        f"all_vo_passivated threshold must be in (0, 1], got {threshold!r}"
      )
    self._threshold = float(threshold)
    self._specie = specie

  def should_stop(self, simulator) -> bool:
    n_unpassivated = 0
    n_passivated = 0
    grid = simulator.grid_crystal
    for idx in simulator.active_event_sites:
      defect = grid[idx].defect
      if defect.chemical_specie != self._specie:
        continue
      if defect.passivation_level > 0:
        n_passivated += 1
      else:
        n_unpassivated += 1
    total = n_unpassivated + n_passivated
    if total == 0:
      # No vacancies present at all: nothing has saturated, so do not stop.
      # (Stopping on a vacuous count would terminate vacancy-free runs at the
      # first dump.)
      return False
    return (n_passivated / total) >= self._threshold

  def describe(self) -> str:
    return f"all_vo_passivated (threshold={self._threshold:.2f})"

  def parameters(self) -> dict:
    return {"threshold": self._threshold, "specie": self._specie}


class GBVOPassivated(StoppingCondition):
  """Stop when (a threshold fraction of) the V_O INSIDE the GB region are passivated.

  Same O(N_active) scan over ``simulator.active_event_sites`` as
  :class:`AllVOPassivated`, but each ``V_O`` site is first classified with
  ``simulator.gb_model.get_site_gb_region(tuple(site.position))`` and only
  sites whose region is in ``regions`` are counted (default
  ``["inner_boundary"]``, the GB core only; pass
  ``["inner_boundary", "outer_boundary"]`` to also include the influence
  zone). Bulk vacancies never influence the decision, so a device whose bulk
  has saturated while GB vacancies still keep the filament open keeps running.

  Guards — both return ``False`` (never stop on a vacuous or undefined count):

  * No ``V_O`` found in the requested regions: same vacuous-count rationale
    as :class:`AllVOPassivated` (stopping would terminate runs at the first
    dump for a condition nothing in the GB has saturated yet).
  * No GB configured: ``gb_model`` attribute missing, ``None``, or with empty
    ``gb_configurations`` — region classification is undefined there and must
    not crash the dump-cadence check.

  Cost: the region lookup only runs for sites already filtered by specie, so
  it is one distance evaluation per ``V_O``, not per active site.
  """

  def __init__(self, threshold: float = 1.0, specie: str = "V_O",
               regions: list[str] = ["inner_boundary"]) -> None:
    """Args:
      threshold: Passivated fraction in (0, 1] that triggers the stop.
      specie: Chemical symbol counted as an oxygen vacancy.
      regions: GB regions included in the count (default: the core only).
    """
    if not 0.0 < threshold <= 1.0:
      raise ValueError(
        f"gb_vo_passivated threshold must be in (0, 1], got {threshold!r}"
      )
    self._threshold = float(threshold)
    self._specie = specie
    # Copy into an immutable tuple: never alias the caller's list (or the
    # shared default above).
    self._regions = tuple(regions)

  def should_stop(self, simulator) -> bool:
    gb_model = getattr(simulator, "gb_model", None)
    if gb_model is None or not getattr(gb_model, "gb_configurations", None):
      # No GB configured: the region filter is undefined here -> never stop.
      return False
    n_unpassivated = 0
    n_passivated = 0
    grid = simulator.grid_crystal
    for idx in simulator.active_event_sites:
      site = grid[idx]
      defect = site.defect
      if defect.chemical_specie != self._specie:
        continue
      if gb_model.get_site_gb_region(tuple(site.position)) not in self._regions:
        continue
      if defect.passivation_level > 0:
        n_passivated += 1
      else:
        n_unpassivated += 1
    total = n_unpassivated + n_passivated
    if total == 0:
      # No vacancies in the requested GB regions: nothing has saturated, so
      # do not stop (same vacuous-count guard as AllVOPassivated).
      return False
    return (n_passivated / total) >= self._threshold

  def describe(self) -> str:
    return (
      f"gb_vo_passivated (threshold={self._threshold:.2f}, "
      f"regions={list(self._regions)})"
    )

  def parameters(self) -> dict:
    return {
      "threshold": self._threshold,
      "specie": self._specie,
      "regions": list(self._regions),
    }
