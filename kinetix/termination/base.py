"""Abstract strategy interface for kMC early-stopping conditions.

A stopping condition answers one question, evaluated periodically (at the
dump-file cadence in ``cli.py`` - never inside the inner kMC step): *has the
user-defined physical condition that justifies terminating the simulation been
met?*

Implementations must be **stateless**: they inspect simulation state only
through the ``simulator`` argument and must never mutate it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StoppingCondition(ABC):
  """Strategy interface for graceful, condition-driven simulation termination."""

  @abstractmethod
  def should_stop(self, simulator) -> bool:
    """Return ``True`` when the simulation should terminate early.

    Parameters
    ----------
    simulator : KMCSimulator
        The live simulator whose state is inspected (read-only).

    Returns
    -------
    bool
        ``True`` if the stopping condition is satisfied.
    """

  @abstractmethod
  def describe(self) -> str:
    """Human-readable description used in the INFO log message."""

  def parameters(self) -> dict:
    """Optional: condition-specific parameters, for logging and metadata.

    Overridden by conditions that carry tunable parameters (e.g. a threshold).
    """
    return {}
