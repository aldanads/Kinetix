# kinetix/termination/__init__.py
"""Graceful, condition-driven early termination for kMC simulations.

Public API: :func:`get_stopping_condition` resolves the parsed
``settings.early_stopping`` config block to a live condition (or ``None``
when the feature is off). The CLI evaluates ``should_stop`` at the
dump-file cadence and breaks out of the main loop on ``True``.
"""

from __future__ import annotations

from kinetix.termination.base import StoppingCondition
from kinetix.termination.conditions import AllVOPassivated, GBVOPassivated
from kinetix.termination.enums import StoppingConditionKey
from kinetix.termination.registry import get_stopping_condition

__all__ = [
  "AllVOPassivated",
  "GBVOPassivated",
  "StoppingCondition",
  "StoppingConditionKey",
  "get_stopping_condition",
]
