# kinetix/termination/enums.py
"""Valid early-stopping condition keywords.

Single source of truth for the ``settings.early_stopping.condition`` YAML
field. The config loader validates the user-supplied keyword against this enum
at load time; the registry maps each member to its concrete implementation.
"""

from __future__ import annotations

from enum import Enum


class StoppingConditionKey(str, Enum):
  """Registered early-stopping condition keywords (YAML ``condition`` values)."""

  #: PZT: stop once (a threshold fraction of) the oxygen vacancies are passivated.
  ALL_VO_PASSIVATED = "all_vo_passivated"

  @classmethod
  def valid_options(cls) -> list[str]:
    """Accepted keyword strings, for error messages and metadata."""
    return [member.value for member in cls]
