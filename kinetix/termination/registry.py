# kinetix/termination/registry.py
"""Keyword -> condition-class map plus the public factory.

To add a condition: implement a :class:`StoppingCondition` in
``conditions.py``, add its keyword to :class:`StoppingConditionKey`, and
register it in ``_REGISTRY``. The registry asserts enum/registry consistency
at import so the two can never drift apart.
"""

from __future__ import annotations

from kinetix.termination.base import StoppingCondition
from kinetix.termination.conditions import AllVOPassivated, GBVOPassivated
from kinetix.termination.enums import StoppingConditionKey

_REGISTRY: dict[str, type[StoppingCondition]] = {
  StoppingConditionKey.ALL_VO_PASSIVATED.value: AllVOPassivated,
  StoppingConditionKey.GB_VO_PASSIVATED.value: GBVOPassivated,
}

_missing = set(StoppingConditionKey.valid_options()) - set(_REGISTRY)
assert not _missing, f"stopping conditions without registry entry: {sorted(_missing)}"


def get_stopping_condition(config) -> StoppingCondition | None:
  """Resolve a parsed early-stopping config block to a live condition.

  Args:
    config: The ``SimulationSettings.early_stopping`` value
      (``EarlyStoppingConfig`` or ``None``). Only ``enabled``, ``condition``
      and ``parameters`` are read, so test doubles work without importing the
      config dataclasses (which would create a circular import).

  Returns:
    A configured :class:`StoppingCondition`, or ``None`` when the block is
    absent or ``enabled`` is false (the simulation then runs to completion).

  Raises:
    ValueError: If ``condition`` is not a registered keyword. The message
      lists the valid options.
  """
  if config is None or not getattr(config, "enabled", False):
    return None
  keyword = getattr(config, "condition", None)
  if keyword is None:
    raise ValueError(
      "early_stopping.enabled is true but no 'condition' keyword was given. "
      f"Valid options: {', '.join(StoppingConditionKey.valid_options())}"
    )
  try:
    StoppingConditionKey(keyword)
  except ValueError:
    raise ValueError(
      f"Unknown early_stopping.condition {keyword!r}. Valid options: "
      f"{', '.join(StoppingConditionKey.valid_options())}"
    ) from None
  cls = _REGISTRY[keyword]
  params = dict(getattr(config, "parameters", None) or {})
  return cls(**params)
