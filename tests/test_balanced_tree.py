"""Unit tests for the binary search tree used by the kMC event selector.

:class:`kinetix.utils.balanced_tree` maps a uniform random draw onto an
event: leaves carry ``(rate, event_type, particle)`` tuples while internal
nodes carry the sum of the rates beneath them (``update_data``). These
tests pin the **current** behavior of ``build_tree`` / ``update_data`` /
``search_value`` — including known sharp edges such as out-of-range
targets — so the data structure can be refactored safely.
"""

from __future__ import annotations

from typing import Any

import pytest

from kinetix.utils.balanced_tree import Node, build_tree, search_value, update_data


def _leaf_values(root: Node | None) -> list[Any]:
  """Collect leaf payloads in left-to-right tree order."""
  if root is None:
    return []
  if root.left is None and root.right is None:
    return [root.data]
  return _leaf_values(root.left) + _leaf_values(root.right)


def _tuple_leaves(rates: list[float]) -> tuple[tuple[float, str, int], ...]:
  """Leaf payloads as the production format (rate, event_type, particle)."""
  return tuple((r, 'migration', i) for i, r in enumerate(rates))


class TestBuildTree:
  def test_empty_input_returns_none(self) -> None:
    assert build_tree(()) is None
    assert build_tree([]) is None

  def test_single_element_returns_leaf(self) -> None:
    root = build_tree((7.0,))
    assert isinstance(root, Node)
    assert root.data == 7.0
    assert root.left is None and root.right is None

  @pytest.mark.parametrize('n', [2, 3, 10, 100])
  def test_leaf_count_and_order_preserved(self, n: int) -> None:
    arr = tuple(float(i + 1) for i in range(n))
    root = build_tree(arr)
    assert _leaf_values(root) == list(arr)

  def test_internal_nodes_start_with_none_data(self) -> None:
    root = build_tree((1.0, 2.0, 3.0))
    assert root.data is None  # only update_data() fills inner nodes


class TestUpdateData:
  def test_none_root_is_a_no_op(self) -> None:
    assert update_data(None) is None

  def test_single_leaf_unchanged(self) -> None:
    leaf = Node((5.0, 'migration', 0))
    assert update_data(leaf) == (5.0, 'migration', 0)  # nothing to sum

  @pytest.mark.parametrize('n', [2, 3, 10, 100])
  def test_root_equals_sum_of_leaf_rates(self, n: int) -> None:
    rates = [float(i + 1) for i in range(n)]
    root = build_tree(_tuple_leaves(rates))
    update_data(root)
    assert root.data == pytest.approx(sum(rates))

  def test_plain_float_leaves_also_sum(self) -> None:
    root = build_tree((1.0, 2.0, 4.0))
    update_data(root)
    assert root.data == pytest.approx(7.0)

  def test_update_is_idempotent(self) -> None:
    root = build_tree(_tuple_leaves([1.0, 2.0, 3.0, 4.0]))
    first = update_data(root)
    second = update_data(root)
    assert first == second == pytest.approx(10.0)


class TestSearchValue:
  """search_value() selects the leaf whose cumulative rate covers target."""

  def _tree(self):
    # Cumulative coverage: (0,1] -> 1, (1,3] -> 2, (3,6] -> 3, (6,10] -> 4.
    root = build_tree(_tuple_leaves([1.0, 2.0, 3.0, 4.0]))
    update_data(root)
    return root

  @pytest.mark.parametrize(
    'target,expected_rate',
    [(0.5, 1.0), (1.0, 1.0), (1.5, 2.0), (3.0, 2.0), (3.5, 3.0), (6.0, 3.0), (6.5, 4.0), (10.0, 4.0)],
  )
  def test_cumulative_leaf_selection(self, target: float, expected_rate: float) -> None:
    leaf = search_value(self._tree(), target)
    assert leaf[0] == pytest.approx(expected_rate)

  def test_returns_full_leaf_tuple(self) -> None:
    leaves = _tuple_leaves([1.0, 2.0, 3.0, 4.0])
    root = build_tree(leaves)
    update_data(root)
    assert search_value(root, 0.5) == leaves[0]

  def test_target_below_zero_selects_first_leaf(self) -> None:
    """Current semantics: any target <= the first leaf selects it."""
    assert search_value(self._tree(), 0.0)[0] == pytest.approx(1.0)
    assert search_value(self._tree(), -1.0)[0] == pytest.approx(1.0)

  def test_target_above_total_raises_unboundlocalerror(self) -> None:
    """Documented sharp edge: out-of-range targets currently raise
    UnboundLocalError (leaf nodes have no left child to compare against).
    Pinned here so a future fix is a deliberate, visible change."""
    with pytest.raises(UnboundLocalError):
      search_value(self._tree(), 10.5)

  def test_duplicate_rates_return_a_valid_leaf(self) -> None:
    root = build_tree(_tuple_leaves([1.0, 1.0, 1.0, 1.0]))
    update_data(root)
    for target in (0.5, 1.0, 2.5, 4.0):
      assert search_value(root, target)[0] == pytest.approx(1.0)

  def test_single_leaf_tree(self) -> None:
    root = build_tree(_tuple_leaves([5.0]))
    update_data(root)
    assert search_value(root, 3.0) == (5.0, 'migration', 0)

  def test_odd_leaf_count(self) -> None:
    root = build_tree(_tuple_leaves([1.0, 2.0, 3.0]))
    update_data(root)
    assert root.data == pytest.approx(6.0)
    assert search_value(root, 4.0)[0] == pytest.approx(3.0)