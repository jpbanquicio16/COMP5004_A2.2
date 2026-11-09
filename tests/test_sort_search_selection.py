"""Selection and search behavior, checked against NumPy and explicit results."""

import numpy as np
import pytest

from numcompute.sort_search import binary_search, quickselect, searchsorted, topk


@pytest.mark.parametrize(
    "values",
    [[7], [4, 1, 4, 2, 4, -3], [2, 2, 2], [1, 2, 3, 4],
     [4, 3, 2, 1], [-np.inf, 1.5, np.inf, -2.5]],
)
def test_quickselect_all_ranks(values):
    arr = np.array(values)
    original = arr.copy()
    for k, expected in enumerate(np.sort(arr)):
        assert quickselect(arr, k, rng=42) == expected
    np.testing.assert_array_equal(arr, original)


def test_quickselect_accepts_generator():
    assert quickselect([9, 1, 3], 1, rng=np.random.default_rng(7)) == 3


@pytest.mark.parametrize(
    "values, k, error",
    [([], 0, ValueError), ([[1, 2]], 0, ValueError), (3, 0, ValueError),
     ([1, np.nan], 0, ValueError), ([1, 2], -1, ValueError),
     ([1, 2], 2, ValueError), ([1, 2], 0.5, TypeError),
     ([1, 2], True, TypeError)],
)
def test_quickselect_invalid_inputs(values, k, error):
    with pytest.raises(error):
        quickselect(values, k)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("ordered", [True, False])
def test_topk_axes_boundaries_and_ties(axis, k, largest, ordered):
    arr = np.array([[4, 4, 1], [4, 2, 2], [1, 2, 2]])
    original = arr.copy()
    values, indices = topk(arr, k, axis=axis, largest=largest, sorted=ordered)

    expected_shape = list(arr.shape)
    expected_shape[axis] = k
    assert values.shape == indices.shape == tuple(expected_shape)
    assert np.issubdtype(indices.dtype, np.integer)
    np.testing.assert_array_equal(values, np.take_along_axis(arr, indices, axis=axis))
    rows = np.moveaxis(arr, axis, -1)
    index_rows = np.moveaxis(indices, axis, -1)
    for row, selected in zip(rows, index_rows):
        expected = sorted(
            range(len(row)),
            key=lambda i: (-int(row[i]) if largest else int(row[i]), i),
        )[:k]
        if ordered:
            np.testing.assert_array_equal(selected, expected)
        else:
            assert set(selected) == set(expected)
    np.testing.assert_array_equal(arr, original)


@pytest.mark.parametrize(
    "arr, expected",
    [
        (np.array([0, 2**64 - 1, 2**64 - 1], dtype=np.uint64), [1, 2, 0]),
        (np.array([-(2**63), 0, 2**63 - 1], dtype=np.int64), [2, 1, 0]),
        (np.array([False, True, True]), [1, 2, 0]),
        (np.array([-np.inf, np.inf, 0.5]), [1, 2, 0]),
    ],
)
def test_topk_preserves_dtype_and_handles_extreme_values(arr, expected):
    values, indices = topk(arr, 3)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, arr[expected])
    assert values.dtype == arr.dtype


def test_topk_three_dimensions():
    arr = np.arange(24).reshape(2, 3, 4)
    values, indices = topk(arr, 2, axis=1)
    np.testing.assert_array_equal(values, arr[:, [2, 1], :])
    np.testing.assert_array_equal(indices, np.broadcast_to(np.array([2, 1])[None, :, None], (2, 2, 4)))


@pytest.mark.parametrize("shape, axis, k", [((0,), 0, 0), ((2, 0), 1, 0), ((0, 3), 1, 2)])
def test_topk_empty_dimensions(shape, axis, k):
    values, indices = topk(np.empty(shape), k, axis=axis)
    expected_shape = list(shape)
    expected_shape[axis] = k
    assert values.shape == indices.shape == tuple(expected_shape)
    assert values.size == indices.size == 0


@pytest.mark.parametrize(
    "arr, k, axis, error",
    [([1, 2], -1, 0, ValueError), ([1, 2], 3, 0, ValueError),
     ([1, 2], 1.0, 0, TypeError), ([1, 2], True, 0, TypeError),
     ([1, 2], 1, 1, np.exceptions.AxisError),
     ([1, 2], 1, -2, np.exceptions.AxisError),
     (3, 0, 0, np.exceptions.AxisError)],
)
def test_topk_invalid_inputs(arr, k, axis, error):
    with pytest.raises(error):
        topk(arr, k, axis=axis)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("arr", [[], [2], [-3, 1, 1, 1, 5], [-np.inf, 0, np.inf, np.nan, np.nan]])
def test_search_functions_match_numpy(arr, side):
    queries = [-np.inf, -4, 1, 2, 5, 6, np.inf, np.nan]
    expected = np.searchsorted(arr, queries, side=side)
    result = searchsorted(arr, queries, side=side)
    np.testing.assert_array_equal(result, expected)
    assert np.issubdtype(result.dtype, np.integer)
    for target, index in zip(queries, expected):
        binary_result = binary_search(arr, target, side=side)
        scalar_result = searchsorted(arr, target, side=side)
        assert binary_result == scalar_result == index
        assert type(binary_result) is int
        assert type(scalar_result) is int


@pytest.mark.parametrize("search", [binary_search, searchsorted])
def test_search_defaults_to_left(search):
    assert search([1, 2, 2, 3], 2) == 1


def test_searchsorted_empty_queries():
    result = searchsorted([1, 2, 3], [])
    np.testing.assert_array_equal(result, [])
    assert np.issubdtype(result.dtype, np.integer)


@pytest.mark.parametrize("search", [binary_search, searchsorted])
@pytest.mark.parametrize("side, expected", [("left", 1), ("right", 3)])
def test_search_strings(search, side, expected):
    assert search(["a", "b", "b", "z"], "b", side=side) == expected


@pytest.mark.parametrize("search", [binary_search, searchsorted])
@pytest.mark.parametrize("arr, side", [([[1, 2]], "left"), (1, "left"), ([1, 2], "middle"), ([], "middle")])
def test_search_invalid_inputs(search, arr, side):
    with pytest.raises(ValueError):
        search(arr, 1, side=side)


def test_binary_search_rejects_non_scalar_target():
    with pytest.raises(ValueError):
        binary_search([1, 2], [1, 2])


def test_searchsorted_rejects_multidimensional_queries():
    with pytest.raises(ValueError):
        searchsorted([1, 2], [[1, 2]])
