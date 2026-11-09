"""Direct sorting contracts, with NumPy's stable order as the oracle."""

import numpy as np
import pytest

from numcompute.sort_search import argsort, merge_sort, sort


@pytest.mark.parametrize('values', [
    [], [7], [3, 1, 2], [1, 2, 3], [3, 2, 1], [4, 4, 4],
    [3, 1, 3, 2, 1, 3], [np.nan, 2, np.nan, 1, np.nan],
    [np.nan, np.nan], [-np.inf, np.nan, np.inf, 0, -np.inf],
    [0.0, -0.0, 0.0, -0.0], [True, False, True], ['z', 'a', 'z', 'b'],
    np.array([2**64 - 1, 0, 2**64 - 2], dtype=np.uint64),
    np.array([2**63 - 1, -(2**63), 2**63 - 2], dtype=np.int64),
    np.array([2, 1, 2], dtype=np.float32),
    np.array([b'b', b'a', b'b']),
])
@pytest.mark.parametrize('return_indices', [False, True])
def test_merge_sort_values_dtype_indices_and_no_mutation(values, return_indices):
    arr = np.asarray(values)
    before = arr.copy()
    expected_indices = np.argsort(arr, kind='stable')
    result = merge_sort(arr, return_indices=return_indices)
    if return_indices:
        result, indices = result
        assert indices.dtype == np.dtype(np.intp)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(result, arr[indices])
    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape
    assert result.dtype == arr.dtype
    np.testing.assert_array_equal(result, arr[expected_indices])
    np.testing.assert_array_equal(arr, before)
    assert not np.shares_memory(arr, result)
    if arr.dtype.kind == 'f':
        np.testing.assert_array_equal(np.signbit(result), np.signbit(arr[expected_indices]))


@pytest.mark.parametrize('size', [0, 1, 2, 3, 17, 128, 1001])
def test_merge_sort_randomized_noncontiguous_input(size):
    rng = np.random.default_rng(42)
    arr = rng.integers(-5, 6, size * 2).astype(float)
    arr[rng.random(arr.size) < 0.2] = np.nan
    arr = arr[::-2]
    arr.setflags(write=False)
    values, indices = merge_sort(arr, return_indices=True)
    expected = np.argsort(arr, kind='stable')
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, arr[expected])


@pytest.mark.parametrize('values', [3, [[1, 2]], np.empty((0, 2)), np.zeros((2, 2, 2))])
def test_merge_sort_rejects_wrong_dimensions(values):
    with pytest.raises(ValueError, match='1-D'):
        merge_sort(values)


@pytest.mark.parametrize('values', [
    np.array([1 + 2j]), np.array([1, None], dtype=object),
    np.array([], dtype=object), np.array(['2026-01-01'], dtype='datetime64[D]'),
    np.array([(1,)], dtype=[('value', int)]),
])
def test_merge_sort_rejects_unsupported_dtypes(values):
    with pytest.raises(TypeError, match='real numbers, booleans, or strings'):
        merge_sort(values)


def test_merge_sort_does_not_delegate_to_numpy_sort(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Educational merge sort must implement sorting itself')
    monkeypatch.setattr(np, 'sort', forbidden)
    monkeypatch.setattr(np, 'argsort', forbidden)
    values, indices = merge_sort([3, 1, 1, 2], return_indices=True)
    np.testing.assert_array_equal(values, [1, 1, 2, 3])
    np.testing.assert_array_equal(indices, [1, 2, 3, 0])


@pytest.mark.parametrize('axis', [0, 1, -1, -2])
def test_numpy_sort_wrappers_are_stable(axis):
    arr = np.array([[3, 1, 1], [2, np.nan, 2], [3, 1, np.nan]])
    before = arr.copy()
    indices = argsort(arr, axis=axis)
    np.testing.assert_array_equal(indices, np.argsort(arr, axis=axis, kind='stable'))
    np.testing.assert_array_equal(sort(arr, axis=axis), np.take_along_axis(arr, indices, axis=axis))
    np.testing.assert_array_equal(arr, before)
