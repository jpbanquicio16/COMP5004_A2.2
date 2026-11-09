"""Ranking contracts and independent, count-based reference checks."""

import numpy as np
import pytest

from numcompute.rank import rank

METHODS = ['average', 'min', 'max', 'dense', 'ordinal']


def reference(values, method):
    result = []
    for i, value in enumerate(values):
        if np.isnan(value):
            result.append(np.nan)
            continue
        less = sum(v < value for v in values)
        equal = sum(v == value for v in values)
        result.append({
            'average': less + (equal + 1) / 2,
            'min': less + 1,
            'max': less + equal,
            'dense': len(set(v for v in values if v < value)) + 1,
            'ordinal': less + sum(v == value for v in values[:i + 1]),
        }[method])
    return np.array(result, dtype=float)


@pytest.mark.parametrize('method, expected', [
    ('average', [4, 1, 2.5, 2.5]), ('min', [4, 1, 2, 2]),
    ('max', [4, 1, 3, 3]), ('dense', [3, 1, 2, 2]),
    ('ordinal', [4, 1, 2, 3]),
])
def test_tie_methods(method, expected):
    np.testing.assert_array_equal(rank([30, 10, 20, 20], method), expected)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('axis', [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize('policy', ['keep', 'omit', 'raise'])
def test_randomized_axes_against_counts(method, axis, policy):
    arr = np.random.default_rng(17).integers(-3, 4, (3, 4, 5)).astype(float)
    if policy != 'raise':
        arr[1, 2, 3] = np.nan
    arr = arr[::-1, :, ::-1]  # non-contiguous view
    before = arr.copy()
    expected = np.apply_along_axis(reference, axis, arr, method)
    out = rank(arr, method, axis, nan_policy=policy)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(arr, before)
    assert out.dtype == np.float64


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('values', [[7], [3, 3, 3], [np.nan, np.nan],
                                    [np.inf, -np.inf, 0, np.inf],
                                    [2**63 - 1, 2**63 - 2, 0],
                                    [True, False, True]])
def test_special_values(method, values):
    np.testing.assert_array_equal(rank(values, method), reference(values, method))


@pytest.mark.parametrize('shape, axis', [((0,), 0), ((2, 0), 1), ((0, 3), 1)])
def test_empty_arrays(shape, axis):
    out = rank(np.empty(shape), axis=axis)
    assert out.shape == shape
    assert out.dtype == np.float64


@pytest.mark.parametrize('kwargs', [{'method': 'bad'}, {'nan_policy': 'bad'},
                                   {'nan_policy': 'raise'}])
def test_invalid_options_or_nan(kwargs):
    with pytest.raises(ValueError):
        rank([1, np.nan], **kwargs)


@pytest.mark.parametrize('axis', [1, -2])
def test_invalid_axis(axis):
    with pytest.raises(np.exceptions.AxisError):
        rank([1, 2], axis=axis)


@pytest.mark.parametrize('axis', [None, 1.5, True])
def test_noninteger_axis(axis):
    with pytest.raises(TypeError):
        rank([1, 2], axis=axis)


def test_scalar_has_no_ranking_axis():
    with pytest.raises(np.exceptions.AxisError):
        rank(5)
