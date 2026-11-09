"""Percentile interpolation, shape, validation and inverse ranking tests."""

import numpy as np
import pytest

from numcompute.rank import _check_q, percentile, percentile_rank

METHODS = ['linear', 'lower', 'higher', 'nearest', 'midpoint']


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('axis', [None, 0, 1, 2, -1, -2])
@pytest.mark.parametrize('q', [0, 50, 100, [0, 10, 25, 50, 75, 90, 100],
                              [[10, 30], [70, 90]], []])
def test_matches_numpy(method, axis, q):
    arr = np.random.default_rng(25).normal(size=(3, 4, 5))[:, ::-1, ::2]
    before = arr.copy()
    actual = percentile(arr, q, axis, method=method)
    expected = np.percentile(arr, q, axis=axis, method=method)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.shape(actual) == np.shape(expected)
    np.testing.assert_array_equal(arr, before)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('policy', ['omit', 'propagate'])
def test_nan_policy_per_slice(method, policy):
    arr = np.array([[1, np.nan, 3], [10, 20, 30]])
    reference = np.nanpercentile if policy == 'omit' else np.percentile
    np.testing.assert_allclose(
        percentile(arr, [0, 25, 50, 100], axis=1, method=method, nan_policy=policy),
        reference(arr, [0, 25, 50, 100], axis=1, method=method), equal_nan=True,
    )


@pytest.mark.parametrize('policy', ['omit', 'propagate'])
def test_all_nan_slice(policy):
    np.testing.assert_allclose(
        percentile([[np.nan, np.nan], [2, 4]], [0, 50, 100], axis=1, nan_policy=policy),
        [[np.nan, 2], [np.nan, 3], [np.nan, 4]], equal_nan=True,
    )


@pytest.mark.parametrize('method', METHODS)
def test_singleton_and_constant(method):
    assert percentile(7, 35, method=method) == 7
    np.testing.assert_array_equal(percentile([4, 4, 4], [0, 32, 100], method=method), [4, 4, 4])


def test_nearest_rounds_half_to_even():
    np.testing.assert_array_equal(percentile([0, 10, 20, 30], [50 / 3, 50, 250 / 3], method='nearest'), [0, 20, 20])


@pytest.mark.parametrize('method', ['linear', 'midpoint'])
def test_large_integers_do_not_overflow(method):
    assert percentile(np.array([2**63 - 2, 2**63 - 1]), 50, method=method) > 9e18


@pytest.mark.parametrize('method', METHODS)
def test_infinite_endpoints_and_constant(method):
    np.testing.assert_array_equal(percentile([-np.inf, 0, np.inf], [0, 100], method=method), [-np.inf, np.inf])
    assert percentile([np.inf, np.inf], 50, method=method) == np.inf


@pytest.mark.parametrize('q', [-1, 101, np.nan, np.inf, -np.inf, [50, np.nan], [[0, 101]]])
def test_invalid_percentiles(q):
    with pytest.raises(ValueError):
        percentile([1, 2], q)


def test_check_q_preserves_shape_and_converts_dtype():
    out = _check_q([[0, 50], [75, 100]])
    assert out.shape == (2, 2)
    assert out.dtype == np.float64


@pytest.mark.parametrize('values, kwargs', [([], {}), (np.empty((0, 3)), {'axis': 1}),
    ([1, 2], {'method': 'bad'}), ([1, 2], {'nan_policy': 'bad'}),
    ([1, np.nan], {'nan_policy': 'raise'})])
def test_invalid_input(values, kwargs):
    with pytest.raises(ValueError):
        percentile(values, 50, **kwargs)


@pytest.mark.parametrize('axis', [1, -2])
def test_invalid_axis(axis):
    with pytest.raises(np.exceptions.AxisError):
        percentile([1, 2], 50, axis=axis)


@pytest.mark.parametrize('axis', [1.5, True, (0,)])
def test_invalid_axis_type(axis):
    with pytest.raises(TypeError):
        percentile([1, 2], 50, axis=axis)


@pytest.mark.parametrize('kind, expected', [('weak', 75), ('strict', 25), ('mean', 50)])
def test_percentile_rank_ties(kind, expected):
    out = percentile_rank([1, 2, 2, 3], 2, kind=kind)
    assert isinstance(out, float)
    assert out == expected


@pytest.mark.parametrize('kind', ['weak', 'strict', 'mean'])
def test_percentile_rank_against_counts(kind):
    arr = np.random.default_rng(5).integers(-10, 10, 100)[::-2]
    before = arr.copy()
    scores = np.array([[-np.inf, -5, 0], [2.5, 8, np.inf]])
    strict = (arr < scores[..., None]).mean(axis=-1) * 100
    weak = (arr <= scores[..., None]).mean(axis=-1) * 100
    expected = {'strict': strict, 'weak': weak, 'mean': (strict + weak) / 2}[kind]
    np.testing.assert_allclose(percentile_rank(arr, scores, kind=kind), expected)
    np.testing.assert_array_equal(arr, before)


@pytest.mark.parametrize('kind', ['weak', 'strict', 'mean'])
def test_percentile_rank_nan_and_empty_scores(kind):
    assert np.isnan(percentile_rank([1, np.nan], 1, kind=kind))
    np.testing.assert_array_equal(percentile_rank([1, np.nan], [0, 2], kind=kind), [np.nan, np.nan])
    assert np.isnan(percentile_rank([1, 2], np.nan, kind=kind))
    assert percentile_rank([1, 2], np.empty((2, 0)), kind=kind).shape == (2, 0)


@pytest.mark.parametrize('values, kind', [([], 'weak'), ([[1, 2]], 'weak'),
                                         (3, 'weak'), ([1, 2], 'bad')])
def test_percentile_rank_invalid(values, kind):
    with pytest.raises(ValueError):
        percentile_rank(values, 1, kind=kind)
