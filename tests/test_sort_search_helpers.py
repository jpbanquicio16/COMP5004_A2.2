"""Contract tests for the validation helpers in sort_search."""

import numpy as np
import pytest

from numcompute.sort_search import _as_1d, _check_axis, _check_k


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param([1, 2, 3], [1, 2, 3], id="python-list"),
        pytest.param(np.array([1, 2, 3]), [1, 2, 3], id="numpy-array"),
        pytest.param(np.array([]), [], id="empty-array"),
        pytest.param(np.array([7]), [7], id="single-element"),
    ],
)
def test_as_1d_returns_one_dimensional_array(value, expected):
    result = _as_1d(value)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(3, id="scalar"),
        pytest.param(np.array(3), id="zero-dimensional-array"),
        pytest.param([[1, 2, 3]], id="row-vector"),
        pytest.param([[1], [2], [3]], id="column-vector"),
        pytest.param(np.zeros((1, 1, 1)), id="three-dimensional-array"),
        pytest.param(np.empty((0, 3)), id="empty-two-dimensional-array"),
    ],
)
def test_as_1d_rejects_invalid_dimensions(value):
    with pytest.raises(ValueError):
        _as_1d(value)


# Despite its name, allow_zero controls the upper bound; both modes accept zero.
@pytest.mark.parametrize(
    "allow_zero", [True, False], ids=["inclusive-upper-bound", "exclusive-upper-bound"]
)
@pytest.mark.parametrize("k", [0, 1, 4])
def test_check_k_accepts_python_integers_in_both_modes(k, allow_zero):
    result = _check_k(k, n=5, allow_zero=allow_zero)

    assert result == k
    assert type(result) is int


def test_check_k_default_mode_includes_n():
    assert _check_k(5, n=5) == 5


def test_check_k_inclusive_mode_accepts_n():
    assert _check_k(5, n=5, allow_zero=True) == 5


@pytest.mark.parametrize(
    "allow_zero, k",
    [
        pytest.param(True, -1, id="inclusive-negative"),
        pytest.param(True, 6, id="inclusive-above-n"),
        pytest.param(False, -1, id="exclusive-negative"),
        pytest.param(False, 5, id="exclusive-at-n"),
        pytest.param(False, 6, id="exclusive-above-n"),
    ],
)
def test_check_k_rejects_out_of_range_values(allow_zero, k):
    with pytest.raises(ValueError):
        _check_k(k, n=5, allow_zero=allow_zero)


@pytest.mark.parametrize(
    "k",
    [
        pytest.param(np.int64(2), id="numpy-int64"),
        pytest.param(2.5, id="fractional-float"),
        pytest.param(2.0, id="integral-float"),
        pytest.param("2", id="numeric-string"),
        pytest.param(True, id="boolean-true"),
        pytest.param(False, id="boolean-false"),
    ],
)
def test_check_k_requires_exact_python_int(k):
    """The implementation rejects NumPy integers and bool (an int subclass)."""
    with pytest.raises(TypeError):
        _check_k(k, n=5)


@pytest.mark.parametrize(
    "axis, expected",
    [(0, 0), (1, 1), (2, 2), (-1, 2), (-2, 1), (-3, 0)],
)
def test_check_axis_normalizes_valid_axes(axis, expected):
    assert _check_axis(axis, ndim=3) == expected


@pytest.mark.parametrize("axis", [3, 4, 10, -4, -10])
def test_check_axis_rejects_out_of_range_axes(axis):
    with pytest.raises(np.exceptions.AxisError):
        _check_axis(axis, ndim=3)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(np.int64(1), id="numpy-int64"),
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="none"),
    ],
)
def test_check_axis_rejects_non_python_int_types(axis):
    with pytest.raises(TypeError):
        _check_axis(axis, ndim=3)
