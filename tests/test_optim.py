"""Tests for numcompute.optim (grad and jacobian)."""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian


# functions where we know the exact answer

def poly(v):
    # f(x) = x0^2 + 3*x1 + x2^3, exact gradient is (2*x0, 3, 3*x2^2)
    return v[0] ** 2 + 3 * v[1] + v[2] ** 3


def vec_fn(v):
    # F(x, y) = [x*y, x+y, y^2, sin x], goes from 2 inputs to 4 outputs
    return np.array([v[0] * v[1], v[0] + v[1], v[1] ** 2, np.sin(v[0])])


# grad: correct answers

def test_grad_central_matches_exact_answer():
    g = grad(poly, [2.0, 1.0, 3.0])
    assert np.allclose(g, [4.0, 3.0, 27.0], atol=1e-3)


def test_grad_forward_matches_exact_answer():
    g = grad(poly, [2.0, 1.0, 3.0], method="forward")
    assert np.allclose(g, [4.0, 3.0, 27.0], atol=1e-3)


def test_central_is_more_accurate_than_forward():
    f = lambda v: np.exp(v[0])
    error_central = abs(grad(f, [1.0], h=1e-3)[0] - np.e)
    error_forward = abs(grad(f, [1.0], h=1e-3, method="forward")[0] - np.e)
    assert error_central < error_forward


def test_error_shrinks_faster_for_central_when_h_is_halved():
    # forward error is about h, so halving h halves it
    # central error is about h*h, so halving h makes it 4 times smaller
    f = lambda v: np.exp(v[0])
    forward_big = abs(grad(f, [1.0], h=0.1, method="forward")[0] - np.e)
    forward_small = abs(grad(f, [1.0], h=0.05, method="forward")[0] - np.e)
    central_big = abs(grad(f, [1.0], h=0.1)[0] - np.e)
    central_small = abs(grad(f, [1.0], h=0.05)[0] - np.e)
    assert 1.8 < forward_big / forward_small < 2.2
    assert 3.6 < central_big / central_small < 4.4


def test_grad_of_quadratic_form():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda v: v @ A @ v
    v = np.array([1.0, 2.0])
    assert np.allclose(grad(f, v), (A + A.T) @ v, atol=1e-6)


def test_grad_with_one_input():
    g = grad(lambda v: v[0] ** 3, [2.0])
    assert g.shape == (1,)
    assert np.allclose(g, [12.0], atol=1e-6)


def test_grad_shape_and_type():
    g = grad(poly, [2.0, 1.0, 3.0])
    assert g.shape == (3,)
    assert g.dtype == np.float64


# grad: input handling

def test_grad_works_with_integer_array():
    # an integer array cannot hold 2.00001, so x must be turned into floats first
    x_int = np.array([2, 1, 3])
    g = grad(poly, x_int)
    assert np.allclose(g, [4.0, 3.0, 27.0], atol=1e-3)
    assert np.array_equal(x_int, [2, 1, 3])  # the callers array is not changed
    assert x_int.dtype.kind == "i"


def test_grad_works_with_plain_list():
    assert np.allclose(grad(poly, [2, 1, 3]), [4, 3, 27], atol=1e-3)


def test_grad_does_not_change_x():
    x = np.array([2.0, 1.0, 3.0])
    grad(poly, x)
    grad(poly, x, method="forward")
    assert np.array_equal(x, [2.0, 1.0, 3.0])


def test_grad_number_of_calls():
    calls = []

    def counted(v):
        calls.append(1)
        return poly(v)

    grad(counted, [2.0, 1.0, 3.0], method="central")
    assert len(calls) == 6  # 2 * n with n = 3

    calls.clear()
    grad(counted, [2.0, 1.0, 3.0], method="forward")
    assert len(calls) == 4  # n + 1 with n = 3


# grad: mistakes

def test_grad_bad_method():
    with pytest.raises(ValueError, match="method"):
        grad(poly, [1.0, 1.0, 1.0], method="sideways")


def test_grad_bad_h_values():
    for bad_h in [0.0, -1e-5, np.nan, np.inf]:
        with pytest.raises(ValueError, match="h must be"):
            grad(poly, [1.0, 1.0, 1.0], h=bad_h)


def test_grad_h_that_is_not_a_number():
    for bad_h in ["1e-5", None]:
        with pytest.raises(TypeError):
            grad(poly, [1.0, 1.0, 1.0], h=bad_h)


def test_grad_bad_x():
    f = lambda v: float(np.sum(v))
    bad_values = [np.ones((2, 2)), np.array([]), 3.0, [1.0, np.nan], [1.0, np.inf]]
    for bad_x in bad_values:
        with pytest.raises(ValueError):
            grad(f, bad_x)


def test_grad_x_with_text():
    with pytest.raises(ValueError):
        grad(lambda v: 0.0, ["a", "b"])


def test_grad_f_not_callable():
    with pytest.raises(TypeError, match="callable"):
        grad("not a function", [1.0])


def test_grad_f_returns_array():
    with pytest.raises(ValueError, match="single number"):
        grad(lambda v: v * 2, [1.0, 2.0])


def test_grad_f_returns_text():
    with pytest.raises(ValueError):
        grad(lambda v: "abc", [1.0, 2.0])


# grad: tricky numbers

def test_grad_nan_from_f_shows_up_in_result():
    # log of a negative number is NaN, we do not hide it
    with np.errstate(invalid="ignore", divide="ignore"):
        g = grad(lambda v: np.log(v[0]), [0.0])
    assert np.isnan(g[0])


def test_grad_at_a_sharp_corner():
    # abs(x) at 0 has no derivative, central says 0 and forward says 1
    f = lambda v: abs(v[0])
    assert grad(f, [0.0], method="central")[0] == pytest.approx(0.0)
    assert grad(f, [0.0], method="forward")[0] == pytest.approx(1.0)


def test_grad_h_too_small_for_big_x():
    with pytest.raises(ValueError, match="too small"):
        grad(lambda v: v[0] ** 2, [1e12], h=1e-5)


def test_grad_bigger_h_fixes_big_x():
    g = grad(lambda v: v[0] ** 2, [1e12], h=1e3)
    assert np.allclose(g, [2e12], rtol=1e-6)


# jacobian

def test_jacobian_two_inputs_four_outputs():
    J = jacobian(vec_fn, [1.0, 2.0])
    expected = np.array([[2.0, 1.0], [1.0, 1.0], [0.0, 4.0], [np.cos(1.0), 0.0]])
    assert J.shape == (4, 2)  # (outputs, inputs)
    assert np.allclose(J, expected, atol=1e-6)


def test_jacobian_three_inputs_two_outputs():
    F = lambda v: np.array([v[0] + v[1], v[1] * v[2]])
    J = jacobian(F, [1.0, 2.0, 3.0])
    assert J.shape == (2, 3)
    assert np.allclose(J, [[1, 1, 0], [0, 3, 2]], atol=1e-6)


def test_jacobian_of_linear_function_is_its_matrix():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    J = jacobian(lambda v: A @ v, [0.5, -1.0, 2.0])
    assert np.allclose(J, A, atol=1e-7)


def test_jacobian_of_one_output_equals_grad():
    f = lambda v: v[0] * v[1]
    J = jacobian(f, [3.0, 5.0])
    assert J.shape == (1, 2)
    assert np.allclose(J[0], grad(f, [3.0, 5.0]))


def test_jacobian_forward_is_close_to_central():
    F = lambda v: np.array([np.sin(v[0]), v[0] * v[1]])
    central = jacobian(F, [0.5, 2.0])
    forward = jacobian(F, [0.5, 2.0], method="forward")
    assert np.allclose(forward, central, atol=1e-3)


def test_jacobian_does_not_change_x():
    x = np.array([1.0, 2.0])
    jacobian(vec_fn, x)
    assert np.array_equal(x, [1.0, 2.0])


def test_jacobian_number_of_calls():
    calls = []

    def counted(v):
        calls.append(1)
        return vec_fn(v)

    jacobian(counted, [1.0, 2.0], method="central")
    assert len(calls) == 4  # 2 * n with n = 2

    calls.clear()
    jacobian(counted, [1.0, 2.0], method="forward")
    assert len(calls) == 3  # n + 1 with n = 2


def test_jacobian_works_with_integer_array():
    J = jacobian(vec_fn, np.array([1, 2]))
    assert np.allclose(J[0], [2.0, 1.0], atol=1e-6)


def test_jacobian_f_returns_2d():
    with pytest.raises(ValueError, match="1D"):
        jacobian(lambda v: np.ones((2, 2)), [1.0, 2.0])


def test_jacobian_bad_arguments():
    with pytest.raises(ValueError, match="method"):
        jacobian(vec_fn, [1.0, 2.0], method="nope")
    with pytest.raises(ValueError):
        jacobian(vec_fn, np.ones((2, 2)))
    with pytest.raises(TypeError, match="callable"):
        jacobian(None, [1.0, 2.0])
