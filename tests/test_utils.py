"""Tests for numcompute.utils."""

import numpy as np
import pytest

from numcompute.utils import (
    batch_indices,
    euclidean_distance,
    k_nearest,
    logsumexp,
    manhattan_distance,
    relu,
    sigmoid,
    softmax,
)


# distances

def test_euclidean_distance_of_3_4_5_triangle():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0


def test_manhattan_distance():
    assert manhattan_distance([0, 0], [3, 4]) == 7.0


def test_distance_for_many_pairs_at_once():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[3.0, 4.0], [1.0, 1.0]])
    assert np.allclose(euclidean_distance(x, y), [5.0, 0.0])
    assert np.allclose(manhattan_distance(x, y), [7.0, 0.0])


def test_distance_matrix_with_broadcasting():
    X = np.random.default_rng(0).normal(size=(5, 3))
    D = euclidean_distance(X[:, None, :], X[None, :, :])
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_distance_matches_a_simple_loop():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 3))
    Y = rng.normal(size=(5, 3))
    D = euclidean_distance(X[:, None, :], Y[None, :, :])
    for i in range(4):
        for j in range(5):
            assert D[i, j] == pytest.approx(np.sqrt(np.sum((X[i] - Y[j]) ** 2)))


def test_distance_with_integer_input():
    assert euclidean_distance(np.array([1, 1]), np.array([4, 5])) == 5.0


def test_distance_with_non_contiguous_input():
    X = np.arange(24, dtype=float).reshape(6, 4)
    every_other_row = X[::2]  # not contiguous in memory
    transposed = X.T  # not contiguous either
    for a in (every_other_row, transposed):
        assert np.allclose(euclidean_distance(a, a[::-1]), euclidean_distance(a.copy(), a[::-1].copy()))


def test_distance_with_nan_gives_nan():
    assert np.isnan(euclidean_distance([0.0, np.nan], [1.0, 1.0]))


def test_distance_does_not_change_input():
    x = np.array([1.0, 2.0])
    y = np.array([4.0, 6.0])
    euclidean_distance(x, y)
    manhattan_distance(x, y)
    assert np.array_equal(x, [1.0, 2.0])
    assert np.array_equal(y, [4.0, 6.0])


def test_distance_with_shapes_that_do_not_match_raises():
    with pytest.raises(ValueError):
        euclidean_distance(np.zeros((2, 3)), np.zeros((2, 4)))


# k_nearest

def test_k_nearest_simple_example():
    X = [[0, 0], [1, 0], [5, 5]]
    idx, dist = k_nearest(X, [[0.9, 0]], k=2)
    assert idx.tolist() == [[1, 0]]
    assert np.allclose(dist, [[0.1, 0.9]])


def test_k_nearest_shapes():
    X = np.random.default_rng(2).normal(size=(10, 3))
    queries = np.random.default_rng(3).normal(size=(4, 3))
    idx, dist = k_nearest(X, queries, k=5)
    assert idx.shape == (4, 5)
    assert dist.shape == (4, 5)
    assert np.all(np.diff(dist, axis=1) >= 0)  # closest first


def test_k_nearest_ties_use_smaller_index_first():
    X = [[1.0], [-1.0], [1.0]]  # all three are distance 1 from 0
    idx, dist = k_nearest(X, [[0.0]], k=3)
    assert idx.tolist() == [[0, 1, 2]]


def test_k_nearest_with_k_equal_to_all_points():
    X = [[0.0], [2.0], [1.0]]
    idx, dist = k_nearest(X, [[0.0]], k=3)
    assert idx.tolist() == [[0, 2, 1]]


def test_k_nearest_bad_k():
    X = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="k must"):
        k_nearest(X, [[0.0]], k=0)
    with pytest.raises(ValueError, match="k must"):
        k_nearest(X, [[0.0]], k=3)


def test_k_nearest_different_number_of_columns():
    with pytest.raises(ValueError):
        k_nearest(np.zeros((3, 2)), np.zeros((1, 5)), k=1)


# sigmoid

def test_sigmoid_known_values():
    assert float(sigmoid(0)) == pytest.approx(0.5)
    assert float(sigmoid(2)) == pytest.approx(1 / (1 + np.exp(-2)))


def test_sigmoid_huge_numbers_do_not_overflow():
    with np.errstate(over="raise"):  # an overflow would now stop the test
        out = sigmoid([-1000.0, 1000.0, -np.inf, np.inf])
    assert np.array_equal(out, [0.0, 1.0, 0.0, 1.0])


def test_sigmoid_is_symmetric():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(sigmoid(-x), 1 - sigmoid(x))


def test_sigmoid_keeps_shape_and_nan():
    out = sigmoid(np.array([[1.0, np.nan], [2.0, 3.0]]))
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 1])


# relu

def test_relu_values():
    assert np.array_equal(relu([-2.0, 0.0, 3.0]), [0.0, 0.0, 3.0])


def test_relu_nan_stays_nan_and_input_is_not_changed():
    x = np.array([-1.0, np.nan, 2.0])
    out = relu(x)
    assert np.isnan(out[1])
    assert x[0] == -1.0


# softmax

def test_softmax_with_huge_numbers():
    out = softmax([1000.0, 1001.0, 1002.0])
    assert np.allclose(out, [0.09003057, 0.24472847, 0.66524096])  # naive softmax gives NaN here


def test_softmax_adds_up_to_one_along_the_axis():
    x = np.random.default_rng(4).normal(size=(3, 5))
    assert np.allclose(softmax(x, axis=-1).sum(axis=-1), 1.0)
    assert np.allclose(softmax(x, axis=0).sum(axis=0), 1.0)


def test_softmax_does_not_change_when_every_score_is_shifted():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(softmax(x), softmax(x + 100.0))


def test_softmax_single_element_is_one():
    assert np.array_equal(softmax([5.0]), [1.0])


def test_softmax_nan_gives_nan():
    assert np.all(np.isnan(softmax([1.0, np.nan, 2.0])))


def test_softmax_empty_raises():
    with pytest.raises(ValueError):
        softmax([])


def test_softmax_bad_axis_raises():
    with pytest.raises(ValueError):
        softmax(np.ones((2, 2)), axis=5)


# logsumexp

def test_logsumexp_with_huge_numbers():
    assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000 + np.log(2))


def test_logsumexp_matches_the_simple_formula_for_small_numbers():
    x = np.array([0.5, 1.0, -2.0])
    assert logsumexp(x) == pytest.approx(np.log(np.sum(np.exp(x))))


def test_logsumexp_axis_none_sums_everything():
    x = np.arange(6, dtype=float).reshape(2, 3)
    assert logsumexp(x) == pytest.approx(np.log(np.sum(np.exp(x))))


def test_logsumexp_axis_and_keepdims():
    x = np.arange(6, dtype=float).reshape(2, 3)
    assert logsumexp(x, axis=0).shape == (3,)
    assert logsumexp(x, axis=1).shape == (2,)
    assert logsumexp(x, axis=1, keepdims=True).shape == (2, 1)
    assert np.allclose(logsumexp(x, axis=1), np.log(np.sum(np.exp(x), axis=1)))


def test_logsumexp_special_values():
    with np.errstate(divide="ignore"):  # log(0) makes numpy warn, the answer is still right
        assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([1.0, np.inf]) == np.inf
    assert np.isnan(logsumexp([1.0, np.nan]))
    assert logsumexp([-np.inf, 0.0]) == pytest.approx(0.0)


def test_logsumexp_empty_raises():
    with pytest.raises(ValueError):
        logsumexp([])


# batch_indices

def test_batches_of_ten_rows_in_fours():
    batches = batch_indices(10, 4)
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_every_row_is_in_exactly_one_batch_even_when_shuffled():
    batches = batch_indices(23, 5, shuffle=True, seed=0)
    all_rows = np.concatenate(batches)
    assert sorted(all_rows.tolist()) == list(range(23))


def test_shuffle_with_same_seed_gives_same_batches():
    a = batch_indices(20, 6, shuffle=True, seed=42)
    b = batch_indices(20, 6, shuffle=True, seed=42)
    assert len(a) == len(b)
    for i in range(len(a)):
        assert np.array_equal(a[i], b[i])


def test_shuffle_with_different_seeds_gives_different_order():
    a = np.concatenate(batch_indices(20, 6, shuffle=True, seed=0))
    b = np.concatenate(batch_indices(20, 6, shuffle=True, seed=1))
    assert not np.array_equal(a, b)


def test_batch_size_bigger_than_data_gives_one_batch():
    batches = batch_indices(3, 10)
    assert len(batches) == 1
    assert batches[0].tolist() == [0, 1, 2]


def test_batch_size_that_divides_evenly():
    assert [len(b) for b in batch_indices(12, 4)] == [4, 4, 4]


def test_no_rows_gives_no_batches():
    assert batch_indices(0, 4) == []


def test_batch_size_one():
    assert len(batch_indices(5, 1)) == 5


def test_batch_size_must_be_at_least_one():
    with pytest.raises(ValueError, match="batch_size"):
        batch_indices(10, 0)
    with pytest.raises(ValueError, match="batch_size"):
        batch_indices(10, -2)


def test_batch_size_that_is_not_an_int_raises_type_error():
    with pytest.raises(TypeError):
        batch_indices(10, 2.5)


def test_batches_can_be_used_to_slice_data():
    X = np.arange(10).reshape(5, 2)
    batches = batch_indices(len(X), 2)
    assert X[batches[0]].tolist() == [[0, 1], [2, 3]]
    assert X[batches[-1]].tolist() == [[8, 9]]
