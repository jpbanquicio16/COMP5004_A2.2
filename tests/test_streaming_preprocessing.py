"""Streaming (reset / partial_fit) tests for numcompute.preprocessing."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    MinMaxScaler,
    NotFittedError,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)

CHUNKINGS = [1, 3, 17]
NUMERIC = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "impute_mean": lambda: SimpleImputer("mean"),
    "impute_constant": lambda: SimpleImputer(fill_value=-1.0),
}


def data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(10, 4, size=(n, 4))
    X[::6, 0] = np.nan
    X[:60, 1] = np.nan           # all NaN in the first chunks
    X[:, 3] = 0.1                # constant with rounding error
    return X


def stream_fit(obj, X, k):
    for chunk in np.array_split(X, k):
        obj.partial_fit(chunk)
    return obj


@pytest.mark.parametrize("name", NUMERIC)
@pytest.mark.parametrize("k", CHUNKINGS)
def test_partial_fit_over_chunks_equals_fit(name, k):
    X = data()
    batch = NUMERIC[name]().fit(X)
    streamed = stream_fit(NUMERIC[name](), X, k)
    np.testing.assert_allclose(streamed.transform(X), batch.transform(X),
                               rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("k", CHUNKINGS)
def test_standard_scaler_attributes_are_chunk_invariant(k):
    X = data(seed=2)
    a, b = StandardScaler().fit(X), stream_fit(StandardScaler(), X, k)
    np.testing.assert_allclose(b.mean_, a.mean_, rtol=1e-13)
    np.testing.assert_allclose(b.scale_, a.scale_, rtol=1e-12)
    np.testing.assert_array_equal(b.n_samples_seen_, (~np.isnan(X)).sum(axis=0))


def test_standard_scaler_keeps_near_constant_rule_when_streaming():
    s = stream_fit(StandardScaler(), np.full((30, 1), 0.1), 7)
    assert s.scale_[0] == 1.0
    np.testing.assert_array_equal(s.transform(np.full((3, 1), 0.1)), 0.0)


@pytest.mark.parametrize("name", ["standard", "minmax"])
def test_all_nan_column_in_a_chunk_does_not_raise_when_streaming(name):
    obj = NUMERIC[name]()
    obj.partial_fit(np.array([[1.0, np.nan], [2.0, np.nan]]))     # no error
    out = obj.transform(np.array([[1.5, 3.0]]))
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1])       # unknown col -> NaN
    obj.partial_fit(np.array([[3.0, 5.0], [4.0, 7.0]]))
    assert np.isfinite(obj.transform(np.array([[1.5, 3.0]]))).all()


def test_imputer_all_nan_column_in_a_chunk_does_not_raise_when_streaming():
    imp = SimpleImputer("mean").partial_fit(np.array([[1.0, np.nan], [2.0, np.nan]]))
    # nothing to impute with yet: the NaN stays NaN (the tree can route it)
    np.testing.assert_array_equal(imp.transform(np.array([[np.nan, np.nan]])),
                                  [[1.5, np.nan]])
    imp.partial_fit(np.array([[np.nan, 8.0]]))
    np.testing.assert_array_equal(imp.transform(np.array([[np.nan, np.nan]])), [[1.5, 8.0]])


def test_batch_fit_still_rejects_an_all_nan_column():
    with pytest.raises(ValueError, match=r"\[1\]"):
        StandardScaler().fit(np.array([[1.0, np.nan], [2.0, np.nan]]))


@pytest.mark.parametrize("name", NUMERIC)
def test_feature_count_locked_by_first_chunk(name):
    obj = NUMERIC[name]().partial_fit(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="features"):
        obj.partial_fit(np.zeros((3, 5)))


@pytest.mark.parametrize("name", NUMERIC)
def test_empty_first_chunk_then_data(name):
    obj = NUMERIC[name]().partial_fit(np.zeros((0, 2)))
    assert obj.transform(np.zeros((0, 2))).shape == (0, 2)
    obj.partial_fit(np.array([[1.0, 2.0], [3.0, 6.0]]))
    assert np.isfinite(obj.transform(np.array([[2.0, 4.0]]))).all()


@pytest.mark.parametrize("name", NUMERIC)
def test_reset_forgets_state(name):
    obj = NUMERIC[name]().partial_fit(np.ones((3, 2)))
    assert obj.reset() is obj
    with pytest.raises(NotFittedError):
        obj.transform(np.ones((1, 2)))
    obj.partial_fit(np.ones((2, 7)))                 # new width allowed after reset


def test_fit_restarts_instead_of_accumulating():
    s = StandardScaler().partial_fit(np.array([[100.0], [200.0]]))
    s.fit(np.array([[1.0], [3.0]]))
    assert s.mean_[0] == 2.0


def test_zero_variance_chunk_then_spread():
    s = StandardScaler().partial_fit(np.full((5, 1), 3.0))
    np.testing.assert_array_equal(s.transform(np.array([[3.0]])), 0.0)
    s.partial_fit(np.array([[1.0], [5.0]]))
    assert s.scale_[0] > 0 and s.scale_[0] != 1.0


def test_minmax_running_range_grows():
    m = MinMaxScaler().partial_fit(np.array([[0.0], [10.0]]))
    m.partial_fit(np.array([[20.0]]))
    np.testing.assert_allclose(m.transform(np.array([[10.0]])), [[0.5]])


def test_imputer_running_mean_uses_every_chunk():
    imp = SimpleImputer("mean").partial_fit(np.array([[1.0], [np.nan]]))
    imp.partial_fit(np.array([[5.0], [6.0]]))
    np.testing.assert_allclose(imp.statistics_, [4.0])


# ---------------- OneHotEncoder ----------------

def test_encoder_new_categories_go_at_the_end():
    enc = OneHotEncoder(handle_unknown="ignore")
    first = np.array([["b", "x"], ["a", "y"]])
    enc.partial_fit(first)
    before = enc.transform(first)
    enc.partial_fit(np.array([["c", "x"], ["a", "z"]]))      # 'c' and 'z' are new
    after = enc.transform(first)
    assert enc.n_features_out_ == 6
    np.testing.assert_array_equal(after[:, :4], before)      # old columns did not move
    np.testing.assert_array_equal(after[:, 4:], 0)
    np.testing.assert_array_equal(enc.transform(np.array([["c", "z"]])), [[0, 0, 0, 0, 1, 1]])
    assert enc.categories_[0].tolist() == ["a", "b", "c"]


def test_encoder_streamed_matches_fit_when_first_chunk_has_everything():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 4, size=(200, 3))
    X[:4] = np.arange(4)[:, None]            # all categories in the first chunk
    batch = OneHotEncoder().fit(X)
    streamed = stream_fit(OneHotEncoder(), X, 9)
    np.testing.assert_array_equal(streamed.transform(X), batch.transform(X))


def test_max_categories_fixes_width_from_the_start():
    enc = OneHotEncoder(handle_unknown="ignore", max_categories=3)
    enc.partial_fit(np.array([["a", "p"]]))
    assert enc.n_features_out_ == 6
    np.testing.assert_array_equal(enc.transform(np.array([["a", "p"]])), [[1, 0, 0, 1, 0, 0]])
    enc.partial_fit(np.array([["b", "q"], ["c", "p"]]))
    assert enc.n_features_out_ == 6
    np.testing.assert_array_equal(enc.transform(np.array([["c", "q"]])), [[0, 0, 1, 0, 1, 0]])


def test_max_categories_overflow_with_ignore_encodes_zeros():
    enc = OneHotEncoder(handle_unknown="ignore", max_categories=2)
    enc.partial_fit(np.array([["a"], ["b"]]))
    enc.partial_fit(np.array([["c"]]))                         # no slot left
    np.testing.assert_array_equal(enc.transform(np.array([["c"]])), [[0, 0]])
    assert enc.categories_[0].tolist() == ["a", "b"]


def test_max_categories_overflow_with_error_raises_and_keeps_state():
    enc = OneHotEncoder(max_categories=2).partial_fit(np.array([["a"]]))
    with pytest.raises(ValueError, match="max_categories"):
        enc.partial_fit(np.array([["b"], ["c"]]))
    assert enc.categories_[0].tolist() == ["a"]


def test_unknown_in_transform_still_raises_by_default():
    enc = OneHotEncoder(max_categories=4).partial_fit(np.array([["a"]]))
    with pytest.raises(ValueError, match="unseen"):
        enc.transform(np.array([["b"]]))


def test_encoder_numeric_and_wider_strings_later():
    enc = OneHotEncoder(handle_unknown="ignore").partial_fit(np.array([["ab"]]))
    enc.partial_fit(np.array([["a much longer name"]]))       # dtype widens
    out = enc.transform(np.array([["a much longer name"], ["ab"]]))
    np.testing.assert_array_equal(out, [[0, 1], [1, 0]])
    num = OneHotEncoder().partial_fit(np.array([[1], [2]])).partial_fit(np.array([[2.5]]))
    assert num.n_features_out_ == 3


def test_encoder_type_family_locked_by_first_chunk():
    enc = OneHotEncoder().partial_fit(np.array([[1], [2]]))
    with pytest.raises(TypeError):
        enc.partial_fit(np.array([["a"]]))


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_bad_max_categories(bad):
    with pytest.raises(ValueError):
        OneHotEncoder(max_categories=bad)
