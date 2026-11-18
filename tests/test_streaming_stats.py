"""Tests for the streaming parts of numcompute.stats:
StreamingStats (Welford / Chan merge), update_stats and StreamingHistogram."""

import numpy as np
import pytest

from numcompute_stream import stats as s
from numcompute_stream.stats import StreamingHistogram, StreamingStats, update_stats

CHUNKINGS = [1, 3, 17]


def data_with_nans(seed=0, n=600, f=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 3, size=(n, f))
    X[::7, 1] = np.nan
    X[:100, 2] = np.nan          # all NaN for the first chunks
    X[:, 3] = 2.5                # constant column
    return X


def stream(obj, X, k, method="update_stats"):
    for chunk in np.array_split(X, k):
        getattr(obj, method)(chunk)
    return obj


# ---------------- moments: chunk invariance ----------------

@pytest.mark.parametrize("k", CHUNKINGS)
def test_mean_and_variance_match_batch_for_any_chunking(k):
    X = data_with_nans()
    st = stream(StreamingStats(), X, k)
    np.testing.assert_allclose(st.mean, np.nanmean(X, axis=0), rtol=1e-13)
    np.testing.assert_allclose(st.variance(), np.nanvar(X, axis=0), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(st.std(ddof=1), np.nanstd(X, axis=0, ddof=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(st.count, (~np.isnan(X)).sum(axis=0))
    np.testing.assert_array_equal(st.min, np.nanmin(X, axis=0))
    np.testing.assert_array_equal(st.max, np.nanmax(X, axis=0))


def test_results_identical_across_chunkings():
    X = data_with_nans(seed=3)
    results = [stream(StreamingStats(), X, k) for k in CHUNKINGS]
    for other in results[1:]:
        np.testing.assert_allclose(other.mean, results[0].mean, rtol=1e-14)
        np.testing.assert_allclose(other.variance(), results[0].variance(), rtol=1e-12)


def test_one_row_at_a_time():
    X = data_with_nans(seed=1, n=150)
    st = stream(StreamingStats(), X, len(X))
    np.testing.assert_allclose(st.mean, np.nanmean(X, axis=0), rtol=1e-13)
    np.testing.assert_allclose(st.variance(), np.nanvar(X, axis=0), rtol=1e-12, atol=1e-24)


def test_module_level_update_stats():
    X = data_with_nans()
    state = None
    for chunk in np.array_split(X, 5):
        state = update_stats(chunk, state)
    assert isinstance(state, StreamingStats)
    np.testing.assert_allclose(state.mean, np.nanmean(X, axis=0))
    with pytest.raises(TypeError):
        update_stats(X, state, bins=8)              # options only for a new state
    assert update_stats(X, bins=8).hist_ is not None


def test_update_stats_is_the_same_as_update_and_chainable():
    st = StreamingStats()
    assert st.update_stats(np.ones((2, 2))) is st
    assert StreamingStats.update is StreamingStats.update_stats
    assert s.StreamingStats is s.Welford


# ---------------- NaN column by column ----------------

def test_all_nan_column_in_a_chunk_does_not_move_it():
    st = StreamingStats().update_stats(np.array([[1.0, 10.0], [3.0, 20.0]]))
    st.update_stats(np.array([[np.nan, 30.0], [np.nan, 40.0]]))
    np.testing.assert_array_equal(st.count, [2, 4])
    np.testing.assert_allclose(st.mean, [2.0, 25.0])


def test_column_with_no_values_yet_is_nan():
    st = StreamingStats().update_stats(np.array([[1.0, np.nan], [2.0, np.nan]]))
    assert np.isnan(st.mean[1]) and np.isnan(st.variance()[1]) and np.isnan(st.min[1])
    assert st.count[1] == 0


def test_layout_and_shape_are_locked_by_first_update():
    st = StreamingStats().update_stats(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="columns"):
        st.update_stats(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="2-D"):
        st.update_stats(np.zeros(3))
    assert st.count.tolist() == [3, 3]              # unchanged


def test_empty_chunk_is_a_noop():
    st = StreamingStats().update_stats(np.zeros((0, 3)))
    assert st.count == 0 and np.isnan(st.mean)
    st.update_stats(np.ones((2, 3))).update_stats(np.zeros((0, 3)))
    assert st.count.tolist() == [2, 2, 2]


def test_reset_forgets_everything():
    st = StreamingStats(bins=8).update_stats(np.ones((4, 2)))
    st.reset()
    assert st.count == 0 and st.n_features_in_ is None
    st.update_stats(np.zeros(3))                    # a different layout is fine now
    assert st.count == 3


# ---------------- numerical stability ----------------

def test_welford_survives_catastrophic_cancellation():
    # 1e9 + N(0, 1): x**2 is ~1e18, so E[x^2] - E[x]^2 cancels away every
    # significant digit while the merge of two-pass chunk sums does not
    rng = np.random.default_rng(0)
    x = 1e9 + rng.normal(size=50_000)
    exact = np.var(x.astype(np.longdouble))
    naive = np.mean(x * x) - np.mean(x) ** 2
    assert abs(naive - exact) / exact > 0.5             # naive formula is garbage
    for k in (1, 7, 1000):
        st = stream(StreamingStats(), x, k)
        assert abs(st.variance() - exact) / exact < 1e-12


def test_huge_and_tiny_values_do_not_overflow():
    for scale in (1e200, 1e-200):
        st = stream(StreamingStats(), np.array([1.0, 3.0, 5.0, 7.0]) * scale, 2, "update")
        assert st.std() / scale == pytest.approx(np.sqrt(5.0))
        assert st.mean / scale == pytest.approx(4.0)


def test_zero_variance_stream():
    st = stream(StreamingStats(), np.full((50, 2), 0.1), 5)
    np.testing.assert_array_equal(st.variance(), 0.0)


# ---------------- histogram and quantiles ----------------

def inverted_cdf(X, q):
    return np.nanquantile(X, q, axis=0, method="inverted_cdf")


@pytest.mark.parametrize("k", CHUNKINGS)
def test_quantiles_within_one_bin_width(k):
    rng = np.random.default_rng(k)
    X = np.column_stack([rng.normal(0, 1, 2000), rng.exponential(5, 2000),
                         rng.integers(0, 10, 2000)]).astype(float)
    X[::9, 0] = np.nan
    h = stream(StreamingHistogram(bins=64), X, k, "update")
    q = np.array([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    err = np.abs(h.quantile(q) - inverted_cdf(X, q))
    assert (err <= h.bin_width_ * (1 + 1e-9)).all()


def test_quantile_zero_and_one_are_exact_min_and_max():
    X = np.random.default_rng(2).normal(size=(500, 2))
    h = stream(StreamingHistogram(bins=16), X, 4, "update")
    np.testing.assert_array_equal(h.quantile(0.0), X.min(axis=0))
    np.testing.assert_array_equal(h.quantile(1.0), X.max(axis=0))


def test_histogram_counts_every_value_exactly_once():
    X = data_with_nans()
    h = stream(StreamingHistogram(bins=32), X, 17, "update")
    np.testing.assert_array_equal(h.count, (~np.isnan(X)).sum(axis=0))
    counts, edges = h.histogram()
    assert counts.shape == (4, 32) and edges.shape == (4, 33)
    # each value lies inside its column's range
    lo, hi = edges[:, 0], edges[:, -1]
    assert (np.nanmin(X, axis=0) >= lo).all() and (np.nanmax(X, axis=0) <= hi).all()


def test_range_doubles_and_merges_bins_in_pairs():
    h = StreamingHistogram(bins=4).update(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    counts, edges = h.histogram()
    np.testing.assert_array_equal(edges, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(counts, [1, 1, 1, 2])
    h.update(np.array([7.5]))                   # beyond 4: range [0, 8], bins of 2
    counts, edges = h.histogram()
    np.testing.assert_array_equal(edges, [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(counts, [2, 3, 0, 1])
    h.update(np.array([-3.0]))                  # below 0: range [-8, 8], bins of 4
    counts, edges = h.histogram()
    np.testing.assert_array_equal(edges, [-8, -4, 0, 4, 8])
    # pairs merge as a whole: the 4.0 sat on the closed right edge of the
    # old last bin, so it stays with [0, 4] (still within one bin width)
    np.testing.assert_array_equal(counts, [0, 1, 5, 1])


def test_zero_variance_first_chunk_gets_unit_range():
    h = StreamingHistogram(bins=8).update(np.full(10, 5.0))
    counts, edges = h.histogram()
    assert edges[0] == 4.5 and edges[-1] == 5.5 and np.isfinite(edges).all()
    assert counts.sum() == 10
    assert h.quantile(0.5) == 5.0
    h.update(np.array([5.0, 9.0]))              # later spread is still handled
    assert h.quantile(1.0) == 9.0 and h.count == 12


def test_all_nan_column_has_nan_quantile_until_data_arrives():
    h = StreamingHistogram(bins=8).update(np.array([[1.0, np.nan], [2.0, np.nan]]))
    assert np.isnan(h.quantile(0.5)[1])
    h.update(np.array([[3.0, 7.0]]))
    assert h.quantile(0.5)[1] == 7.0


def test_sliding_window_keeps_only_last_chunks():
    rng = np.random.default_rng(5)
    chunks = [rng.normal(loc=10 * i, size=(100, 2)) for i in range(8)]
    h = StreamingHistogram(bins=32, window=3)
    for c in chunks:
        h.update(c)
    recent = np.vstack(chunks[-3:])
    np.testing.assert_array_equal(h.count, [300, 300])
    np.testing.assert_array_equal(h.quantile(0.0), recent.min(axis=0))
    np.testing.assert_array_equal(h.quantile(1.0), recent.max(axis=0))
    err = np.abs(h.quantile(0.5) - inverted_cdf(recent, 0.5))
    assert (err <= h.bin_width_).all()


def test_window_eviction_survives_range_growth():
    h = StreamingHistogram(bins=8, window=2)
    h.update(np.array([0.0, 1.0]))
    h.update(np.array([100.0]))                 # forces several doublings
    h.update(np.array([50.0]))                  # evicts the first chunk
    assert h.count == 2
    counts, _ = h.histogram()
    assert counts.min() >= 0 and counts.sum() == 2


def test_streaming_stats_quantiles_need_bins():
    st = StreamingStats().update_stats(np.arange(10.0))
    with pytest.raises(ValueError, match="bins"):
        st.quantile(0.5)
    st = StreamingStats(bins=16).update_stats(np.arange(10.0))
    assert abs(st.quantile(0.5) - 4.0) <= st.hist_.bin_width_


@pytest.mark.parametrize("bins", [0, 3, 2.0, True])
def test_bad_bins(bins):
    with pytest.raises(ValueError):
        StreamingHistogram(bins=bins)


def test_bad_window_and_quantile():
    with pytest.raises(ValueError):
        StreamingHistogram(window=0)
    h = StreamingHistogram()
    with pytest.raises(ValueError):
        h.quantile(0.5)                         # nothing seen yet
    h.update([1.0, 2.0])
    for q in (-0.1, 1.5, [[0.5]]):
        with pytest.raises(ValueError):
            h.quantile(q)


def test_infinity_rejected_and_state_unchanged():
    st = StreamingStats(bins=8).update_stats(np.ones((2, 2)))
    with pytest.raises(ValueError):
        st.update_stats(np.array([[1.0, np.inf]]))
    assert st.count.tolist() == [2, 2] and st.hist_.count.tolist() == [2, 2]


def test_one_dimensional_stream_gives_plain_numbers():
    h = StreamingHistogram(bins=16).update(np.arange(10.0))
    assert isinstance(h.quantile(0.5), float) and isinstance(h.count, int)
    assert isinstance(h.bin_width_, float) and h.quantile([0.1, 0.9]).shape == (2,)
    st = StreamingStats().update_stats(np.arange(10.0))
    assert isinstance(st.mean, float) and isinstance(st.std(), float)
