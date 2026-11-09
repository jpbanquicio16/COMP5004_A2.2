"""
Tests for numcompute.stats.
Run from the project root: py -m unittest discover -s tests -p "test_stats.py" -v
"""
import unittest

import numpy as np

from numcompute import stats as s

REDUCTIONS = ('mean', 'median', 'var', 'std', 'min', 'max')


class TestReductions(unittest.TestCase):
    def test_known_values(self):
        # [2, 4, 6]: population variance = (4 + 0 + 4) / 3, sample = 8 / 2
        x = [2, 4, 6]
        cases = [(s.mean, 4), (s.median, 4), (s.var, 8 / 3),
                 (s.std, np.sqrt(8 / 3)), (s.min, 2), (s.max, 6)]
        for fn, expected in cases:
            with self.subTest(function=fn.__name__):
                self.assertAlmostEqual(float(fn(x)), expected)
        self.assertAlmostEqual(float(s.var(x, ddof=1)), 4)
        self.assertAlmostEqual(float(s.std(x, ddof=1)), 2)
        self.assertIs(s.variance, s.var)

    def test_axes_and_keepdims_match_numpy(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        for name in REDUCTIONS:
            for axis in (None, 0, -1, (0, 2)):
                for keepdims in (False, True):
                    with self.subTest(name=name, axis=axis, keepdims=keepdims):
                        np.testing.assert_allclose(
                            getattr(s, name)(x, axis, keepdims=keepdims),
                            getattr(np, name)(x, axis=axis, keepdims=keepdims))

    def test_constant_and_single_value(self):
        for x in ([7], [7] * 10):
            self.assertEqual(s.mean(x), 7)
            self.assertEqual(s.median(x), 7)
            self.assertEqual(s.var(x), 0)
            self.assertEqual(s.std(x), 0)

    def test_non_contiguous_input_not_modified(self):
        x = np.arange(100.0).reshape(10, 10)[::-2, ::3]
        original = x.copy()
        for name in REDUCTIONS:
            np.testing.assert_allclose(getattr(s, name)(x, axis=0),
                                       getattr(np, name)(original, axis=0))
        np.testing.assert_array_equal(x, original)

    def test_random_data_with_nans_matches_numpy(self):
        x = np.random.default_rng(4).normal(size=(20, 8))
        x[::3, 2] = np.nan
        for name in REDUCTIONS:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    getattr(s, name)(x, 0, nan_policy='omit'),
                    getattr(np, 'nan' + name)(x, axis=0), rtol=1e-12)

    def test_negative_and_decimal_values(self):
        x = [-1.5, 0.5, 2.5, 4.5]
        self.assertAlmostEqual(float(s.mean(x)), 1.5)
        self.assertAlmostEqual(float(s.median(x)), 1.5)
        self.assertEqual(s.min(x), -1.5)
        self.assertEqual(s.max(x), 4.5)


class TestNaNHandling(unittest.TestCase):
    def test_three_policies(self):
        for name in REDUCTIONS:
            fn = getattr(s, name)
            with self.subTest(name=name):
                self.assertTrue(np.isnan(fn([1, np.nan, 3])))      # default
                self.assertAlmostEqual(float(fn([1, np.nan, 3], nan_policy='omit')),
                                       float(getattr(np, name)([1.0, 3.0])))
                with self.assertRaises(ValueError):
                    fn([1, np.nan], nan_policy='raise')

    def test_all_nan_column_gives_nan(self):
        x = [[np.nan, 1], [np.nan, 3]]
        for name in REDUCTIONS:
            out = getattr(s, name)(x, axis=0, nan_policy='omit')
            self.assertTrue(np.isnan(out[0]))
            self.assertTrue(np.isfinite(out[1]))


class TestErrors(unittest.TestCase):
    def test_ddof(self):
        for fn in (s.var, s.std):
            # not enough values -> NaN
            self.assertTrue(np.isnan(fn([1], ddof=1)))
            # negative -> ValueError, float/bool -> TypeError
            for value, error in [(-1, ValueError), (0.5, TypeError), (True, TypeError)]:
                with self.assertRaises(error):
                    fn([1, 2], ddof=value)

    def test_empty_input(self):
        for name in REDUCTIONS:
            with self.assertRaises(ValueError):
                getattr(s, name)([])

    def test_non_numeric_or_infinite_input(self):
        cases = [(['hello'], TypeError), ([1j], TypeError), ([np.inf], ValueError)]
        for name in REDUCTIONS:
            for values, error in cases:
                with self.assertRaises(error):
                    getattr(s, name)(values)

    def test_bad_axis_and_policy(self):
        with self.assertRaises(ValueError):
            s.mean([1], nan_policy='ignore')
        with self.assertRaises((ValueError, IndexError)):
            s.mean([1, 2], axis=4)


class TestQuantile(unittest.TestCase):
    def test_interpolation_methods(self):
        # the median of [10, 20, 30, 40] sits between 20 and 30
        for method, expected in [('linear', 25), ('lower', 20), ('higher', 30), ('midpoint', 25)]:
            with self.subTest(method=method):
                self.assertEqual(s.quantile([10, 20, 30, 40], 0.5, interpolation=method), expected)

    def test_shapes_and_axes_match_numpy(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        qs = [0, 0.25, 0.5, 1]
        for axis in (None, 0, -1, (0, 2)):
            for keepdims in (False, True):
                np.testing.assert_allclose(
                    s.quantile(x, qs, axis, keepdims=keepdims),
                    np.quantile(x, qs, axis=axis, keepdims=keepdims))

    def test_nan_handling(self):
        self.assertEqual(s.quantile([1, np.nan, 3], 0.5), 2)  # omit is the default
        self.assertTrue(np.isnan(s.quantile([np.nan], 0.5)))
        self.assertTrue(np.isnan(s.quantile([1, np.nan], 0.5, nan_policy='propagate')))

    def test_invalid_input(self):
        # q is a fraction (not 0-100), finite, and at most 1-D
        for q in (-0.1, 1.1, np.nan, np.inf, [], [[0.5]]):
            with self.assertRaises(ValueError):
                s.quantile([1, 2], q)
        with self.assertRaises(ValueError):
            s.quantile([1], 0.5, interpolation='bad')
        with self.assertRaises(ValueError):
            s.quantile([], 0.5)


class TestHistogram(unittest.TestCase):
    def test_explicit_edges(self):
        # bins are [0, 2) and [2, 4]; the last bin includes 4
        counts, edges = s.histogram([0, 1, 2, 3, 4], bins=[0, 2, 4])
        np.testing.assert_array_equal(counts, [2, 3])
        np.testing.assert_array_equal(edges, [0, 2, 4])

    def test_matches_numpy(self):
        x = np.random.default_rng(9).normal(size=100)
        for ours, theirs in zip(s.histogram(x, bins=7, range=(-2, 2)),
                                np.histogram(x, bins=7, range=(-2, 2))):
            np.testing.assert_allclose(ours, theirs)

    def test_density_integrates_to_one(self):
        counts, edges = s.histogram([0, 1, 2, 3], bins=[0, 1, 4], density=True)
        self.assertAlmostEqual(float(np.sum(counts * np.diff(edges))), 1)

    def test_nan_handling(self):
        np.testing.assert_array_equal(s.histogram([1, np.nan, 3], bins=[0, 2, 4])[0], [1, 1])
        with self.assertRaises(ValueError):   # all NaN and no edges to fall back on
            s.histogram([np.nan])
        with self.assertRaises(ValueError):
            s.histogram([1, np.nan], nan_policy='raise')

    def test_invalid_input(self):
        for bins in (0, True, 2.5, [1, 1], [2, 1], [0, np.inf]):
            with self.assertRaises(ValueError):
                s.histogram([1, 2], bins=bins)
        with self.assertRaises(ValueError):    # low >= high
            s.histogram([1], range=(2, 1))
        with self.assertRaises(ValueError):    # nothing falls in the bins
            s.histogram([5], bins=[0, 1], density=True)
        with self.assertRaises(ValueError):
            s.histogram([])


class TestWelford(unittest.TestCase):
    def test_one_value_at_a_time(self):
        w = s.Welford()
        for value in [2, 4, 6]:
            self.assertIs(w.update(value), w)   # chainable
        self.assertEqual(w.count, 3)
        self.assertAlmostEqual(w.mean, 4)
        self.assertAlmostEqual(w.variance(), 8 / 3)
        self.assertAlmostEqual(w.std(ddof=1), 2)

    def test_batches_match_numpy(self):
        x = np.random.default_rng(42).normal(100, 3, size=103)
        w = s.Welford()
        for chunk in np.array_split(x, 7):
            w.update(chunk)
        self.assertAlmostEqual(w.mean, np.mean(x))
        self.assertAlmostEqual(w.variance(), np.var(x))

    def test_empty_and_nan(self):
        w = s.Welford().update([]).update([np.nan])
        self.assertEqual(w.count, 0)
        self.assertTrue(np.isnan(w.mean))
        self.assertTrue(np.isnan(w.variance()))
        w.update([1, np.nan, 3])     # NaN skipped
        self.assertEqual((w.count, w.mean), (2, 2))

    def test_rejected_input_and_options(self):
        w = s.Welford('raise').update([1, 2])
        for bad in ([np.nan], [np.inf], [[1]], ['x']):
            with self.assertRaises((ValueError, TypeError)):
                w.update(bad)
            self.assertEqual((w.count, w.mean), (2, 1.5))   # state unchanged
        with self.assertRaises(ValueError):
            s.Welford('bad')
        with self.assertRaises(ValueError):
            w.variance(-1)
        self.assertIs(s.StreamingStats, s.Welford)


if __name__ == '__main__':
    unittest.main()