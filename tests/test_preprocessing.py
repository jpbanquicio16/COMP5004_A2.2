"""Unit tests for numcompute.preprocessing."""

import unittest

import numpy as np

from numcompute.preprocessing import (
    Imputer,
    MinMaxScaler,
    NotFittedError,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _example():
    return np.array([[1.0, 10.0, 5.0],
                     [2.0, 20.0, 5.0],
                     [3.0, 40.0, 5.0],
                     [4.0, 50.0, 5.0]])


class TestStandardScaler(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        Z = StandardScaler().fit_transform(_example()[:, :2])
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)

    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        X = rng.normal(5, 3, size=(50, 4))
        Z = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(Z, (X - X.mean(0)) / X.std(0))

    def test_shape_preserved_and_float_output(self):
        X = np.arange(12).reshape(4, 3)  # integers in
        Z = StandardScaler().fit_transform(X)
        self.assertEqual(Z.shape, X.shape)
        self.assertEqual(Z.dtype, np.float64)

    def test_fit_returns_self_and_fit_transform_equals_two_steps(self):
        s = StandardScaler()
        self.assertIs(s.fit(_example()), s)
        np.testing.assert_array_equal(
            StandardScaler().fit_transform(_example()),
            StandardScaler().fit(_example()).transform(_example()),
        )

    def test_constant_column_maps_to_zero(self):
        Z = StandardScaler().fit_transform(_example())
        np.testing.assert_array_equal(Z[:, 2], 0.0)
        self.assertTrue(np.isfinite(Z).all())

    def test_constant_column_with_rounding_error(self):
        Z = StandardScaler().fit_transform(np.full((3, 1), 0.1))
        np.testing.assert_array_equal(Z, 0.0)

    def test_nan_ignored_in_fit_and_kept_in_transform(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]])
        s = StandardScaler().fit(X)
        np.testing.assert_allclose(s.mean_, [2.0, 4.0])
        Z = s.transform(X)
        self.assertTrue(np.isnan(Z[1, 0]))
        self.assertEqual(np.isnan(Z).sum(), 1)

    def test_all_nan_column_raises(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan]])
        with self.assertRaisesRegex(ValueError, r"\[1\]"):
            StandardScaler().fit(X)

    def test_transform_before_fit(self):
        with self.assertRaises(NotFittedError):
            StandardScaler().transform(_example())
        # NotFittedError is a ValueError, so the group convention holds
        with self.assertRaises(ValueError):
            StandardScaler().transform(_example())

    def test_input_not_modified(self):
        X = _example()
        before = X.copy()
        StandardScaler().fit_transform(X)
        np.testing.assert_array_equal(X, before)

    def test_wrong_number_of_features(self):
        s = StandardScaler().fit(_example())
        with self.assertRaisesRegex(ValueError, "3 features"):
            s.transform(np.zeros((2, 5)))

    def test_1d_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            StandardScaler().fit(np.array([1.0, 2.0, 3.0]))

    def test_non_numeric_input_rejected(self):
        with self.assertRaises(TypeError):
            StandardScaler().fit(np.array([["a", "b"]]))

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            StandardScaler().fit(np.array([[1.0], [np.inf]]))

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            StandardScaler().fit(np.empty((0, 3)))

    def test_very_large_and_very_small_values_do_not_overflow(self):
        expected = np.array([-1.0, 0.0, 1.0]) * np.sqrt(1.5)
        for scale in (1e200, 1e-200):
            X = np.array([[1.0], [3.0], [5.0]]) * scale
            Z = StandardScaler().fit_transform(X)
            np.testing.assert_allclose(Z.ravel(), expected, atol=1e-12)


class TestMinMaxScaler(unittest.TestCase):
    def test_default_range(self):
        Z = MinMaxScaler().fit_transform(_example()[:, :2])
        np.testing.assert_allclose(Z.min(axis=0), 0)
        np.testing.assert_allclose(Z.max(axis=0), 1)

    def test_custom_range(self):
        Z = MinMaxScaler((-1, 1)).fit_transform(_example()[:, :2])
        np.testing.assert_allclose(Z.min(axis=0), -1)
        np.testing.assert_allclose(Z.max(axis=0), 1)

    def test_known_values(self):
        Z = MinMaxScaler().fit_transform(np.array([[0.0], [5.0], [10.0]]))
        np.testing.assert_allclose(Z.ravel(), [0.0, 0.5, 1.0])

    def test_constant_column_maps_to_lower_bound(self):
        Z = MinMaxScaler((2, 4)).fit_transform(_example())
        np.testing.assert_array_equal(Z[:, 2], 2.0)

    def test_values_outside_fitted_range_extrapolate(self):
        m = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
        np.testing.assert_allclose(m.transform(np.array([[-10.0], [20.0]])).ravel(), [-1, 2])

    def test_nan_ignored_and_preserved(self):
        X = np.array([[0.0], [np.nan], [10.0]])
        Z = MinMaxScaler().fit_transform(X)
        self.assertTrue(np.isnan(Z[1, 0]))
        np.testing.assert_allclose(Z[[0, 2], 0], [0, 1])

    def test_invalid_feature_range(self):
        for bad in [(1, 1), (2, 1), (0, np.inf)]:
            with self.assertRaises(ValueError):
                MinMaxScaler(bad)
        for bad in [5, ("a", "b"), (0, 1, 2)]:
            with self.assertRaises(TypeError):
                MinMaxScaler(bad)

    def test_transform_before_fit(self):
        with self.assertRaises(NotFittedError):
            MinMaxScaler().transform(_example())

    def test_wrong_number_of_features(self):
        m = MinMaxScaler().fit(_example())
        with self.assertRaises(ValueError):
            m.transform(np.zeros((2, 2)))

    def test_all_nan_column_raises(self):
        with self.assertRaises(ValueError):
            MinMaxScaler().fit(np.array([[np.nan], [np.nan]]))

    def test_input_not_modified(self):
        X = _example()
        before = X.copy()
        MinMaxScaler().fit_transform(X)
        np.testing.assert_array_equal(X, before)

    def test_huge_values_stay_in_range(self):
        X = np.array([[-1e300], [0.0], [1e300]])
        Z = MinMaxScaler().fit_transform(X)
        self.assertTrue(np.isfinite(Z).all())


class TestSimpleImputer(unittest.TestCase):
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 8.0]])

    def test_constant_default_zero(self):
        out = SimpleImputer().fit_transform(self.X)
        np.testing.assert_array_equal(out, [[1, 0], [0, 4], [3, 8]])

    def test_constant_custom(self):
        out = SimpleImputer(fill_value=-9).fit_transform(self.X)
        self.assertEqual(out[0, 1], -9)

    def test_mean_strategy(self):
        np.testing.assert_allclose(
            SimpleImputer("mean").fit_transform(self.X), [[1, 6], [2, 4], [3, 8]]
        )

    def test_learns_from_fit_data_only(self):
        imp = SimpleImputer("mean").fit(self.X)
        out = imp.transform(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(out, [[2, 6]])

    def test_no_nan_left_and_input_unchanged(self):
        X = self.X.copy()
        out = SimpleImputer().fit_transform(X)
        self.assertFalse(np.isnan(out).any())
        self.assertTrue(np.isnan(X).any())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            SimpleImputer("mode")
        with self.assertRaises(ValueError):
            SimpleImputer(fill_value=np.nan)
        with self.assertRaises(TypeError):
            SimpleImputer(fill_value="x")

    def test_all_nan_column_with_mean_raises(self):
        with self.assertRaises(ValueError):
            SimpleImputer("mean").fit(np.array([[np.nan], [np.nan]]))

    def test_all_nan_column_with_constant_is_fine(self):
        out = SimpleImputer().fit_transform(np.array([[np.nan], [np.nan]]))
        np.testing.assert_array_equal(out, [[0.0], [0.0]])

    def test_transform_before_fit(self):
        with self.assertRaises(NotFittedError):
            SimpleImputer().transform(self.X)

    def test_wrong_number_of_features(self):
        imp = SimpleImputer().fit(self.X)
        with self.assertRaises(ValueError):
            imp.transform(np.zeros((2, 3)))


class TestOneHotEncoder(unittest.TestCase):
    def test_strings_basic(self):
        X = np.array([["red"], ["blue"], ["red"]])
        enc = OneHotEncoder().fit(X)
        np.testing.assert_array_equal(enc.categories_[0], ["blue", "red"])
        np.testing.assert_array_equal(
            enc.transform(X), [[0, 1], [1, 0], [0, 1]]
        )

    def test_multiple_columns_shape_and_order(self):
        X = np.array([["red", "S"], ["blue", "M"], ["red", "L"]])
        enc = OneHotEncoder().fit(X)
        out = enc.transform(X)
        self.assertEqual(out.shape, (3, 5))
        self.assertEqual(enc.n_features_out_, 5)
        np.testing.assert_array_equal(
            out,
            [[0, 1, 0, 0, 1],
             [1, 0, 0, 1, 0],
             [0, 1, 1, 0, 0]],
        )

    def test_each_row_sums_to_number_of_features(self):
        rng = np.random.default_rng(1)
        X = rng.integers(0, 4, size=(40, 3))
        out = OneHotEncoder().fit_transform(X)
        np.testing.assert_array_equal(out.sum(axis=1), 3)
        self.assertTrue(set(np.unique(out)) <= {0.0, 1.0})

    def test_matches_naive_loop_reference(self):
        rng = np.random.default_rng(2)
        X = rng.integers(0, 5, size=(30, 4))
        out = OneHotEncoder().fit_transform(X)
        blocks = []
        for j in range(X.shape[1]):
            cats = np.unique(X[:, j])
            blocks.append((X[:, [j]] == cats[None, :]).astype(float))
        np.testing.assert_array_equal(out, np.hstack(blocks))

    def test_columns_with_different_category_counts(self):
        X = np.array([[0, 5], [1, 5], [2, 5]])
        enc = OneHotEncoder().fit(X)
        self.assertEqual([len(c) for c in enc.categories_], [3, 1])
        self.assertEqual(enc.transform(X).shape, (3, 4))

    def test_same_value_in_two_columns_is_independent(self):
        X = np.array([[1, 2], [2, 1]])
        out = OneHotEncoder().fit_transform(X)
        np.testing.assert_array_equal(out, [[1, 0, 0, 1], [0, 1, 1, 0]])

    def test_unknown_raises_by_default(self):
        enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
        with self.assertRaisesRegex(ValueError, "unseen"):
            enc.transform(np.array([["c"]]))

    def test_value_seen_only_in_other_feature_is_unknown(self):
        enc = OneHotEncoder().fit(np.array([["a", "x"], ["b", "y"]]))
        with self.assertRaises(ValueError):
            enc.transform(np.array([["x", "y"]]))  # "x" never occurred in feature 0

    def test_unknown_ignore_gives_zeros(self):
        enc = OneHotEncoder("ignore").fit(np.array([["a"], ["b"]]))
        out = enc.transform(np.array([["c"], ["a"]]))
        np.testing.assert_array_equal(out, [[0, 0], [1, 0]])

    def test_numeric_float_input(self):
        out = OneHotEncoder().fit_transform(np.array([[1.5], [2.5], [1.5]]))
        np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0]])

    def test_transform_empty_array(self):
        enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
        self.assertEqual(enc.transform(np.empty((0, 1), dtype=str)).shape, (0, 2))

    def test_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            OneHotEncoder().fit(np.array([[1.0], [np.nan]]))

    def test_object_dtype_rejected(self):
        with self.assertRaises(TypeError):
            OneHotEncoder().fit(np.array([["a"], [None]], dtype=object))

    def test_type_family_mismatch(self):
        enc = OneHotEncoder().fit(np.array([[1], [2]]))
        with self.assertRaises(TypeError):
            enc.transform(np.array([["1"]]))

    def test_1d_rejected(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            OneHotEncoder().fit(np.array(["a", "b"]))

    def test_empty_fit_rejected(self):
        with self.assertRaises(ValueError):
            OneHotEncoder().fit(np.empty((0, 2)))

    def test_wrong_number_of_features(self):
        enc = OneHotEncoder().fit(np.array([["a", "b"]]))
        with self.assertRaises(ValueError):
            enc.transform(np.array([["a"]]))

    def test_transform_before_fit(self):
        with self.assertRaises(NotFittedError):
            OneHotEncoder().transform(np.array([["a"]]))

    def test_invalid_handle_unknown(self):
        with self.assertRaises(ValueError):
            OneHotEncoder("skip")

    def test_input_not_modified(self):
        X = np.array([["a"], ["b"]])
        before = X.copy()
        OneHotEncoder().fit_transform(X)
        np.testing.assert_array_equal(X, before)


class TestEdgeCases(unittest.TestCase):
    """Edge cases listed in the assignment: empty arrays, all-equal values,
    duplicates, NaNs and non-contiguous strides."""

    def test_non_contiguous_views_give_same_result_as_copies(self):
        base = np.arange(40, dtype=float).reshape(8, 5) ** 1.5
        views = {
            "column slice": base[:, ::2],
            "row slice": base[::2],
            "transpose": base.T[:, :6],
            "fortran order": np.asfortranarray(base),
        }
        for name, view in views.items():
            for cls in (StandardScaler, MinMaxScaler):
                with self.subTest(view=name, cls=cls.__name__):
                    np.testing.assert_allclose(
                        cls().fit_transform(view),
                        cls().fit_transform(np.ascontiguousarray(view)),
                    )

    def test_non_contiguous_strings_for_onehot(self):
        base = np.array([["a", "x", "b"], ["c", "y", "d"], ["a", "x", "b"]])
        view = base[:, ::2]
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(
            OneHotEncoder().fit_transform(view),
            OneHotEncoder().fit_transform(np.ascontiguousarray(view)),
        )

    def test_non_contiguous_imputer(self):
        base = np.array([[1.0, 9.0, np.nan], [np.nan, 9.0, 3.0]])
        out = SimpleImputer(fill_value=-1).fit_transform(base[:, ::2])
        np.testing.assert_array_equal(out, [[1, -1], [-1, 3]])

    def test_transform_of_zero_rows(self):
        X = _example()
        for obj in (StandardScaler(), MinMaxScaler(), SimpleImputer()):
            obj.fit(X)
            self.assertEqual(obj.transform(np.empty((0, 3))).shape, (0, 3))

    def test_all_equal_everything(self):
        X = np.full((5, 3), 7.0)
        np.testing.assert_array_equal(StandardScaler().fit_transform(X), 0.0)
        np.testing.assert_array_equal(MinMaxScaler().fit_transform(X), 0.0)

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(StandardScaler().fit_transform(X), 0.0)
        np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), [[1, 1]])

    def test_single_feature(self):
        Z = StandardScaler().fit_transform(np.array([[1.0], [2.0], [3.0]]))
        self.assertEqual(Z.shape, (3, 1))

    def test_python_lists_are_accepted(self):
        out = StandardScaler().fit_transform([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(out.shape, (3, 2))
        out = OneHotEncoder().fit_transform([["a"], ["b"]])
        self.assertEqual(out.shape, (2, 2))

    def test_duplicate_rows_encode_identically(self):
        X = np.array([["a", "x"]] * 4)
        out = OneHotEncoder().fit_transform(X)
        self.assertTrue((out == out[0]).all())

    def test_nan_in_every_row_of_one_column(self):
        X = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, np.nan]])
        Z = StandardScaler().fit_transform(X)  # column 1 has one valid value
        np.testing.assert_array_equal(Z[1, 1], 0.0)
        self.assertTrue(np.isnan(Z[[0, 2], 1]).all())

    def test_imputer_alias_matches_brief_name(self):
        self.assertIs(Imputer, SimpleImputer)


class TestRemainingBranches(unittest.TestCase):
    """Small error paths that were left"""

    def test_zero_features_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            StandardScaler().fit(np.empty((3, 0)))
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            OneHotEncoder().fit(np.empty((3, 0)))

    def test_base_class_methods_must_be_overridden(self):
        from numcompute.preprocessing import _BaseTransformer
        base = _BaseTransformer()
        with self.assertRaises(NotImplementedError):
            base.fit(np.zeros((2, 2)))
        with self.assertRaises(NotImplementedError):
            base.transform(np.zeros((2, 2)))


class TestChaining(unittest.TestCase):
    """The assignment's pipeline example, chained by hand (no Pipeline yet)."""

    def test_scale_then_encode(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 10.0]])
        scaled = StandardScaler().fit_transform(X)
        out = OneHotEncoder().fit_transform(scaled)
        self.assertEqual(out.shape[0], 3)
        np.testing.assert_array_equal(out.sum(axis=1), 2)

    def test_impute_then_scale(self):
        X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 8.0]])
        out = StandardScaler().fit_transform(SimpleImputer("mean").fit_transform(X))
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
