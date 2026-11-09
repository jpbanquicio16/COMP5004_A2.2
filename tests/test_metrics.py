"""
Tests for metrics.
Run from the project root:  py -m unittest discover -s .\tests -p "test_metrics.py" -v
"""
import unittest

import numpy as np

from numcompute import metrics as m

CLASSIFIERS = (m.accuracy, m.precision, m.recall, m.f1)

class TestClassification(unittest.TestCase):
    def setUp(self):
        # TP=2, FN=1, FP=1, TN=2
        self.y_true = [1, 1, 1, 0, 0, 0]
        self.y_pred = [1, 1, 0, 1, 0, 0]

    def test_binary_known_values(self):
        for fn in CLASSIFIERS:
            with self.subTest(metric=fn.__name__):
                self.assertAlmostEqual(fn(self.y_true, self.y_pred), 2 / 3)

    def test_precision_and_recall_differ(self):
        # TP=1, FP=1, FN=2 -> precision 1/2, recall 1/3, f1 = 2 / 5
        y_true, y_pred = [1, 1, 1, 0, 0], [1, 0, 0, 1, 0]
        self.assertAlmostEqual(m.precision(y_true, y_pred), 0.5)
        self.assertAlmostEqual(m.recall(y_true, y_pred), 1 / 3)
        self.assertAlmostEqual(m.f1(y_true, y_pred), 0.4)

    def test_confusion_matrix(self):
        np.testing.assert_array_equal(
            m.confusion_matrix(self.y_true, self.y_pred), [[2, 1], [1, 2]])

    def test_confusion_matrix_custom_label_order(self):
        # label 2 never appears but still gets a row and column
        result = m.confusion_matrix([0, 0, 1], [0, 1, 1], labels=[1, 0, 2])
        np.testing.assert_array_equal(result, [[1, 0, 0], [1, 1, 0], [0, 0, 0]])

    def test_string_and_boolean_labels(self):
        y_true, y_pred = ['yes', 'yes', 'no'], ['yes', 'no', 'no']
        self.assertEqual(m.precision(y_true, y_pred, pos_label='yes'), 1)
        self.assertEqual(m.recall(y_true, y_pred, pos_label='yes'), 0.5)
        self.assertEqual(m.f1([True, False], [True, False]), 1)

    def test_perfect_and_completely_wrong(self):
        for fn in CLASSIFIERS:
            self.assertEqual(fn([0, 1], [0, 1]), 1)
            self.assertEqual(fn([0, 1], [1, 0]), 0)


class TestAveraging(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 0, 1, 1, 2, 2]
        self.y_pred = [0, 1, 1, 1, 0, 2]

    def test_per_class_scores(self):
        np.testing.assert_allclose(
            m.precision(self.y_true, self.y_pred, average=None), [0.5, 2 / 3, 1])
        np.testing.assert_allclose(
            m.recall(self.y_true, self.y_pred, average=None), [0.5, 1, 0.5])
        np.testing.assert_allclose(
            m.f1(self.y_true, self.y_pred, average=None), [0.5, 0.8, 2 / 3])

    def test_macro_average(self):
        expected = (0.5 + 2 / 3 + 1) / 3      # plain mean of per-class precision
        self.assertAlmostEqual(
            m.precision(self.y_true, self.y_pred, average='macro'), expected)

    def test_weighted_average(self):
        y_true, y_pred = [0, 0, 0, 1], [0, 0, 1, 1]
        self.assertAlmostEqual(m.precision(y_true, y_pred, average='weighted'), 0.875)
        self.assertAlmostEqual(m.recall(y_true, y_pred, average='weighted'), 0.75)

    def test_micro_average_equals_accuracy(self):
        rng = np.random.default_rng(42)
        y_true, y_pred = rng.integers(0, 5, (2, 100))
        for fn in (m.precision, m.recall, m.f1):
            self.assertAlmostEqual(fn(y_true, y_pred, average='micro'),
                                   m.accuracy(y_true, y_pred))


class TestClassificationErrors(unittest.TestCase):
    def test_zero_division(self):
        # no positives anywhere -> every denominator is 0
        for fn in (m.precision, m.recall, m.f1):
            self.assertEqual(fn([0, 0], [0, 0]), 0)
            self.assertEqual(fn([0, 0], [0, 0], zero_division=1), 1)

    def test_bad_data_raises(self):
        bad_pairs = [([], []), ([1], [1, 2]), ([[1]], [[1]]), ([np.nan], [1]), ([1], [np.inf])]
        for fn in (*CLASSIFIERS, m.confusion_matrix):
            for y_true, y_pred in bad_pairs:
                with self.assertRaises(ValueError):
                    fn(y_true, y_pred)
            with self.assertRaises(TypeError):     # numbers mixed with strings
                fn([1], ['1'])

    def test_bad_options_raise(self):
        for fn in (m.precision, m.recall, m.f1):
            with self.assertRaises(ValueError):    # 3 classes with binary
                fn([0, 1, 2], [0, 1, 2])
            with self.assertRaises(ValueError):
                fn([0, 1], [0, 1], average='bad')
            with self.assertRaises(ValueError):
                fn([0, 1], [0, 1], zero_division=2)
            with self.assertRaises(ValueError):    # pos_label not in the data
                fn([0, 1], [0, 1], pos_label=3)

    def test_bad_labels_argument_raises(self):
        for labels in ([0, 0, 1], [0], []):        # duplicates, missing, empty
            with self.assertRaises(ValueError):
                m.confusion_matrix([0, 1], [0, 1], labels)


class TestMSE(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(m.mse([50, 70, 90], [55, 65, 80]), 50)
        self.assertEqual(m.mse([1, 2], [1, 2]), 0)

    def test_axis_matches_numpy(self):
        y_true, y_pred = np.arange(12.0).reshape(3, 4), np.ones((3, 4))
        for axis in (None, 0, 1, (0, 1)):
            np.testing.assert_allclose(m.mse(y_true, y_pred, axis=axis),
                                       np.mean((y_true - y_pred) ** 2, axis=axis))

    def test_nan_policies(self):
        self.assertEqual(m.mse([1, np.nan, 4], [2, 5, 2], nan_policy='omit'), 2.5)
        self.assertTrue(np.isnan(m.mse([np.nan], [2], nan_policy='omit')))
        self.assertTrue(np.isnan(m.mse([1, np.nan], [2, 3], nan_policy='propagate')))
        with self.assertRaises(ValueError):    # default is raise
            m.mse([np.nan], [1])

    def test_bad_input_raises(self):
        for y_true, y_pred in [([], []), ([1], [1, 2]), ([[1]], [1]), ([np.inf], [0])]:
            with self.assertRaises(ValueError):
                m.mse(y_true, y_pred)
        with self.assertRaises(TypeError):
            m.mse(['a'], ['b'])
        with self.assertRaises(ValueError):
            m.mse([1], [2], nan_policy='bad')

    def test_small_integer_types_do_not_wrap(self):
        a, b = np.array([100], dtype=np.int8), np.array([-100], dtype=np.int8)
        self.assertAlmostEqual(m.mse(a, b), 40000)


class TestROC(unittest.TestCase):
    def test_known_curve(self):
        # the classic 4-sample example
        fpr, tpr, thresholds = m.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
        np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
        np.testing.assert_allclose(thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])
        self.assertEqual(m.auc(fpr, tpr), 0.75)

    def test_all_scores_tied(self):
        # tied scores collapse to one point -> diagonal line, AUC 0.5
        fpr, tpr, _ = m.roc_curve([0, 1, 0, 1], [0.5] * 4)
        np.testing.assert_array_equal(fpr, [0, 1])
        np.testing.assert_array_equal(tpr, [0, 1])
        self.assertEqual(m.auc(fpr, tpr), 0.5)

    def test_perfect_and_reversed_scores(self):
        for scores, expected in [([0, 1], 1), ([1, 0], 0)]:
            fpr, tpr, _ = m.roc_curve([0, 1], scores)
            self.assertEqual(m.auc(fpr, tpr), expected)

    def test_string_labels(self):
        fpr, tpr, _ = m.roc_curve(['no', 'yes'], [0.2, 0.8], pos_label='yes')
        self.assertEqual(m.auc(fpr, tpr), 1)

    def test_roc_bad_input_raises(self):
        cases = [([1, 1], [0.1, 0.2]),           # one class only
                 ([0, 1, 2], [0.1, 0.2, 0.3]),   # three classes
                 ([0, 1], [np.nan, 0.2]),        # NaN score
                 ([0, 1], [0.2])]                # length mismatch
        for y_true, scores in cases:
            with self.assertRaises(ValueError):
                m.roc_curve(y_true, scores)

    def test_auc_bad_input_raises(self):
        cases = [([1, 0], [0, 1]),        # x decreasing
                 ([0], [0]),              # one point
                 ([0, 1], [0]),           # length mismatch
                 ([0, 1], [0, np.inf])]   # inf
        for x, y in cases:
            with self.assertRaises(ValueError):
                m.auc(x, y)


if __name__ == '__main__':
    unittest.main()