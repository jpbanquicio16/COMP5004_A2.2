"""Tests for the streaming metric classes in numcompute.metrics."""

import numpy as np
import pytest

from numcompute import metrics as m

CHUNKINGS = [1, 3, 17]


def labels(seed=0, n=600, k=3):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, k, n)
    p = np.where(rng.random(n) < 0.7, t, rng.integers(0, k, n))
    return t, p


def stream(metric, t, p, k):
    for a, b in zip(np.array_split(t, k), np.array_split(p, k)):
        metric.update(a, b)
    return metric


# ---------------- streaming == batch ----------------

@pytest.mark.parametrize("k", CHUNKINGS)
def test_confusion_matrix_and_accuracy_equal_batch(k):
    t, p = labels()
    np.testing.assert_array_equal(stream(m.ConfusionMatrix(), t, p, k).result(),
                                  m.confusion_matrix(t, p))
    assert stream(m.Accuracy(), t, p, k).result() == m.accuracy(t, p)


@pytest.mark.parametrize("k", CHUNKINGS)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.parametrize("cls, fn", [(m.Precision, m.precision), (m.Recall, m.recall),
                                     (m.F1, m.f1)])
def test_averaged_metrics_equal_batch(cls, fn, average, k):
    t, p = labels(seed=k)
    got = stream(cls(average=average), t, p, k).result()
    np.testing.assert_allclose(got, fn(t, p, average=average), rtol=0, atol=1e-15)


@pytest.mark.parametrize("cls, fn", [(m.Precision, m.precision), (m.Recall, m.recall),
                                     (m.F1, m.f1)])
def test_binary_with_string_labels(cls, fn):
    t = np.array(["yes", "no", "yes", "yes", "no", "no"] * 5)
    p = np.array(["yes", "yes", "no", "yes", "no", "no"] * 5)
    got = stream(cls(pos_label="yes"), t, p, 4).result()
    assert got == fn(t, p, pos_label="yes")


def test_batch_score_now_comes_from_counts():
    # the batch functions and the streaming classes share one formula
    tp, support, predicted = m._counts_from_matrix([[3, 1], [2, 4]])
    np.testing.assert_array_equal(tp, [3, 4])
    np.testing.assert_array_equal(support, [4, 6])
    np.testing.assert_array_equal(predicted, [5, 5])
    got = m._score_from_counts(tp, support, predicted, np.array([0, 1]), "binary", 1, 0, "f1")
    assert got == pytest.approx(8 / 11)


# ---------------- growing labels ----------------

def test_label_list_grows_and_matrix_is_padded():
    cm = m.ConfusionMatrix()
    cm.update([1, 3], [1, 3])
    cm.update([2], [0])                          # two new labels, sorted in between
    assert cm.classes_.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(cm.result(), m.confusion_matrix([1, 3, 2], [1, 3, 0]))


def test_new_class_late_matches_batch():
    t, p = labels(n=300, k=2)
    t2, p2 = np.r_[t, [5, 5, 5]], np.r_[p, [5, 0, 5]]
    got = stream(m.F1(average="macro"), t2, p2, 6).result()
    assert got == pytest.approx(m.f1(t2, p2, average="macro"))


def test_mixing_strings_and_numbers_raises():
    cm = m.ConfusionMatrix().update([1, 2], [1, 2])
    with pytest.raises(TypeError):
        cm.update(["a"], ["a"])


# ---------------- window ----------------

@pytest.mark.parametrize("window", [1, 7, 50, 1000])
def test_window_counts_only_last_samples(window):
    t, p = labels(n=230)
    cm = stream(m.ConfusionMatrix(window=window), t, p, 9)
    want = m.confusion_matrix(t[-window:], p[-window:], labels=cm.classes_)
    np.testing.assert_array_equal(cm.result(), want)
    assert cm.n_samples_ == min(window, len(t))


def test_window_eviction_changes_accuracy():
    acc = m.Accuracy(window=4)
    acc.update([1, 1, 1, 1], [1, 1, 1, 1])
    assert acc.result() == 1.0
    acc.update([1, 1], [0, 0])                   # two right ones fall out
    assert acc.result() == 0.5
    acc.update([0, 0, 0, 0], [0, 0, 0, 0])
    assert acc.result() == 1.0


def test_window_survives_label_growth():
    cm = m.ConfusionMatrix(window=3)
    cm.update([5, 5], [5, 5])
    cm.update([1], [1])                          # 1 sorts before 5: indices remap
    cm.update([5], [1])
    np.testing.assert_array_equal(cm.result(), m.confusion_matrix([5, 1, 5], [5, 1, 1]))


# ---------------- before data, empty chunks, reset ----------------

def test_results_before_any_data():
    assert np.isnan(m.Accuracy().result())
    assert np.isnan(m.F1().result())
    assert m.Precision(average=None).result().size == 0
    assert m.ConfusionMatrix().result().shape == (0, 0)
    assert np.isnan(m.AUC().result())


def test_empty_chunk_is_ignored():
    acc = m.Accuracy().update([], [])
    assert acc.n_samples_ == 0
    acc.update([1, 0], [1, 1]).update(np.array([]), np.array([]))
    assert acc.result() == 0.5


def test_reset():
    f = stream(m.F1(), *labels(k=2), 3)
    assert f.reset() is f
    assert np.isnan(f.result()) and f.classes_ is None


def test_bad_options():
    with pytest.raises(ValueError):
        m.Precision(average="bad")
    with pytest.raises(ValueError):
        m.Recall(zero_division=2)
    with pytest.raises(ValueError):
        m.Accuracy(window=0)
    with pytest.raises(ValueError):
        m.AUC(bins=0)
    with pytest.raises(ValueError):
        m.Accuracy().update([1, 2], [1])         # length mismatch


def test_binary_average_with_three_classes_raises_at_result():
    f = m.F1().update([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        f.result()


# ---------------- AUC ----------------

@pytest.mark.parametrize("k", CHUNKINGS)
def test_auc_with_tied_scores_equals_batch(k):
    rng = np.random.default_rng(k)
    y = rng.integers(0, 2, 1500)
    score = np.round(np.clip(0.35 * y + rng.random(1500) * 0.65, 0, 1), 2)  # many ties
    a = stream(m.AUC(), y, score, k)
    fpr, tpr, _ = m.roc_curve(y, score)
    assert a.result() == pytest.approx(m.auc(fpr, tpr), abs=1e-12)


def test_auc_tree_like_leaf_probabilities():
    # few distinct values, as from leaf frequencies: ties must count half
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    s = np.array([1 / 3, 1 / 3, 0.75, 0.75, 0.75, 0.0, 1.0, 1 / 3])
    fpr, tpr, _ = m.roc_curve(y, s)
    assert m.AUC().update(y, s).result() == pytest.approx(m.auc(fpr, tpr))


def test_auc_continuous_scores_close_to_batch():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 5000)
    s = 1 / (1 + np.exp(-(y + rng.normal(0, 1.2, 5000))))
    fpr, tpr, _ = m.roc_curve(y, s)
    assert abs(m.AUC().update(y, s).result() - m.auc(fpr, tpr)) < 1e-3


def test_auc_nan_until_both_classes_seen():
    a = m.AUC().update([1, 1], [0.9, 0.8])
    assert np.isnan(a.result())
    a.update([0], [0.1])
    assert a.result() == 1.0


def test_auc_memory_is_fixed():
    a = m.AUC(bins=100)
    for _ in range(5):
        a.update(np.r_[0, 1, 1, 0], np.r_[0.1, 0.9, 0.6, 0.4])
    assert a.pos_counts_.shape == (100,) and a.n_samples_ == 20


def test_auc_window():
    a = m.AUC(window=4)
    a.update([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])     # perfect
    a.update([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8])     # perfectly wrong, evicts the rest
    assert a.result() == 0.0


def test_auc_string_labels_and_bad_scores():
    a = m.AUC(pos_label="yes").update(["no", "yes"], [0.2, 0.7])
    assert a.result() == 1.0
    with pytest.raises(ValueError):
        a.update(["no"], [1.5])
    with pytest.raises(ValueError):
        a.update(["no", "yes"], [0.5])
