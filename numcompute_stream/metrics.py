"""
numcompute_stream.metrics

Classification and regression metrics, numpy only.

- labels are non-empty 1-D numeric or string arrays (no NaN / inf)
- confusion matrix: rows = actual labels, columns = predicted labels
- precision / recall / f1 are binary by default (positive label 1)
"""
import warnings

import numpy as np

__all__ = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'mse',
           'roc_curve', 'auc',
           'Accuracy', 'Precision', 'Recall', 'F1', 'ConfusionMatrix', 'AUC']


def _check_labels(y, name):
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f'{name} must be a nonempty 1-D array')
    if arr.dtype.kind not in 'biufU':
        raise TypeError(f'{name} must contain numeric or string labels')
    if arr.dtype.kind in 'biuf' and not np.isfinite(arr).all():
        raise ValueError(f'{name} must not contain NaN or infinity')
    return arr


def _check_pair(y_true, y_pred):
    t, p = _check_labels(y_true, 'y_true'), _check_labels(y_pred, 'y_pred')
    if t.shape != p.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if (t.dtype.kind == 'U') != (p.dtype.kind == 'U'):
        raise TypeError('actual and predicted labels must both be numbers or both strings')
    return t, p


def _encode(y_true, y_pred, labels=None):
    # Map labels to integer positions 0..C-1 so np.bincount can count them.
    t, p = _check_pair(y_true, y_pred)
    seen = np.unique(np.concatenate((t, p)))
    if labels is None:
        classes = seen
    else:
        classes = _check_labels(labels, 'labels')
        if np.unique(classes).size != classes.size:
            raise ValueError('labels must not contain duplicates')
        if not np.isin(seen, classes).all():
            raise ValueError('labels must include every observed label')
    # searchsorted needs sorted classes; `order` maps back to the user's order
    order = np.argsort(classes, kind='stable')
    t_idx = order[np.searchsorted(classes[order], t)]
    p_idx = order[np.searchsorted(classes[order], p)]
    return classes, t_idx, p_idx


def accuracy(y_true, y_pred):
    """Fraction of correct predictions, a float in [0, 1].
    Raises ValueError for empty, mismatched, non-1-D or NaN/inf labels."""
    t, p = _check_pair(y_true, y_pred)
    return float(np.mean(t == p))


def confusion_matrix(y_true, y_pred, labels=None):
    """Counts of actual (rows) vs predicted (columns) labels, shape (C, C).

    labels: optional order for the rows/columns (no duplicates, must include
    every label in the data). Default is the sorted labels found in the data.
    """
    classes, t_idx, p_idx = _encode(y_true, y_pred, labels)
    c = classes.size
    # turn each (actual, predicted) pair into one number, then count in one go
    return np.bincount(t_idx * c + p_idx, minlength=c * c).reshape(c, c)


def _check_score_options(average, zero_division):
    if average not in ('binary', 'micro', 'macro', 'weighted', None):
        raise ValueError('average must be binary, micro, macro, weighted or None')
    if zero_division not in (0, 1):
        raise ValueError('zero_division must be 0 or 1')


def _score(y_true, y_pred, average, pos_label, zero_division, kind):
    # Shared by precision/recall/f1. Per class we need tp, support (how many
    # really are that class) and predicted (how many were predicted as it):
    #   precision = tp / predicted, recall = tp / support,
    #   f1 = 2tp / (support + predicted)
    _check_score_options(average, zero_division)

    classes, t_idx, p_idx = _encode(y_true, y_pred)
    c = classes.size
    support = np.bincount(t_idx, minlength=c).astype(float)
    predicted = np.bincount(p_idx, minlength=c).astype(float)
    tp = np.bincount(t_idx[t_idx == p_idx], minlength=c).astype(float)
    return _score_from_counts(tp, support, predicted, classes, average, pos_label,
                              zero_division, kind)


def _score_from_counts(tp, support, predicted, classes, average, pos_label,
                       zero_division, kind):
    # the rest of _score, working only from the (tp, support, predicted)
    # counts, so the streaming classes share the same maths
    c = classes.size
    if average == 'binary':
        if c > 2:
            raise ValueError('binary averaging needs at most two classes')
        mask = classes == pos_label
        if c == 2 and not mask.any():
            raise ValueError('pos_label is not one of the observed classes')
        tp, support, predicted = tp[mask].sum(), support[mask].sum(), predicted[mask].sum()
    elif average == 'micro':
        tp, support, predicted = tp.sum(), support.sum(), predicted.sum()

    top, bottom = {'precision': (tp, predicted),
                   'recall': (tp, support),
                   'f1': (2 * tp, support + predicted)}[kind]
    # divide, using zero_division wherever the bottom is 0
    result = np.full(np.shape(bottom), float(zero_division))
    np.divide(top, bottom, out=result, where=bottom != 0)

    if average is None:
        return result
    if average == 'weighted':
        return float(np.sum(result * support / support.sum()))
    return float(result.mean())  # macro (for binary/micro it's a single number)


def precision(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """TP / (TP + FP). average: 'binary', 'micro', 'macro', 'weighted' or None
    (one score per class, sorted order). zero_division: 0 or 1, used when the
    denominator is 0. Returns a float, or an array when average=None."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'precision')


def recall(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """TP / (TP + FN). Parameters are the same as precision()."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'recall')


def f1(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """2TP / (2TP + FP + FN). Parameters are the same as precision()."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'f1')


def mse(y_true, y_pred, *, axis=None, nan_policy='raise'):
    """Mean squared error (same shapes required, no broadcasting).

    nan_policy: 'raise' (default), 'omit' (drop pairs where either value is
    NaN) or 'propagate' (NaN in -> NaN out). Too-big results come back as inf.
    """
    if nan_policy not in ('raise', 'omit', 'propagate'):
        raise ValueError('invalid nan_policy')
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.dtype.kind not in 'biuf' or p.dtype.kind not in 'biuf':
        raise TypeError('regression inputs must contain real numbers')
    if t.shape != p.shape or t.size == 0:
        raise ValueError('regression inputs must be nonempty and have equal shapes')

    # convert to float first so small int types can't wrap around
    t, p = t.astype(float), p.astype(float)
    if np.isinf(t).any() or np.isinf(p).any():
        raise ValueError('regression inputs must not contain infinity')
    if nan_policy == 'raise' and (np.isnan(t) | np.isnan(p)).any():
        raise ValueError('regression inputs contain NaN')

    # a NaN in either array makes the squared error NaN, 'omit' then skips it
    with warnings.catch_warnings(), np.errstate(over='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN mean -> NaN
        squares = (t - p) ** 2
        result = np.nanmean(squares, axis=axis) if nan_policy == 'omit' \
            else np.mean(squares, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def roc_curve(y_true, y_score, *, pos_label=1):
    """Bonus: ROC curve for binary labels.

    Returns (fpr, tpr, thresholds). The first point is (0, 0) at threshold inf,
    then one point per distinct score (tied scores share a point).
    A sample is predicted positive when score >= threshold.
    """
    t = _check_labels(y_true, 'y_true')
    score = np.asarray(y_score)
    if score.dtype.kind not in 'biuf':
        raise TypeError('y_score must be real numeric')
    score = score.astype(float)
    if score.shape != t.shape or not np.isfinite(score).all():
        raise ValueError('y_score must be finite and match y_true shape')
    if np.unique(t).size != 2 or not np.any(t == pos_label):
        raise ValueError('ROC needs two classes and an observed pos_label')

    # walk from the highest score to the lowest
    order = np.argsort(-score, kind='stable')
    sorted_scores = score[order]
    is_pos = t[order] == pos_label

    # a threshold only matters where the score changes: keep the last sample
    # of each run of equal scores
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), t.size - 1]
    tp = np.cumsum(is_pos)[ends]
    fp = ends + 1 - tp
    return (np.r_[0.0, fp / (~is_pos).sum()],
            np.r_[0.0, tp / is_pos.sum()],
            np.r_[np.inf, sorted_scores[ends]])


def auc(x, y):
    """Bonus: area under a curve (trapezoid rule). x must be non-decreasing,
    both arrays finite, 1-D, equal length, at least 2 points."""
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype.kind not in 'biuf' or y.dtype.kind not in 'biuf':
        raise TypeError('curve coordinates must be real numeric')
    x, y = x.astype(float), y.astype(float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError('x and y must be equal-length 1-D arrays with at least two points')
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.any(np.diff(x) < 0):
        raise ValueError('coordinates must be finite and x nondecreasing')
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))

# ---------------------------------------------------------------------------
# streaming metrics
# ---------------------------------------------------------------------------

def _count_matrix(t_idx, p_idx, c):
    # same counting as confusion_matrix, on label indices
    return np.bincount(t_idx * c + p_idx, minlength=c * c).reshape(c, c)


def _counts_from_matrix(cm):
    # per class: tp (diagonal), support (row sums), predicted (column sums)
    cm = np.asarray(cm, dtype=float)
    return np.diag(cm).copy(), cm.sum(axis=1), cm.sum(axis=0)


def _check_window(window):
    if window is not None and (isinstance(window, (bool, np.bool_))
                               or not isinstance(window, (int, np.integer)) or window < 1):
        raise ValueError('window must be None or an int >= 1')
    return None if window is None else int(window)


def _check_stream_labels(y, name):
    # like _check_labels, but an empty chunk is allowed (returns None)
    arr = np.asarray(y)
    if arr.ndim == 1 and arr.size == 0:
        return None
    return _check_labels(arr, name)


class _Ring:
    """Fixed-size ring buffer of the last `size` rows of a few int arrays."""

    def __init__(self, size, n_fields):
        self.size = size
        self.data = np.zeros((n_fields, size), dtype=np.intp)
        self.pos = 0        # next slot to write
        self.n = 0          # filled slots (writes start at 0, so data[:, :n])

    def push(self, *fields):
        new = np.stack(fields)[:, -self.size:]          # only the last `size` matter
        m = new.shape[1]
        self.data[:, (self.pos + np.arange(m)) % self.size] = new
        self.pos = (self.pos + m) % self.size
        self.n = min(self.n + m, self.size)

    def filled(self):
        return self.data[:, :self.n]


class ConfusionMatrix:
    """Streaming confusion matrix: rows = actual, columns = predicted.

    classes_ holds the sorted labels seen so far. When a chunk brings a new
    label the matrix is padded with a zero row and column in its sorted
    position, so result() always lines up with classes_ (and equals the
    batch confusion_matrix on all the data seen).

    window=None counts everything; window=W counts only the last W samples
    (kept in a ring buffer and recounted with np.bincount on each update).
    Memory O(C^2) (+ O(W) with a window).
    """

    def __init__(self, *, window=None):
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget all samples and labels. Returns self."""
        self.classes_ = None
        self._cm = np.zeros((0, 0), dtype=np.int64)
        self._ring = None if self.window is None else _Ring(self.window, 2)
        return self

    def _grow(self, labels):
        # add new labels in sorted position; pad the matrix and remap the ring
        if self.classes_ is None:
            self.classes_ = labels
            self._cm = np.zeros((labels.size, labels.size), dtype=np.int64)
            return
        if (self.classes_.dtype.kind == 'U') != (labels.dtype.kind == 'U'):
            raise TypeError('labels must stay all numbers or all strings')
        if np.isin(labels, self.classes_).all():
            return
        merged = np.union1d(self.classes_, labels)
        where = np.searchsorted(merged, self.classes_)       # old index -> new
        cm = np.zeros((merged.size, merged.size), dtype=np.int64)
        cm[np.ix_(where, where)] = self._cm
        if self._ring is not None:
            self._ring.data = where[self._ring.data]
        self.classes_, self._cm = merged, cm

    def update(self, y_true, y_pred):
        """Add a chunk of (actual, predicted) labels. Empty chunks are ignored."""
        t = _check_stream_labels(y_true, 'y_true')
        p = _check_stream_labels(y_pred, 'y_pred')
        if t is None and p is None:
            return self
        t, p = _check_pair(y_true, y_pred)
        self._grow(np.unique(np.concatenate((t, p))))
        t_idx = np.searchsorted(self.classes_, t)
        p_idx = np.searchsorted(self.classes_, p)
        c = self.classes_.size
        if self._ring is None:
            self._cm = self._cm + _count_matrix(t_idx, p_idx, c)
        else:
            self._ring.push(t_idx, p_idx)
            self._cm = _count_matrix(*self._ring.filled(), c)
        return self

    @property
    def n_samples_(self):
        """Samples behind result() (all of them, or the last W)."""
        return int(self._cm.sum())

    def result(self):
        """The (C, C) matrix in classes_ order ((0, 0) before any data)."""
        return self._cm.copy()


class _MatrixMetric:
    # base for metrics computed from a streaming confusion matrix

    def __init__(self, *, window=None):
        self.window = _check_window(window)
        self._matrix = ConfusionMatrix(window=window)

    def update(self, y_true, y_pred):
        """Add a chunk of labels. Returns self."""
        self._matrix.update(y_true, y_pred)
        return self

    def reset(self):
        """Forget everything. Returns self."""
        self._matrix.reset()
        return self

    @property
    def classes_(self):
        return self._matrix.classes_

    @property
    def n_samples_(self):
        return self._matrix.n_samples_

    def result(self):
        raise NotImplementedError


class Accuracy(_MatrixMetric):
    """Streaming accuracy: trace / total of the running confusion matrix.
    result() is NaN until a sample has been seen."""

    def result(self):
        cm = self._matrix.result()
        total = cm.sum()
        return float(np.trace(cm) / total) if total else float('nan')


class _AveragedMetric(_MatrixMetric):
    _kind = None

    def __init__(self, *, average='binary', pos_label=1, zero_division=0, window=None):
        _check_score_options(average, zero_division)
        super().__init__(window=window)
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division

    def result(self):
        """Same value as the batch function on the samples seen.
        NaN (or an empty array for average=None) before any sample."""
        cm = self._matrix.result()
        if cm.sum() == 0:
            return np.zeros(0) if self.average is None else float('nan')
        tp, support, predicted = _counts_from_matrix(cm)
        return _score_from_counts(tp, support, predicted, self.classes_, self.average,
                                  self.pos_label, self.zero_division, self._kind)


class Precision(_AveragedMetric):
    """Streaming precision, same options as precision()."""
    _kind = 'precision'


class Recall(_AveragedMetric):
    """Streaming recall, same options as recall()."""
    _kind = 'recall'


class F1(_AveragedMetric):
    """Streaming F1, same options as f1()."""
    _kind = 'f1'


class AUC:
    """Streaming ROC AUC from two fixed score histograms.

    update(y_true, y_score): y_score is the probability (in [0, 1]) that the
    sample is pos_label. Each score goes into one of `bins` equal bins, one
    histogram for positives and one for negatives. result() sweeps the
    threshold from the top bin down (cumulative sums give the ROC points)
    and applies the trapezoid rule. Scores in the same bin count as tied,
    which is exactly right for real ties (tree leaf probabilities often are)
    and costs at most the pairs that share a bin otherwise.

    Memory O(bins) (+ O(W) with window=W, recounted with np.bincount).
    result() is NaN until both classes have been seen.
    """

    def __init__(self, *, pos_label=1, bins=1000, window=None):
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) \
                or bins < 1:
            raise ValueError('bins must be a positive integer')
        self.pos_label = pos_label
        self.bins = int(bins)
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget everything. Returns self."""
        self.pos_counts_ = np.zeros(self.bins, dtype=np.int64)
        self.neg_counts_ = np.zeros(self.bins, dtype=np.int64)
        self._ring = None if self.window is None else _Ring(self.window, 2)
        return self

    def update(self, y_true, y_score):
        """Add a chunk of labels and positive-class scores. Returns self."""
        t = _check_stream_labels(y_true, 'y_true')
        score = np.asarray(y_score)
        if t is None:
            if score.size:
                raise ValueError('y_score must match y_true shape')
            return self
        if score.dtype.kind not in 'biuf':
            raise TypeError('y_score must be real numeric')
        score = score.astype(float)
        if score.shape != t.shape or not np.isfinite(score).all():
            raise ValueError('y_score must be finite and match y_true shape')
        if score.min() < 0 or score.max() > 1:
            raise ValueError('y_score must be probabilities in [0, 1]')
        is_pos = (t == self.pos_label).astype(np.intp)
        b = np.minimum((score * self.bins).astype(np.intp), self.bins - 1)
        if self._ring is None:
            self.pos_counts_ = self.pos_counts_ + np.bincount(b[is_pos == 1], minlength=self.bins)
            self.neg_counts_ = self.neg_counts_ + np.bincount(b[is_pos == 0], minlength=self.bins)
        else:
            self._ring.push(is_pos, b)
            flag, bb = self._ring.filled()
            self.pos_counts_ = np.bincount(bb[flag == 1], minlength=self.bins)
            self.neg_counts_ = np.bincount(bb[flag == 0], minlength=self.bins)
        return self

    @property
    def n_samples_(self):
        return int(self.pos_counts_.sum() + self.neg_counts_.sum())

    def roc_curve(self):
        """(fpr, tpr) at every bin edge, highest threshold first."""
        p, n = self.pos_counts_.sum(), self.neg_counts_.sum()
        if p == 0 or n == 0:
            raise ValueError('ROC needs at least one positive and one negative')
        tpr = np.r_[0.0, np.cumsum(self.pos_counts_[::-1]) / p]
        fpr = np.r_[0.0, np.cumsum(self.neg_counts_[::-1]) / n]
        return fpr, tpr

    def result(self):
        """Area under the ROC curve, NaN until both classes have been seen."""
        if self.pos_counts_.sum() == 0 or self.neg_counts_.sum() == 0:
            return float('nan')
        return auc(*self.roc_curve())
