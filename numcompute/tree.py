"""Decision tree classifier with streaming (buffered retraining) support.

Shapes
------
X : (n_samples, n_features) numeric array. NaN is allowed, +/-inf is rejected.
y : (n_samples,) labels of any hashable/sortable type. Missing labels
    (None / NaN) and infinities are rejected.
predict(X)       -> (n_samples,)            labels (same dtype as training y)
predict_proba(X) -> (n_samples, n_classes)  columns follow ``classes_``

Streaming
---------
``partial_fit(X_chunk, y_chunk)`` appends the chunk to an internal buffer.
Once the buffer holds at least ``buffer_size`` samples, the tree is retrained
from scratch on the buffered data and the buffer is cleared. Between retrains
``predict`` uses the most recently trained tree. The feature count is locked by
the first chunk seen (or by ``fit``) and later chunks must match it.

Missing values
--------------
At every split the training data decides whether NaN goes left or right (the
side with the lower impurity). The choice is stored in the node and reused at
prediction time. A feature that is all NaN (or constant) can't give a split.
If a feature had no NaN in training, NaN is sent to the larger child.

Example
-------
>>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
>>> for X_chunk, y_chunk in stream:
...     tree.partial_fit(X_chunk, y_chunk)
"""

import numpy as np

__all__ = ["DecisionTreeClassifier", "NotFittedError", "gini", "entropy"]


class NotFittedError(ValueError):
    """predict/score called before the tree has been trained."""

_EPS = 1e-12


# ---------------------------------------------------------------------------
# impurity functions. They take class counts with the class axis LAST, so they
# work on one node (k,) or on a whole grid of candidate splits (..., k).
# ---------------------------------------------------------------------------

def gini(counts):
    """Gini impurity from class counts (last axis). 0 = pure. Empty -> 0."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1)
    safe = np.where(n > 0, n, 1.0)
    p = counts / safe[..., None]
    return np.where(n > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)[()]


def entropy(counts):
    """Shannon entropy (bits) from class counts (last axis). Empty -> 0."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1)
    safe = np.where(n > 0, n, 1.0)
    p = counts / safe[..., None]
    logp = np.log2(np.where(p > 0, p, 1.0))        # log2(1)=0, avoids log2(0)
    return -(p * logp).sum(axis=-1)


_CRITERIA = {"gini": gini, "entropy": entropy}

def _check_X(X, name="X"):
    try:
        arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {type(X).__name__}: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}"
        )
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values; NaN is allowed, inf is not")
    return arr


def _check_y(y, n_samples, name="y"):
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,), got shape {arr.shape}")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"X and {name} have inconsistent lengths: {n_samples} vs {arr.shape[0]}"
        )
    if arr.dtype.kind in "fc":
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite labels")
    elif arr.dtype.kind == "O":
        bad = [v is None or (isinstance(v, float) and not np.isfinite(v)) for v in arr]
        if any(bad):
            raise ValueError(f"{name} contains missing (None/NaN) or infinite labels")
    return arr

class DecisionTreeClassifier:
    """Depth-limited CART-style classifier (Gini or entropy), streaming-ready.

    Parameters
    ----------
    max_depth : int, default 5          Maximum depth (root = depth 0).
    min_samples_split : int, default 2  Fewest samples a node needs to be split.
    max_features : None | int | float | "sqrt" | "log2"
        Features examined per split (random subset). None = all features.
    buffer_size : int, default 500      Samples to collect before retraining.
    criterion : "gini" | "entropy"
    random_state : int | None           Seed for the feature subsampling.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", random_state=None):
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError(f"max_depth must be an int >= 0, got {max_depth!r}")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be an int >= 2, got {min_samples_split!r}"
            )
        if not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
            raise ValueError(f"buffer_size must be an int >= 1, got {buffer_size!r}")
        if criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(_CRITERIA)}, got {criterion!r}"
            )
        self._validate_max_features(max_features)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = random_state

        self._impurity = _CRITERIA[criterion]
        self._reset_buffer()
        self.n_features_in_ = None
        self.classes_ = None
        self.n_retrains_ = 0
        self._nodes = None                      

    # Helpers
    @staticmethod
    def _validate_max_features(mf):
        if mf is None or mf in ("sqrt", "log2"):
            return
        if isinstance(mf, bool):
            raise ValueError(f"invalid max_features: {mf!r}")
        if isinstance(mf, (int, np.integer)) and mf >= 1:
            return
        if isinstance(mf, (float, np.floating)) and 0.0 < mf <= 1.0:
            return
        raise ValueError(
            "max_features must be None, 'sqrt', 'log2', an int >= 1 or a float in "
            f"(0, 1], got {mf!r}"
        )

    def _n_candidate_features(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            m = int(np.sqrt(n_features))
        elif mf == "log2":
            m = int(np.log2(n_features))
        elif isinstance(mf, (int, np.integer)):
            m = int(mf)
        else:
            m = int(mf * n_features)
        return min(n_features, max(1, m))

    def _reset_buffer(self):
        self._X_chunks = []
        self._y_chunks = []
        self.buffer_count_ = 0

    # -- split search (vectorised over positions, features and classes) ---

    def _best_split(self, X, yi, k, feats):
        """Best (feature, threshold, nan_left, gain) over ``feats`` or None.

        X : (n, n_features) float, yi : (n,) class indices in [0, k).
        All candidate thresholds of all candidate features are scored at once
        using cumulative one-hot class counts on the sorted columns.
        """
        n = yi.shape[0]
        Xs = X[:, feats]                                        # (n, F)
        order = np.argsort(Xs, axis=0, kind="stable")           # NaN sorts last
        vs = np.take_along_axis(Xs, order, axis=0)              # (n, F)
        onehot = (yi[:, None] == np.arange(k)).astype(float)    # (n, k)
        cum = np.cumsum(onehot[order], axis=0)                  # (n, F, k)

        total = onehot.sum(axis=0)                              # (k,)
        isnan = np.isnan(Xs)                                    # (n, F)
        nan_cnt = isnan.T.astype(float) @ onehot                # (F, k)

        left = cum[:-1]                                         # (n-1, F, k)
        valid = vs[:-1] < vs[1:]                                # False if NaN involved
        if not valid.any():
            return None

        # option 0: NaN goes right, option 1: NaN goes left
        L = np.stack([left, left + nan_cnt[None]], axis=1)      # (n-1, 2, F, k)
        R = total - L
        nL = L.sum(axis=-1)
        nR = R.sum(axis=-1)
        weighted = (nL * self._impurity(L) + nR * self._impurity(R)) / n
        weighted = np.where(valid[:, None, :], weighted, np.inf)

        has_nan = nan_cnt.sum(axis=1) > 0                       # (F,)
        weighted[:, 1, ~has_nan] = np.inf                       # option 1 pointless

        flat = weighted.reshape(-1, weighted.shape[-1])         # (2(n-1), F)
        # per feature: best (position, option); first minimum wins ties
        best_row = np.argmin(flat, axis=0)
        best_per_f = flat[best_row, np.arange(flat.shape[1])]
        fi = int(np.argmin(best_per_f))
        best_score = best_per_f[fi]
        if not np.isfinite(best_score):
            return None

        parent = float(self._impurity(total))
        gain = parent - best_score
        if gain <= _EPS:
            return None

        pos, option = divmod(int(best_row[fi]), 2)      # row = pos * 2 + option
        lo, hi = vs[pos, fi], vs[pos + 1, fi]
        thr = 0.5 * (lo + hi)
        if not (lo <= thr < hi):                                # float rounding guard
            thr = lo
        if has_nan[fi]:
            nan_left = bool(option)
        else:                                                   # unseen NaN -> bigger child
            n_left = int(cum[pos, fi].sum())
            nan_left = n_left >= (n - int(isnan[:, fi].sum())) - n_left
        return int(feats[fi]), float(thr), nan_left, float(gain)

    # -- tree construction -------------------------------------------------

    def _build(self, X, yi, k, depth, rng):
        counts = np.bincount(yi, minlength=k).astype(float)
        node = len(self._feature)
        self._feature.append(-1)
        self._threshold.append(np.nan)
        self._nan_left.append(False)
        self._left.append(-1)
        self._right.append(-1)
        self._counts.append(counts)

        n = yi.shape[0]
        if (depth >= self.max_depth or n < self.min_samples_split
                or np.count_nonzero(counts) <= 1):
            return node

        n_feat = X.shape[1]
        m = self._n_candidate_features(n_feat)
        feats = np.arange(n_feat) if m == n_feat else np.sort(
            rng.choice(n_feat, size=m, replace=False))

        split = self._best_split(X, yi, k, feats)
        if split is None:
            return node
        f, thr, nan_left, _ = split

        col = X[:, f]
        go_left = np.where(np.isnan(col), nan_left, col <= thr)
        self._feature[node] = f
        self._threshold[node] = thr
        self._nan_left[node] = nan_left
        self._left[node] = self._build(X[go_left], yi[go_left], k, depth + 1, rng)
        self._right[node] = self._build(X[~go_left], yi[~go_left], k, depth + 1, rng)
        return node

    # -- public API -------------------------------------------------------

    def fit(self, X, y):
        """Train a new tree from scratch on (X, y). Clears the stream buffer."""
        X = _check_X(X)
        y = _check_y(y, X.shape[0])
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature")

        classes, yi = np.unique(y, return_inverse=True)
        yi = yi.ravel()
        self._feature, self._threshold, self._nan_left = [], [], []
        self._left, self._right, self._counts = [], [], []
        rng = np.random.default_rng(self.random_state)
        self._build(X, yi, len(classes), 0, rng)

        self._nodes = {
            "feature": np.array(self._feature, dtype=np.intp),
            "threshold": np.array(self._threshold, dtype=float),
            "nan_left": np.array(self._nan_left, dtype=bool),
            "left": np.array(self._left, dtype=np.intp),
            "right": np.array(self._right, dtype=np.intp),
            "counts": np.vstack(self._counts),
        }
        del self._feature, self._threshold, self._nan_left
        del self._left, self._right, self._counts
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self._reset_buffer()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Add a chunk to the buffer; retrain once it holds >= buffer_size samples."""
        X = _check_X(X_chunk, "X_chunk")
        y = _check_y(y_chunk, X.shape[0], "y_chunk")
        if self.n_features_in_ is None:
            if X.shape[1] == 0:
                raise ValueError("X_chunk must have at least one feature")
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, expected {self.n_features_in_} "
                "(the feature count is fixed after the first chunk)"
            )
        if X.shape[0] == 0:
            return self

        self._X_chunks.append(X.copy())
        self._y_chunks.append(y.copy())
        self.buffer_count_ += X.shape[0]

        if self.buffer_count_ >= self.buffer_size:
            Xb = np.concatenate(self._X_chunks, axis=0)
            yb = np.concatenate(self._y_chunks, axis=0)
            self.fit(Xb, yb)                    # fit also clears the buffer
            self.n_retrains_ += 1
        return self

    def _leaf_ids(self, X):
        """Route every row to its leaf at once (loop over depth, not samples)."""
        nd = self._nodes
        n = X.shape[0]
        node = np.zeros(n, dtype=np.intp)
        rows = np.arange(n)
        for _ in range(self.max_depth + 1):
            feat = nd["feature"][node]
            active = feat >= 0
            if not active.any():
                break
            vals = X[rows, np.where(active, feat, 0)]
            go_left = np.where(np.isnan(vals), nd["nan_left"][node],
                               vals <= nd["threshold"][node])
            nxt = np.where(go_left, nd["left"][node], nd["right"][node])
            node = np.where(active, nxt, node)
        return node

    def _check_predict_input(self, X):
        if self._nodes is None:
            raise NotFittedError(
                "tree has not been trained yet; call fit() or feed partial_fit() "
                "at least buffer_size samples"
            )
        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return X

    def predict(self, X):
        """Predicted labels, shape (n_samples,), from the most recently trained tree."""
        X = self._check_predict_input(X)
        leaves = self._leaf_ids(X)
        return self.classes_[self._nodes["counts"][leaves].argmax(axis=1)]

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, n_classes)."""
        X = self._check_predict_input(X)
        counts = self._nodes["counts"][self._leaf_ids(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """Accuracy on (X, y)."""
        X = self._check_predict_input(X)
        y = _check_y(y, X.shape[0])
        if X.shape[0] == 0:
            return 0.0
        return float(np.mean(self.predict(X) == y))

    @property
    def is_fitted(self):
        return self._nodes is not None

    @property
    def depth_(self):
        """Depth of the trained tree (a single leaf has depth 0)."""
        if self._nodes is None:
            raise NotFittedError("tree has not been trained yet")
        nd = self._nodes
        depth = np.zeros(len(nd["feature"]), dtype=int)
        for i in range(len(depth)):             # children always have larger index
            if nd["feature"][i] >= 0:
                depth[nd["left"][i]] = depth[nd["right"][i]] = depth[i] + 1
        return int(depth.max())
