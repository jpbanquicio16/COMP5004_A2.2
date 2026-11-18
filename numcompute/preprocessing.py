"""
numcompute.preprocessing

Feature scaling, encoding and imputation.

- X is always 2D shape (n_samples, n_features)
- fit -> self, transform -> new array, fit_transform does both
- learned attrs end with _ like mean_
- transform before fit raises NotFittedError
- no python loops, all vectorized
"""

from __future__ import annotations

import warnings

import numpy as np

from .stats import StreamingStats

__all__ = [
    "NotFittedError",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "Imputer",
]

_EPS = np.finfo(np.float64).eps
_UNKNOWN_POLICIES = ("error", "ignore")
_IMPUTE_STRATEGIES = ("constant", "mean")


class NotFittedError(ValueError):
    # raised when transform called before fit
    pass



# Private helpers




def _check_numeric_2d(X, *, name: str = "X") -> np.ndarray:
    # check X is numeric 2D, return as float64. NaN ok, inf not ok
    arr = np.asarray(X)
    if arr.dtype.kind not in ("b", "i", "u", "f"):
        raise TypeError(
            f"{name} must contain numbers, got dtype {arr.dtype}"
        )
    if arr.ndim != 2:
        hint = " (use X.reshape(-1, 1) for a single feature)" if arr.ndim == 1 else ""
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features), "
            f"got shape {arr.shape}{hint}"
        )
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature, got shape {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinity; clip or remove it first")
    return arr


def _check_fit_data(X, *, name: str = "X") -> np.ndarray:
    # same as above but also need at least 1 sample
    arr = _check_numeric_2d(X, name=name)
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one sample, got shape {arr.shape}")
    return arr


def _check_no_empty_columns(X: np.ndarray) -> None:
    # raise if any column is all NaN
    empty = np.isnan(X).all(axis=0)
    if empty.any():
        cols = np.flatnonzero(empty).tolist()
        raise ValueError(
            f"column(s) {cols} contain only NaN, so statistics cannot be "
            f"learned; remove them or impute first"
        )


def _check_n_features(X: np.ndarray, expected: int) -> None:
    # check num features matches fitted
    if X.shape[1] != expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but this transformer was fitted "
            f"with {expected} features"
        )


def _nan_std(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    # std ignoring NaN, scaled to avoid overflow
    dev = X - mean
    peak = np.nanmax(np.abs(dev), axis=0)
    safe = np.where(peak > 0, peak, 1.0)
    return peak * np.sqrt(np.nanmean((dev / safe) ** 2, axis=0))


class _BaseTransformer:
    # base class: fit_transform, fitted check and the streaming plumbing

    _fitted_attr = "n_features_in_"
    _state_attrs = ("n_features_in_",)

    def fit(self, X):
        raise NotImplementedError

    def partial_fit(self, X_chunk):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def reset(self):
        """
        Forget everything learned, as if the object was just created.

        Returns:
            self
        """
        for attr in self._state_attrs:
            self.__dict__.pop(attr, None)
        return self

    def fit_transform(self, X):
        """
        Fit and transform.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            array: transformed X, same as fit(X).transform(X)

        Time: depends on child
        Space: depends on child
        """
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, self._fitted_attr):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit(X) or "
                f"partial_fit(X_chunk) before transform(X)"
            )

    def _start_chunk(self, X) -> bool:
        # lock the feature count on the first chunk, check it afterwards.
        # Returns True for the first chunk so the child can set up its state.
        if hasattr(self, "n_features_in_"):
            _check_n_features(X, self.n_features_in_)
            return False
        self.n_features_in_ = X.shape[1]
        return True



# Public API





class StandardScaler(_BaseTransformer):
    """
    Standardize features to mean 0 and std 1.

    Params:
        None

    Attributes:
        mean_: mean per column, shape (n_features,)
        scale_: std per column, shape (n_features,), 1.0 for constant cols
        var_: variance per column, shape (n_features,)
        n_samples_seen_: non-NaN values seen per column, shape (n_features,)
        n_features_in_: int

    Notes:
        NaN ignored when fitting and stays NaN after transform. fit() raises
        for an all-NaN column; partial_fit() does not (the column keeps NaN
        statistics until it sees a value, and transforms to NaN until then).
        A column whose std is tiny next to its mean (std <= 10 eps |mean|)
        counts as constant: scale 1 and centred on its minimum, so it maps to 0.

        Streaming uses numcompute.stats.StreamingStats (Welford / Chan merge),
        so partial_fit over any chunking gives the same mean_ and scale_ as
        fit on all the data.

    Time: O(n_samples * n_features) for fit / partial_fit
    Space: O(n_features)
    """

    _state_attrs = ("n_features_in_", "mean_", "scale_", "var_", "n_samples_seen_",
                    "_stats")

    def fit(self, X):
        """
        Learn mean and std per column.

        Params:
            X: array shape (n_samples, n_features), numeric, can have NaN

        Returns:
            self: fitted scaler

        Raises:
            ValueError: if X not 2D, empty, has inf, or all NaN col
            TypeError: if X not numeric

        Time: O(n_samples * n_features)
        Space: O(n_features)
        """
        X = _check_fit_data(X)
        _check_no_empty_columns(X)
        return self.reset().partial_fit(X)

    def partial_fit(self, X_chunk):
        """
        Update the running mean and std with one chunk.

        Params:
            X_chunk: array shape (n_rows, n_features), may have 0 rows or NaN

        Returns:
            self

        Raises:
            ValueError: if not 2D, has inf, or the feature count changed
            TypeError: if not numeric

        Time: O(n_rows * n_features)
        Space: O(n_features)
        """
        X = _check_numeric_2d(X_chunk, name="X_chunk")
        if self._start_chunk(X):
            self._stats = StreamingStats()
        if X.shape[0]:
            self._stats.update_stats(X)
        self._refresh()
        return self

    def _refresh(self):
        # same rule as the batch fit, on the running statistics
        stats = self._stats
        if stats.n_features_in_ is None:                     # no rows yet
            count = np.zeros(self.n_features_in_, dtype=np.int64)
            mean = std = var = np.full(self.n_features_in_, np.nan)
        else:
            count, mean, std, var = stats.count, stats.mean, stats.std(), stats.variance()
        constant = (count > 0) & (std <= 10 * _EPS * np.abs(mean))
        mean = np.where(constant, stats.min, mean)
        self.mean_ = mean
        self.scale_ = np.where(constant | (count == 0), 1.0, std)
        self.var_ = var
        self.n_samples_seen_ = count

    def transform(self, X):
        """
        Standardize X: (x - mean) / std

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            array: float array shape (n_samples, n_features), new array, same shape as input

        Raises:
            NotFittedError: if fit not called yet
            ValueError: if X not 2D, wrong num features, or has inf
            TypeError: if not numeric

        Time: O(n_samples * n_features)
        Space: O(n_samples * n_features) for output
        """
        self._check_is_fitted()
        X = _check_numeric_2d(X)
        _check_n_features(X, self.n_features_in_)
        return (X - self.mean_) / self.scale_


class MinMaxScaler(_BaseTransformer):
    """
    Scale each feature to a range like 0-1.

    Params:
        feature_range: tuple (lo, hi), default (0,1), lo < hi must hold

    Attributes:
        data_min_: min per col, shape (n_features,)
        data_max_: max per col, shape (n_features,)
        data_range_: max-min per col, shape (n_features,)
        n_features_in_: int

    Raises:
        ValueError: if feature_range not finite or lo >= hi
        TypeError: if feature_range not a pair of numbers

    Notes:
        Streaming keeps a running np.fmin / np.fmax per column (fmin ignores
        NaN), so any chunking gives the same result as fit.

    Time: O(n_samples * n_features)
    Space: O(n_features)
    """

    _state_attrs = ("n_features_in_", "data_min_", "data_max_", "data_range_")

    def __init__(self, feature_range=(0.0, 1.0)):
        try:
            lo, hi = feature_range
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "feature_range must be a pair of numbers (lo, hi), "
                f"got {feature_range!r}"
            ) from exc
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(
                f"feature_range must be finite with lo < hi, got ({lo}, {hi})"
            )
        self.feature_range = (lo, hi)

    def fit(self, X):
        """
        Learn min and max per column.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            self

        Raises:
            ValueError: if X not 2D, empty, inf, or all NaN col
            TypeError: if not numeric

        Time: O(n_samples * n_features)
        Space: O(n_features)
        """
        X = _check_fit_data(X)
        _check_no_empty_columns(X)
        return self.reset().partial_fit(X)

    def partial_fit(self, X_chunk):
        """
        Update the running min and max with one chunk.

        Params:
            X_chunk: array shape (n_rows, n_features), may have 0 rows or NaN

        Returns:
            self

        Time: O(n_rows * n_features)
        Space: O(n_features)
        """
        X = _check_numeric_2d(X_chunk, name="X_chunk")
        if self._start_chunk(X):
            self.data_min_ = np.full(X.shape[1], np.nan)
            self.data_max_ = np.full(X.shape[1], np.nan)
        if X.shape[0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN col -> NaN, ignored by fmin
                self.data_min_ = np.fmin(self.data_min_, np.nanmin(X, axis=0))
                self.data_max_ = np.fmax(self.data_max_, np.nanmax(X, axis=0))
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, X):
        """
        Scale X to feature_range.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            array: float shape (n_samples, n_features), new array

        Raises:
            NotFittedError: if not fitted
            ValueError: if X not 2D, wrong features, inf
            TypeError: if not numeric

        Time: O(n_samples * n_features)
        Space: O(n_samples * n_features)
        """
        self._check_is_fitted()
        X = _check_numeric_2d(X)
        _check_n_features(X, self.n_features_in_)
        lo, hi = self.feature_range
        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        return (X - self.data_min_) / safe_range * (hi - lo) + lo


class SimpleImputer(_BaseTransformer):
    """
    Replace NaN with constant or mean.

    Params:
        strategy: "constant" or "mean", default "constant"
        fill_value: number for constant strategy, default 0.0, must be finite

    Attributes:
        statistics_: replacement per col, shape (n_features,)
        n_features_in_: int

    Raises:
        ValueError: if strategy invalid or fill_value not finite
        TypeError: if fill_value not number

    Notes:
        Streaming keeps a running sum and count of the non-NaN values per
        column; statistics_ = sum / count. A column with no values yet has
        statistic NaN, so its NaN stay NaN (the tree can still route them).

    Time: O(n_samples * n_features)
    Space: O(n_features)
    """

    _state_attrs = ("n_features_in_", "statistics_", "_sum", "_count")

    def __init__(self, strategy: str = "constant", fill_value: float = 0.0):
        if strategy not in _IMPUTE_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_IMPUTE_STRATEGIES}, got {strategy!r}"
            )
        if isinstance(fill_value, bool) or not isinstance(
            fill_value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"fill_value must be a number, got {type(fill_value).__name__}"
            )
        if not np.isfinite(fill_value):
            raise ValueError(f"fill_value must be finite, got {fill_value}")
        self.strategy = strategy
        self.fill_value = float(fill_value)

    def fit(self, X):
        """
        Learn replacement value per column.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            self

        Raises:
            ValueError: if X not 2D, empty, inf, or all NaN col for mean
            TypeError: if not numeric

        Time: O(n_samples * n_features)
        Space: O(n_features)
        """
        X = _check_fit_data(X)
        if self.strategy == "mean":
            _check_no_empty_columns(X)
        return self.reset().partial_fit(X)

    def partial_fit(self, X_chunk):
        """
        Update the running sum and count with one chunk.

        Params:
            X_chunk: array shape (n_rows, n_features), may have 0 rows or NaN

        Returns:
            self

        Time: O(n_rows * n_features)
        Space: O(n_features)
        """
        X = _check_numeric_2d(X_chunk, name="X_chunk")
        n_features = X.shape[1]
        if self._start_chunk(X):
            self._sum = np.zeros(n_features)
            self._count = np.zeros(n_features, dtype=np.int64)
        self._sum = self._sum + np.nansum(X, axis=0)
        self._count = self._count + (~np.isnan(X)).sum(axis=0)
        if self.strategy == "constant":
            stats = np.full(n_features, self.fill_value)
        else:
            stats = np.divide(self._sum, self._count, out=np.full(n_features, np.nan),
                              where=self._count > 0)
        self.statistics_ = stats
        return self

    def transform(self, X):
        """
        Replace NaN with learned value.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            array: float shape (n_samples, n_features), no NaN, new array

        Raises:
            NotFittedError: if not fitted
            ValueError: if X not 2D, wrong features, inf
            TypeError: if not numeric

        Time: O(n_samples * n_features)
        Space: O(n_samples * n_features)
        """
        self._check_is_fitted()
        X = _check_numeric_2d(X)
        _check_n_features(X, self.n_features_in_)
        return np.where(np.isnan(X), self.statistics_, X)


# Imputer name for assignment
Imputer = SimpleImputer


class OneHotEncoder(_BaseTransformer):
    """
    One-hot encode categorical features.

    Params:
        handle_unknown: "error" or "ignore", default "error"
            error raises if unseen category, ignore encodes as zeros
        max_categories: None or int k >= 1, default None
            None: the output grows by one column per new category.
            k: every feature gets exactly k columns from the start, so
            n_features_out_ = n_features * k never changes. Categories fill
            a feature's slots in the order they are first seen. A category
            beyond k raises with handle_unknown="error" and is encoded as
            zeros with "ignore".

    Attributes:
        categories_: list of arrays, categories per feature in column order
            (sorted within the first chunk, later ones in order of arrival)
        n_features_in_: int, num input features
        n_features_out_: int, total output cols

    Raises:
        ValueError: if handle_unknown or max_categories invalid

    Notes:
        X can be numbers or strings but not mixed object dtype. NaN not allowed.
        Streaming: partial_fit learns categories it has not seen before and
        adds their columns at the END of the output, so earlier columns never
        move. transform does not learn, so in a test-then-train loop a new
        category meets transform first: use handle_unknown="ignore" there.
        A model whose feature count is fixed (the tree) needs max_categories.

    Time: O(n_samples * n_features * log(n_categories))
    Space: O(n_samples * n_features_out_)
    """

    _state_attrs = ("n_features_in_", "n_features_out_", "categories_")

    def __init__(self, handle_unknown: str = "error", max_categories=None):
        if handle_unknown not in _UNKNOWN_POLICIES:
            raise ValueError(
                f"handle_unknown must be one of {_UNKNOWN_POLICIES}, "
                f"got {handle_unknown!r}"
            )
        if max_categories is not None and (
            isinstance(max_categories, bool)
            or not isinstance(max_categories, (int, np.integer))
            or max_categories < 1
        ):
            raise ValueError(
                f"max_categories must be None or an int >= 1, got {max_categories!r}"
            )
        self.handle_unknown = handle_unknown
        self.max_categories = None if max_categories is None else int(max_categories)
        self.reset()

    def reset(self):
        """
        Forget all categories. Returns self.
        """
        super().reset()
        self._family = None
        # one entry per learned category: its feature, value and output column
        self._feature_of_output = np.zeros(0, dtype=np.intp)
        self._value_of_output = None
        self._output_col = np.zeros(0, dtype=np.intp)
        self._n_used = None
        # lookup tables rebuilt whenever categories are added: sorted unique
        # values, sorted keys (feature * n_values + value code) and their columns
        self._values = None
        self._pair_keys = np.zeros(0, dtype=np.intp)
        self._pair_cols = np.zeros(0, dtype=np.intp)
        return self

    @staticmethod
    def _check_categorical_2d(X):
        # check X is 2D numeric or string, return array and family
        arr = np.asarray(X)
        if arr.dtype.kind in ("b", "i", "u", "f"):
            family = "numeric"
        elif arr.dtype.kind in ("U", "S"):
            family = "string"
        else:
            raise TypeError(
                f"X must contain numbers or strings, got dtype {arr.dtype}; "
                f"convert it with astype(str) or astype(float)"
            )
        if arr.ndim != 2:
            hint = " (use X.reshape(-1, 1) for a single feature)" if arr.ndim == 1 else ""
            raise ValueError(
                f"X must be 2-D with shape (n_samples, n_features), "
                f"got shape {arr.shape}{hint}"
            )
        if arr.shape[1] == 0:
            raise ValueError(f"X must have at least one feature, got shape {arr.shape}")
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            raise ValueError(
                "X contains NaN; impute numeric data first (e.g. SimpleImputer)"
            )
        return arr, family

    @staticmethod
    def _flatten(arr):
        # flatten col-major and get feature index per element
        n, f = arr.shape
        return arr.T.reshape(-1), np.repeat(np.arange(f), n)

    def _check_family(self, family):
        if family != self._family:
            raise TypeError(
                f"X holds {family} values but the encoder was fitted on "
                f"{self._family} values"
            )

    def _lookup(self, flat, feature):
        # (known, output column) for every (value, feature) pair, all at once
        if self._values is None:
            return np.zeros(flat.shape, dtype=bool), np.zeros(flat.shape, dtype=np.intp)
        values, pair_keys = self._values, self._pair_keys
        n_values = values.size

        pos = np.minimum(np.searchsorted(values, flat), n_values - 1)
        value_known = values[pos] == flat
        keys = feature * n_values + pos
        out_col = np.minimum(np.searchsorted(pair_keys, keys), pair_keys.size - 1)
        known = value_known & (pair_keys[out_col] == keys)
        return known, self._pair_cols[out_col]

    def _rebuild(self):
        # sorted lookup keys (feature * n_values + value code) -> output column
        values = np.unique(self._value_of_output)
        n_values = values.size
        pair_keys = self._feature_of_output * n_values + np.searchsorted(values, self._value_of_output)
        order = np.argsort(pair_keys, kind="stable")
        self._values = values
        self._pair_keys = pair_keys[order]
        self._pair_cols = self._output_col[order]

    def _refresh_categories(self):
        f = self.n_features_in_
        if self._value_of_output is None:
            self.categories_ = [np.array([]) for _ in range(f)]
            return
        order = np.lexsort((self._output_col, self._feature_of_output))
        counts = np.bincount(self._feature_of_output, minlength=f)
        self.categories_ = np.split(self._value_of_output[order], np.cumsum(counts)[:-1])

    def fit(self, X):
        """
        Learn categories per column.

        Params:
            X: array shape (n_samples, n_features), numeric or string

        Returns:
            self

        Raises:
            ValueError: if X not 2D, empty, or has NaN
            TypeError: if X not numeric/string

        Time: O(n_samples * n_features * log)
        Space: O(n_categories)
        """
        arr, _ = self._check_categorical_2d(X)
        if arr.shape[0] == 0:
            raise ValueError(f"X must have at least one sample, got shape {arr.shape}")
        return self.reset().partial_fit(arr)

    def partial_fit(self, X_chunk):
        """
        Learn the categories in one chunk that have not been seen before.

        New categories are sorted (by feature, then value) and given the next
        free columns: at the end of the output, or in the feature's reserved
        slots when max_categories is set. Existing columns never move.

        Params:
            X_chunk: array shape (n_rows, n_features), numeric or string

        Returns:
            self

        Raises:
            ValueError: if not 2D, has NaN, wrong feature count, or a feature
                needs more than max_categories columns (handle_unknown="error")
            TypeError: if not numeric/string or the type family changed

        Time: O(n_rows * n_features * log)
        Space: O(n_categories)
        """
        arr, family = self._check_categorical_2d(X_chunk)
        k = self.max_categories
        if self._family is None:
            self._family = family
        else:
            self._check_family(family)
        if self._start_chunk(arr):
            self._n_used = np.zeros(arr.shape[1], dtype=np.intp)
            self.n_features_out_ = arr.shape[1] * k if k else 0
        f = self.n_features_in_

        if arr.shape[0]:
            flat, feature = self._flatten(arr)
            # distinct (feature, value) pairs, sorted by feature then value
            values, codes = np.unique(flat, return_inverse=True)
            n_values = values.size
            pair_keys = np.unique(feature * n_values + codes.ravel())
            feature_of_output = pair_keys // n_values
            value_of_output = values[pair_keys % n_values]
            # keep only the pairs not learned before
            known, _ = self._lookup(value_of_output, feature_of_output)
            feature_of_output, value_of_output = feature_of_output[~known], value_of_output[~known]

            if feature_of_output.size:
                # position of each new pair among the new pairs of its feature
                rank = np.arange(feature_of_output.size) - np.searchsorted(
                    feature_of_output, feature_of_output)
                if k:
                    slot = self._n_used[feature_of_output] + rank
                    over = slot >= k
                    if over.any():
                        if self.handle_unknown == "error":
                            bad = np.unique(feature_of_output[over]).tolist()
                            raise ValueError(
                                f"feature(s) {bad} have more than max_categories={k} "
                                f"categories; raise max_categories or use "
                                f"handle_unknown='ignore' to encode extras as zeros"
                            )
                        feature_of_output = feature_of_output[~over]
                        value_of_output, slot = value_of_output[~over], slot[~over]
                    out_col = feature_of_output * k + slot
                else:
                    out_col = self.n_features_out_ + np.arange(feature_of_output.size)
                    self.n_features_out_ += int(feature_of_output.size)
                self._n_used = self._n_used + np.bincount(feature_of_output, minlength=f)
                self._feature_of_output = np.concatenate([self._feature_of_output,
                                                          feature_of_output])
                self._output_col = np.concatenate([self._output_col, out_col])
                self._value_of_output = value_of_output if self._value_of_output is None \
                    else np.concatenate([self._value_of_output, value_of_output])
                self._rebuild()
        self._refresh_categories()
        return self

    def transform(self, X):
        """
        One-hot encode X.

        Params:
            X: array shape (n_samples, n_features)

        Returns:
            array: float shape (n_samples, n_features_out_), 0 or 1 values

        Raises:
            NotFittedError: if not fitted
            ValueError: if X not 2D, wrong features, NaN, or unseen category with error policy
            TypeError: if type family different from fit

        Time: O(n_samples * n_features * log)
        Space: O(n_samples * n_features_out_)
        """
        self._check_is_fitted()
        arr, family = self._check_categorical_2d(X)
        self._check_family(family)
        _check_n_features(arr, self.n_features_in_)
        n, f = arr.shape
        flat, feature = self._flatten(arr)
        known, out_col = self._lookup(flat, feature)

        if self.handle_unknown == "error" and not known.all():
            bad = np.flatnonzero(~known)
            shown = ", ".join(
                f"{flat[i].item()!r} in feature {int(feature[i])}" for i in bad[:5]
            )
            more = f" (and {bad.size - 5} more)" if bad.size > 5 else ""
            raise ValueError(
                f"found unseen categories during transform: {shown}{more}; "
                f"use handle_unknown='ignore' to encode them as zeros"
            )

        rows = np.tile(np.arange(n), f)
        out = np.zeros((n, self.n_features_out_), dtype=np.float64)
        out[rows[known], out_col[known]] = 1.0
        return out
