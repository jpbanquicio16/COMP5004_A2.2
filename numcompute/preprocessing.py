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

import numpy as np

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
    # base class with fit_transform and fitted check

    _fitted_attr = "n_features_in_"

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

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
                f"{type(self).__name__} is not fitted yet; call fit(X) "
                f"before transform(X)"
            )



# Public API





class StandardScaler(_BaseTransformer):
    """
    Standardize features to mean 0 and std 1.

    Params:
        None

    Attributes:
        mean_: mean per column, shape (n_features,)
        scale_: std per column, shape (n_features,), 1.0 for constant cols
        n_features_in_: int

    Notes:
        NaN ignored when fitting and stays NaN after transform. All NaN col raises error.

    Time: O(n_samples * n_features) for fit
    Space: O(n_features)
    """

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
        mean = np.nanmean(X, axis=0)
        std = _nan_std(X, mean)
        constant = std <= 10 * _EPS * np.abs(mean)
        mean = np.where(constant, np.nanmin(X, axis=0), mean)
        self.mean_ = mean
        self.scale_ = np.where(constant, 1.0, std)
        self.n_features_in_ = X.shape[1]
        return self

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

    Time: O(n_samples * n_features)
    Space: O(n_features)
    """

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
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.n_features_in_ = X.shape[1]
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

    Time: O(n_samples * n_features)
    Space: O(n_features)
    """

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
        n_features = X.shape[1]
        if self.strategy == "constant":
            stats = np.full(n_features, self.fill_value)
        else:
            _check_no_empty_columns(X)
            stats = np.nanmean(X, axis=0)
        self.statistics_ = stats
        self.n_features_in_ = n_features
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

    Attributes:
        categories_: list of arrays, sorted distinct values per feature
        n_features_in_: int, num input features
        n_features_out_: int, total output cols = sum len(categories)

    Raises:
        ValueError: if handle_unknown invalid

    Notes:
        X can be numbers or strings but not mixed object dtype. NaN not allowed.

    Time: O(n_samples * n_features * log(n_categories))
    Space: O(n_samples * n_features_out_)
    """

    def __init__(self, handle_unknown: str = "error"):
        if handle_unknown not in _UNKNOWN_POLICIES:
            raise ValueError(
                f"handle_unknown must be one of {_UNKNOWN_POLICIES}, "
                f"got {handle_unknown!r}"
            )
        self.handle_unknown = handle_unknown

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
        arr, family = self._check_categorical_2d(X)
        n, f = arr.shape
        if n == 0:
            raise ValueError(f"X must have at least one sample, got shape {arr.shape}")
        flat, feature = self._flatten(arr)

        values, codes = np.unique(flat, return_inverse=True)
        n_values = values.size
        pair_keys = np.unique(feature * n_values + codes)

        feature_of_output = pair_keys // n_values
        counts = np.bincount(feature_of_output, minlength=f)
        self.categories_ = np.split(
            values[pair_keys % n_values], np.cumsum(counts)[:-1]
        )
        self._values = values
        self._pair_keys = pair_keys
        self._family = family
        self.n_features_in_ = f
        self.n_features_out_ = int(pair_keys.size)
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
        if family != self._family:
            raise TypeError(
                f"X holds {family} values but the encoder was fitted on "
                f"{self._family} values"
            )
        _check_n_features(arr, self.n_features_in_)
        n, f = arr.shape
        flat, feature = self._flatten(arr)

        values, pair_keys = self._values, self._pair_keys
        n_values = values.size

        pos = np.minimum(np.searchsorted(values, flat), n_values - 1)
        value_known = values[pos] == flat
        keys = feature * n_values + pos
        out_col = np.minimum(np.searchsorted(pair_keys, keys), pair_keys.size - 1)
        known = value_known & (pair_keys[out_col] == keys)

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
