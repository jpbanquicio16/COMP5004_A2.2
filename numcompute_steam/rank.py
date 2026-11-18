"""
numcompute.rank
===============

Ranking with tie handling, and percentiles.

Conventions
-----------
- Ranks are 1-based (smallest value gets rank 1), as in most statistics texts.
- ``axis=-1`` default for ranking; ``axis=None`` (flatten) default for percentiles.
- NaN handling is controlled by ``nan_policy`` (see each function).
"""

from __future__ import annotations

import numpy as np

from numcompute.sort_search import _check_axis, argsort

__all__ = [
    "rank",
    "percentile",
    "percentile_rank",
]

_RANK_METHODS = ("average", "min", "max", "dense", "ordinal")
_NAN_POLICIES = ("keep", "omit", "raise")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rank_1d(x: np.ndarray, method: str) -> np.ndarray:
    """Rank a NaN-free 1-D array. Core logic shared by ``rank``."""
    result = np.empty(x.size, dtype=float)
    if x.size == 0:
        return result
    order = argsort(x, stable=True)
    if method == "ordinal":
        result[order] = np.arange(1, x.size + 1)
        return result

    values = x[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], x.size]
    if method == "average":
        ranks = (starts + 1 + ends) / 2.0
    elif method == "min":
        ranks = starts + 1
    elif method == "max":
        ranks = ends
    else:  # dense; public rank validates the method
        ranks = np.arange(1, starts.size + 1)
    result[order] = np.repeat(ranks, ends - starts)
    return result


def _check_q(q) -> np.ndarray:
    """Validate percentiles are in [0, 100]; return as float ndarray."""
    q = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q) | (q < 0) | (q > 100)):
        raise ValueError("q must contain finite percentiles in [0, 100]")
    return q


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(x, method: str = "average", axis: int = -1, *, nan_policy: str = "keep"):
    """Rank values along ``axis`` with configurable tie handling.

    Parameters
    ----------
    x : array_like, shape (..., n, ...)
    method : {"average", "min", "max", "dense", "ordinal"}, default "average"
        How tied values are ranked. For ``[10, 20, 20, 30]``:
        average -> [1, 2.5, 2.5, 4]; min -> [1, 2, 2, 4];
        max -> [1, 3, 3, 4]; dense -> [1, 2, 2, 3]; ordinal -> [1, 2, 3, 4].
    axis : int, default -1
    nan_policy : {"keep", "omit", "raise"}, default "keep"
        keep: NaN inputs get NaN ranks; other values ranked among themselves.
        omit: same output as "keep" (NaNs excluded from ranking).
        raise: raise ``ValueError`` if any NaN is present.

    Returns
    -------
    ndarray of float, same shape as ``x``
        Float so "average" ranks and NaNs are representable.

    Raises
    ------
    ValueError
        If ``method`` or ``nan_policy`` is invalid, or NaNs found with
        ``nan_policy="raise"``.
    np.exceptions.AxisError
        If ``axis`` is out of range.

    Notes
    -----
    Time O(n log n) per slice (dominated by the stable argsort),
    space O(n).
    """
    if method not in _RANK_METHODS:
        raise ValueError(f"method must be one of {_RANK_METHODS}")
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}")
    arr = np.asarray(x)
    axis = _check_axis(axis, arr.ndim)
    missing = np.isnan(arr)
    if nan_policy == "raise" and missing.any():
        raise ValueError("x must not contain NaN")

    rows = np.moveaxis(arr, axis, -1)
    masks = np.moveaxis(missing, axis, -1)
    result = np.full(rows.shape, np.nan, dtype=float)
    for index in np.ndindex(rows.shape[:-1]):
        valid = ~masks[index]
        result[index][valid] = _rank_1d(rows[index][valid], method)
    return np.moveaxis(result, -1, axis)


def percentile(x, q, axis=None, *, method: str = "linear", nan_policy: str = "propagate"):
    """Compute the q-th percentile(s) of ``x``.

    Parameters
    ----------
    x : array_like
    q : float or array_like of float
        Percentile(s) in ``[0, 100]``.
    axis : int or None, default None
        None flattens ``x`` first.
    method : {"linear", "lower", "higher", "nearest", "midpoint"}, default "linear"
        Interpolation between order statistics; "linear" matches
        ``np.percentile``'s default.
    nan_policy : {"propagate", "omit", "raise"}, default "propagate"
        propagate: a slice containing NaN produces NaN percentiles.
        omit: exclude NaNs; an all-NaN slice produces NaN percentiles.
        raise: reject any NaN in the input.

    Returns
    -------
    scalar or ndarray
        Shape is ``q.shape`` + ``x.shape`` with ``axis`` removed.

    Raises
    ------
    ValueError
        If ``x`` is empty, any ``q`` is outside [0, 100], ``method`` is
        invalid, or NaNs found with ``nan_policy="raise"``.

    Notes
    -----
    Time O(n log n) per slice via sorting (O(n) possible with
    partitioning). Space O(n).
    """
    methods = ("linear", "lower", "higher", "nearest", "midpoint")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}")
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'")
    q = _check_q(q)
    arr = np.asarray(x)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    axis = _check_axis(axis, arr.ndim)
    if arr.size == 0:
        raise ValueError("x must not be empty")
    if nan_policy == "raise" and np.isnan(arr).any():
        raise ValueError("x must not contain NaN")

    rows = np.moveaxis(arr, axis, -1)
    result = np.full(q.shape + rows.shape[:-1], np.nan, dtype=float)
    for index in np.ndindex(rows.shape[:-1]):
        row = rows[index]
        missing = np.isnan(row)
        if missing.any():
            if nan_policy == "propagate":
                continue
            row = row[~missing]
        if row.size == 0:
            continue
        values = row[argsort(row, stable=True)]
        positions = (q / 100.0) * (row.size - 1)
        lower = np.floor(positions).astype(np.intp)
        upper = np.ceil(positions).astype(np.intp)
        if method == "lower":
            interpolated = values[lower]
        elif method == "higher":
            interpolated = values[upper]
        elif method == "nearest":
            interpolated = values[np.rint(positions).astype(np.intp)]
        else:
            weight = positions - lower if method == "linear" else 0.5
            lo = values[lower].astype(float)
            hi = values[upper].astype(float)
            # Weighted endpoints avoid overflowing integer subtraction or sums.
            with np.errstate(invalid="ignore"):
                interpolated = (1 - weight) * lo + weight * hi
            interpolated = np.where((lower == upper) | (lo == hi), lo, interpolated)
        result[(slice(None),) * q.ndim + index] = interpolated
    return result.item() if result.ndim == 0 else result


def percentile_rank(x, score, *, kind: str = "weak"):
    """Percentage of values in ``x`` at or below ``score`` (inverse of percentile).

    Parameters
    ----------
    x : array_like, shape (n,)
    score : scalar or array_like
    kind : {"weak", "strict", "mean"}, default "weak"
        weak: % of values <= score; strict: % of values < score;
        mean: average of weak and strict (fair to ties).
        NaNs in ``x`` propagate to every result; NaN scores return NaN.

    Returns
    -------
    float or ndarray of float
        Values in [0, 100].

    Raises
    ------
    ValueError
        If ``x`` is empty or ``kind`` is invalid.

    Notes
    -----
    Time O(n log n + m log n) for m scores (sort once, then binary search).
    Space O(n).
    """
    if kind not in ("weak", "strict", "mean"):
        raise ValueError("kind must be 'weak', 'strict', or 'mean'")
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    scores = np.asarray(score)
    if np.isnan(arr).any():
        result = np.full(scores.shape, np.nan, dtype=float)
    else:
        values = arr[argsort(arr, stable=True)]
        if kind == "weak":
            counts = np.searchsorted(values, scores, side="right")
        elif kind == "strict":
            counts = np.searchsorted(values, scores, side="left")
        else:
            counts = (np.searchsorted(values, scores, side="left") / 2.0
                      + np.searchsorted(values, scores, side="right") / 2.0)
        result = np.where(np.isnan(scores), np.nan, counts / arr.size * 100.0)
    return result.item() if result.ndim == 0 else result
