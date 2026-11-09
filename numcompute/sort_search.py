"""Sorting, selection and insertion searches using Python and NumPy.

Sorting is stable by default. Selection uses zero-based positions; searches
require ascending input and return insertion positions, not match indicators.
Functions return new outputs without modifying their inputs. Merge sort and
binary search use educational from-scratch loops; quickselect iteratively
narrows vectorised partitions. NumPy-backed alternatives handle bulk work.
"""

from __future__ import annotations

import operator

import numpy as np 

__all__ = ["merge_sort", "sort", "argsort", "quickselect", "topk",
           "binary_search", "searchsorted"]

# priv validation helpers 

def _as_1d(x,name: str = "x") -> np.ndarray: 
    """ 
    convert x into 1d ndarray or raise value error
    """

    arr = np.asarray(x)

    if arr.ndim != 1: 
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    
    return arr

def _as_index(value, name: str) -> int:
    """Accept Python/NumPy integer indices, excluding booleans and floats."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def _check_k(k: int, n: int, *, allow_n: bool = True) -> int:
    """Validate 0 <= k <= n (or k < n when allow_n=False)."""
    k = _as_index(k, "k")
    if k < 0 or (k > n if allow_n else k >= n):
        comparison = "<=" if allow_n else "<"
        raise ValueError(f"k must satisfy 0 <= k {comparison} {n}, got {k}")
    return k
        


def _check_axis(axis: int, ndim: int) -> int: 
    """
    normalise axis, can be negative. 
    Raise exceptions.AxisError if invalid
    """

    axis = _as_index(axis, "axis")
    
    original_axis = axis
    if axis < 0: 
        axis = axis + ndim
    
    if axis < 0 or axis >= ndim:
        raise np.exceptions.AxisError(original_axis, ndim=ndim)
    
    return axis


# Sorting Algorithms 
def merge_sort(x, *, return_indices=False):
    """
    Stable merge sort implemented from scratch (1-D).
 
    Parameters
    ----------
    x : array_like, shape (n,)
        Real numbers, booleans, or strings to sort. NaNs are placed last
        and retain their original order.
    return_indices : bool, default False
        If True, also return the permutation that sorts ``x``.
 
    Returns
    -------
    sorted_x : ndarray, shape (n,)
        Sorted copy of ``x``.
    indices : ndarray of int, shape (n,)
        Only if ``return_indices`` is True; ``x[indices] == sorted_x``.
 
    Raises
    ------
    ValueError
        If ``x`` is not 1-D.
    TypeError
        If ``x`` has an unsupported dtype (including complex, object,
        structured, and datetime arrays).
 
    Notes
    -----
    Time O(n log n), space O(n). Stable: equal elements keep input order.
    This educational implementation uses recursive splitting and merge loops;
    ``sort`` and ``argsort`` provide the NumPy-backed alternatives used by
    ``rank``. The input is not modified.
    """

    arr = _as_1d(x)
    if arr.dtype.kind not in "buifUS":
        raise TypeError("x must contain real numbers, booleans, or strings")
    
    values = arr.tolist()
    indices = list(range(len(arr)))

    def _merge_sort(val, index): 
        if len(val) <= 1:
            return val, index

        mid = len(val) // 2

        left_val, left_index = _merge_sort(
            val[:mid],
            index[:mid]
        )

        right_val, right_index = _merge_sort(
            val[mid:],
            index[mid:]
        )

        return _merge(
            left_val,
            left_index,
            right_val,
            right_index
        )

    _, sorted_indices = _merge_sort(values, indices)
    sorted_indices = np.asarray(sorted_indices, dtype=np.intp)
    sorted_x = arr[sorted_indices]
    if return_indices:
        return sorted_x, sorted_indices
    return sorted_x
    
def _merge(left_vals, left_index, right_vals, right_index):
    """Merge sorted value/index lists stably, placing real NaNs last.

    Return two new lists in O(len(left_vals) + len(right_vals)) time and space.
    Preferring the left value on ties preserves the original input order.
    """
    merged_vals = []
    merged_index = []

    i = 0
    j = 0

    while i < len(left_vals) and j < len(right_vals):

        left = left_vals[i]
        right = right_vals[j]
        # NaN is unequal to itself. Prefer the left side for equal values
        # (including two NaNs) to preserve stability across merge boundaries.
        if right != right or (left == left and left <= right):
            merged_vals.append(left_vals[i])
            merged_index.append(left_index[i])
            i += 1
        else:
            merged_vals.append(right_vals[j])
            merged_index.append(right_index[j])
            j += 1

    merged_vals.extend(left_vals[i:])
    merged_index.extend(left_index[i:])

    merged_vals.extend(right_vals[j:])
    merged_index.extend(right_index[j:])

    return merged_vals, merged_index

def sort(x, axis: int = -1, *, stable: bool = True):
    """Vectorised sort along ``axis`` (NumPy-backed fast path).
 
    Parameters
    ----------
    x : array_like, shape (..., n, ...)
    axis : int, default -1
    stable : bool, default True
        Use a stable algorithm so ties keep their input order.
 
    Returns
    -------
    ndarray, same shape as ``x``
 
    Raises
    ------
    TypeError
        If ``axis`` is not an integer index (booleans and floats are rejected),
        or input elements cannot be ordered.
    np.exceptions.AxisError
        If ``axis`` is out of range.
 
    Notes
    -----
    Time O(n log n) per slice, space O(size of x).
    """


    arr = np.asarray(x)

    axis = _check_axis(axis, arr.ndim)

    if stable:
        return np.sort(arr, axis=axis, kind="stable")

    return np.sort(arr, axis=axis)


 
 
def argsort(x, axis: int = -1, *, stable: bool = True):
    """Indices that would sort ``x`` along ``axis``.
 
    Parameters
    ----------
    x : array_like
    axis : int, default -1
    stable : bool, default True
 
    Returns
    -------
    ndarray of int, same shape as ``x``
 
    Raises
    ------
    TypeError
        If ``axis`` is not an integer index (booleans and floats are rejected),
        or input elements cannot be ordered.
    np.exceptions.AxisError
        If ``axis`` is out of range.
 
    Notes
    -----
    Time O(n log n) per slice, space O(size of x).
    """

    arr = np.asarray(x)
    axis = _check_axis(axis, arr.ndim)

    if stable:
        return np.argsort(arr, axis=axis, kind="stable")
    else:
        return np.argsort(arr, axis=axis)
 
# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
 
def quickselect(x, k: int, *, rng=None):
    """Return the k-th smallest element (0-indexed) of a 1-D array.
 
    Implemented from scratch: random pivot, three-way partition using
    vectorised boolean masks.
 
    Parameters
    ----------
    x : array_like, shape (n,)
    k : int
        Rank of the element to return, ``0 <= k < n``.
    rng : np.random.Generator or int or None
        Source of randomness for pivot choice (pass a seed for reproducibility).
 
    Returns
    -------
    scalar
        The value that would sit at index ``k`` in ``np.sort(x)``.
 
    Raises
    ------
    ValueError
        If ``x`` is empty, not 1-D, contains NaN, or ``k`` is out of range.
    TypeError
        If ``k`` is not an integer index (booleans and floats are rejected),
        or input elements cannot be compared.
 
    Notes
    -----
    Time O(n) expected, O(n^2) worst case. Space O(n) for the partitions.
    """
    arr = _as_1d(x)
    k = _check_k(k, arr.size, allow_n=False)
    if np.issubdtype(arr.dtype, np.inexact) and np.isnan(arr).any():
        raise ValueError("x must not contain NaN")

    generator = np.random.default_rng(rng)
    while arr.size > 1:
        pivot = arr[generator.integers(arr.size)]
        lower = arr[arr < pivot]
        equal_count = np.count_nonzero(arr == pivot)

        if k < lower.size:
            arr = lower
        elif k < lower.size + equal_count:
            return pivot
        else:
            k -= lower.size + equal_count
            arr = arr[arr > pivot]

    return arr[0]
 
 
def topk(x, k: int, axis: int = -1, *, largest: bool = True, sorted: bool = True):
    """Top-k values and their indices along ``axis`` (argpartition-based).
 
    Parameters
    ----------
    x : array_like, shape (..., n, ...)
    k : int
        Number of elements to return, ``0 <= k <= n``.
    axis : int, default -1
    largest : bool, default True
        If False, return the k smallest instead.
    sorted : bool, default True
        If True, results are ordered (descending if ``largest``, else
        ascending). Ties are broken by smaller original index first.
 
    Returns
    -------
    values : ndarray, shape (..., k, ...)
    indices : ndarray of int, shape (..., k, ...)
 
    Raises
    ------
    ValueError
        If ``k`` is negative or larger than ``x.shape[axis]``.
    TypeError
        If ``k`` or ``axis`` is not an integer index (booleans and floats are
        rejected), or input elements cannot be ordered.
    np.exceptions.AxisError
        If ``axis`` is out of range.
 
    Notes
    -----
    Time O(n + k log k) per slice when ``sorted`` else O(n).
    Space O(size of x).
    """
    arr = np.asarray(x)
    axis = _check_axis(axis, arr.ndim)
    n = arr.shape[axis]
    k = _check_k(k, n)

    if k == 0:
        shape = list(arr.shape)
        shape[axis] = 0
        return np.empty(shape, dtype=arr.dtype), np.empty(shape, dtype=np.intp)

    # Include indices in the partition keys so ties at the cutoff are stable.
    # For largest=True, larger keys must prefer smaller original indices.
    keys = np.empty(arr.shape, dtype=[("value", arr.dtype), ("index", np.intp)])
    keys["value"] = arr
    index_shape = [1] * arr.ndim
    index_shape[axis] = n
    original_indices = np.arange(n, dtype=np.intp).reshape(index_shape)
    keys["index"] = -original_indices if largest else original_indices

    cutoff = n - k if largest else k - 1
    partition = np.argpartition(keys, cutoff, axis=axis, order=["value", "index"])
    positions = np.arange(n - k, n) if largest else np.arange(k)
    indices = np.take(partition, positions, axis=axis)

    if sorted:
        selected_keys = np.take_along_axis(keys, indices, axis=axis)
        order = np.argsort(selected_keys, axis=axis, order=["value", "index"])
        if largest:
            order = np.flip(order, axis=axis)
        indices = np.take_along_axis(indices, order, axis=axis)

    return np.take_along_axis(arr, indices, axis=axis), indices
 
 
# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
 
def binary_search(arr, target, *, side: str = "left") -> int:
    """Insertion index of a single ``target`` in a sorted 1-D array.
 
    Implemented from scratch with an iterative loop.
 
    Parameters
    ----------
    arr : array_like, shape (n,)
        Sorted in ascending order (not checked, for O(log n) cost).
    target : scalar
    side : {"left", "right"}, default "left"
        "left": first index i with ``arr[i] >= target``.
        "right": first index i with ``arr[i] > target``.
 
    Returns
    -------
    int
        Index in ``[0, n]``.
 
    Raises
    ------
    ValueError
        If ``arr`` is not 1-D, ``target`` is not scalar, or ``side`` is invalid.
    TypeError
        If the target and array elements cannot be compared.
 
    Notes
    -----
    Time O(log n), space O(1).
    """
    arr = _as_1d(arr, name="arr")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if np.ndim(target) != 0:
        raise ValueError(f"target must be a scalar, got shape {np.shape(target)}")

    low, high = 0, arr.size
    target_is_nan = target != target
    while low < high:
        mid = (low + high) // 2
        value = arr[mid]
        # Match NumPy's placement of real NaNs after ordinary values.
        if value != value:
            move_right = target_is_nan and side == "right"
        elif target_is_nan:
            move_right = True
        else:
            move_right = value < target or (side == "right" and value == target)

        if move_right:
            low = mid + 1
        else:
            high = mid

    return low
 
 
def searchsorted(arr, values, *, side: str = "left"):
    """Vectorised insertion indices for many ``values`` in sorted ``arr``.
 
    Parameters
    ----------
    arr : array_like, shape (n,)
        Sorted ascending.
    values : array_like, shape (m,) or scalar
    side : {"left", "right"}, default "left"
 
    Returns
    -------
    ndarray of int, shape (m,) (or int if ``values`` is scalar)
 
    Raises
    ------
    ValueError
        If ``arr`` is not 1-D, ``values`` is neither scalar nor 1-D,
        or ``side`` is invalid.
    TypeError
        If the values and array elements cannot be ordered together.
 
    Notes
    -----
    Time O(m log n), space O(m).
    """
    arr = _as_1d(arr, name="arr")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    values = np.asarray(values)
    if values.ndim > 1:
        raise ValueError(f"values must be a scalar or 1-D, got shape {values.shape}")

    result = np.searchsorted(arr, values, side=side)
    return int(result) if values.ndim == 0 else result
 
