"""
numcompute.io

Loading and saving CSV files as numpy arrays.

- loaded data is always 2D even for 1 row/col
- missing values are NaN for float and "" for string
- uses numpy.genfromtxt, no python loops
- iter_csv_chunks is for big files that dont fit in memory
"""

from __future__ import annotations

import io as _stdio
import itertools
import warnings

import numpy as np

__all__ = [
    "load_csv",
    "iter_csv_chunks",
]

_NAN_POLICIES = ("keep", "omit", "fill")



# Private helpers


def _check_delimiter(delimiter) -> str:
    # delimiter must be non empty string
    if not isinstance(delimiter, str):
        raise TypeError(
            f"delimiter must be a str, got {type(delimiter).__name__}"
        )
    if delimiter == "":
        raise ValueError("delimiter must be a non-empty string")
    return delimiter


def _check_dtype(dtype) -> np.dtype:
    # only float and string dtypes allowed because int/bool can't hold NaN
    try:
        dt = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"dtype {dtype!r} is not a valid NumPy dtype") from exc
    if dt.kind not in ("f", "U", "S"):
        raise ValueError(
            f"dtype must be a float or string type (so missing values can "
            f"be represented), got {dt}"
        )
    return dt


def _check_usecols(usecols):
    # usecols must be None, one int, or a list of ints (0 or more).
    # numpy would give a confusing error otherwise
    if usecols is None:
        return None
    if isinstance(usecols, (int, np.integer)) and not isinstance(usecols, bool):
        if usecols < 0:
            raise ValueError(f"usecols must be >=0, got {usecols}")
        return usecols
    try:
        cols = list(usecols)
    except TypeError:
        raise TypeError("usecols must be None, int, or iterable of ints")
    for c in cols:
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
            raise TypeError(f"usecols must contain ints, got {c!r}")
        if c < 0:
            raise ValueError(f"usecols values must be >=0, got {c}")
    return cols


def _as_tokens(missing_values) -> list[str]:
    # convert missing_values to list of strings
    if missing_values is None:
        return []
    if isinstance(missing_values, str):
        return [missing_values]
    try:
        return [str(t) for t in missing_values]
    except TypeError as exc:
        raise TypeError(
            "missing_values must be None, a string, or an iterable of strings"
        ) from exc


def _missing_mask(data: np.ndarray, tokens: list[str]) -> np.ndarray:
    # True where value is missing
    if data.dtype.kind == "f":
        return np.isnan(data)
    # for string : empty string or any token in missing_values
    return np.isin(data, ["", *tokens])


def _validate_options(delimiter, skip_header, dtype, missing_values,
                      nan_policy, fill_value):
    # validate all options and return normalized values
    delimiter = _check_delimiter(delimiter)
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}"
        )
    if isinstance(skip_header, bool) or not isinstance(skip_header, (int, np.integer)):
        raise TypeError(
            f"skip_header must be an int, got {type(skip_header).__name__}"
        )
    if skip_header < 0:
        raise ValueError(f"skip_header must be >= 0, got {skip_header}")
    if nan_policy == "fill" and fill_value is None:
        raise ValueError('nan_policy="fill" requires a fill_value')
    dt = _check_dtype(dtype)
    tokens = _as_tokens(missing_values)
    # for float data the fill value has to be something that can become a number
    if nan_policy == "fill" and dt.kind == "f":
        try:
            float(fill_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"fill_value {fill_value!r} cannot be converted to float for dtype {dt}"
            )
    return delimiter, int(skip_header), dt, tokens


def _parse(source, *, delimiter, skip_header, usecols, dt, tokens,
           nan_policy, fill_value, encoding, allow_empty, where):
    # parse source and apply nan_policy
    kwargs = dict(
        delimiter=delimiter,
        skip_header=skip_header,
        usecols=usecols,
        dtype=dt,
        encoding=encoding,
        ndmin=2,
    )
    if dt.kind == "f":
        # empty fields and extra tokens both become NaN
        kwargs.update(missing_values=tokens or None, filling_values=np.nan)

    with warnings.catch_warnings():
        # genfromtxt warns on empty input, we handle it ourselves
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.genfromtxt(source, **kwargs)

    if data.size == 0:
        if allow_empty:
            return data
        raise ValueError(f"no data found in {where} (is the file empty?)")

    if nan_policy == "keep":
        return data

    mask = _missing_mask(data, tokens)

    if nan_policy == "omit":
        return data[~mask.any(axis=1)]

    # nan_policy == "fill"
    if dt.kind == "f":
        out = data.copy()
    else:
        fill = str(fill_value)
        # make dtype wide enough so fill value not cut
        width = max(data.dtype.itemsize // np.dtype(f"{dt.kind}1").itemsize,
                    len(fill))
        out = data.astype(f"{dt.kind}{width}")
        fill_value = fill
    out[mask] = fill_value
    return out



# Public API

def load_csv(
    path,
    *,
    delimiter: str = ",",
    skip_header: int = 0,
    usecols=None,
    dtype=float,
    missing_values=None,
    nan_policy: str = "keep",
    fill_value=None,
    encoding: str = "utf-8",
) -> np.ndarray:
    """
    Load csv file into 2D array.

    Params:
        path: file path to read
        delimiter: separator like , or tab, default ","
        skip_header: how many lines to skip from top, default 0
        usecols: which columns to read (ints, 0 or more), None means all
        dtype: float or string type, float by default. int not allowed because cant hold NaN
        missing_values: extra strings to treat as missing like "NA" or "?"
        nan_policy: what to do with missing - keep, omit, fill. default keep
        fill_value: value to use if nan_policy is fill, must give it then
        encoding: file encoding, default utf-8

    Returns:
        array: 2D numpy array with shape (n_samples, n_features)
        even if file has 1 row or 1 col it is still 2D. if omit, rows can be 0.

    Example:
        >>> load_csv("data.csv").shape
        (100, 5)
        >>> load_csv("data.csv", nan_policy="omit").shape
        (98, 5)  # 2 rows had missing

    Raises:
        ValueError: if delimiter empty, skip_header negative, bad nan_policy/dtype,
                    negative usecols, no fill_value for fill (or one that is not a
                    number for float data), empty file, or rows with different lengths
        TypeError: if delimiter not str, or usecols / missing_values wrong type
        FileNotFoundError: if file not found

    Time: O(n_samples * n_features) one pass
    Space: O(n_samples * n_features) whole file in memory
    """
    # load csv file into 2D numpy array
    delimiter, skip_header, dt, tokens = _validate_options(
        delimiter, skip_header, dtype, missing_values, nan_policy, fill_value
    )
    usecols = _check_usecols(usecols)
    return _parse(
        path, delimiter=delimiter, skip_header=skip_header, usecols=usecols,
        dt=dt, tokens=tokens, nan_policy=nan_policy, fill_value=fill_value,
        encoding=encoding, allow_empty=False, where=repr(path),
    )


def iter_csv_chunks(
    path,
    chunk_size: int,
    *,
    delimiter: str = ",",
    skip_header: int = 0,
    usecols=None,
    dtype=float,
    missing_values=None,
    nan_policy: str = "keep",
    fill_value=None,
    encoding: str = "utf-8",
):
    """
    Read csv file in chunks, for big files.

    Params:
        path: file path
        chunk_size: how many rows per chunk, must be >=1
        delimiter, skip_header, usecols, dtype, missing_values, nan_policy,
            fill_value, encoding: same as load_csv
        skip_header is only skipped once at start

    Yields:
        chunk: 2D array shape (rows_in_chunk, n_features).
        All chunks have same number of cols. If omit, chunk can be smaller than chunk_size.
        Empty chunks are skipped. vstack of all chunks == load_csv

    Example:
        >>> for chunk in iter_csv_chunks("big.csv", 1000):
        ...     process(chunk)

    Raises:
        ValueError: if chunk_size <1 or any option invalid
        TypeError: if chunk_size not int
        FileNotFoundError: if file not found. File check happens right away not on first chunk.

    Time: O(n_samples * n_features) total
    Space: O(chunk_size * n_features) only one chunk in memory
    """
    # read csv in chunks, for big files
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(
            f"chunk_size must be an int, got {type(chunk_size).__name__}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    delimiter, skip_header, dt, tokens = _validate_options(
        delimiter, skip_header, dtype, missing_values, nan_policy, fill_value
    )
    usecols = _check_usecols(usecols)
    # check file exists now, not later
    with open(path, encoding=encoding):
        pass
    chunk_size = int(chunk_size)

    def _blocks():
        with open(path, encoding=encoding) as fh:
            for _ in range(skip_header):
                next(fh, None)
            while True:
                lines = list(itertools.islice(fh, chunk_size))
                if not lines:
                    return
                block = _parse(
                    _stdio.StringIO("".join(lines)),
                    delimiter=delimiter, skip_header=0, usecols=usecols,
                    dt=dt, tokens=tokens, nan_policy=nan_policy,
                    fill_value=fill_value, encoding=encoding,
                    allow_empty=True, where=repr(path),
                )
                if block.size:
                    yield block

    return _blocks()
