"""
numcompute_stream.stats

Descriptive statistics and a streaming mean/variance, numpy only.

- data must be numbers, inf is not allowed, empty input raises ValueError
- nan_policy: 'propagate' (NaN in -> NaN out), 'omit' (skip NaN) or 'raise'
- axis=None reduces everything, keepdims=True keeps reduced axes as length 1
"""

"""
### `stats.py`
- All statistical functions redesigned for streaming
- Implement chunk-based versions of:
- Mean, variance, quantiles
- Histograms (optionally with sliding window)
- Add `up
"""
import warnings
from collections import deque

import numpy as np

__all__ = ['mean', 'median', 'var', 'variance', 'std', 'min', 'max',
           'histogram', 'quantile', 'Welford', 'StreamingStats',
           'StreamingHistogram', 'update_stats']


def _check_data(data, nan_policy):
    # check the input and return it as a float array
    if nan_policy not in ('propagate', 'omit', 'raise'):
        raise ValueError("nan_policy must be 'propagate', 'omit' or 'raise'")
    arr = np.asarray(data)
    if arr.dtype.kind not in 'biuf':
        raise TypeError('data must contain real numbers')
    arr = arr.astype(float)
    if arr.size == 0:
        raise ValueError('data must not be empty')
    if np.isinf(arr).any():
        raise ValueError('data must not contain infinity')
    if nan_policy == 'raise' and np.isnan(arr).any():
        raise ValueError('data contains NaN')
    return arr


def _check_ddof(ddof):
    # True/False count as ints in python, so check for bool first
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError('ddof must be a non-negative integer')
    if ddof < 0:
        raise ValueError('ddof must be non-negative')


def _apply(name, data, axis, nan_policy, keepdims, **kwargs):
    # run a numpy function, using the nan version (np.nanmean etc.) for 'omit'
    arr = _check_data(data, nan_policy)
    func = getattr(np, ('nan' if nan_policy == 'omit' else '') + name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slice -> NaN is fine
        return func(arr, axis=axis, keepdims=keepdims, **kwargs)


def _spread(name, data, axis, ddof, nan_policy, keepdims):
    # var and std: same as numpy, but NaN when there are too few values
    _check_ddof(ddof)
    result = _apply(name, data, axis, nan_policy, keepdims, ddof=ddof)
    arr = np.asarray(data, dtype=float)
    valid = ~np.isnan(arr) if nan_policy == 'omit' else np.ones(arr.shape, dtype=bool)
    count = valid.sum(axis=axis, keepdims=keepdims)
    return np.where(count > ddof, result, np.nan)


def mean(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Arithmetic mean along axis. Raises TypeError for non-numbers and
    ValueError for empty/inf data, a bad policy, or NaN with 'raise'."""
    return _apply('mean', data, axis, nan_policy, keepdims)


def median(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Median along axis (middle two are averaged for even counts)."""
    return _apply('median', data, axis, nan_policy, keepdims)


def var(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Variance, divided by (count - ddof). ddof=1 gives the sample variance.
    Returns NaN when count <= ddof."""
    return _spread('var', data, axis, ddof, nan_policy, keepdims)


variance = var


def std(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Standard deviation, same ddof rules as var()."""
    return _spread('std', data, axis, ddof, nan_policy, keepdims)


def min(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Minimum along axis."""
    return _apply('min', data, axis, nan_policy, keepdims)


def max(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Maximum along axis."""
    return _apply('max', data, axis, nan_policy, keepdims)


def quantile(data, q, axis=None, *, interpolation='linear', nan_policy='omit',
             keepdims=False):
    """Quantiles of data. q is a fraction between 0 and 1 (not 0-100),
    either a single number or a 1-D array.

    interpolation: 'linear', 'lower', 'higher' or 'midpoint'.
    The result has shape q.shape + the reduced shape of data.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or q.size == 0 or not np.all((q >= 0) & (q <= 1)):
        raise ValueError('q must be a nonempty scalar or 1-D array in [0, 1]')
    if interpolation not in ('linear', 'lower', 'higher', 'midpoint'):
        raise ValueError('unsupported interpolation method')
    return _apply('quantile', data, axis, nan_policy, keepdims,
                  q=q, method=interpolation)


def histogram(data, bins=10, range=None, *, nan_policy='omit', density=False):
    """Count values in bins. Data is flattened, NaN is skipped by default.

    bins: a positive int, or a 1-D array of strictly increasing edges.
    range: (low, high), only used when bins is an int.
    density=True scales the counts so the histogram integrates to 1.
    The last bin includes its right edge. Returns (counts, edges).
    """
    if nan_policy not in ('omit', 'raise'):
        raise ValueError("histogram nan_policy must be 'omit' or 'raise'")
    arr = _check_data(data, nan_policy).ravel()
    arr = arr[~np.isnan(arr)]

    if np.ndim(bins) == 0:
        # number of bins
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer or increasing edges')
        if arr.size == 0 and range is None:
            raise ValueError('all-NaN data needs explicit edges or a range')
    else:
        # explicit edges
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all() \
                or not np.all(np.diff(bins) > 0):
            raise ValueError('bin edges must be finite and strictly increasing')

    if range is not None:
        r = np.asarray(range, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all() or r[0] >= r[1]:
            raise ValueError('range must be a finite (low, high) with low < high')

    counts, edges = np.histogram(arr, bins=bins, range=range)
    if density:
        if counts.sum() == 0:
            raise ValueError('density is undefined when no values fall in the bins')
        counts = counts / counts.sum() / np.diff(edges)
    return counts, edges




# ---------------------------------------------------------------------------
# streaming statistics
# ---------------------------------------------------------------------------

def _as_chunk(values, nan_policy, ndim, n_features):
    """Validate one streaming update and return it as a float (n, F) array.

    A number or 1-D array is a batch of samples of ONE variable (one column).
    A 2-D array is (n_samples, n_features). ndim / n_features are the layout
    fixed by the first update (None before that). Returns None for an empty
    update, which leaves the state unchanged.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in 'biuf':
        raise TypeError('values must contain real numbers')
    if arr.ndim > 2:
        raise ValueError('values must be a scalar, 1-D or 2-D')
    kind = 2 if arr.ndim == 2 else 1
    if ndim is not None and kind != ndim:
        raise ValueError(f'expected {ndim}-D input (fixed by the first update), '
                         f'got shape {arr.shape}')
    arr = arr.reshape(-1, 1) if kind == 1 else arr
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f'chunk has {arr.shape[1]} columns, expected {n_features}')
    if arr.size == 0:
        return None
    return _check_data(arr, nan_policy)


def _column_extremes(arr, valid):
    # per-column min / max of the valid entries, NaN for a column with none
    has = valid.any(axis=0)
    lo = np.where(valid, arr, np.inf).min(axis=0)
    hi = np.where(valid, arr, -np.inf).max(axis=0)
    return np.where(has, lo, np.nan), np.where(has, hi, np.nan)


def _pow2_scale(a):
    # power of two s with s <= a < 2s (0 where a == 0). Dividing by a power of
    # two is exact, so rescaling never loses precision; it only prevents
    # overflow / underflow when squaring huge or tiny values.
    _, e = np.frexp(a)
    return np.where(a > 0, np.ldexp(1.0, e - 1), 0.0)


def _check_bins(bins):
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) \
            or bins < 2 or bins % 2:
        raise ValueError('bins must be an even integer >= 2 (bins are merged in pairs)')
    return int(bins)


class Welford:
    """Running mean and variance without keeping the data (Welford's method),
    one column or many.

    update() takes a number, a 1-D batch (samples of one variable) or a 2-D
    chunk (n_samples, n_features); the first non-empty update fixes which.
    1-D streams give floats, 2-D streams give one value per column.
    Each batch is merged in one vectorised step (Chan's parallel formula):
        total = count + n
        delta = batch_mean - mean
        M2   += batch_M2 + delta**2 * count * n / total
        mean += delta * n / total
    NaN is skipped column by column, so a column that is all NaN in one batch
    simply does not move.

    nan_policy: 'omit' (default) skips NaN, 'raise' throws ValueError.
    bins: None, or an even int to also keep a StreamingHistogram so that
    quantile() and histogram() work.
    If the input is rejected nothing is changed.
    """

    def __init__(self, nan_policy='omit', *, bins=None):
        if nan_policy not in ('omit', 'raise'):
            raise ValueError("nan_policy must be 'omit' or 'raise'")
        self.nan_policy = nan_policy
        self.bins = None if bins is None else _check_bins(bins)
        self.reset()

    def reset(self):
        """Forget everything seen so far. Returns self."""
        self._ndim = None
        self._count = None      # values seen per column
        self._shift = None      # reference value K per column (its first value)
        self._mean = None       # mean - K per column
        self._m2 = None         # sum of squared differences from the mean / scale**2
        self._scale = None      # power of two >= max |x - K| seen, 0 if none
        self._min = None
        self._max = None
        self.hist_ = None if self.bins is None else StreamingHistogram(self.bins)
        return self

    @property
    def n_features_in_(self):
        return None if self._count is None else self._count.size

    def update(self, values):
        """Add a number, a 1-D batch or a 2-D chunk. Returns self so calls can
        be chained. If the input is rejected nothing is changed."""
        arr = _as_chunk(values, self.nan_policy, self._ndim, self.n_features_in_)
        if arr is None:
            return self
        if self._ndim is None:
            f = arr.shape[1]
            self._ndim = 2 if np.ndim(values) == 2 else 1
            self._count = np.zeros(f, dtype=np.int64)
            self._shift, self._mean = np.zeros(f), np.zeros(f)
            self._m2, self._scale = np.zeros(f), np.zeros(f)
            self._min, self._max = np.full(f, np.nan), np.full(f, np.nan)

        valid = ~np.isnan(arr)
        n = valid.sum(axis=0)
        batch_min, batch_max = _column_extremes(arr, valid)
        self._min, self._max = np.fmin(self._min, batch_min), np.fmax(self._max, batch_max)
        if self.hist_ is not None:
            self.hist_.update(arr if self._ndim == 2 else arr[:, 0])

        # Shifted data: every column is measured from a fixed reference K (its
        # first value), and the mean is stored as mean - K, a small number with
        # full precision. So delta stays exact even for 1e9 + N(0, 1).
        first = arr[np.argmax(valid, axis=0), np.arange(arr.shape[1])]
        self._shift = np.where((n > 0) & (self._count == 0), first, self._shift)
        dev = np.where(valid, arr - self._shift, 0.0)

        # Work in units of a power of two (exact) so squares of huge or tiny
        # values neither overflow nor underflow; rescale the old M2 to match.
        scale = np.maximum(self._scale, _pow2_scale(np.abs(dev).max(axis=0)))
        safe = np.where(scale > 0, scale, 1.0)
        ratio = np.where(scale > 0, self._scale / safe, 1.0)
        m2 = self._m2 * ratio * ratio
        mean = self._mean / safe
        dev = dev / safe

        # merge the batch into the running totals (a column with n == 0 keeps
        # its values: every term below is multiplied by n)
        batch_mean = dev.sum(axis=0) / np.maximum(n, 1)
        batch_m2 = np.sum(np.where(valid, dev - batch_mean, 0.0) ** 2, axis=0)
        total = self._count + n
        safe_total = np.maximum(total, 1)
        delta = batch_mean - mean
        self._m2 = m2 + batch_m2 + delta ** 2 * self._count * n / safe_total
        self._mean = (mean + delta * n / safe_total) * safe
        self._scale = scale
        self._count = total
        return self

    update_stats = update

    # -- results -------------------------------------------------------------

    def _out(self, arr):
        # floats for 1-D streams, (F,) arrays for 2-D streams
        if self._ndim is None:
            return float('nan')
        return float(arr[0]) if self._ndim == 1 else arr

    @property
    def count(self):
        """Number of non-NaN values seen (per column for 2-D streams)."""
        if self._ndim is None:
            return 0
        return int(self._count[0]) if self._ndim == 1 else self._count.copy()

    @property
    def mean(self):
        """Current mean, NaN if nothing was added yet."""
        if self._ndim is None:
            return float('nan')
        return self._out(np.where(self._count > 0, self._shift + self._mean, np.nan))

    def _spread(self, ddof):
        # M2 / (count - ddof) in scaled units, and where it is defined
        _check_ddof(ddof)
        if self._ndim is None:
            return None
        ok = self._count > ddof
        return ok, self._m2 / np.where(ok, self._count - ddof, 1)

    def variance(self, ddof=0):
        """Variance, NaN when count <= ddof."""
        res = self._spread(ddof)
        if res is None:
            return float('nan')
        ok, v = res
        with np.errstate(over='ignore'):          # a variance can exceed float range
            return self._out(np.where(ok, v * self._scale * self._scale, np.nan))

    def std(self, ddof=0):
        """Standard deviation, NaN when count <= ddof (no overflow for huge data)."""
        res = self._spread(ddof)
        if res is None:
            return float('nan')
        ok, v = res
        return self._out(np.where(ok, np.sqrt(v) * self._scale, np.nan))

    @property
    def min(self):
        """Running minimum (np.fmin), NaN where nothing has been seen."""
        return self._out(self._min) if self._ndim else float('nan')

    @property
    def max(self):
        """Running maximum (np.fmax), NaN where nothing has been seen."""
        return self._out(self._max) if self._ndim else float('nan')

    def _need_hist(self):
        if self.hist_ is None:
            raise ValueError('create the object with bins=... to track quantiles')
        return self.hist_

    def quantile(self, q):
        """Approximate quantiles from the histogram (needs bins=...)."""
        return self._need_hist().quantile(q)

    def histogram(self):
        """(counts, edges) of the histogram (needs bins=...)."""
        return self._need_hist().histogram()


StreamingStats = Welford


def update_stats(X_chunk, state=None, **options):
    """Functional form of StreamingStats.update_stats.

    >>> state = None
    >>> for chunk in stream:
    ...     state = update_stats(chunk, state)
    >>> state.mean, state.std()

    options (nan_policy, bins) are only used when a new state is created.
    """
    if state is None:
        state = StreamingStats(**options)
    elif options:
        raise TypeError('options can only be given when state is None')
    return state.update_stats(X_chunk)


def _merge_pairs(counts, up, down):
    # Double the range of the columns flagged up/down. counts is (..., F, B).
    # Neighbouring bins (0,1), (2,3), ... are added together; growing up puts
    # them in the lower half, growing down puts them in the upper half.
    merged = counts[..., 0::2] + counts[..., 1::2]
    empty = np.zeros_like(merged)
    grown_up = np.concatenate([merged, empty], axis=-1)
    grown_down = np.concatenate([empty, merged], axis=-1)
    return np.where(up[:, None], grown_up, np.where(down[:, None], grown_down, counts))


class StreamingHistogram:
    """Fixed-bin histogram per column for streaming data, with quantiles.

    The first chunk that has data in a column sets that column's range to
    [min, max] (a constant chunk gets a unit-width range around its value, so
    nothing is divided by zero). Later values outside the range double it,
    merging bins in pairs, as often as needed. Memory stays O(F * bins).

    quantile() walks the cumulative counts to the bin holding the target rank
    and interpolates linearly inside it. The answer lies in the same bin as the
    exact sample quantile, so the error is at most one bin width
    (bin_width_, which is current range / bins).

    Parameters
    ----------
    bins : even int, default 256.
    window : None (whole stream) or int W >= 1: only the last W chunks count.
        Each chunk's histogram is kept in a deque; the oldest is subtracted
        when it drops out.
    """

    def __init__(self, bins=256, window=None):
        self.bins = _check_bins(bins)
        if window is not None and (isinstance(window, (bool, np.bool_))
                                   or not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError('window must be None or an int >= 1')
        self.window = None if window is None else int(window)
        self.reset()

    def reset(self):
        """Forget everything seen so far. Returns self."""
        self._ndim = None
        self._lo = None          # (F,) left edge of bin 0, NaN until data arrives
        self._width = None       # (F,) bin width
        self.counts_ = None      # (F, bins) int64
        self._min = None
        self._max = None
        self._chunks = deque()   # per-chunk (counts, min, max) when windowed
        return self

    @property
    def n_features_in_(self):
        return None if self._lo is None else self._lo.size

    @property
    def bin_width_(self):
        """Current bin width per column (the quantile error bound)."""
        return self._out(self._width)

    def _out(self, arr):
        # 1-D streams give plain numbers (or q-shaped arrays), like StreamingStats
        if self._ndim is None:
            return float('nan')
        if self._ndim == 2:
            return arr
        out = arr[..., 0]
        return out.item() if out.ndim == 0 else out

    def update(self, values):
        """Add a chunk (number, 1-D samples or 2-D (n, F)). Returns self."""
        arr = _as_chunk(values, 'omit', self._ndim, self.n_features_in_)
        if arr is None:
            return self
        f, b = arr.shape[1], self.bins
        if self._ndim is None:
            self._ndim = 2 if np.ndim(values) == 2 else 1
            self._lo, self._width = np.full(f, np.nan), np.full(f, np.nan)
            self.counts_ = np.zeros((f, b), dtype=np.int64)
            self._min, self._max = np.full(f, np.nan), np.full(f, np.nan)

        valid = ~np.isnan(arr)
        has = valid.any(axis=0)
        cmin, cmax = _column_extremes(arr, valid)

        # first data in a column fixes its range (unit width if constant)
        new = has & np.isnan(self._lo)
        span = cmax - cmin
        flat = ~(span > 0)
        self._lo = np.where(new, np.where(flat, cmin - 0.5, cmin), self._lo)
        self._width = np.where(new, np.where(flat, 1.0, span) / b, self._width)

        # double the range until the chunk fits (a few passes at most per
        # chunk; each pass is vectorised over columns)
        while True:
            hi = self._lo + b * self._width
            down = has & (cmin < self._lo)
            up = has & ~down & (cmax > hi)
            if not (down.any() or up.any()):
                break
            self.counts_ = _merge_pairs(self.counts_, up, down)
            if self._chunks:
                old = _merge_pairs(np.stack([c for c, _, _ in self._chunks]), up, down)
                self._chunks = deque((c, lo, mx) for c, (_, lo, mx) in zip(old, self._chunks))
            self._lo = np.where(down, self._lo - b * self._width, self._lo)
            self._width = np.where(up | down, 2.0 * self._width, self._width)

        # count the chunk: one bincount over (column, bin) keys
        idx = np.floor((np.where(valid, arr, 0.0) - np.where(has, self._lo, 0.0))
                       / np.where(has, self._width, 1.0))
        idx = np.clip(np.nan_to_num(idx), 0, b - 1).astype(np.intp)
        keys = (np.arange(f) * b + idx)[valid]
        chunk = np.bincount(keys, minlength=f * b).reshape(f, b)
        self.counts_ = self.counts_ + chunk

        if self.window is None:
            self._min, self._max = np.fmin(self._min, cmin), np.fmax(self._max, cmax)
        else:
            self._chunks.append((chunk, cmin, cmax))
            if len(self._chunks) > self.window:
                self.counts_ = self.counts_ - self._chunks.popleft()[0]
            mins = np.stack([lo for _, lo, _ in self._chunks])
            maxs = np.stack([hi for _, _, hi in self._chunks])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN column
                self._min, self._max = np.nanmin(mins, axis=0), np.nanmax(maxs, axis=0)
        return self

    update_stats = update

    @property
    def count(self):
        """Values currently counted (per column for 2-D streams)."""
        if self._ndim is None:
            return 0
        return self._out(self.counts_.sum(axis=1))

    def histogram(self):
        """(counts, edges): counts (F, bins), edges (F, bins + 1); 1-D streams
        give (bins,) and (bins + 1,)."""
        if self._ndim is None:
            raise ValueError('no data has been added yet')
        edges = self._lo[:, None] + np.arange(self.bins + 1) * self._width[:, None]
        if self._ndim == 1:
            return self.counts_[0].copy(), edges[0]
        return self.counts_.copy(), edges

    def quantile(self, q):
        """Approximate quantiles, q in [0, 1] (scalar or 1-D).

        Result shape: q.shape + (F,) for 2-D streams, q.shape for 1-D streams.
        NaN for a column with no data. q=0 and q=1 give the exact min / max.
        """
        q = np.asarray(q, dtype=float)
        if q.ndim > 1 or q.size == 0 or not np.all((q >= 0) & (q <= 1)):
            raise ValueError('q must be a nonempty scalar or 1-D array in [0, 1]')
        if self._ndim is None:
            raise ValueError('no data has been added yet')
        counts = self.counts_.astype(float)                    # (F, B)
        total = counts.sum(axis=1)                             # (F,)
        cum = np.cumsum(counts, axis=1)
        target = q.reshape(-1, 1) * total                      # (Q, F)
        # first bin whose cumulative count reaches the target rank
        j = np.minimum((cum[None] < target[..., None]).sum(axis=-1), self.bins - 1)
        cols = np.arange(counts.shape[0])
        c = counts[cols, j]
        before = cum[cols, j] - c
        inside = np.divide(target - before, c, out=np.zeros_like(target), where=c > 0)
        est = self._lo + (j + inside) * self._width
        est = np.minimum(np.maximum(est, self._min), self._max)   # q=0/1 exact
        est = np.where(total > 0, est, np.nan)
        est = est.reshape(q.shape + (counts.shape[0],))
        return self._out(est)
