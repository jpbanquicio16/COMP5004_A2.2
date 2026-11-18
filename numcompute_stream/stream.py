"""
numcompute_stream.stream

Prequential ("test-then-train") training for streaming models.

Every chunk is first scored by the current model and only then used for
training, so every prediction is made on data the model has not seen:

>>> trainer = StreamTrainer(pipe, rolling_window=500)
>>> trainer.run(zip(X_chunks, y_chunks))
>>> trainer.history_["rolling_accuracy"]

The model needs partial_fit and predict (predict_proba for AUC). It can be a
DecisionTreeClassifier or a Pipeline ending in one. A retrain is detected
when the final model's n_retrains_ goes up. Until the first retrain the model
cannot predict, so those chunks score NaN (warm-up).

Progress is reported through the standard logging module (logger
"numcompute_stream.stream"); call logging.basicConfig(level=logging.INFO) to see it.
"""

import logging
import time
from collections import deque

import numpy as np

from .metrics import AUC, Accuracy

__all__ = ["StreamTrainer", "model_nbytes", "HISTORY_KEYS"]

logger = logging.getLogger(__name__)

#: columns of StreamTrainer.history_, in order
HISTORY_KEYS = (
    "chunk",                # chunk number, 0-based
    "n_samples",            # rows in the chunk
    "seen",                 # rows seen so far, this chunk included
    "accuracy",             # accuracy on this chunk (NaN in warm-up)
    "cumulative_accuracy",  # accuracy over every scored row so far
    "rolling_accuracy",     # accuracy over the last rolling_window scored rows
    "auc",                  # cumulative ROC AUC for pos_label (NaN until both classes)
    "retrained",            # did this chunk trigger a retrain?
    "fit_time",             # seconds spent in partial_fit for this chunk
    "retrain_time",         # fit_time if retrained else 0
    "buffer_count",         # rows waiting in the model's buffer after the chunk
    "memory_bytes",         # bytes held in the model's numpy arrays
)

_INT_KEYS = ("chunk", "n_samples", "seen", "buffer_count", "memory_bytes")


def model_nbytes(model):
    """Total .nbytes of every numpy array reachable from the model's attributes.

    Walks attributes, lists, tuples, dicts and nested objects (each object is
    counted once). Unlike tracemalloc it does not depend on the allocator, so
    the same model state always gives the same number.
    """
    seen = set()

    def walk(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, dict):
            return sum(walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple, deque, set, frozenset)):
            return sum(walk(v) for v in obj)
        if hasattr(obj, "__dict__") and not isinstance(obj, type) and not callable(obj):
            return walk(vars(obj))
        return 0

    return walk(model)


def _final_estimator(model):
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


class StreamTrainer:
    """Drive a streaming model chunk by chunk and record how it does.

    Parameters
    ----------
    model : object with partial_fit(X, y) and predict(X); predict_proba(X)
        and classes_ are used for AUC when present.
    rolling_window : int, default 500
        Rows behind rolling_accuracy (a ring buffer, see metrics.Accuracy).
    pos_label : default 1
        Positive class for AUC. The probability column is looked up in the
        model's classes_ on every chunk, because the classes can change
        between retrains (e.g. a buffer that held only one class).
    auc_bins : int, default 1000
    log_every : int, default 1
        Log every log_every-th chunk (retrains are always logged).

    Attributes
    ----------
    history_ : dict of numpy arrays, one entry per chunk (see HISTORY_KEYS)
    n_chunks_, n_seen_ : chunks / rows processed by step() or run()
    last_y_true_, last_y_pred_ : labels and predictions of the last scored chunk
    """

    def __init__(self, model, *, rolling_window=500, pos_label=1, auc_bins=1000,
                 log_every=1):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model needs a {method} method, got {type(model).__name__}")
        if isinstance(log_every, bool) or not isinstance(log_every, (int, np.integer)) \
                or log_every < 1:
            raise ValueError("log_every must be an int >= 1")
        self.model = model
        self.rolling_window = rolling_window
        self.pos_label = pos_label
        self.auc_bins = auc_bins
        self.log_every = int(log_every)
        self.reset()

    def reset(self):
        """Clear the history and metrics (the model itself is not touched)."""
        self._cumulative = Accuracy()
        self._rolling = Accuracy(window=self.rolling_window)
        self._auc = AUC(pos_label=self.pos_label, bins=self.auc_bins)
        self._rows = {key: [] for key in HISTORY_KEYS}
        self.n_chunks_ = 0
        self.n_seen_ = 0
        self.last_y_true_ = None
        self.last_y_pred_ = None
        self._last_fit_time = 0.0
        return self

    @property
    def final_estimator_(self):
        """The model itself, or the last step of a Pipeline."""
        return _final_estimator(self.model)

    @property
    def is_warm(self):
        """True once the model can predict (after the first retrain or a fit)."""
        return bool(getattr(self.final_estimator_, "is_fitted", True))

    # -- the two halves of a step ---------------------------------------------

    def score_chunk(self, X_chunk, y_chunk):
        """Accuracy of the current model on a chunk it has NOT trained on.

        Also updates the cumulative / rolling accuracy and AUC. Returns NaN
        during warm-up (no tree trained yet) or for an empty chunk.
        """
        y = np.asarray(y_chunk)
        if y.size == 0 or not self.is_warm:
            return float("nan")
        pred = np.asarray(self.model.predict(X_chunk))
        self._cumulative.update(y, pred)
        self._rolling.update(y, pred)
        self._update_auc(X_chunk, y)
        self.last_y_true_, self.last_y_pred_ = y, pred
        return float(np.mean(pred == y))

    def _update_auc(self, X_chunk, y):
        if not callable(getattr(self.model, "predict_proba", None)):
            return
        classes = getattr(self.final_estimator_, "classes_", None)
        if classes is None:
            return
        proba = np.asarray(self.model.predict_proba(X_chunk))
        match = np.flatnonzero(np.isin(np.asarray(classes), [self.pos_label]))
        # pos_label missing from this tree (one-class buffer): probability 0
        score = proba[:, match[0]] if match.size else np.zeros(len(y))
        self._auc.update(y, np.clip(score, 0.0, 1.0))

    def fit_chunk(self, X_chunk, y_chunk):
        """Train on a chunk. Returns True if it triggered a retrain
        (detected through the final model's n_retrains_)."""
        final = self.final_estimator_
        before = getattr(final, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        self._last_fit_time = time.perf_counter() - start
        return getattr(final, "n_retrains_", 0) > before

    # -- driving the stream ---------------------------------------------------

    def step(self, X_chunk, y_chunk):
        """Score the chunk, then train on it, then record a history row.
        Returns the row as a dict."""
        n = len(y_chunk)
        acc = self.score_chunk(X_chunk, y_chunk)
        retrained = self.fit_chunk(X_chunk, y_chunk)
        final = self.final_estimator_
        self.n_seen_ += n
        row = {
            "chunk": self.n_chunks_,
            "n_samples": n,
            "seen": self.n_seen_,
            "accuracy": acc,
            "cumulative_accuracy": self._cumulative.result(),
            "rolling_accuracy": self._rolling.result(),
            "auc": self._auc.result(),
            "retrained": retrained,
            "fit_time": self._last_fit_time,
            "retrain_time": self._last_fit_time if retrained else 0.0,
            "buffer_count": int(getattr(final, "buffer_count_", 0)),
            "memory_bytes": model_nbytes(self.model),
        }
        for key in HISTORY_KEYS:
            self._rows[key].append(row[key])
        self.n_chunks_ += 1
        self._log(row)
        return row

    def run(self, chunks, max_chunks=None):
        """Test-then-train over an iterable of (X_chunk, y_chunk). Returns self."""
        for i, (X_chunk, y_chunk) in enumerate(chunks):
            if max_chunks is not None and i >= max_chunks:
                break
            self.step(X_chunk, y_chunk)
        logger.info("done: %d chunks, %d rows, %d retrains, cumulative accuracy %.3f",
                    self.n_chunks_, self.n_seen_, int(np.sum(self._rows["retrained"])),
                    self._cumulative.result())
        return self

    def _log(self, row):
        if not (row["retrained"] or row["chunk"] % self.log_every == 0):
            return
        event = (f"retrain {row['retrain_time'] * 1e3:6.1f} ms" if row["retrained"]
                 else f"buffer {row['buffer_count']:>6d} rows")
        logger.info("chunk %4d | rows %7d | acc %s | rolling %s | cum %s | %s | mem %6.1f KB",
                    row["chunk"], row["seen"], _fmt(row["accuracy"]),
                    _fmt(row["rolling_accuracy"]), _fmt(row["cumulative_accuracy"]),
                    event, row["memory_bytes"] / 1024)

    # -- results --------------------------------------------------------------

    @property
    def history_(self):
        """dict of numpy arrays, one value per chunk, keys as HISTORY_KEYS."""
        out = {}
        for key in HISTORY_KEYS:
            values = self._rows[key]
            if key == "retrained":
                out[key] = np.array(values, dtype=bool)
            elif key in _INT_KEYS:
                out[key] = np.array(values, dtype=np.int64)
            else:
                out[key] = np.array(values, dtype=float)
        return out

    def to_csv(self, path):
        """Write history_ as CSV (header row, then one row per chunk).
        Readable back with numcompute_stream.io.load_csv(path, skip_header=1)."""
        hist = self.history_
        table = np.column_stack([hist[k].astype(float) for k in HISTORY_KEYS]) \
            if self.n_chunks_ else np.zeros((0, len(HISTORY_KEYS)))
        np.savetxt(path, table, delimiter=",", fmt="%.10g",
                   header=",".join(HISTORY_KEYS), comments="")
        return path


def _fmt(value):
    return "  -  " if np.isnan(value) else f"{value:.3f}"
