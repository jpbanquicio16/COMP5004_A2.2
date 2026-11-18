# NumCompute Stream

A NumPy-only toolkit for **streaming machine learning**, built around a from-scratch
decision tree that retrains on a buffer. Every component (preprocessing, statistics,
metrics, model, pipeline) accepts data chunk by chunk through `.partial_fit()` or `.update()`.
Only NumPy and matplotlib are used.

## Setup

```bash
git clone https://github.com/jpbanquicio16/COMP5004_A2.2.git
cd COMP5004_A2.2
git checkout jp-individual
pip install numpy matplotlib pytest jupyter    # jupyter only needed for the demo notebook
```

Python 3.10+ is recommended (developed on 3.12). Run everything from the repository root.

## Layout

```
numcompute_stream/
  io.py             load_csv, iter_csv_chunks (CSV -> arrays, NaN handling, chunked reading)
  preprocessing.py  StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder (all partial_fit)
  stats.py          Welford running mean/var, StreamingHistogram, update_stats, quantiles
  metrics.py        streaming Accuracy / Precision / Recall / F1 / ConfusionMatrix / AUC, rolling windows
  tree.py           DecisionTreeClassifier with buffered retraining
  pipeline.py       Pipeline with partial_fit / transform / predict
  stream.py         StreamTrainer: prequential (test-then-train) loop with per-chunk logging
  visualise.py      plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth
  sort_search.py, rank.py, optim.py, utils.py   supporting modules from earlier assignments
tests/              pytest suite (1,100+ tests, including streaming and edge cases)
benchmark/          loop vs vectorised and buffer_size benchmarks
demo/               stream_demo.ipynb and its dataset stream_data.csv
```

## Quick start

```python
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
for X_chunk, y_chunk in stream:
    pipe.partial_fit(X_chunk, y_chunk)   # buffers; retrains once >= 500 rows are held
    y_pred = pipe.predict(X_new)         # uses the most recently trained tree
```

`DecisionTreeClassifier(max_depth, min_samples_split, max_features, buffer_size, criterion="gini"|"entropy")`
also has `.fit(X, y)` for ordinary batch training. Until the first retrain `.predict()`
raises `NotFittedError`.

### Prequential evaluation and logging

```python
import logging
from numcompute_stream.io import iter_csv_chunks
from numcompute_stream.stream import StreamTrainer
from numcompute_stream import visualise as viz

logging.basicConfig(level=logging.INFO, format="%(message)s")
trainer = StreamTrainer(pipe, rolling_window=500, log_every=10)
trainer.run((c[:, :-1], c[:, -1].astype(int))
            for c in iter_csv_chunks("demo/stream_data.csv", 100, skip_header=1))

h = trainer.history_    # per-chunk arrays: accuracy, rolling_accuracy, auc, fit_time, memory_bytes, ...
viz.plot_metric_over_time(h["rolling_accuracy"], "Rolling accuracy", "accuracy",
                          markers=h["retrained"].astype(bool), save_path="acc.png")
trainer.to_csv("history.csv")
```

Each chunk is scored by the current model first and used for training afterwards. Chunks
seen before the first retrain have no prediction and score NaN (warm-up).

### Plots (`visualise.py`)

| Function | Purpose |
|---|---|
| `plot_metric_over_time(values, title, ylabel, markers=...)` | one metric per chunk, optional retrain markers |
| `compare_models(m1, m2, labels)` | two runs on one axis, e.g. different `buffer_size` |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | latest chunk, mistakes highlighted |

All accept `save_path="file.png"` to save and `show=True` to display inline.

## Running things

```bash
python -m pytest -q                       # full test suite
jupyter notebook demo/stream_demo.ipynb   # streaming demo (CSV -> chunks -> partial_fit -> plots)
python benchmark/tree_benchmarks.py       # tree: (a) loop vs vectorised, (b) buffer_size sweep
python benchmark/tree_benchmarks.py --repeats 5 --markdown benchmark/tree_results.md
```

The other benchmark scripts (`benchmark_section1.py` ... `section4_benchmarks.py`) compare
loop-based and vectorised versions of the earlier modules; run them the same way.

The demo notebook trains on `demo/stream_data.csv` (6,000 rows, 4 features, 3% missing values,
5% label noise, and a concept drift at the midpoint) and compares `buffer_size` of 100, 500 and 2000.
The `buffer_size` tradeoff in short: a small buffer adapts fast after drift but retrains often;
a large buffer is cheaper overall but stays wrong longer after the data changes
(see `benchmark/tree_results.md`).

## Design notes

- **Vectorised splits:** the tree finds the best split with sorted columns and cumulative class counts, with no per-threshold Python loop.
- **Numerical stability:** NaN-aware splits (missing rows are routed to the better side), deterministic tie-breaking, Welford updates for running variance, zero-variance columns handled by the scaler.
- **Errors:** invalid shapes and types raise `ValueError` / `TypeError` with a message naming the offending argument.
