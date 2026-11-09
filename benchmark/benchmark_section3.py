"""Section 3: reproducible loop vs NumPy statistics/metrics benchmarks.
"""
import argparse
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from numcompute import stats, metrics


def mean_loop(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def variance_loop(values):
    average = mean_loop(values)
    total = 0.0
    for value in values:
        total += (value - average) ** 2
    return total / len(values)


def histogram_loop(values, edges):
    counts = np.zeros(len(edges) - 1, dtype=int)
    for value in values:
        for i in range(len(counts)):
            if edges[i] <= value < edges[i + 1] or (i == len(counts)-1 and value == edges[-1]):
                counts[i] += 1
                break
    return counts


def accuracy_loop(actual, predicted):
    correct = 0
    for a, p in zip(actual, predicted):
        correct += int(a == p)
    return correct / len(actual)


def mse_loop(actual, predicted):
    total = 0.0
    for a, p in zip(actual, predicted):
        total += (a - p) ** 2
    return total / len(actual)


def confusion_loop(actual, predicted):
    counts = np.zeros((2, 2), dtype=int)
    for a, p in zip(actual, predicted):
        counts[a, p] += 1
    return counts


def best_time(function, repeats):
    """Warm up once, then return the smallest elapsed time in seconds."""
    function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return min(timings)


def run_benchmarks(n_rows=20000, repeats=5, seed=0):
    """Print comparisons and return rows (name, loop seconds, NumPy seconds).

    Both versions receive identical inputs and repeat counts. Result equality
    is checked before timing. Data generation is excluded from the timings.
    Python loops are intentional reference baselines, not core implementations.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 2:
        raise ValueError('n_rows must be an integer >= 2')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be an integer >= 1')
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n_rows)
    actual = rng.integers(0, 2, size=n_rows)
    predicted = rng.integers(0, 2, size=n_rows)
    targets = rng.normal(70, 10, size=n_rows)
    estimates = targets + rng.normal(0, 5, size=n_rows)
    edges = np.linspace(-4, 4, 17)
    cases = [
        ('Mean', lambda: mean_loop(values), lambda: stats.mean(values)),
        ('Variance', lambda: variance_loop(values), lambda: stats.var(values)),
        ('Histogram', lambda: histogram_loop(values, edges), lambda: stats.histogram(values, bins=edges)[0]),
        ('Accuracy', lambda: accuracy_loop(actual, predicted), lambda: metrics.accuracy(actual, predicted)),
        ('MSE', lambda: mse_loop(targets, estimates), lambda: metrics.mse(targets, estimates)),
        ('Confusion matrix', lambda: confusion_loop(actual, predicted),
         lambda: metrics.confusion_matrix(actual, predicted, labels=[0, 1])),
    ]
    print(f'Section 3: stats and metrics | rows={n_rows} | seed={seed} | repeats={repeats}')
    print('Timing: best of repeats after one warm-up; same repeats for both versions.')
    print(f"{'Component':<20} {'Loop (s)':>12} {'NumPy (s)':>12} {'Speed-up':>12}")
    print('-' * 60)
    rows = []
    for name, loop_fn, numpy_fn in cases:
        np.testing.assert_allclose(loop_fn(), numpy_fn(), rtol=1e-10, atol=1e-12,
                                   err_msg=f'{name}: results differ')
        loop_time = best_time(loop_fn, repeats)
        numpy_time = best_time(numpy_fn, repeats)
        rows.append((name, loop_time, numpy_time))
        ratio = loop_time / numpy_time if numpy_time else float('inf')
        print(f'{name:<20} {loop_time:>12.6f} {numpy_time:>12.6f} {ratio:>11.2f}x')
    print('\nAll result comparisons passed.')
    print('A speed-up below 1 means NumPy was slower in this run; timings vary by machine.')
    print('Environment:', platform.platform())
    print('Processor:', platform.processor() or 'not reported')
    print(f'Python {platform.python_version()} | NumPy {np.__version__}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Section 3: stats and metrics benchmarks')
    parser.add_argument('--n-rows', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.n_rows < 2 or args.repeats < 1:
        parser.error('--n-rows must be >= 2 and --repeats must be >= 1')
    run_benchmarks(args.n_rows, args.repeats, args.seed)


if __name__ == '__main__':
    main()
