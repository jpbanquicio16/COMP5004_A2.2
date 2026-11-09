# section 1 benchmark - loop vs vectorised
# testing my numcompute.preprocessing stuff
# run from root: python benchmark/benchmark_section1.py
import argparse
import pathlib
import platform
import sys
import time

import numpy as np

# add project root to path so we can import numcompute
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from numcompute.preprocessing import (  
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)

def measure_time(func, repeats):
    # takes best time out of repeats
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


# loop versions -  for comparison

def standardize_loop(A):
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        col = A[:, j]
        # mean
        mean = 0.0
        for v in col:
            mean += v
        mean = mean / len(col)
        # std
        var = 0.0
        for v in col:
            var += (v - mean) ** 2
        std = (var / len(col)) ** 0.5
        for i in range(len(col)):
            out[i, j] = (col[i] - mean) / std
    return out


def minmax_loop(A):
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        col = A[:, j]
        lo = col[0]
        hi = col[0]
        for v in col:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        for i in range(len(col)):
            out[i, j] = (col[i] - lo) / (hi - lo)
    return out


def impute_mean_loop(A):
    out = A.copy()
    for j in range(A.shape[1]):
        total = 0.0
        count = 0
        for v in A[:, j]:
            if v == v:  # NaN check
                total += v
                count += 1
        mean = total / count if count != 0 else 0
        for i in range(A.shape[0]):
            if A[i, j] != A[i, j]:
                out[i, j] = mean
    return out


def onehot_loop(A):
    categories = [np.unique(A[:, j]) for j in range(A.shape[1])]
    total_cols = sum(len(c) for c in categories)
    out = np.zeros((A.shape[0], total_cols))
    for i in range(A.shape[0]):
        offset = 0
        for j, cats in enumerate(categories):
            idx = int(np.searchsorted(cats, A[i, j]))
            out[i, offset + idx] = 1.0
            offset += len(cats)
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark Section 1")
    parser.add_argument("--n-rows", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    numeric = rng.normal(size=(args.n_rows, 10))
    with_nan = numeric.copy()
    with_nan[rng.random(with_nan.shape) < 0.05] = np.nan
    categorical = rng.integers(0, 20, size=(args.n_rows, 5))

    cases = [
        ("StandardScaler", numeric, standardize_loop,
         lambda A: StandardScaler().fit_transform(A), np.allclose),
        ("MinMaxScaler", numeric, minmax_loop,
         lambda A: MinMaxScaler().fit_transform(A), np.allclose),
        ("SimpleImputer(mean)", with_nan, impute_mean_loop,
         lambda A: SimpleImputer("mean").fit_transform(A), np.allclose),
        ("OneHotEncoder", categorical, onehot_loop,
         lambda A: OneHotEncoder().fit_transform(A), np.array_equal),
    ]

    print(f"Data: {args.n_rows} rows (10 numeric / 5 categorical), seed 0\n")
    print(f"{'Component':<20} {'Loop (s)':<10} {'Vectorised (s)':<15} {'Speed-up'}")
    print("-" * 60)

    for name, data, loop_fn, vec_fn, check_fn in cases:
        loop_result = loop_fn(data)
        vec_result = vec_fn(data)
        if not check_fn(loop_result, vec_result):
            raise AssertionError(f"{name} results don't match")

        t_loop = measure_time(lambda: loop_fn(data), 3)
        t_vec = measure_time(lambda: vec_fn(data), args.repeats)
        speedup = t_loop / t_vec if t_vec != 0 else 0
        print(f"{name:<20} {t_loop:<10.4f} {t_vec:<15.4f} {speedup:.0f}x")

    print()
    print("Environment:", platform.platform())
    print(f"Python {platform.python_version()} | NumPy {np.__version__}")

if __name__ == "__main__":
    main()
