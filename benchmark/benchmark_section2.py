"""Reproducible sort/search/rank comparisons; run from any working directory.

    python benchmark/benchmark_section2.py --sizes 100 1000 --repeats 5
    python benchmark/benchmark_section2.py --output benchmark/results/section2

Only NumPy is required. See benchmark/README.md for methodology and limitations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numcompute.rank as ranking
import numcompute.sort_search as ss

DISTRIBUTIONS = ("random", "ascending", "descending", "duplicates", "constant")
METHODS = ("average", "min", "max", "dense", "ordinal")
TIMING_NOTES = (
    "Seconds per call; median, IQR and minimum of equally many samples. "
    "Each side is independently calibrated to the target batch duration (at most "
    "10000 calls); samples alternate baseline/custom order. Warm-ups and calibration "
    "are discarded. GC retains its initial setting. Data generation, correctness "
    "checks and search input sorting are excluded. Public API validation, internal "
    "allocation/conversion and percentile_rank sorting are included. Quickselect "
    "receives the same integer pivot seed on every call, including RNG construction. "
    "Ratio = baseline median / custom median; below 1 means custom is slower."
)


def rank_reference(x, method="average", axis=-1):
    """Independent Python stable sort/group baseline, O(n log n) per slice.

    Conversion to Python scalars and output allocation are part of this API call.
    NaNs retain their positions and are excluded from ranking.
    """
    rows = np.moveaxis(np.asarray(x), axis, -1)
    result = np.full(rows.shape, np.nan)
    for index in np.ndindex(rows.shape[:-1]):
        values = rows[index].tolist()
        order = sorted((i for i, v in enumerate(values) if v == v), key=values.__getitem__)
        start, dense = 0, 1
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            for position in range(start, end):
                if method == "ordinal":
                    value = position + 1
                elif method == "average":
                    value = (start + 1 + end) / 2
                elif method == "min":
                    value = start + 1
                elif method == "max":
                    value = end
                else:
                    value = dense
                result[index][order[position]] = value
            start, dense = end, dense + 1
    return np.moveaxis(result, -1, axis)


def full_sort_topk(x, k, largest=True):
    # Benchmark inputs are finite float64: negation preserves their ordering.
    # Stable sorting of -x retains smaller indices for tied largest values.
    indices = np.argsort(-x if largest else x, kind="stable")[:k]
    return x[indices], indices


def percentile_rank_reference(x, scores, kind):
    """Direct counts per score, O(n*m); includes output allocation."""
    scores = np.asarray(scores)
    out = np.empty(scores.shape)
    for index in np.ndindex(scores.shape):
        score = scores[index]
        if np.isnan(score) or np.isnan(x).any():
            out[index] = np.nan
        elif kind == "strict":
            out[index] = np.count_nonzero(x < score) / x.size * 100
        elif kind == "weak":
            out[index] = np.count_nonzero(x <= score) / x.size * 100
        else:
            out[index] = (np.count_nonzero(x < score) / 2
                          + np.count_nonzero(x <= score) / 2) / x.size * 100
    return out.item() if out.ndim == 0 else out


def check_equal(expected, actual):
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(expected) != len(actual):
            raise AssertionError("Output tuple differs")
        for a, b in zip(expected, actual):
            check_equal(a, b)
        return
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        raise AssertionError(f"Output shapes differ: {a.shape} != {b.shape}")
    if a.dtype.kind in "iub":
        if b.dtype.kind not in "iub":
            raise AssertionError("Expected integer indices")
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)


def check_topk(expected, actual, *, x, ordered):
    values, indices = actual
    if indices.dtype.kind not in "iu" or indices.shape != expected[1].shape:
        raise AssertionError("Invalid top-k index output")
    check_equal(x[indices], values)
    check_equal(expected[1] if ordered else np.sort(expected[1]),
                indices if ordered else np.sort(indices))


@dataclass
class Case:
    operation: str
    baseline_label: str
    data: np.ndarray
    distribution: str
    options: dict
    baseline: object
    custom: object
    check: object = check_equal


def make_data(n, distribution, seed):
    # Independent streams make inputs invariant to case ordering or size subsets.
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, DISTRIBUTIONS.index(distribution)]))
    if distribution == "duplicates":
        x = rng.integers(-5, 6, n).astype(float)
    elif distribution == "constant":
        x = np.full(n, 3.0)
    else:
        x = rng.normal(size=n)
        if distribution in ("ascending", "descending"):
            x.sort()
            if distribution == "descending":
                x = x[::-1].copy()
    x.setflags(write=False)
    return x


def make_cases(x, distribution, pivot_seed, queries, seed=0):
    """Yield lazy cases so only one dataset's closures are retained at a time."""
    def case(name, label, baseline, custom, options=None, check=check_equal):
        return Case(name, label, x, distribution, options or {}, baseline, custom, check)

    yield case("merge_sort", "NumPy stable sort", partial(np.sort, x, kind="stable"), partial(ss.merge_sort, x))
    yield case("merge_sort_indices", "NumPy stable argsort + gather",
               lambda: full_sort_topk(x, x.size, largest=False), partial(ss.merge_sort, x, return_indices=True))
    for name in ("sort", "argsort"):
        yield case(name, "direct NumPy (wrapper overhead)",
                   partial(getattr(np, name), x, kind="stable"), partial(getattr(ss, name), x))
    for k in sorted({0, x.size // 2, x.size - 1}):
        for label, baseline in (("NumPy partition", lambda k=k: np.partition(x, k)[k]),
                                ("NumPy full sort", lambda k=k: np.sort(x)[k])):
            yield case("quickselect", label, baseline, partial(ss.quickselect, x, k, rng=pivot_seed),
                       {"k": k, "pivot_seed": pivot_seed})
    for k in sorted({1, max(1, x.size // 10), x.size}):
        for largest in (False, True):
            for ordered in (False, True):
                yield case("topk", "NumPy full stable sort + gather",
                           partial(full_sort_topk, x, k, largest),
                           partial(ss.topk, x, k, largest=largest, sorted=ordered),
                           {"k": k, "largest": largest, "sorted": ordered},
                           partial(check_topk, x=x, ordered=ordered))
    sorted_x = np.sort(x)
    sorted_x.setflags(write=False)
    # Includes an exact hit and misses outside the data range for batches >= 3.
    query_rng = np.random.default_rng(np.random.SeedSequence([seed, x.size, queries]))
    scores = query_rng.uniform(sorted_x[0] - 1, sorted_x[-1] + 1, queries)
    probes = [x[0], sorted_x[0] - 1, sorted_x[-1] + 1]
    scores[:min(3, queries)] = probes[:min(3, queries)]
    scores.setflags(write=False)
    for side in ("left", "right"):
        yield case("binary_search", "NumPy scalar searchsorted",
                   partial(np.searchsorted, sorted_x, x[0], side=side),
                   partial(ss.binary_search, sorted_x, x[0], side=side), {"side": side, "queries": 1})
        yield case("searchsorted", "Python loop of binary_search",
                   lambda side=side: np.array([ss.binary_search(sorted_x, v, side=side) for v in scores]),
                   partial(ss.searchsorted, sorted_x, scores, side=side), {"side": side, "queries": queries})
        yield case("searchsorted", "direct NumPy (wrapper overhead)",
                   partial(np.searchsorted, sorted_x, scores, side=side),
                   partial(ss.searchsorted, sorted_x, scores, side=side), {"side": side, "queries": queries})
    for method in METHODS:
        yield case("rank", "Python stable sort/group", partial(rank_reference, x, method),
                   partial(ranking.rank, x, method), {"method": method, "axis": -1, "nan_policy": "keep"})
    for method in ("linear", "lower", "higher", "nearest", "midpoint"):
        q = [0, 25, 50, 75, 100]
        yield case("percentile", "NumPy percentile", partial(np.percentile, x, q, method=method),
                   partial(ranking.percentile, x, q, method=method), {"method": method, "q": q})
    yield case("percentile", "NumPy percentile", partial(np.percentile, x, 50),
               partial(ranking.percentile, x, 50), {"method": "linear", "q": 50})
    for kind in ("weak", "strict", "mean"):
        for score in (x[0], scores):
            yield case("percentile_rank", "direct counts O(n*m)",
                       partial(percentile_rank_reference, x, score, kind),
                       partial(ranking.percentile_rank, x, score, kind=kind),
                       {"kind": kind, "queries": int(np.size(score))})


def axis_nan_cases(n, seed):
    """Same element count, different slice lengths/axes, including strided views."""
    width = min(10, n)
    x = make_data(n, "duplicates", seed)[: n // width * width].reshape(-1, width)
    missing = x.copy()
    missing.flat[::7] = np.nan
    for data, label, policy in ((x, "many_short_rows", "keep"),
                                (x.T, "few_long_rows_strided", "keep"),
                                (missing, "nan_every_7", "omit")):
        data.setflags(write=False)
        for axis in (0, 1):
            yield Case("sort", "direct NumPy (wrapper overhead)", data, label, {"axis": axis},
                       partial(np.sort, data, axis=axis, kind="stable"), partial(ss.sort, data, axis=axis))
            for method in METHODS:
                yield Case("rank", "Python stable sort/group", data, label,
                           {"axis": axis, "method": method, "nan_policy": policy},
                           partial(rank_reference, data, method, axis),
                           partial(ranking.rank, data, method, axis, nan_policy=policy))
            yield Case("percentile", "NumPy nanpercentile" if policy == "omit" else "NumPy percentile",
                       data, label, {"axis": axis, "q": [25, 50, 75],
                                     "nan_policy": "omit" if policy == "omit" else "propagate"},
                       partial(np.nanpercentile if policy == "omit" else np.percentile, data, [25, 50, 75], axis=axis),
                       partial(ranking.percentile, data, [25, 50, 75], axis=axis,
                               nan_policy="omit" if policy == "omit" else "propagate"))


def elapsed_batch(function, calls):
    start = time.perf_counter_ns()
    for _ in range(calls):
        function()
    return (time.perf_counter_ns() - start) * 1e-9


def calibrate(function, target_seconds):
    calls = 1
    while True:
        elapsed = elapsed_batch(function, calls)
        if elapsed >= target_seconds or calls >= 10000:
            return calls
        calls = min(10000, calls * 10)


def summary(samples):
    q1, median, q3 = np.percentile(samples, [25, 50, 75])
    return {"median_s": float(median), "iqr_s": float(q3 - q1), "min_s": float(min(samples))}


def measure(case, repeats, warmups, target_seconds):
    # Fail immediately on incorrect results; never emit a misleading timing row.
    case.check(case.baseline(), case.custom())
    for _ in range(warmups):
        case.baseline()
        case.custom()
    functions = (case.baseline, case.custom)
    counts = [calibrate(fn, target_seconds) for fn in functions]
    samples = [[], []]
    for repeat in range(repeats):
        for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            samples[i].append(elapsed_batch(functions[i], counts[i]) / counts[i])
    baseline, custom = summary(samples[0]), summary(samples[1])
    return {
        "operation": case.operation, "baseline_label": case.baseline_label,
        "n": int(case.data.size), "shape": list(case.data.shape), "dtype": str(case.data.dtype),
        "strides": list(case.data.strides), "distribution": case.distribution, "options": case.options,
        "input_sha256": hashlib.sha256(case.data.tobytes()).hexdigest(),
        "baseline": {**baseline, "calls_per_sample": counts[0], "samples_s": samples[0]},
        "custom": {**custom, "calls_per_sample": counts[1], "samples_s": samples[1]},
        "ratio": baseline["median_s"] / custom["median_s"],
    }


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif sys.platform.startswith("linux"):
        try:
            cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name"))
        except (OSError, StopIteration):
            pass
    status = command_output(["git", "status", "--porcelain"])
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "python_implementation": platform.python_implementation(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu or "not reported",
        "logical_cpus": os.cpu_count(), "executable": sys.executable,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("sort_search.py", "rank.py", "benchmark/benchmark_section2.py")},
        "gc_enabled": gc.isenabled(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "timer": vars(time.get_clock_info("perf_counter")),
    }


def markdown_report(report):
    lines = ["# Section 2 benchmarks", "", TIMING_NOTES, "", "## Environment", "", "```json",
             json.dumps(report["environment"], indent=2), "```", "", "## Configuration", "", "```json",
             json.dumps(report["configuration"], indent=2), "```", "", "Command: `" + report["command"] + "`",
             "", "## Results", "", "All correctness comparisons passed. Times are milliseconds per call.", "",
             "| Operation | Baseline | Shape / data / options | Baseline median (IQR) ms | Custom median (IQR) ms | Baseline/custom |",
             "|---|---|---|---:|---:|---:|"]
    for row in report["results"]:
        b, c = row["baseline"], row["custom"]
        detail = f'{row["shape"]} / {row["distribution"]} / {json.dumps(row["options"], sort_keys=True)}'
        lines.append(f'| {row["operation"]} | {row["baseline_label"]} | {detail} | '
                     f'{b["median_s"] * 1000:.6f} ({b["iqr_s"] * 1000:.6f}) | '
                     f'{c["median_s"] * 1000:.6f} ({c["iqr_s"] * 1000:.6f}) | {row["ratio"]:.3f}x |')
    return "\n".join(lines) + "\n"


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--distributions", choices=DISTRIBUTIONS, nargs="+", default=list(DISTRIBUTIONS))
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pivot-seed", type=int, default=17)
    parser.add_argument("--queries", type=positive_int, default=100)
    parser.add_argument("--batch-ms", type=float, default=1.0, help="target duration per sample; each side calibrated separately")
    parser.add_argument("--output", type=Path, help="output prefix for .json raw timings and .md table")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.pivot_seed < 0:
        parser.error("seeds must be nonnegative")
    if not np.isfinite(args.batch_ms) or args.batch_ms <= 0:
        parser.error("--batch-ms must be finite and positive")
    config = {**vars(args), "output": str(args.output) if args.output else None}
    report = {"schema_version": 1, "configuration": config, "environment": environment(),
              "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]),
              "timing_notes": TIMING_NOTES, "results": []}
    print("Section 2: sort_search and rank\n" + TIMING_NOTES, flush=True)
    print("Environment:", json.dumps(report["environment"], sort_keys=True), flush=True)
    print("Configuration:", json.dumps(config, sort_keys=True), flush=True)
    print(f'{"Operation":<20} {"n":>7} {"Data":<24} {"Base ms (IQR)":>20} {"Custom ms (IQR)":>20} {"Ratio":>8}', flush=True)
    def run(case):
        try:
            row = measure(case, args.repeats, args.warmups, args.batch_ms / 1000)
        except Exception as error:
            raise RuntimeError(f"Failed {case.operation}, {case.distribution}, {case.options}, baseline={case.baseline_label}") from error
        report["results"].append(row)
        base_text = f'{row["baseline"]["median_s"] * 1000:.5f} ({row["baseline"]["iqr_s"] * 1000:.5f})'
        custom_text = f'{row["custom"]["median_s"] * 1000:.5f} ({row["custom"]["iqr_s"] * 1000:.5f})'
        print(f'{row["operation"]:<20} {row["n"]:>7} {row["distribution"]:<24} '
              f'{base_text:>20} {custom_text:>20} '
              f'{row["ratio"]:>7.2f}x  {row["options"]} [{row["baseline_label"]}]', flush=True)
    for n in args.sizes:
        for distribution in args.distributions:
            for case in make_cases(make_data(n, distribution, args.seed), distribution, args.pivot_seed, args.queries, args.seed):
                run(case)
        for case in axis_nan_cases(n, args.seed):
            run(case)
    print(f'All {len(report["results"])} correctness comparisons passed.', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        Path(str(args.output) + ".md").write_text(markdown_report(report), encoding="utf-8")
        print(f"Saved {args.output}.json and {args.output}.md", flush=True)
    return report


if __name__ == "__main__":
    main()
