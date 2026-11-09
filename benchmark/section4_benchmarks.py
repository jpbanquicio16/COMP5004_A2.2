"""Section 4: reproducible loop vs NumPy benchmarks for optim, utils and pipeline.

Run from the project root:
    python benchmark/section4_benchmarks.py
    python benchmark/section4_benchmarks.py --n-points 400 --repeats 7 --markdown benchmark/section4_results.md
"""
import argparse
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from numcompute import optim, utils
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import MinMaxScaler, SimpleImputer, StandardScaler


# ---------------------------------------------------------------------------
# utils: pure Python loop baselines (they get plain lists, not NumPy arrays)
# ---------------------------------------------------------------------------

def euclidean_matrix_loop(X, Y):
    out = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            total = 0.0
            for k in range(len(X[i])):
                total += (X[i][k] - Y[j][k]) ** 2
            out[i, j] = math.sqrt(total)
    return out


def manhattan_matrix_loop(X, Y):
    out = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            total = 0.0
            for k in range(len(X[i])):
                total += abs(X[i][k] - Y[j][k])
            out[i, j] = total
    return out


def k_nearest_loop(X, queries, k):
    all_indices = []
    all_distances = []
    for query in queries:
        distances = []
        for row in X:
            total = 0.0
            for a, b in zip(query, row):
                total += (a - b) ** 2
            distances.append(math.sqrt(total))
        order = sorted(range(len(distances)), key=lambda i: distances[i])[:k]  # sorted is stable
        all_indices.append(order)
        all_distances.append([distances[i] for i in order])
    return np.array(all_indices), np.array(all_distances)


def sigmoid_loop(values):
    out = []
    for v in values:
        if v >= 0:
            out.append(1.0 / (1.0 + math.exp(-v)))
        else:
            e = math.exp(v)
            out.append(e / (1.0 + e))
    return out


def relu_loop(values):
    out = []
    for v in values:
        out.append(v if v > 0 else 0.0)
    return out


def softmax_loop(rows):
    out = []
    for row in rows:
        biggest = max(row)
        exps = [math.exp(v - biggest) for v in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def logsumexp_loop(rows):
    out = []
    for row in rows:
        biggest = max(row)
        total = 0.0
        for v in row:
            total += math.exp(v - biggest)
        out.append(biggest + math.log(total))
    return out


def batches_loop(n_samples, batch_size):
    batches = []
    start = 0
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        batches.append(list(range(start, end)))
        start = end
    return batches


# ---------------------------------------------------------------------------
# optim: the function we differentiate is f(x) = sum(x_i^2 + sin(x_i))
# ---------------------------------------------------------------------------

def f_loop(x):
    total = 0.0
    for v in x:
        total += v * v + math.sin(v)
    return total


def f_numpy(x):
    return np.sum(x ** 2 + np.sin(x))


def grad_loop(f, x, h, method='central'):
    n = len(x)
    result = [0.0] * n
    f_at_x = f(x) if method == 'forward' else None
    for i in range(n):
        up = list(x)
        up[i] = x[i] + h
        if method == 'central':
            down = list(x)
            down[i] = x[i] - h
            f_down = f(down)
        else:
            down = x
            f_down = f_at_x
        result[i] = (f(up) - f_down) / (up[i] - down[i])
    return result


def make_vector_function(A):
    # F(x) = A @ x + sin(x[:m]), goes from n inputs to m outputs
    m = A.shape[0]
    A_list = A.tolist()

    def F_loop(x):
        out = []
        for i in range(m):
            total = 0.0
            for j in range(len(x)):
                total += A_list[i][j] * x[j]
            out.append(total + math.sin(x[i]))
        return out

    def F_numpy(x):
        return A @ x + np.sin(x[:m])

    return F_loop, F_numpy


def jacobian_loop(F, x, h):
    n = len(x)
    columns = []
    for i in range(n):
        up = list(x)
        up[i] = x[i] + h
        down = list(x)
        down[i] = x[i] - h
        width = up[i] - down[i]
        f_up = F(up)
        f_down = F(down)
        columns.append([(a - b) / width for a, b in zip(f_up, f_down)])
    m = len(columns[0])
    return [[columns[j][i] for j in range(n)] for i in range(m)]  # rows are outputs


# ---------------------------------------------------------------------------
# pipeline: impute (mean) -> StandardScaler -> MinMaxScaler
# ---------------------------------------------------------------------------

def preprocess_fit_loop(rows):
    """Fit and transform with plain loops. Returns the new rows and what was learned."""
    n_rows, n_cols = len(rows), len(rows[0])
    means, stds, lows, highs = [], [], [], []

    data = [list(row) for row in rows]
    for j in range(n_cols):                      # step 1: fill NaN with the column mean
        total, count = 0.0, 0
        for i in range(n_rows):
            if data[i][j] == data[i][j]:         # a NaN is never equal to itself
                total += data[i][j]
                count += 1
        mean = total / count
        for i in range(n_rows):
            if data[i][j] != data[i][j]:
                data[i][j] = mean

    for j in range(n_cols):                      # step 2: standardise (population std)
        total = 0.0
        for i in range(n_rows):
            total += data[i][j]
        mean = total / n_rows
        squares = 0.0
        for i in range(n_rows):
            squares += (data[i][j] - mean) ** 2
        std = math.sqrt(squares / n_rows)
        for i in range(n_rows):
            data[i][j] = (data[i][j] - mean) / std
        means.append(mean)
        stds.append(std)

    for j in range(n_cols):                      # step 3: min-max to 0..1
        low = min(data[i][j] for i in range(n_rows))
        high = max(data[i][j] for i in range(n_rows))
        for i in range(n_rows):
            data[i][j] = (data[i][j] - low) / (high - low)
        lows.append(low)
        highs.append(high)
    return data, (means, stds, lows, highs)


def column_means_of(rows):
    n_cols = len(rows[0])
    means = []
    for j in range(n_cols):
        total, count = 0.0, 0
        for row in rows:
            if row[j] == row[j]:
                total += row[j]
                count += 1
        means.append(total / count)
    return means


def preprocess_transform_loop(rows, learned):
    """Apply what was learned (no fitting). The mean used for filling is the first one learned."""
    means, stds, lows, highs, fill_means = learned
    out = []
    for row in rows:
        new_row = []
        for j in range(len(row)):
            v = row[j] if row[j] == row[j] else fill_means[j]
            v = (v - means[j]) / stds[j]
            v = (v - lows[j]) / (highs[j] - lows[j])
            new_row.append(v)
        out.append(new_row)
    return out


def make_pipeline():
    return Pipeline([
        ('fill', SimpleImputer(strategy='mean')),
        ('std', StandardScaler()),
        ('minmax', MinMaxScaler()),
    ])


# ---------------------------------------------------------------------------
# timing and printing
# ---------------------------------------------------------------------------

def best_time(function, repeats):
    """Warm up once, then return the smallest elapsed time in seconds."""
    function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return min(timings)


def check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def case(name, loop, numpy_version, rtol=1e-10, atol=1e-12, same=None):
    """One comparison. 'same' can replace the default numeric check."""
    return {'name': name, 'loop': loop, 'numpy': numpy_version, 'rtol': rtol, 'atol': atol, 'same': same}


def same_batches(loop_result, numpy_result):
    numpy_lists = [[int(i) for i in batch] for batch in numpy_result]
    assert loop_result == numpy_lists, 'Batch indices: results differ'


def run_benchmarks(n_points=200, n_values=20000, n_inputs=100, n_rows=5000, repeats=5, seed=0):
    """Print the comparisons and return rows (name, loop seconds, NumPy seconds).

    Both versions receive identical inputs and repeat counts. Result equality
    is checked before timing. Data generation is excluded from the timings.
    The loop versions get plain Python lists. Python loops are intentional
    reference baselines, not core implementations.
    """
    check_int('n_points', n_points, 3)
    check_int('n_values', n_values, 10)
    check_int('n_inputs', n_inputs, 4)
    check_int('n_rows', n_rows, 10)
    check_int('repeats', repeats, 1)
    rng = np.random.default_rng(seed)

    # --- inputs (the loop versions get plain lists) ---
    d = 8
    X = rng.normal(size=(n_points, d))
    Y = rng.normal(size=(n_points, d))
    n_queries = max(2, n_points // 2)
    queries = rng.normal(size=(n_queries, d))
    X_list, Y_list, queries_list = X.tolist(), Y.tolist(), queries.tolist()
    k = 5

    values = rng.normal(scale=5.0, size=n_values)
    values_list = values.tolist()
    matrix = values[: (n_values // 10) * 10].reshape(-1, 10)
    matrix_list = matrix.tolist()
    batch_size = 32

    x = rng.normal(size=n_inputs)
    x_list = x.tolist()
    h = 1e-5
    jac_n = max(2, n_inputs // 2)
    jac_m = max(1, jac_n * 2 // 3)
    A = rng.normal(size=(jac_m, jac_n))
    xj = rng.normal(size=jac_n)
    xj_list = xj.tolist()
    F_loop, F_numpy = make_vector_function(A)

    raw = rng.normal(50, 10, size=(n_rows, 8))
    raw[rng.random(raw.shape) < 0.05] = np.nan   # about 5% missing values
    raw_list = raw.tolist()
    pipe_new = rng.normal(50, 10, size=(n_rows, 8))   # "new" data for the transform case
    pipe_new[rng.random(pipe_new.shape) < 0.05] = np.nan
    pipe_new_list = pipe_new.tolist()

    # the loop transform needs what the loop fit learned (this is not timed)
    _, learned_core = preprocess_fit_loop(raw_list)
    learned = learned_core + (column_means_of(raw_list),)
    fitted_pipe = make_pipeline().fit(raw)

    cases = [
        case('utils: euclidean distance matrix',
             lambda: euclidean_matrix_loop(X_list, Y_list),
             lambda: utils.euclidean_distance(X[:, None, :], Y[None, :, :])),
        case('utils: manhattan distance matrix',
             lambda: manhattan_matrix_loop(X_list, Y_list),
             lambda: utils.manhattan_distance(X[:, None, :], Y[None, :, :])),
        case('utils: k_nearest',
             lambda: k_nearest_loop(X_list, queries_list, k),
             lambda: utils.k_nearest(X, queries, k),
             same=lambda a, b: (np.testing.assert_array_equal(a[0], b[0], err_msg='k_nearest: indices differ'),
                                np.testing.assert_allclose(a[1], b[1], rtol=1e-10, atol=1e-12,
                                                           err_msg='k_nearest: distances differ'))),
        case('utils: sigmoid',
             lambda: sigmoid_loop(values_list), lambda: utils.sigmoid(values)),
        case('utils: relu',
             lambda: relu_loop(values_list), lambda: utils.relu(values)),
        case('utils: softmax (rows of 10)',
             lambda: softmax_loop(matrix_list), lambda: utils.softmax(matrix)),
        case('utils: logsumexp (rows of 10)',
             lambda: logsumexp_loop(matrix_list), lambda: utils.logsumexp(matrix, axis=1)),
        case('utils: batch_indices',
             lambda: batches_loop(n_values, batch_size),
             lambda: utils.batch_indices(n_values, batch_size),
             same=same_batches),
        case('optim: grad (central)',
             lambda: grad_loop(f_loop, x_list, h),
             lambda: optim.grad(f_numpy, x, h), rtol=1e-6, atol=1e-6),
        case('optim: grad (forward)',
             lambda: grad_loop(f_loop, x_list, h, 'forward'),
             lambda: optim.grad(f_numpy, x, h, 'forward'), rtol=1e-6, atol=1e-6),
        case('optim: jacobian (central)',
             lambda: jacobian_loop(F_loop, xj_list, h),
             lambda: optim.jacobian(F_numpy, xj, h), rtol=1e-6, atol=1e-6),
        case('pipeline: fit_transform (3 steps)',
             lambda: preprocess_fit_loop(raw_list)[0],
             lambda: make_pipeline().fit_transform(raw), rtol=1e-9, atol=1e-9),
        case('pipeline: transform (fitted)',
             lambda: preprocess_transform_loop(pipe_new_list, learned),
             lambda: fitted_pipe.transform(pipe_new), rtol=1e-9, atol=1e-9),
    ]

    print(f'Section 4: optim, utils and pipeline | seed={seed} | repeats={repeats}')
    print(f'Sizes: n_points={n_points} (d={d}, {n_queries} queries, k={k}) | n_values={n_values} | '
          f'n_inputs={n_inputs} (jacobian {jac_m}x{jac_n}) | n_rows={n_rows}')
    print('Timing: best of repeats after one warm-up; same repeats for both versions.')
    print('Loop versions get plain Python lists, converted before timing.')
    print(f"{'Component':<36} {'Loop (s)':>12} {'NumPy (s)':>12} {'Speed-up':>12}")
    print('-' * 76)
    rows = []
    for item in cases:
        loop_result, numpy_result = item['loop'](), item['numpy']()
        if item['same'] is not None:
            item['same'](loop_result, numpy_result)
        else:
            np.testing.assert_allclose(loop_result, numpy_result, rtol=item['rtol'], atol=item['atol'],
                                       err_msg=f"{item['name']}: results differ")
        loop_time = best_time(item['loop'], repeats)
        numpy_time = best_time(item['numpy'], repeats)
        rows.append((item['name'], loop_time, numpy_time))
        ratio = loop_time / numpy_time if numpy_time else float('inf')
        print(f"{item['name']:<36} {loop_time:>12.6f} {numpy_time:>12.6f} {ratio:>11.2f}x")

    print('\nAll result comparisons passed.')
    print('A speed-up below 1 means NumPy was slower in this run; timings vary by machine.')
    print('Loop = pure Python on lists, NumPy = the numcompute function.')
    print('Environment:', platform.platform())
    print('Processor:', platform.processor() or 'not reported')
    print(f'Python {platform.python_version()} | NumPy {np.__version__}')
    return rows


def write_markdown(path, rows, args):
    lines = [
        '## Section 4 benchmarks: optim, utils, pipeline',
        '',
        f'seed={args.seed}, repeats={args.repeats}, n_points={args.n_points}, n_values={args.n_values}, '
        f'n_inputs={args.n_inputs}, n_rows={args.n_rows}. '
        f'Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}.',
        '',
        '| Component | Loop (s) | NumPy (s) | Speed-up |',
        '|---|---|---|---|',
    ]
    for name, loop_time, numpy_time in rows:
        lines.append(f'| {name} | {loop_time:.6f} | {numpy_time:.6f} | {loop_time / numpy_time:.2f}x |')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Markdown table written to {path}')


def main():
    parser = argparse.ArgumentParser(description='Section 4: optim, utils and pipeline benchmarks')
    parser.add_argument('--n-points', type=int, default=200, help='points for the distance and k_nearest cases')
    parser.add_argument('--n-values', type=int, default=20000, help='values for sigmoid, relu, softmax, logsumexp, batches')
    parser.add_argument('--n-inputs', type=int, default=100, help='number of inputs for grad (jacobian uses half)')
    parser.add_argument('--n-rows', type=int, default=5000, help='rows for the pipeline cases')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--markdown', default=None, help='optional path for a markdown copy of the table')
    args = parser.parse_args()
    try:
        rows = run_benchmarks(args.n_points, args.n_values, args.n_inputs, args.n_rows,
                              args.repeats, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if args.markdown:
        write_markdown(args.markdown, rows, args)


if __name__ == '__main__':
    main()
