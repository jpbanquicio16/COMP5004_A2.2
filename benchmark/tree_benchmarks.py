"""Tree benchmarks: (a) loop vs vectorised, (b) buffer_size sweep on a stream.

Run from the project root:
    python benchmark/tree_benchmarks.py
    python benchmark/tree_benchmarks.py --repeats 5 --markdown benchmark/tree_results.md
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from numcompute_stream.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# loop-based baselines: same algorithms as the vectorised library code, written
# with plain Python loops over features, thresholds and samples. They must give
# identical results (tests/test_loop_tree.py), so the speed comparison is fair.
# ---------------------------------------------------------------------------

class LoopDecisionTree(DecisionTreeClassifier):
    """Gini tree whose split search and predict use Python loops.

    Same tie rules as the vectorised tree: stable sort (NaN last), thresholds
    scanned in ascending order, NaN-right before NaN-left, first best wins.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.criterion != "gini":
            raise ValueError("LoopDecisionTree supports criterion='gini' only")

    def _best_split_block(self, X, yi, k, feats):
        n = yi.shape[0]
        total = np.zeros(k)
        for label in yi:                                  # class counts, sample by sample
            total[label] += 1
        parent = float(self._impurity(total))
        best = None                                       # (score, f, pos, option)
        for fi, f in enumerate(feats):
            col = X[:, f]
            order = np.argsort(col, kind="stable")        # NaN sorts last
            nan_cnt = np.zeros(k)
            for r in range(n):
                if np.isnan(col[r]):
                    nan_cnt[yi[r]] += 1
            has_nan = nan_cnt.sum() > 0
            left = np.zeros(k)
            f_best = None                                 # (score, pos, option)
            for pos in range(n - 1):
                left[yi[order[pos]]] += 1
                lo, hi = col[order[pos]], col[order[pos + 1]]
                if not lo < hi:                           # tie or NaN: no threshold here
                    continue
                for option in ((0, 1) if has_nan else (0,)):
                    L = left + nan_cnt if option else left
                    R = total - L
                    score = (L.sum() * self._impurity(L)
                             + R.sum() * self._impurity(R)) / n
                    if f_best is None or score < f_best[0]:
                        f_best = (score, pos, option)
            if f_best is not None and (best is None or f_best[0] < best[0]):
                best = (f_best[0], f, fi, f_best[1], f_best[2], order, col, has_nan)
        if best is None:
            return None
        score, f, fi, pos, option, order, col, has_nan = best
        gain = parent - score
        if gain <= 1e-12:
            return None
        lo, hi = col[order[pos]], col[order[pos + 1]]
        thr = 0.5 * (lo + hi)
        if not (lo <= thr < hi):
            thr = lo
        if has_nan:
            nan_left = bool(option)
        else:                                             # unseen NaN -> bigger child
            n_left = pos + 1
            nan_left = n_left >= n - n_left
        return int(f), float(thr), nan_left, float(gain)

    def predict(self, X):
        X = self._check_predict_input(X)
        nd = self._nodes
        assert nd is not None
        out = np.empty(X.shape[0], dtype=np.intp)
        for i, x in enumerate(X):                         # one row at a time
            node = 0
            while nd["feature"][node] >= 0:
                v = x[nd["feature"][node]]
                left = nd["nan_left"][node] if np.isnan(v) else v <= nd["threshold"][node]
                node = nd["left"][node] if left else nd["right"][node]
            out[i] = nd["counts"][node].argmax()
        assert self.classes_ is not None
        return self.classes_[out]


def welford_per_sample(X):
    """Running mean and (population) variance, one row at a time. Returns (mean, var)."""
    X = np.asarray(X, dtype=float)
    n, mean, m2 = 0, np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for row in X:
        n += 1
        delta = row - mean
        mean = mean + delta / n
        m2 = m2 + delta * (row - mean)
    return mean, m2 / n


def confusion_per_sample(y_true, y_pred, n_classes):
    """Confusion matrix (actual rows, predicted columns) built one pair at a time."""
    out = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        out[t, p] += 1
    return out


def make_stream(n, n_features=20, seed=0, drift=False):
    """Yes/no features; label = f0 AND f1 with 8% noise (flips halfway if drift)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, n_features))
    y = X[:, 0] & X[:, 1]
    if drift:
        y = np.where(np.arange(n) >= n // 2, X[:, 2] | X[:, 3], y)
    y = np.where(rng.random(n) < 0.08, 1 - y, y)
    return X, y


def best_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def make_continuous(n, n_features=10, seed=0):
    """Gaussian features, 3 classes: (x0 + x1 > 0) + (x2 > 1)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1)


def bench_loop_vs_vectorised(repeats):
    """Rows: (data, n_train, fit loop, fit vec, predict loop, predict vec, agreement)."""
    rows = []
    cases = [("0/1", make_stream, (200, 1000, 5000)),
             ("continuous", make_continuous, (200, 1000, 3000))]
    for name, maker, sizes in cases:
        for n in sizes:
            X, y = maker(n)
            Xt, _ = maker(20000, seed=1)
            fast, slow = DecisionTreeClassifier(max_depth=5), LoopDecisionTree(max_depth=5)
            t_fit_v = best_time(lambda: fast.fit(X, y), repeats)
            t_fit_l = best_time(lambda: slow.fit(X, y), max(1, repeats // 2))
            t_pred_v = best_time(lambda: fast.predict(Xt), repeats)
            t_pred_l = best_time(lambda: slow.predict(Xt), repeats)
            for field in fast.tree_._fields:
                assert np.array_equal(getattr(fast.tree_, field), getattr(slow.tree_, field),
                                      equal_nan=True), field
            agree = float(np.mean(fast.predict(Xt) == slow.predict(Xt)))
            assert agree == 1.0
            rows.append((name, n, t_fit_l, t_fit_v, t_pred_l, t_pred_v, agree))
    return rows


def _recovery_rows(chunk_acc, chunk, change_row):
    """Rows after the change until a 3-chunk average gets within 0.03 of the
    accuracy just before it (None if it never does)."""
    c0 = change_row // chunk
    pre = np.nanmean(chunk_acc[max(0, c0 - 20):c0])
    for c in range(c0, len(chunk_acc) - 2):
        if np.nanmean(chunk_acc[c:c + 3]) >= pre - 0.03:
            return (c - c0) * chunk
    return None


def bench_buffer_sizes(repeats, n=20000, chunk=50, seeds=5):
    """Mean over `seeds` streams. Rows: (buffer, retrains, total retrain s,
    s / retrain, accuracy, peak model bytes, recovery rows or None)."""
    from numcompute_stream.stream import model_nbytes
    rows = []
    for buf in (50, 100, 250, 500, 1000, 2500):
        acc_s, tot_s, mem_s, rec_s, nr = [], [], [], [], 0
        for seed in range(seeds):
            X, y = make_stream(n, seed=seed, drift=True)

            def run():
                t = DecisionTreeClassifier(max_depth=5, buffer_size=buf)
                retrain, correct, seen, peak = 0.0, 0, 0, 0
                chunk_acc = []
                for i in range(0, n, chunk):
                    Xc, yc = X[i:i + chunk], y[i:i + chunk]
                    if t.is_fitted:                 # test-then-train accuracy
                        ok = int((t.predict(Xc) == yc).sum())
                        correct += ok
                        seen += len(yc)
                        chunk_acc.append(ok / len(yc))
                    else:
                        chunk_acc.append(np.nan)
                    t0 = time.perf_counter()
                    before = t.n_retrains_
                    t.partial_fit(Xc, yc)
                    if t.n_retrains_ > before:
                        retrain += time.perf_counter() - t0
                    peak = max(peak, model_nbytes(t))
                return t, retrain, correct / max(seen, 1), peak, np.array(chunk_acc)
            best = None
            for _ in range(repeats):
                res = run()
                if best is None or res[1] < best[1]:
                    best = res
            assert best is not None
            t, retrain, acc, peak, chunk_acc = best
            nr = t.n_retrains_
            acc_s.append(acc); tot_s.append(retrain); mem_s.append(peak)
            rec = _recovery_rows(chunk_acc, chunk, n // 2)
            if rec is not None:
                rec_s.append(rec)
        rec = float(np.mean(rec_s)) if len(rec_s) == seeds else None
        rows.append((buf, nr, float(np.mean(tot_s)), float(np.mean(tot_s)) / max(nr, 1),
                     float(np.mean(acc_s)), float(np.mean(mem_s)), rec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--markdown", type=str, default=None)
    args = ap.parse_args()

    out = ["# Tree benchmarks", "",
           f"Python {sys.version.split()[0]}, NumPy {np.__version__}, best of {args.repeats} runs.", "",
           "## (a) Loop vs vectorised (max_depth=5; 20 yes/no features or 10 Gaussian features, "
           "3 classes; predict on 20,000 rows)", "",
           "| data | n_train | fit loop (ms) | fit vec (ms) | fit speed-up | predict loop (ms) "
           "| predict vec (ms) | predict speed-up | identical tree + predictions |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, n, fl, fv, pl, pv, ag in bench_loop_vs_vectorised(args.repeats):
        out.append(f"| {name} | {n} | {fl*1e3:.1f} | {fv*1e3:.1f} | {fl / fv:.1f}x "
                   f"| {pl*1e3:.1f} | {pv*1e3:.1f} | {pl / pv:.1f}x | yes ({ag:.0%}) |")
    out += ["", "## (b) buffer_size sweep (20,000-row stream, 50-row chunks, label rule changes "
            "at row 10,000; mean of 5 seeds)", "",
            "| buffer_size | retrains | total retrain time (s) | time / retrain (ms) "
            "| prequential accuracy | peak model memory (KB) | rows to recover after change |",
            "|---|---|---|---|---|---|---|"]
    for buf, nr, tot, per, acc, mem, rec in bench_buffer_sizes(args.repeats):
        rec_s = "never" if rec is None else f"{rec:.0f}"
        out.append(f"| {buf} | {nr} | {tot:.3f} | {per*1e3:.1f} | {acc:.3f} | {mem/1024:.1f} | {rec_s} |")
    text = "\n".join(out)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")


if __name__ == "__main__":
    main()
