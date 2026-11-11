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
from numcompute.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# loop-based baseline: same algorithm, plain Python loops over features,
# thresholds and samples (NaN handling omitted: benchmark data has no NaN)
# ---------------------------------------------------------------------------

class LoopDecisionTree(DecisionTreeClassifier):
    """The course's original style: Python loops over rows for gain and predict."""

    def _best_feature(self, X, y):
        n = len(y)
        k = int(y.max()) + 1
        best_f, best_gain = None, 0.0
        for f in range(X.shape[1]):
            yes, no = np.zeros(k), np.zeros(k)
            for r in range(n):                       # count classes row by row
                (yes if X[r, f] == 1 else no)[y[r]] += 1
            if yes.sum() == 0 or no.sum() == 0:
                continue
            total = yes + no
            before = 1.0 - sum((c / n) ** 2 for c in total)
            gl = 1.0 - sum((c / yes.sum()) ** 2 for c in yes)
            gr = 1.0 - sum((c / no.sum()) ** 2 for c in no)
            gain = before - (yes.sum() * gl + no.sum() * gr) / n
            if gain > best_gain:
                best_f, best_gain = f, gain
        return best_f, False

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for x in X:                                  # one row at a time
            node = self.root
            while node.label is None:
                node = node.left if x[node.feature] == 1 else node.right
            out.append(node.label)
        return self.classes_[np.array(out)]


# ---------------------------------------------------------------------------

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


def bench_loop_vs_vectorised(repeats):
    rows = []
    for n in (200, 1000, 5000):
        X, y = make_stream(n)
        Xt, _ = make_stream(20000, seed=1)
        kw = dict(max_depth=5)
        fast, slow = DecisionTreeClassifier(**kw), LoopDecisionTree(**kw)
        t_fit_v = best_time(lambda: fast.fit(X, y), repeats)
        t_fit_l = best_time(lambda: slow.fit(X, y), max(1, repeats // 2))
        t_pred_v = best_time(lambda: fast.predict(Xt), repeats)
        t_pred_l = best_time(lambda: slow.predict(Xt), repeats)
        agree = float(np.mean(fast.predict(Xt) == slow.predict(Xt)))
        rows.append((n, t_fit_l, t_fit_v, t_pred_l, t_pred_v, agree))
    return rows


def bench_buffer_sizes(repeats, n=20000, chunk=50):
    X, y = make_stream(n, drift=True)
    Xtest, ytest = make_stream(n, seed=9, drift=True)
    rows = []
    for buf in (50, 100, 250, 500, 1000, 2500):
        def run():
            t = DecisionTreeClassifier(max_depth=5, buffer_size=buf)
            retrain, correct, seen = 0.0, 0, 0
            for i in range(0, n, chunk):
                Xc, yc = X[i:i + chunk], y[i:i + chunk]
                if t.is_fitted:                 # test-then-train accuracy
                    correct += int((t.predict(Xc) == yc).sum())
                    seen += len(yc)
                t0 = time.perf_counter()
                before = t.n_retrains_
                t.partial_fit(Xc, yc)
                if t.n_retrains_ > before:
                    retrain += time.perf_counter() - t0
            return t, retrain, correct / max(seen, 1)
        best = None
        for _ in range(repeats):
            res = run()
            if best is None or res[1] < best[1]:
                best = res
        t, retrain, acc = best
        rows.append((buf, t.n_retrains_, retrain, retrain / max(t.n_retrains_, 1), acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--markdown", type=str, default=None)
    args = ap.parse_args()

    out = ["# Tree benchmarks", "", "## (a) Loop vs vectorised (max_depth=5, 20 yes/no features)", "",
           "| n_train | fit loop (s) | fit vec (s) | fit speed-up | predict loop (s) "
           "| predict vec (s) | predict speed-up | prediction agreement |",
           "|---|---|---|---|---|---|---|---|"]
    for n, fl, fv, pl, pv, ag in bench_loop_vs_vectorised(args.repeats):
        out.append(f"| {n} | {fl:.4f} | {fv:.4f} | {fl / fv:.1f}x | {pl:.4f} | {pv:.5f} "
                   f"| {pl / pv:.1f}x | {ag:.3f} |")
    out += ["", "## (b) buffer_size sweep (20,000-sample stream, 50-row chunks, "
            "concept flips at the midpoint)", "",
            "| buffer_size | retrains | total retrain time (s) | time / retrain (s) "
            "| prequential accuracy |", "|---|---|---|---|---|"]
    for buf, nr, tot, per, acc in bench_buffer_sizes(args.repeats):
        out.append(f"| {buf} | {nr} | {tot:.4f} | {per:.5f} | {acc:.3f} |")
    text = "\n".join(out)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n")


if __name__ == "__main__":
    main()
