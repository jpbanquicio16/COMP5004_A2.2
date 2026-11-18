"""The loop-based benchmark tree must build exactly the same tree as the
vectorised one (that is what makes the speed comparison fair)."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from tree_benchmarks import (  # noqa: E402
    LoopDecisionTree,
    confusion_per_sample,
    welford_per_sample,
)

from numcompute.metrics import confusion_matrix  # noqa: E402
from numcompute.stats import StreamingStats  # noqa: E402
from numcompute.tree import DecisionTreeClassifier  # noqa: E402


def datasets():
    rng = np.random.default_rng(11)
    binary = rng.integers(0, 2, size=(300, 6))
    yield "binary", binary, binary[:, 0] & binary[:, 1]
    cont = rng.normal(size=(250, 5))
    yield "continuous", cont, (cont[:, 0] + cont[:, 1] > 0).astype(int) + (cont[:, 2] > 1)
    nan = cont.copy()
    nan[rng.random(nan.shape) < 0.1] = np.nan
    yield "continuous+NaN", nan, (np.nan_to_num(nan[:, 0]) > 0).astype(int)
    ties = rng.integers(0, 4, size=(200, 3)).astype(float)
    yield "many ties", ties, rng.integers(0, 3, 200)


@pytest.mark.parametrize("name, X, y", list(datasets()), ids=lambda v: v if isinstance(v, str) else "")
def test_loop_tree_builds_identical_tree(name, X, y):
    fast = DecisionTreeClassifier(max_depth=5).fit(X, y)
    slow = LoopDecisionTree(max_depth=5).fit(X, y)
    for a, b in zip(fast.tree_, slow.tree_):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))


def test_loop_tree_is_gini_only():
    with pytest.raises(ValueError):
        LoopDecisionTree(criterion="entropy")


def test_small_loop_baselines_agree():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(500, 3))
    mean, var = welford_per_sample(X)
    st = StreamingStats()
    for c in np.array_split(X, 9):
        st.update_stats(c)
    np.testing.assert_allclose(mean, st.mean)
    np.testing.assert_allclose(var, st.variance())
    t, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    np.testing.assert_array_equal(confusion_per_sample(t, p, 4), confusion_matrix(t, p))
