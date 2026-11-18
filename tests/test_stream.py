"""Tests for streaming end to end: Pipeline.partial_fit and StreamTrainer,
including the streaming edge cases (chunk sizes, labels, data, timing)."""

import logging

import numpy as np
import pytest

from numcompute_stream import metrics as m
from numcompute_stream.io import load_csv
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stream import HISTORY_KEYS, StreamTrainer, model_nbytes
from numcompute_stream.tree import DecisionTreeClassifier, NotFittedError


def make_stream(n=2000, seed=0, drift=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int)
    if drift:
        y[n // 2:] = (X[n // 2:, 1] > 0)
    return X, y


def chunks(X, y, size):
    return [(X[i:i + size], y[i:i + size]) for i in range(0, len(y), size)]


def tree_pipe(buffer_size=200, **kw):
    return Pipeline([("impute", SimpleImputer("mean")), ("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size, **kw))])


# ---------------- Pipeline.partial_fit ----------------

class Recorder:
    """Transformer / model that logs every call it gets."""

    def __init__(self, log, name, add=0):
        self.log, self.name, self.add = log, name, add

    def partial_fit(self, X, y=None):
        self.log.append((self.name, "partial_fit", np.asarray(X).copy(),
                         None if y is None else np.asarray(y).copy()))
        return self

    def transform(self, X):
        self.log.append((self.name, "transform"))
        return np.asarray(X) + self.add

    def predict(self, X):
        return np.zeros(len(X))


def test_pipeline_partial_fit_order_and_data_flow():
    log = []
    pipe = Pipeline([("a", Recorder(log, "a", add=1)), ("b", Recorder(log, "b", add=10)),
                     ("model", Recorder(log, "model"))])
    assert pipe.partial_fit(np.zeros((2, 1)), np.array([0, 1])) is pipe
    assert [e[:2] for e in log] == [("a", "partial_fit"), ("a", "transform"),
                                    ("b", "partial_fit"), ("b", "transform"),
                                    ("model", "partial_fit")]
    np.testing.assert_array_equal(log[2][2], 1.0)          # b learns from a's output
    np.testing.assert_array_equal(log[4][2], 11.0)         # model gets both transforms
    np.testing.assert_array_equal(log[4][3], [0, 1])       # ... and y


def test_pipeline_partial_fit_needs_partial_fit_everywhere():
    class FitOnly:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler), ("old", FitOnly()), ("tree", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="'old'.*partial_fit"):
        pipe.partial_fit(np.zeros((3, 2)), np.zeros(3))
    assert not hasattr(scaler, "n_features_in_")            # nothing was updated


def test_pipeline_streaming_equals_batch_scaling():
    X, y = make_stream(600, drift=False)
    pipe = tree_pipe(buffer_size=10_000)
    for Xc, yc in chunks(X, y, 70):
        pipe.partial_fit(Xc, yc)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, X.mean(axis=0))


def test_pipeline_predict_proba_and_classes():
    X, y = make_stream(400, drift=False)
    pipe = tree_pipe(buffer_size=400).partial_fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (400, 2) and pipe.classes_.tolist() == [0, 1]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_pipeline_feature_count_change_raises():
    pipe = tree_pipe().partial_fit(np.zeros((5, 4)), np.zeros(5))
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit(np.zeros((5, 3)), np.zeros(5))


def test_pipeline_encoder_width_change_raises_with_hint():
    pipe = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore")),
                     ("tree", DecisionTreeClassifier(buffer_size=100))])
    pipe.partial_fit(np.array([["a"], ["b"]]), np.array([0, 1]))
    with pytest.raises(ValueError, match="max_categories"):
        pipe.partial_fit(np.array([["c"]]), np.array([1]))


def test_pipeline_encoder_with_max_categories_streams_fine():
    pipe = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore", max_categories=3)),
                     ("tree", DecisionTreeClassifier(buffer_size=6, max_depth=3))])
    pipe.partial_fit(np.array([["a"], ["b"], ["a"]]), np.array([0, 1, 0]))
    pipe.partial_fit(np.array([["c"], ["c"], ["b"]]), np.array([2, 2, 1]))   # retrain
    assert pipe.predict(np.array([["c"], ["a"]])).tolist() == [2, 0]


# ---------------- StreamTrainer basics ----------------

def test_history_has_every_key_with_one_value_per_chunk():
    X, y = make_stream(1000)
    tr = StreamTrainer(tree_pipe()).run(chunks(X, y, 100))
    h = tr.history_
    assert tuple(h) == HISTORY_KEYS
    assert all(len(v) == 10 for v in h.values())
    assert h["retrained"].dtype == bool and h["seen"][-1] == 1000
    np.testing.assert_array_equal(h["chunk"], np.arange(10))


def test_warm_up_scores_nan_until_first_retrain():
    X, y = make_stream(1000)
    tr = StreamTrainer(tree_pipe(buffer_size=300))
    assert np.isnan(tr.score_chunk(X[:10], y[:10]))         # nothing trained yet
    tr.run(chunks(X, y, 100))
    h = tr.history_
    assert np.isnan(h["accuracy"][:3]).all()                # chunks 0-2 fill the buffer
    assert h["retrained"][2] and np.isfinite(h["accuracy"][3:]).all()


def test_test_then_train_order():
    log = []

    class Model:
        n_retrains_ = 0
        is_fitted = True

        def predict(self, X):
            log.append(("predict", int(X[0, 0])))
            return np.zeros(len(X))

        def partial_fit(self, X, y):
            log.append(("fit", int(X[0, 0])))
            return self

    X = np.repeat(np.arange(3), 2).reshape(-1, 1).astype(float)
    StreamTrainer(Model()).run(chunks(X, np.zeros(6), 2))
    assert log == [("predict", 0), ("fit", 0), ("predict", 1), ("fit", 1),
                   ("predict", 2), ("fit", 2)]


def test_predictions_are_always_on_unseen_rows():
    # a lookup model that only knows rows it trained on scores 0 on new rows
    class Memorise:
        n_retrains_ = 0
        is_fitted = True

        def __init__(self):
            self.seen = {}

        def partial_fit(self, X, y):
            self.seen.update({float(a): int(b) for a, b in zip(X[:, 0], y)})
            return self

        def predict(self, X):
            return np.array([self.seen.get(float(a), -1) for a in X[:, 0]])

    X = np.arange(40, dtype=float).reshape(-1, 1)
    tr = StreamTrainer(Memorise()).run(chunks(X, np.ones(40, dtype=int), 8))
    assert (tr.history_["accuracy"] == 0).all()


def test_retrains_detected_through_n_retrains():
    X, y = make_stream(1000)
    pipe = tree_pipe(buffer_size=250)
    h = StreamTrainer(pipe).run(chunks(X, y, 50)).history_
    assert h["retrained"].sum() == pipe.final_estimator.n_retrains_ == 4
    assert (h["retrain_time"][h["retrained"]] > 0).all()
    assert (h["retrain_time"][~h["retrained"]] == 0).all()
    assert (h["buffer_count"][h["retrained"]] == 0).all()


def test_cumulative_and_rolling_accuracy_match_batch_metrics():
    X, y = make_stream(1200)
    tr = StreamTrainer(tree_pipe(buffer_size=200), rolling_window=150)
    tr.run(chunks(X, y, 100))
    h = tr.history_
    preds, truth = [], []
    model = tree_pipe(buffer_size=200)                       # replay by hand
    for Xc, yc in chunks(X, y, 100):
        if model.final_estimator.is_fitted:
            preds.append(model.predict(Xc))
            truth.append(yc)
        model.partial_fit(Xc, yc)
    p, t = np.concatenate(preds), np.concatenate(truth)
    assert h["cumulative_accuracy"][-1] == pytest.approx(m.accuracy(t, p))
    assert h["rolling_accuracy"][-1] == pytest.approx(m.accuracy(t[-150:], p[-150:]))
    np.testing.assert_array_equal(tr.last_y_pred_, p[-100:])


def test_concept_change_drops_then_recovers():
    X, y = make_stream(4000, seed=1)
    h = StreamTrainer(tree_pipe(buffer_size=500)).run(chunks(X, y, 100)).history_
    acc = h["accuracy"]
    assert np.nanmean(acc[10:20]) > 0.9                     # first concept learnt
    assert np.nanmin(acc[20:25]) < 0.75                     # change at row 2000 hurts
    assert np.nanmean(acc[-10:]) > 0.9                      # retrains recover


def test_memory_is_deterministic_and_follows_the_buffer():
    X, y = make_stream(1000)
    runs = [StreamTrainer(tree_pipe(buffer_size=300)).run(chunks(X, y, 100)).history_
            for _ in range(2)]
    mem = runs[0]["memory_bytes"]
    np.testing.assert_array_equal(mem, runs[1]["memory_bytes"])
    assert mem[1] > mem[0]                                  # buffer grows
    assert mem[2] < mem[1]                                  # retrain empties it


def test_model_nbytes_counts_arrays_once():
    class Holder:
        pass

    h = Holder()
    a = np.zeros(10)
    h.a, h.same, h.items = a, a, [np.zeros(5), {"x": np.zeros(2, dtype=np.int8)}]
    assert model_nbytes(h) == 80 + 40 + 2


def test_to_csv_round_trip(tmp_path):
    X, y = make_stream(600)
    tr = StreamTrainer(tree_pipe()).run(chunks(X, y, 100))
    path = tr.to_csv(tmp_path / "history.csv")
    table = load_csv(path, skip_header=1)
    assert table.shape == (6, len(HISTORY_KEYS))
    assert open(path).readline().strip().split(",") == list(HISTORY_KEYS)
    np.testing.assert_allclose(table[:, HISTORY_KEYS.index("rolling_accuracy")],
                               tr.history_["rolling_accuracy"], equal_nan=True)


def test_logging_reports_chunks_and_retrains(caplog):
    X, y = make_stream(600)
    with caplog.at_level(logging.INFO, logger="numcompute.stream"):
        StreamTrainer(tree_pipe(buffer_size=200), log_every=100).run(chunks(X, y, 100))
    text = caplog.text
    assert "chunk    0" in text and text.count("| retrain") == 3 and "done: 6 chunks" in text


def test_bad_arguments():
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(object())
    with pytest.raises(ValueError):
        StreamTrainer(DecisionTreeClassifier(), log_every=0)


def test_reset_clears_history_not_model():
    X, y = make_stream(400)
    tree = DecisionTreeClassifier(buffer_size=100)
    tr = StreamTrainer(tree).run(chunks(X, y, 100))
    tr.reset()
    assert tr.n_chunks_ == 0 and len(tr.history_["chunk"]) == 0 and tree.is_fitted


# ---------------- edge cases: chunk sizes ----------------

def test_empty_chunk():
    X, y = make_stream(300)
    tr = StreamTrainer(tree_pipe(buffer_size=100))
    tr.run(chunks(X, y, 100) + [(np.zeros((0, 4)), np.zeros(0, dtype=int))])
    h = tr.history_
    assert h["n_samples"][-1] == 0 and np.isnan(h["accuracy"][-1])
    assert not h["retrained"][-1]


def test_chunks_of_one_row():
    X, y = make_stream(120, drift=False)
    h = StreamTrainer(DecisionTreeClassifier(buffer_size=40)).run(chunks(X, y, 1)).history_
    assert len(h["chunk"]) == 120 and h["retrained"].sum() == 3
    assert set(np.unique(h["accuracy"][40:])) <= {0.0, 1.0}


def test_chunk_larger_than_the_buffer():
    X, y = make_stream(1000, drift=False)
    tree = DecisionTreeClassifier(buffer_size=100)
    h = StreamTrainer(tree).run([(X, y)]).history_
    assert h["retrained"].tolist() == [True] and h["buffer_count"][0] == 0
    assert tree.tree_.counts[0].sum() == 1000              # trained on the whole chunk


def test_buffer_size_one():
    X, y = make_stream(30, drift=False)
    tree = DecisionTreeClassifier(buffer_size=1)
    h = StreamTrainer(tree).run(chunks(X, y, 1)).history_
    assert h["retrained"].all() and tree.n_retrains_ == 30
    assert tree.tree_.n_nodes == 1                          # one row -> one leaf


# ---------------- edge cases: labels ----------------

def test_single_class_buffer_then_two_classes():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.r_[np.zeros(20, dtype=int), (np.arange(20) % 2)]
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=20), pos_label=1)
    tr.run(chunks(X, y, 10))
    h = tr.history_
    assert tr.final_estimator_.classes_.tolist() == [0, 1]
    assert np.isfinite(h["accuracy"][2:]).all()            # predicting with a 1-class tree
    assert np.isfinite(h["auc"][-1])


def test_new_class_after_retrain_and_proba_column_by_pos_label():
    # phase 1 sees classes {1, 2}, phase 2 sees {0, 1}: the column of
    # pos_label=1 moves from 0 to 1. A fixed column would give AUC < 1.
    y = np.r_[np.tile([1, 2], 20), np.tile([0, 1], 20)]
    X = (y == 1).astype(float).reshape(-1, 1)
    tr = StreamTrainer(DecisionTreeClassifier(buffer_size=20, max_depth=1), pos_label=1)
    tr.run(chunks(X, y, 10))
    assert tr.final_estimator_.classes_.tolist() == [0, 1]
    assert tr.history_["auc"][-1] == 1.0
    assert np.nanmin(tr.history_["accuracy"]) < 1.0         # class 0 unseen at first


# ---------------- edge cases: data ----------------

def test_all_nan_column_streams_through_imputer_scaler_tree():
    X, y = make_stream(600, drift=False)
    X[:250, 2] = np.nan                                     # all NaN for 2.5 chunks
    h = StreamTrainer(tree_pipe(buffer_size=200)).run(chunks(X, y, 100)).history_
    assert np.nanmean(h["accuracy"]) > 0.9


def test_zero_variance_chunk():
    X, y = make_stream(400, drift=False)
    X[:100, 1] = 7.0                                        # constant in chunk 0
    pipe = tree_pipe(buffer_size=200)
    StreamTrainer(pipe).run(chunks(X, y, 100))
    scaler = pipe.named_steps["scale"]
    assert np.isfinite(scaler.scale_).all() and scaler.scale_[1] > 0


# ---------------- edge cases: timing and setup ----------------

def test_predicting_during_warm_up_raises_on_the_model_but_not_the_trainer():
    X, y = make_stream(100)
    tree = DecisionTreeClassifier(buffer_size=500).partial_fit(X, y)
    with pytest.raises(NotFittedError):
        tree.predict(X)
    assert np.isnan(StreamTrainer(tree).score_chunk(X, y))


def test_rolling_window_evicts_old_chunks():
    X, y = make_stream(3000, seed=2)
    h = StreamTrainer(tree_pipe(buffer_size=300), rolling_window=200).run(
        chunks(X, y, 100)).history_
    # right after the concept change the rolling accuracy forgets the good
    # past quickly, while the cumulative one barely moves
    i = 16
    assert h["rolling_accuracy"][i] < h["cumulative_accuracy"][i] - 0.15


def test_feature_count_change_mid_stream_raises():
    X, y = make_stream(400)
    tr = StreamTrainer(tree_pipe(buffer_size=100)).run(chunks(X, y, 100))
    with pytest.raises(ValueError, match="features"):
        tr.step(np.zeros((10, 5)), np.zeros(10, dtype=int))
