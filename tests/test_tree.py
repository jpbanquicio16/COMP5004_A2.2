"""Tests for numcompute.tree (DecisionTreeClassifier), incl. streaming.

Features are yes/no (1/0) columns, as in the course tree.
"""

import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier, NotFittedError, entropy, gini

# the course's twelve students: submitted, attended, study_group -> passed
STUDENTS = np.array([
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
])
XS, YS = STUDENTS[:, :3], STUDENTS[:, 3]


def make_data(n=400, seed=0, flip=0.05):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 5))
    y = (X[:, 0] & X[:, 1])
    y = np.where(rng.random(n) < flip, 1 - y, y)
    return X, y


# ---------------- impurity ----------------

def test_gini_pure_and_even():
    assert gini(np.array([1, 1, 1])) == 0.0
    assert gini(np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_entropy_pure_and_even():
    assert entropy(np.array([0, 0])) == 0.0
    assert entropy(np.array([0, 1, 0, 1])) == pytest.approx(1.0)


def test_impurity_empty_is_zero():
    assert gini(np.array([], dtype=int)) == 0.0
    assert entropy(np.array([], dtype=int)) == 0.0


def test_course_numbers_for_twelve_students():
    assert gini(YS) == pytest.approx(1 - ((7 / 12) ** 2 + (5 / 12) ** 2))


# ---------------- fit / predict ----------------

def test_fit_returns_self_and_predict_shape():
    t = DecisionTreeClassifier()
    assert t.fit(XS, YS) is t
    assert t.predict(XS).shape == (12,)


def test_course_tree_root_is_submitted_then_attended():
    t = DecisionTreeClassifier(max_depth=2).fit(XS, YS)
    assert t.root.feature == 0
    assert t.root.left.feature == 1


def test_course_tree_gets_only_the_fluke_wrong():
    t = DecisionTreeClassifier(max_depth=2).fit(XS, YS)
    assert (t.predict(XS) != YS).sum() == 1


def test_entropy_matches_gini_on_course_data():
    a = DecisionTreeClassifier(criterion="gini").fit(XS, YS).predict(XS)
    b = DecisionTreeClassifier(criterion="entropy").fit(XS, YS).predict(XS)
    assert np.array_equal(a, b)


def test_deeper_tree_memorises_training_data():
    assert DecisionTreeClassifier(max_depth=3).fit(XS, YS).score(XS, YS) == 1.0


def test_string_and_multiclass_labels():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 0], [0, 1], [0, 0]])
    y = np.array(["a", "b", "c", "a", "b", "c"])
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert list(t.predict(X)) == list(y)
    assert list(t.classes_) == ["a", "b", "c"]


def test_negative_labels():
    X = np.array([[1], [1], [0], [0]])
    y = np.array([-1, -1, 1, 1])
    assert list(DecisionTreeClassifier().fit(X, y).predict(X)) == [-1, -1, 1, 1]


def test_list_input_accepted():
    t = DecisionTreeClassifier().fit(XS.tolist(), YS.tolist())
    assert t.predict(XS.tolist()).shape == (12,)


# ---------------- hyperparameters ----------------

def test_max_depth_zero_is_single_leaf():
    t = DecisionTreeClassifier(max_depth=0).fit(XS, YS)
    assert t.depth_ == 0 and len(set(t.predict(XS))) == 1


@pytest.mark.parametrize("d", [1, 2, 3])
def test_max_depth_respected(d):
    X, y = make_data()
    assert DecisionTreeClassifier(max_depth=d).fit(X, y).depth_ <= d


def test_min_samples_split_blocks_split():
    assert DecisionTreeClassifier(min_samples_split=13).fit(XS, YS).depth_ == 0


def test_min_samples_split_allows_split():
    assert DecisionTreeClassifier(min_samples_split=12).fit(XS, YS).depth_ >= 1


def test_max_features_reproducible():
    X, y = make_data()
    a = DecisionTreeClassifier(max_features=2, random_state=1).fit(X, y).predict(X)
    b = DecisionTreeClassifier(max_features=2, random_state=1).fit(X, y).predict(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mf", ["sqrt", "log2", 1, 0.5, None])
def test_max_features_options_accepted(mf):
    X, y = make_data()
    assert DecisionTreeClassifier(max_features=mf, random_state=0).fit(X, y).is_fitted


@pytest.mark.parametrize("kwargs", [
    {"max_depth": -1}, {"max_depth": 2.5}, {"min_samples_split": 1},
    {"buffer_size": 0}, {"criterion": "mse"}, {"max_features": 0},
    {"max_features": 1.5}, {"max_features": "all"},
])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


# ---------------- validation ----------------

def test_predict_before_fit():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict(XS)


def test_x_must_be_2d():
    with pytest.raises(ValueError, match="2-D"):
        DecisionTreeClassifier().fit(np.zeros(5), np.zeros(5, dtype=int))


def test_non_numeric_x():
    with pytest.raises(TypeError):
        DecisionTreeClassifier().fit([["a", "b"], ["c", "d"]], [0, 1])


def test_length_mismatch():
    with pytest.raises(ValueError, match="inconsistent"):
        DecisionTreeClassifier().fit(np.zeros((4, 2)), np.zeros(3, dtype=int))


def test_infinite_x_rejected():
    with pytest.raises(ValueError, match="infinite"):
        DecisionTreeClassifier().fit(np.array([[1.0], [np.inf]]), [0, 1])


def test_nan_label_rejected():
    with pytest.raises(ValueError, match="labels"):
        DecisionTreeClassifier().fit(np.zeros((2, 1)), np.array([0.0, np.nan]))


def test_none_label_rejected():
    with pytest.raises(ValueError, match="labels"):
        DecisionTreeClassifier().fit(np.zeros((2, 1)), np.array([0, None], dtype=object))


def test_empty_fit():
    with pytest.raises(ValueError, match="empty"):
        DecisionTreeClassifier().fit(np.zeros((0, 2)), np.zeros(0, dtype=int))


def test_predict_wrong_feature_count():
    t = DecisionTreeClassifier().fit(XS, YS)
    with pytest.raises(ValueError, match="features"):
        t.predict(np.zeros((2, 7)))


# ---------------- NaN / edge cases ----------------

def test_nan_in_training_and_prediction():
    X, y = make_data()
    X = X.astype(float)
    X[::7, 0] = np.nan
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert t.predict(X).shape == (len(X),)


def test_nan_goes_to_side_with_better_gain():
    X = np.array([[1], [1], [1], [0], [0], [0], [np.nan], [np.nan]])
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1])        # NaN rows look like "yes"
    t = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert t.root.nan_left is True
    assert t.predict(np.array([[np.nan]]))[0] == 1


def test_nan_goes_right_when_it_looks_like_no():
    X = np.array([[1], [1], [1], [0], [0], [0], [np.nan], [np.nan]])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    t = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert t.root.nan_left is False
    assert t.predict(np.array([[np.nan]]))[0] == 0


def test_unseen_nan_follows_larger_child():
    X = np.array([[1], [1], [1], [0]])
    y = np.array([1, 1, 1, 0])
    t = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert t.predict(np.array([[np.nan]]))[0] == 1


def test_all_nan_feature_gives_no_split():
    X = np.full((10, 1), np.nan)
    y = np.array([0, 1] * 5)
    assert DecisionTreeClassifier().fit(X, y).depth_ == 0


def test_all_nan_feature_ignored_when_other_is_useful():
    X = np.column_stack([np.full(8, np.nan), [1, 1, 1, 1, 0, 0, 0, 0]])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).root.feature == 1


def test_constant_feature_no_split():
    X = np.ones((10, 2))
    y = np.array([0, 1] * 5)
    assert DecisionTreeClassifier().fit(X, y).depth_ == 0


def test_single_class_data():
    t = DecisionTreeClassifier().fit(XS, np.ones(12, dtype=int))
    assert set(t.predict(XS)) == {1}


def test_identical_rows_conflicting_labels():
    X = np.ones((6, 2))
    y = np.array([0, 0, 0, 1, 1, 0])
    assert DecisionTreeClassifier().fit(X, y).predict(X[:1])[0] == 0


def test_tied_gain_picks_first_feature():
    X = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    y = np.array([1, 1, 0, 0])
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).root.feature == 0


# ---------------- streaming ----------------

def test_partial_fit_buffers_until_full():
    X, y = make_data(300)
    t = DecisionTreeClassifier(buffer_size=200)
    t.partial_fit(X[:100], y[:100])
    assert not t.is_fitted and t.buffer_count_ == 100
    with pytest.raises(NotFittedError):
        t.predict(X)


def test_partial_fit_retrains_and_clears_buffer():
    X, y = make_data(300)
    t = DecisionTreeClassifier(buffer_size=200)
    t.partial_fit(X[:100], y[:100]).partial_fit(X[100:200], y[100:200])
    assert t.is_fitted and t.buffer_count_ == 0 and t.n_retrains_ == 1
    assert t.X_buffer is None and t.y_buffer is None


def test_single_big_chunk_triggers_retrain():
    X, y = make_data(300)
    t = DecisionTreeClassifier(buffer_size=100).partial_fit(X, y)
    assert t.n_retrains_ == 1 and t.buffer_count_ == 0


def test_predict_uses_last_trained_tree_between_retrains():
    X, y = make_data(300)
    t = DecisionTreeClassifier(buffer_size=100)
    t.partial_fit(X[:100], y[:100])
    before = t.predict(X)
    t.partial_fit(X[100:150], 1 - y[100:150])        # buffered only
    assert t.n_retrains_ == 1
    assert np.array_equal(before, t.predict(X))


def test_retrain_uses_only_buffered_data():
    X0 = np.array([[0], [1]] * 20)
    y0 = np.array([0, 1] * 20)
    t = DecisionTreeClassifier(buffer_size=40, max_depth=1).partial_fit(X0, y0)
    assert t.predict(np.array([[1]]))[0] == 1
    t.partial_fit(X0, 1 - y0)                          # flipped labels
    assert t.predict(np.array([[1]]))[0] == 0


def test_stream_reaches_good_accuracy():
    X, y = make_data(1200, seed=3)
    t = DecisionTreeClassifier(max_depth=3, buffer_size=300)
    for i in range(0, 900, 50):
        t.partial_fit(X[i:i + 50], y[i:i + 50])
    assert t.n_retrains_ == 3
    assert t.score(X[900:], y[900:]) > 0.85


def test_feature_count_locked_by_first_chunk():
    t = DecisionTreeClassifier(buffer_size=1000)
    t.partial_fit(np.zeros((5, 3)), np.zeros(5, dtype=int))
    with pytest.raises(ValueError, match="features"):
        t.partial_fit(np.zeros((5, 4)), np.zeros(5, dtype=int))


def test_empty_chunk_is_noop():
    t = DecisionTreeClassifier(buffer_size=10)
    t.partial_fit(np.zeros((0, 2)), np.zeros(0, dtype=int))
    assert t.buffer_count_ == 0 and not t.is_fitted


def test_chunk_validation():
    t = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        t.partial_fit(np.zeros((3, 2)), np.zeros(2, dtype=int))
    with pytest.raises(ValueError):
        t.partial_fit(np.zeros(3), np.zeros(3, dtype=int))


def test_partial_fit_copies_chunk():
    X = np.zeros((4, 1))
    t = DecisionTreeClassifier(buffer_size=100).partial_fit(X, np.zeros(4, dtype=int))
    X[:] = 9
    assert t.X_buffer.max() == 0


def test_chunk_with_nan():
    X, y = make_data(200)
    X = X.astype(float)
    X[::5, 1] = np.nan
    t = DecisionTreeClassifier(buffer_size=100)
    for i in range(0, 200, 50):
        t.partial_fit(X[i:i + 50], y[i:i + 50])
    assert t.predict(X).shape == (200,)


def test_fit_clears_buffer():
    t = DecisionTreeClassifier(buffer_size=1000).partial_fit(XS[:5], YS[:5])
    t.fit(XS, YS)
    assert t.buffer_count_ == 0 and t.is_fitted


def test_new_class_appears_in_later_retrain():
    t = DecisionTreeClassifier(buffer_size=4, max_depth=2)
    t.partial_fit(np.array([[0], [1], [0], [1]]), np.array([0, 1, 0, 1]))
    t.partial_fit(np.array([[0], [1], [0], [1]]), np.array([0, 2, 0, 2]))
    assert list(t.classes_) == [0, 2]


def test_works_inside_pipeline():
    try:
        from numcompute.pipeline import Pipeline
        from numcompute.preprocessing import MinMaxScaler
    except (ImportError, SyntaxError) as exc:
        pytest.skip(f"pipeline/preprocessing not importable: {exc}")
    X, y = make_data()
    pipe = Pipeline([("scale", MinMaxScaler()), ("model", DecisionTreeClassifier(max_depth=3))])
    pipe.fit(X, y)
    assert (pipe.predict(X) == y).mean() > 0.9
