"""Tests for numcompute.pipeline (Pipeline)."""

import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import (
    MinMaxScaler,
    NotFittedError,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


# small fake classes so we can test the plumbing without a real model

class AddOne:
    # transformer: adds 1 to everything. fit has no y, like the real ones
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) + 1


class Doubler:
    # transformer: doubles everything
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) * 2


class MeanModel:
    # model: learns the mean of y, predicts it for every row
    def fit(self, X, y=None):
        self.X_seen = np.asarray(X)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class NoYModel:
    # model that learns without labels, fit(X) only
    def fit(self, X):
        self.n_rows_ = len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X))


X_TRAIN = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


# normal use

def test_fit_transform_matches_doing_the_steps_by_hand():
    pipe = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    result = pipe.fit_transform(X_TRAIN)

    step1 = StandardScaler().fit_transform(X_TRAIN)
    by_hand = MinMaxScaler().fit_transform(step1)
    assert np.allclose(result, by_hand)
    assert np.allclose(result, [[0, 0], [0.5, 0.5], [1, 1]])


def test_transform_new_data_uses_what_was_learned_in_fit():
    pipe = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    pipe.fit(X_TRAIN)
    # 4 is bigger than anything in training so the answer goes above 1
    result = pipe.transform(np.array([[4.0, 15.0]]))
    assert np.allclose(result, [[1.5, 0.25]])


def test_imputer_then_scaler_removes_nan():
    X = np.array([[1.0, np.nan], [2.0, 20.0], [3.0, 30.0]])
    pipe = Pipeline([("fill", SimpleImputer(strategy="mean")), ("std", StandardScaler())])
    out = pipe.fit_transform(X)
    assert not np.isnan(out).any()
    assert out.shape == (3, 2)


def test_one_hot_encoder_inside_pipeline():
    colours = np.array([["red"], ["green"], ["red"], ["blue"]])
    pipe = Pipeline([("encode", OneHotEncoder())])
    out = pipe.fit_transform(colours)
    assert out.shape == (4, 3)
    assert np.array_equal(out.sum(axis=1), [1, 1, 1, 1])


def test_example_from_the_spec():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("encode", OneHotEncoder()),
    ])
    X_tr = pipe.fit_transform(X_TRAIN)

    # same as doing the two steps by hand
    by_hand = OneHotEncoder().fit_transform(StandardScaler().fit_transform(X_TRAIN))
    assert np.array_equal(X_tr, by_hand)
    assert X_tr.shape == (3, 6)


def test_fit_returns_self():
    pipe = Pipeline([("std", StandardScaler())])
    assert pipe.fit(X_TRAIN) is pipe


def test_predict_goes_through_transformers_then_model():
    y = np.array([0.0, 0.0, 1.0])
    pipe = Pipeline([("std", StandardScaler()), ("model", MeanModel())])
    pipe.fit(X_TRAIN, y)
    predictions = pipe.predict(np.array([[5.0, 50.0], [6.0, 60.0]]))
    assert np.allclose(predictions, [1 / 3, 1 / 3])


def test_model_receives_the_scaled_data_not_the_raw_data():
    model = MeanModel()
    pipe = Pipeline([("std", StandardScaler()), ("model", model)])
    pipe.fit(X_TRAIN, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(model.X_seen, StandardScaler().fit_transform(X_TRAIN))


def test_model_without_y_in_fit_works_when_y_is_not_given():
    pipe = Pipeline([("add", AddOne()), ("model", NoYModel())])
    pipe.fit(X_TRAIN)
    assert pipe.named_steps["model"].n_rows_ == 3


def test_y_is_not_passed_to_transformers():
    # the real transformers have fit(X) only, so passing y to them would crash
    pipe = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    pipe.fit(X_TRAIN, np.array([1, 2, 3]))  # should not raise


def test_fitting_twice_uses_the_latest_data():
    pipe = Pipeline([("mm", MinMaxScaler())])
    pipe.fit(X_TRAIN)
    pipe.fit(X_TRAIN * 10)
    out = pipe.transform(np.array([[30.0, 300.0]]))
    assert np.allclose(out, [[1.0, 1.0]])


def test_input_array_is_not_changed():
    X = X_TRAIN.copy()
    pipe = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    pipe.fit_transform(X)
    pipe.transform(X)
    assert np.array_equal(X, X_TRAIN)


def test_named_steps():
    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler), ("model", MeanModel())])
    assert pipe.named_steps["scale"] is scaler


def test_changing_the_original_list_does_not_change_the_pipeline():
    steps = [("add", AddOne())]
    pipe = Pipeline(steps)
    steps.append(("double", Doubler()))
    assert len(pipe.steps) == 1


def test_pipeline_inside_a_pipeline():
    inner = Pipeline([("add", AddOne()), ("double", Doubler())])
    outer = Pipeline([("inner", inner), ("mm", MinMaxScaler())])
    out = outer.fit_transform(X_TRAIN)
    assert out.shape == X_TRAIN.shape


def test_works_with_classes_that_have_the_methods():
    pipe = Pipeline([("add", AddOne()), ("double", Doubler())])
    out = pipe.fit_transform(np.array([[1.0, 2.0]]))
    assert np.array_equal(out, [[4.0, 6.0]])  # (x + 1) * 2


def test_tuple_of_steps_is_accepted():
    pipe = Pipeline((("add", AddOne()), ("double", Doubler())))
    assert np.array_equal(pipe.fit_transform(np.array([[1.0]])), [[4.0]])


# mistakes

def test_empty_steps_raises():
    with pytest.raises(ValueError, match="empty"):
        Pipeline([])


def test_x_and_y_with_different_lengths_raise():
    pipe = Pipeline([("model", MeanModel())])
    with pytest.raises(ValueError, match="same number of rows"):
        pipe.fit(X_TRAIN, np.array([1.0, 2.0]))


def test_transform_before_fit_raises_not_fitted_error():
    # the scaler itself raises this, the pipeline just passes it on
    pipe = Pipeline([("std", StandardScaler())])
    with pytest.raises(NotFittedError):
        pipe.transform(X_TRAIN)


def test_wrong_number_of_features_at_transform_time_raises():
    pipe = Pipeline([("std", StandardScaler())])
    pipe.fit(X_TRAIN)
    with pytest.raises(ValueError, match="features"):
        pipe.transform(np.ones((2, 5)))


def test_step_without_the_needed_method_raises_attribute_error():
    with pytest.raises(AttributeError):
        Pipeline([("bad", object())]).fit(X_TRAIN)


def test_model_in_the_middle_raises_type_error():
    with pytest.raises(TypeError, match="not the last step"):
        Pipeline([("model", MeanModel()), ("std", StandardScaler())])


def test_class_instead_of_object_raises_type_error():
    # StandardScaler instead of StandardScaler()
    pipe = Pipeline([("scale", StandardScaler)])
    with pytest.raises(TypeError):
        pipe.fit(X_TRAIN)


def test_transform_when_last_step_is_a_model_raises_attribute_error():
    pipe = Pipeline([("std", StandardScaler()), ("model", MeanModel())])
    pipe.fit(X_TRAIN, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(AttributeError):
        pipe.transform(X_TRAIN)


def test_predict_when_last_step_is_a_transformer_raises_attribute_error():
    pipe = Pipeline([("std", StandardScaler())])
    pipe.fit(X_TRAIN)
    with pytest.raises(AttributeError):
        pipe.predict(X_TRAIN)
