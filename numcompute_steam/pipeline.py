__all__ = ["Pipeline"]

"""
## `pipeline.py`
- Must support `partial_fit()` for models and transformers
- Support incremental transformation + prediction in chained pipeline
```python
pipe = Pipeline([
('scale', StandardScaler()),
('model', DecisionTreeClassifier(max_depth=5, buffer_size=500))
])
pipe.partia
"""

class Pipeline:
    """
    Input:
        steps: list of (name, object) pairs

    Attributes:
        steps: list of (name, object) pairs
        named_steps: dict, name -> object

    Error Raised:
        ValueError: if steps is empty
        TypeError: if a step that is not the last one has no transform method
        (any other missing method gives python's own AttributeError)
    """
    def __init__(self, steps):

        # Validating input steps during initialzation
        if len(steps) == 0:
            raise ValueError("steps must not be empty")
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"step {name!r} is not the last step, so it needs a transform method")

        self.steps = list(steps) # Better to create a copy
        self.named_steps = dict(steps)
        self._model_width = None  # columns reaching the last step while streaming

    @property
    def final_estimator(self):
        """The last step (usually the model)."""
        return self.steps[-1][1]

    @property
    def classes_(self):
        """The final model's classes_ (they can change between retrains)."""
        return self.final_estimator.classes_

    def fit(self, X, y=None):

        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")

        data = X
        for name, step in self.steps[:-1]:
            step.fit(data)
            data = step.transform(data)

        # Getting the last step , usually predict
        last = self.steps[-1][1]
        if hasattr(last, "predict") and y is not None: #as we dont have a model yet
            last.fit(data, y)  # a model learns from y(answers)
        else:
            last.fit(data) # just fits if there is no model
        self._model_width = None
        return self

    def partial_fit(self, X_chunk, y_chunk=None):
        """
        Learn from one chunk: every transformer runs partial_fit then
        transform, and the last step gets partial_fit(data, y_chunk).

        Raises:
            TypeError: if any step has no partial_fit (checked before anything
                is updated)
            ValueError: if X and y lengths differ, or the number of columns
                reaching the last step changes between chunks (e.g. a
                OneHotEncoder without max_categories met a new category)
        """
        if y_chunk is not None and len(X_chunk) != len(y_chunk):
            raise ValueError(
                f"X and y must have the same number of rows, got {len(X_chunk)} and {len(y_chunk)}"
            )
        missing = [name for name, step in self.steps if not hasattr(step, "partial_fit")]
        if missing:
            raise TypeError(
                f"step(s) {missing} have no partial_fit method, so this pipeline cannot "
                f"learn from a stream; use fit() or replace those steps"
            )

        data = X_chunk
        for name, step in self.steps[:-1]:
            step.partial_fit(data)
            data = step.transform(data)
        self._check_model_width(data)

        last = self.steps[-1][1]
        if y_chunk is not None:
            last.partial_fit(data, y_chunk)
        else:
            last.partial_fit(data)
        return self

    def _check_model_width(self, data):
        # the tree locks its feature count after the first chunk; catch a
        # transformer that changes width here, where we can name the culprit
        shape = getattr(data, "shape", None)
        if shape is None or len(shape) != 2 or len(self.steps) < 2:
            return
        width = shape[1]
        if self._model_width is None:
            self._model_width = width
        elif width != self._model_width:
            name = self.steps[-2][0]
            raise ValueError(
                f"the data reaching the last step changed from {self._model_width} to "
                f"{width} columns (after step {name!r}); the model's feature count is "
                f"fixed after the first chunk. A OneHotEncoder needs max_categories=k "
                f"so its width is fixed from the start"
            )

    def transform(self, X):
        data = X
        for name, step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        data = X
        for name, step in self.steps[:-1]: #run every step except last , which is usually model training
            data = step.transform(data)
        return self.steps[-1][1].predict(data)

    def predict_proba(self, X):
        """Class probabilities from the final model, columns follow classes_."""
        data = X
        for name, step in self.steps[:-1]:
            data = step.transform(data)
        return self.steps[-1][1].predict_proba(data)
