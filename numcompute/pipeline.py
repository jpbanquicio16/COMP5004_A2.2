__all__ = ["Pipeline"]

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
        return self

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
