### `tree.py`

"""
- `DecisionTreeClassifier`: depth-limited, Gini or entropy-based
- `.fit(X, y)`: trains the tree from scratch on the given data
- `.partial_fit(X_chunk, y_chunk)` supports streaming via **buffered retraining**:
- Each call appends the chunk to an internal data buffer
- Once the buffer holds at least `buffer_size` samples, the tree is retrained on
the buffered data and the buffer is cleared
- Between retrains, `.predict()` uses the most recently trained tree
- Support config: `max_depth`, `min_samples_split`, `max_features`, `buffer_size`
```python
tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
for X_chunk, y_chunk in stream:
tree.parti

X: numeric (n_samples, n_features); y: (n_samples,). predict(X) returns (n_samples,).
 Keep the feature count fixed after the first training chunk.

Permit NaN in X, but reject missing labels and infinities. At each split, 
choose whether missing feature values go left or right during training; 
store that choice in the node and use it for prediction. 
A feature containing only NaNs cannot provide a split.

"""

import numpy as np


def gini(y):
    """How mixed a group is: 0 when everyone agrees, 0.5 at half and half."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - (p ** 2).sum()


def entropy(y):
    """The same idea in bits: 0 when everyone agrees, 1 at half and half."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]                          # log2(0) is undefined
    return -(p * np.log2(p)).sum()

class Node:
    """A question with two children, or a leaf with an answer."""

    def __init__(self, feature=None, left=None, right=None, label=None):
        self.feature = feature            # the question: X[feature] == 1 ?
        self.left = left                  # where the yes rows went
        self.right = right                # where the no rows went
        self.label = label                # set only on a leaf

class DecisionTree:
    def __init__(self, max_depth=2, max_buffer=500, criterion="gini"):
        self.max_depth = max_depth
        self.max_buffer = max_buffer
        self.impurity = gini if criterion == "gini" else entropy
    def _gain(self, y, mask):
        """Information gain: impurity before the question minus after it."""
        before = self.impurity(y)
        after = (mask.sum() * self.impurity(y[mask])
                 + (~mask).sum() * self.impurity(y[~mask])) / len(y)
        return before - after
    def _best_feature(self, X, y):
        """Ask every feature; keep the one whose gain is largest."""
        best_f, best_gain = None, 0.0
        for f in range(X.shape[1]):
            mask = X[:, f] == 1
            if mask.all() or not mask.any():    # everyone on one side
                continue
            g = self._gain(y, mask)
            if g > best_gain:
                best_f, best_gain = f, g
        return best_f
    def _build(self, X, y, depth):
        if (depth >= self.max_depth             # deep enough
                or len(np.unique(y)) == 1):     # everyone agrees
            return Node(label=np.bincount(y).argmax())
        f = self._best_feature(X, y)
        if f is None:                           # no question helps
            return Node(label=np.bincount(y).argmax())
        mask = X[:, f] == 1
        return Node(feature=f,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))
    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), 0)
        return self
    def partial_fit(self, X_chunk, y_chunk): 

        ### Append X Chunk to feature buffer

        ### Append y chunk to label buffer 

        ### if buffer sample count >= buffer size 

            ### fit(buffered_X, buffered_y)

            ### Clear both buffers 
        
        ### return self


    def predict(self, X):
        out = []
        for x in np.asarray(X):
            node = self.root
            while node.label is None:           # still a question
                node = node.left if x[node.feature] == 1 else node.right
            out.append(node.label)
        return np.array(out)

