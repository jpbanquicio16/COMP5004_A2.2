import numpy as np

__all__ = [
    "euclidean_distance",
    "manhattan_distance",
    "k_nearest",
    "sigmoid",
    "relu",
    "softmax",
    "logsumexp",
    "batch_indices",
]

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))


def manhattan_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sum(np.abs(x - y), axis=-1)


def k_nearest(X, queries, k):
    X = np.asarray(X, dtype=float)
    queries = np.asarray(queries, dtype=float)
    if k < 1 or k > len(X): # checking the validity of k
        raise ValueError(f"k must satisfy 1 <= k <= {len(X)}, got {k}")

    dist = euclidean_distance(queries[:, None, :], X[None, :, :])  # (n_queries, n_samples)
    indices = np.argsort(dist, axis=1, kind="stable")[:, :k]
    distances = np.sort(dist, axis=1)[:, :k]
    return indices, distances

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))  # keeps between 0 and 1, so that it never overflows
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z)) # two version of the formula


def relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    shift = x - np.max(x, axis=axis, keepdims=True)  # the biggest value becomes 0
    e = np.exp(shift)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    largest = np.max(a, axis=axis, keepdims=True)
    total = np.sum(np.exp(a - largest), axis=axis, keepdims=True)
    return np.squeeze(np.log(total) + largest, axis=axis) # following the formula

def batch_indices(n_samples, batch_size, shuffle=False, seed=None):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    order = np.arange(n_samples)
    if shuffle:
        order = np.random.default_rng(seed).permutation(n_samples) 

    batches = []
    for start in range(0, n_samples, batch_size):
        batches.append(order[start:start + batch_size])
    return batches
