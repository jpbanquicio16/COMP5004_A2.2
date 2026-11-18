import numpy as np

__all__ = ["grad", "jacobian"]


# Helper functions used for validating inputs and further transforming them


def _check_inputs(func, x, h, method):
    # validate the arguments , return x as a float copy as int would eat up decimals
    if not callable(func):
        raise TypeError("the function must be callable")
    x = np.array(x, dtype=float)  # np.array makes a copy so the callers array is safe
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x must be a non-empty 1D array, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must not contain NaN or inf")
    if h <= 0 or not np.isfinite(h):
        raise ValueError(f"h must be a positive finite number, got {h}")
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got {method!r}")
    return x


def _one_number(value):
    # a function must return exactly one number and as type float
    value = np.asarray(value, dtype=float)
    if value.size != 1:
        raise ValueError(f"f must return a single number, got shape {value.shape}")
    return value.item()


def _as_vector(value):
    # a function must return a 1D array, a single number counts as an array of length 1
    out = np.atleast_1d(np.array(value, dtype=float))
    if out.ndim != 1:
        raise ValueError(f"F must return a 1D array or a single number, got shape {out.shape}")
    return out





def grad(f, x, h=1e-5, method="central"):
    """
    Input:
        f: a function, takes 1D float array shape (n,) and returns one number
        x: an array , the point where we start
        h: step size, positive number, default 1e-5
        method: "central"(default) or "forward" 

    Output:
        array: shape (n,) float, result[i] is the slope of f along input i

    Error Raised:
        TypeError: if function isnt callable
        ValueError: if x not 1D, empty or has NaN/inf, h not positive, bad method,
                    f returns more than one number, or h is too small for x
                    (x + h rounds back to x, happens for very big x)

    Time Complexity: 2n calls of f for central, n + 1 for forward where n = len(x)
    Space: O(n)
    """
    x = _check_inputs(f, x, h, method)
    n = len(x)
    result = np.zeros(n)

    # As forward compares with f(x), we only need to calculate it once
    fx = None
    if method == "forward":
        fx = _one_number(f(x))

    for i in range(n):
        up = x.copy()
        up[i] = x[i] + h
        if method == "central":
            down = x.copy()
            down[i] = x[i] - h
            f_down = _one_number(f(down))
        else:
            down = x
            f_down = fx

        width = up[i] - down[i]  # calculate tghe distance between the two points
        if width == 0:
            raise ValueError(f"h={h} is too small for x[{i}]={x[i]}, x + h rounds back to x. Try a bigger h.")
        result[i] = (_one_number(f(up)) - f_down) / width
    return result


def jacobian(F, x, h=1e-5, method="central"):
    """
    Input:
        F: function, takes 1D float array  and returns 1D array .
        x: an array , the point where we start
        h: step size, positive number, default 1e-5
        method: "central"(default) or "forward"

    Output:
        array: shape (m, n) float. Rows are outputs, columns are inputs.

    Error's Raised:
        TypeError: if F is not callable
        ValueError: same as grad, or F returns more than 1 dimension

    Time: 2n calls of F for central, n + 1 for forward (n = len(x))
    Space: O(n * m)
    """
    x = _check_inputs(F, x, h, method)
    n = len(x)

    fx = None
    if method == "forward":
        fx = _as_vector(F(x))

    columns = []
    for i in range(n):
        up = x.copy()
        up[i] = x[i] + h
        if method == "central":
            down = x.copy()
            down[i] = x[i] - h
            f_down = _as_vector(F(down))
        else:
            down = x
            f_down = fx

        width = up[i] - down[i]  # computing the distance between the two points
        if width == 0:
            raise ValueError(f"h={h} is too small for x[{i}]={x[i]}, x + h rounds back to x. Try a bigger h.")
        columns.append((_as_vector(F(up)) - f_down) / width)

    return np.column_stack(columns)
