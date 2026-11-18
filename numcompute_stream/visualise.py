"""
numcompute_stream.visualise

Reusable matplotlib plots for streaming results (scripts, notebooks, logs).

Every function takes:
    ax=None         draw into this Axes (a new figure is made when None)
    save_path=None  also save the figure there (PNG, PDF, ... by extension)
    show=False      call plt.show() at the end
and returns the Axes. (Returning the Figure would make Jupyter display the
plot twice: once from the inline backend and once as the cell's value.)

- plot_metric_over_time(values, title, ylabel, markers=...): one metric per
  chunk; markers draws a vertical hairline at every retrain.
- compare_models(metric1, metric2, labels): two runs of the same metric, e.g.
  trees with different buffer_size values.
- plot_predictions_vs_ground_truth(y_true, y_pred): true and predicted labels
  side by side by sample index, mismatches highlighted.
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth",
           "SERIES_COLORS"]

#: categorical colours in fixed order (validated CVD-safe as adjacent pairs)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
_INK = "#0b0b0b"          # titles, values
_INK_2 = "#52514e"        # axis labels, legend text
_MUTED = "#898781"        # tick labels, reference lines
_GRID = "#e1e0d9"         # hairline grid
_AXIS = "#c3c2b7"         # baseline / spines
_SURFACE = "#fcfcfb"
_CRITICAL = "#d03b3b"     # status colour: a wrong prediction


def _axes(ax, figsize=(9, 3.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_2, length=3)
    return ax


def _finish(ax, title, xlabel, ylabel, save_path, show, legend=False, outside=False):
    if title:
        ax.set_title(title, color=_INK, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_2)
    if legend:
        where = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0)) if outside else dict(loc="best")
        ax.legend(frameon=False, labelcolor=_INK_2, fontsize=9, **where)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight",
                          facecolor=ax.figure.get_facecolor())
    if show:
        plt.show()
    return ax


def _x_values(x, n):
    x = np.arange(n) if x is None else np.asarray(x)
    if x.shape != (n,):
        raise ValueError(f"x must have one value per point ({n}), got shape {x.shape}")
    return x


def _marker_positions(markers, x):
    # markers: boolean mask (one per point) or a list of x positions
    if markers is None:
        return np.zeros(0)
    m = np.asarray(markers)
    if m.dtype == bool:
        if m.shape != x.shape:
            raise ValueError("a boolean markers mask needs one value per point")
        return x[m]
    return m.astype(float).ravel()


def _metric(values, name):
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D sequence")
    return v


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="value", *,
                          x=None, xlabel="chunk", markers=None, marker_label="retrain",
                          label=None, color=None, ax=None, save_path=None, show=False):
    """Plot one metric per chunk (NaN, e.g. warm-up, leaves a gap).

    Params:
        metric_values: 1-D, one value per chunk
        x: optional x positions (default 0, 1, 2, ...), e.g. rows seen
        markers: boolean mask (one per chunk) or x positions; a vertical
            hairline is drawn at each, e.g. history_["retrained"]
        label / color: legend entry and line colour for this series

    Returns:
        the matplotlib Axes
    """
    values = _metric(metric_values, "metric_values")
    xs = _x_values(x, values.size)
    ax = _axes(ax)
    for i, pos in enumerate(_marker_positions(markers, xs)):
        ax.axvline(pos, color=_MUTED, linewidth=0.6, alpha=0.5, zorder=1,
                   label=marker_label if i == 0 else None)
    ax.plot(xs, values, color=color or SERIES_COLORS[0], linewidth=2, label=label, zorder=3)
    has_legend = label is not None or (markers is not None and len(_marker_positions(markers, xs)))
    return _finish(ax, title, xlabel, ylabel, save_path, show, legend=bool(has_legend))


def compare_models(metric1, metric2, labels=("model 1", "model 2"), *,
                   title="Model comparison", ylabel="value", x=None, xlabel="chunk",
                   ax=None, save_path=None, show=False):
    """Two runs of the same metric on one axis, each labelled at its end.

    Params:
        metric1, metric2: 1-D, one value per chunk (lengths may differ when
            x is None; with x both must match it)
        labels: names of the two runs

    Returns:
        the matplotlib Axes
    """
    if len(labels) != 2:
        raise ValueError("labels must name the two models")
    ax = _axes(ax)
    for values, label, color in zip((metric1, metric2), labels, SERIES_COLORS[:2]):
        v = _metric(values, "metric")
        xs = _x_values(x, v.size)
        ax.plot(xs, v, color=color, linewidth=2, label=str(label))
        last = np.flatnonzero(~np.isnan(v))
        if last.size:                                  # direct label at the end
            i = last[-1]
            ax.annotate(str(label), (xs[i], v[i]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=_INK_2)
    return _finish(ax, title, xlabel, ylabel, save_path, show, legend=True)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title="Predictions vs ground truth",
                                     index=None, ax=None, save_path=None, show=False):
    """True and predicted label for every sample, lined up by sample index.

    Each class gets a row; the true label is a hollow circle, the prediction
    a filled dot. Where they differ the prediction is drawn in red with a
    thin red connector, and the title reports how many are wrong.

    Params:
        y_true, y_pred: 1-D labels of the same length (numbers or strings)
        index: optional sample numbers for the x axis (default 0, 1, ...)

    Returns:
        the matplotlib Axes
    """
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.ndim != 1 or t.shape != p.shape or t.size == 0:
        raise ValueError("y_true and y_pred must be nonempty 1-D arrays of equal length")
    classes = np.unique(np.concatenate([t, p]))
    ti, pi = np.searchsorted(classes, t), np.searchsorted(classes, p)
    xs = _x_values(index, t.size)
    wrong = ti != pi

    ax = _axes(ax, figsize=(9, 1.2 + 0.45 * classes.size))
    ax.grid(False)
    ax.vlines(xs[wrong], ti[wrong], pi[wrong], color=_CRITICAL, linewidth=1, alpha=0.6, zorder=1)
    ax.scatter(xs, ti, s=42, facecolors="none", edgecolors=_INK_2, linewidths=1,
               label="true", zorder=2)
    ax.scatter(xs[~wrong], pi[~wrong], s=14, color=SERIES_COLORS[0], label="predicted",
               zorder=3)
    ax.scatter(xs[wrong], pi[wrong], s=22, color=_CRITICAL, label="wrong prediction",
               zorder=4)
    ax.set_yticks(np.arange(classes.size))
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_ylim(-0.6, classes.size - 0.4)
    n_wrong = int(wrong.sum())
    full = f"{title}  ({n_wrong} of {t.size} wrong)" if title else None
    return _finish(ax, full, "sample", "label", save_path, show, legend=True, outside=True)
