"""Tests for numcompute.visualise (headless, Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


ACC = np.r_[np.nan, np.nan, 0.8, 0.85, 0.9, 0.6, 0.88]


def test_every_function_returns_axes():
    assert isinstance(plot_metric_over_time(ACC, "t", "acc"), Axes)
    assert isinstance(compare_models(ACC, ACC - 0.1, ["a", "b"]), Axes)
    assert isinstance(plot_predictions_vs_ground_truth([0, 1, 1], [0, 1, 0]), Axes)


def test_draws_into_the_given_axes():
    fig, (a1, a2, a3) = plt.subplots(3)
    assert plot_metric_over_time(ACC, ax=a1) is a1
    assert compare_models(ACC, ACC, ["a", "b"], ax=a2) is a2
    assert plot_predictions_vs_ground_truth([0, 1], [1, 1], ax=a3) is a3
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("fn, args", [
    (plot_metric_over_time, (ACC,)),
    (compare_models, (ACC, ACC, ["a", "b"])),
    (plot_predictions_vs_ground_truth, ([0, 1, 1], [0, 0, 1])),
])
def test_save_path_writes_a_file(tmp_path, fn, args):
    path = tmp_path / "plot.png"
    fn(*args, save_path=path)
    assert path.exists() and path.stat().st_size > 1000


def test_show_calls_plt_show(monkeypatch):
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    plot_metric_over_time(ACC, show=True)
    assert called == [True]


def test_retrain_markers_from_mask_or_positions():
    retrained = np.array([False, False, True, False, False, True, False])
    ax = plot_metric_over_time(ACC, markers=retrained)
    vlines = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1 and len(ln.get_xdata()) == 2]
    assert sorted(ln.get_xdata()[0] for ln in vlines) == [2, 5]
    ax = plot_metric_over_time(ACC, markers=[1, 3, 4])
    assert len(ax.lines) == 4                          # 3 markers + the metric
    assert ax.get_legend() is not None


def test_metric_line_keeps_nan_gaps_and_custom_x():
    ax = plot_metric_over_time(ACC, x=np.arange(7) * 100, xlabel="rows seen")
    line = ax.lines[-1]
    assert np.isnan(line.get_ydata()[:2]).all() and line.get_xdata()[-1] == 600
    assert ax.get_xlabel() == "rows seen"


def test_compare_models_labels_both_runs():
    ax = compare_models([1, 2, 3], [3, 2, 1], ("buffer 100", "buffer 1000"))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["buffer 100", "buffer 1000"]
    assert {t.get_text() for t in ax.texts} == {"buffer 100", "buffer 1000"}


def test_predictions_plot_highlights_mismatches():
    y_true = np.array(["cat", "dog", "dog", "cat", "bird"])
    y_pred = np.array(["cat", "cat", "dog", "dog", "bird"])
    ax = plot_predictions_vs_ground_truth(y_true, y_pred)
    assert "2 of 5 wrong" in ax.get_title(loc="left")
    wrong = [c for c in ax.collections if c.get_label() == "wrong prediction"][0]
    assert len(wrong.get_offsets()) == 2
    np.testing.assert_array_equal(np.asarray(wrong.get_offsets())[:, 0], [1, 3])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["bird", "cat", "dog"]


def test_bad_input():
    with pytest.raises(ValueError):
        plot_metric_over_time([])
    with pytest.raises(ValueError):
        plot_metric_over_time([1, 2], x=[1, 2, 3])
    with pytest.raises(ValueError):
        plot_metric_over_time([1, 2], markers=[True])
    with pytest.raises(ValueError):
        compare_models([1], [2], ["only one"])
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0])
