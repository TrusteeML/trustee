"""Tests for the low-level plotting helpers.

Every helper writes a file and must leave no matplotlib figure open. The
figure-count assertions guard a leak where a stray plt.figure() call was
superseded by plt.subplots(), so plt.close() only closed the second one.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trustee.utils import plot


@pytest.fixture(autouse=True)
def no_leaked_figures():
    """Fail any test whose helper leaves a figure behind."""
    plt.close("all")
    yield
    open_figures = plt.get_fignums()
    plt.close("all")
    assert open_figures == [], f"helper leaked {len(open_figures)} matplotlib figure(s)"


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "figure.pdf")


class TestPlotHeatmap:
    def test_writes_a_file(self, out):
        plot.plot_heatmap(np.array([[1.0, 0.5], [0.5, 1.0]]), labels=["a", "b"], path=out)
        assert os.path.getsize(out) > 0

    def test_works_without_labels(self, out):
        plot.plot_heatmap(np.array([[1.0, 0.0], [0.0, 1.0]]), path=out)

    def test_handles_a_single_cell(self, out):
        plot.plot_heatmap(np.array([[1.0]]), labels=["only"], path=out)


class TestPlotLines:
    def test_writes_a_file(self, out):
        plot.plot_lines([1, 2, 3], [[1, 4, 9]], labels=["squares"], path=out)
        assert os.path.getsize(out) > 0

    def test_plots_several_series(self, out):
        plot.plot_lines([1, 2, 3], [[1, 2, 3], [3, 2, 1]], labels=["up", "down"], path=out)

    def test_accepts_axis_limits_and_titles(self, out):
        plot.plot_lines(
            [1, 2, 3],
            [[1, 2, 3]],
            xlim=(0, 4),
            ylim=(0, 5),
            title="t",
            xlabel="x",
            ylabel="y",
            size=(4, 2),
            path=out,
        )


class TestPlotBars:
    def test_writes_a_file(self, out):
        plot.plot_bars(["a", "b", "c"], [[1, 2, 3]], labels=["series"], path=out)
        assert os.path.getsize(out) > 0

    def test_plots_grouped_series(self, out):
        plot.plot_bars(["a", "b"], [[1, 2], [3, 4]], labels=["one", "two"], path=out)

    def test_accepts_limits_and_labels(self, out):
        plot.plot_bars(["a"], [[1]], ylim=(0, 2), xlabel="x", ylabel="y", title="t", path=out)


class TestPlotStackedBars:
    def test_writes_a_file(self, out):
        plot.plot_stacked_bars(["a", "b"], [[1, 2], [3, 4]], labels=["lo", "hi"], path=out)
        assert os.path.getsize(out) > 0

    def test_accepts_a_placeholder_series(self, out):
        plot.plot_stacked_bars(["a", "b"], [[10, 20]], y_placeholder=[100], ylim=(0, 100), path=out)

    def test_accepts_a_range_for_x(self, out):
        plot.plot_stacked_bars(range(3), [[1, 2, 3]], path=out)


class TestPlotStackedBarsSplit:
    def test_writes_a_file(self, out):
        plot.plot_stacked_bars_split(["a", "b"], [[1, 2]], [[3, 4]], labels=["l", "r"], path=out)
        assert os.path.getsize(out) > 0

    def test_accepts_a_placeholder_and_limits(self, out):
        plot.plot_stacked_bars_split(
            ["a", "b"],
            [[10, 20]],
            [[30, 40]],
            y_placeholder=[100],
            ylim=(0, 100),
            xlabel="x",
            ylabel="y",
            title="t",
            path=out,
        )


class TestPlotLinesAndBars:
    def test_writes_a_file(self, out):
        plot.plot_lines_and_bars(
            ["a", "b"],
            [[1, 2]],
            [[3, 4]],
            labels=["line", "bar"],
            ylim=(0, 5),
            path=out,
        )
        assert os.path.getsize(out) > 0

    def test_accepts_a_second_x_axis(self, out):
        plot.plot_lines_and_bars(
            ["a", "b"],
            [[1, 2]],
            [[3, 4]],
            second_x_axis=[10, 20],
            second_x_axis_label="other",
            labels=["line", "bar"],
            path=out,
        )

    def test_builds_a_patch_legend_from_label_to_colour(self, out):
        plot.plot_lines_and_bars(
            ["a", "b"],
            [[1, 2]],
            [[3, 4]],
            legend={"CDF": "#d75d5b", "Samples": "#c8c5c3"},
            labels=["line", "bar"],
            path=out,
        )

    def test_accepts_per_x_colors(self, out):
        plot.plot_lines_and_bars(
            ["a", "b"],
            [[1, 2]],
            [[3, 4]],
            colors_by_x=["#d75d5b", "#a7c3cd"],
            labels=["line", "bar"],
            path=out,
        )
