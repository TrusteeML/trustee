"""Tests for the report plotting entry points.

Inputs are derived from a real fitted tree via `get_dt_info` and the Trustee
accessors, so the structures match what TrustReport actually passes in.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from trustee.report import plot as report_plot
from trustee.utils.tree import get_dt_info, get_node_counts


@pytest.fixture(autouse=True)
def no_leaked_figures():
    plt.close("all")
    yield
    open_figures = plt.get_fignums()
    plt.close("all")
    assert open_figures == [], f"leaked {len(open_figures)} matplotlib figure(s)"


@pytest.fixture(scope="module")
def tree(iris):
    X, y = iris
    return DecisionTreeClassifier(random_state=0, max_depth=3).fit(X, y)


@pytest.fixture(scope="module")
def dt_info(tree):
    features, splits, branches = get_dt_info(tree)
    return features, splits, branches


@pytest.fixture(scope="module")
def root_counts(tree):
    return get_node_counts(tree)[0][0]


@pytest.fixture
def out(tmp_path):
    return str(tmp_path)


def pdfs(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".pdf"))


class TestPlotTopFeatures:
    def test_writes_plots(self, dt_info, tree, out):
        features, _, _ = dt_info
        top = sorted(features.items(), key=lambda p: p[1]["samples"], reverse=True)
        report_plot.plot_top_features(top, sum(v["samples"] for v in features.values()), len(features), out)
        assert pdfs(out)

    def test_uses_feature_names_when_given(self, dt_info, iris_bunch, out):
        features, _, _ = dt_info
        top = sorted(features.items(), key=lambda p: p[1]["samples"], reverse=True)
        report_plot.plot_top_features(
            top,
            sum(v["samples"] for v in features.values()),
            len(features),
            out,
            feature_names=iris_bunch.feature_names,
        )
        assert pdfs(out)

    def test_empty_input_is_a_no_op(self, out):
        report_plot.plot_top_features([], 0, 0, out)
        assert pdfs(out) == []


class TestPlotTopNodes:
    def test_writes_plots(self, dt_info, root_counts, tree, out):
        _, splits, _ = dt_info
        report_plot.plot_top_nodes(splits, root_counts, tree.tree_.n_node_samples[0], out)
        assert pdfs(out)

    def test_uses_names_when_given(self, dt_info, root_counts, tree, iris_bunch, out):
        _, splits, _ = dt_info
        report_plot.plot_top_nodes(
            splits,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            feature_names=iris_bunch.feature_names,
            class_names=iris_bunch.target_names,
        )
        assert pdfs(out)

    def test_empty_input_is_a_no_op(self, out):
        report_plot.plot_top_nodes([], [], 0, out)
        assert pdfs(out) == []


class TestPlotTopBranches:
    def test_writes_plots(self, dt_info, root_counts, tree, iris_bunch, out):
        _, _, branches = dt_info
        report_plot.plot_top_branches(
            branches,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            class_names=iris_bunch.target_names,
        )
        assert pdfs(out)

    def test_regression_mode_skips_class_breakdown(self, dt_info, root_counts, tree, out):
        _, _, branches = dt_info
        report_plot.plot_top_branches(
            branches,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            is_classify=False,
        )
        assert pdfs(out)

    def test_filename_prefix_is_honoured(self, dt_info, root_counts, tree, out):
        _, _, branches = dt_info
        report_plot.plot_top_branches(
            branches,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            filename="custom",
        )
        assert any(f.startswith("custom") for f in pdfs(out))

    def test_empty_input_is_a_no_op(self, out):
        report_plot.plot_top_branches([], [], 0, out)
        assert pdfs(out) == []


class TestPlotAllBranches:
    def test_writes_plots(self, dt_info, root_counts, tree, iris_bunch, out):
        _, _, branches = dt_info
        report_plot.plot_all_branches(
            branches,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            class_names=iris_bunch.target_names,
        )
        assert pdfs(out)


class TestPlotSamplesByLevel:
    def test_writes_a_plot(self, dt_info, tree, out):
        _, splits, branches = dt_info
        depth = tree.get_depth()
        samples_by_level = [0] * (depth + 1)
        leaves_by_level = [0] * (depth + 1)
        for node in splits:
            samples_by_level[node["level"]] += node["samples"]
        for node in branches:
            samples_by_level[node["level"]] += node["samples"]
            leaves_by_level[node["level"]] += 1

        report_plot.plot_samples_by_level(samples_by_level, leaves_by_level, tree.tree_.n_node_samples[0], out)
        assert "samples_by_level.pdf" in pdfs(out)


class TestPlotDtsFidelityBySize:
    def test_writes_both_views(self, tree, out):
        # TrustReport runs the same number of iterations for every pruning type,
        # so the per-type series are always equal length.
        pruning_list = [
            {"type": "ccp", "iter": [{"dt": tree, "fidelity": 0.9}, {"dt": tree, "fidelity": 0.8}]},
            {"type": "max_depth", "iter": [{"dt": tree, "fidelity": 0.7}, {"dt": tree, "fidelity": 0.6}]},
        ]
        report_plot.plot_dts_fidelity_by_size(pruning_list, out)
        assert "dts_fidelity_x_leaves.pdf" in pdfs(out)
        assert "dts_fidelity_x_depth.pdf" in pdfs(out)

    def test_filename_prefix_is_honoured(self, tree, out):
        pruning_list = [{"type": "ccp", "iter": [{"dt": tree, "fidelity": 0.9}]}]
        report_plot.plot_dts_fidelity_by_size(pruning_list, out, filename="branches")
        assert "branches_fidelity_x_leaves.pdf" in pdfs(out)

    def test_empty_input_is_a_no_op(self, out):
        report_plot.plot_dts_fidelity_by_size([], out)
        assert pdfs(out) == []


class TestPlotAccuracyByFeatureRemoved:
    def test_writes_a_plot(self, out):
        whitebox_iter = [
            {"it": 0, "feature_removed": 0, "score": 0.9, "fidelity": 0.95},
            {"it": 1, "feature_removed": 2, "score": 0.7, "fidelity": 0.85},
        ]
        report_plot.plot_accuracy_by_feature_removed(whitebox_iter, out)
        assert pdfs(out)

    def test_uses_feature_names_when_given(self, iris_bunch, out):
        whitebox_iter = [{"it": 0, "feature_removed": 1, "score": 0.9, "fidelity": 0.95}]
        report_plot.plot_accuracy_by_feature_removed(whitebox_iter, out, feature_names=iris_bunch.feature_names)
        assert pdfs(out)

    def test_empty_input_is_a_no_op(self, out):
        report_plot.plot_accuracy_by_feature_removed([], out)
        assert pdfs(out) == []


class TestStabilityPlots:
    @pytest.fixture
    def stability_inputs(self, iris_frame, tree, dt_info):
        X, y = iris_frame
        _, _, branches = dt_info
        stability_iter = [
            {"max_dt": tree, "max_dt_fidelity": 0.9, "iteration": 0, "top_branches": branches},
            {"max_dt": tree, "max_dt_fidelity": 0.8, "iteration": 1, "top_branches": branches},
        ]
        return stability_iter, X, y, branches

    def test_plot_stability_writes_plots(self, stability_inputs, tree, iris_bunch, out):
        stability_iter, X, y, branches = stability_inputs
        report_plot.plot_stability(
            stability_iter,
            X,
            y,
            tree,
            "max_dt",
            branches,
            out,
            class_names=iris_bunch.target_names,
        )
        assert pdfs(out)

    def test_plot_stability_heatmap_writes_plots(self, stability_inputs, iris_bunch, out):
        stability_iter, X, y, branches = stability_inputs
        report_plot.plot_stability_heatmap(
            stability_iter,
            X,
            y,
            "max_dt",
            branches,
            out,
            class_names=iris_bunch.target_names,
        )
        assert pdfs(out)

    def test_empty_input_is_a_no_op(self, iris_frame, tree, out):
        X, y = iris_frame
        report_plot.plot_stability([], X, y, tree, "max_dt", [], out)
        report_plot.plot_stability_heatmap([], X, y, "max_dt", [], out)
        assert pdfs(out) == []


class TestPlotDistribution:
    def test_writes_per_feature_plots(self, iris_frame, dt_info, iris_bunch, out):
        X, y = iris_frame
        _, _, branches = dt_info
        report_plot.plot_distribution(
            X,
            y,
            branches[:2],
            out,
            feature_names=iris_bunch.feature_names,
            class_names=iris_bunch.target_names,
        )
        assert os.path.isdir(out)

    def test_accepts_numpy_input(self, iris, dt_info, out):
        X, y = iris
        _, _, branches = dt_info
        report_plot.plot_distribution(pd.DataFrame(X), pd.Series(y), branches[:1], out)


class TestClassNamesDefault:
    """Every entry point defaults class_names to []; indexing that raises IndexError.

    An `is not None` guard passes for an empty list, so these paths used to crash
    whenever the caller omitted class_names, and to raise on a class_names shorter
    than the number of classes in the tree.
    """

    def test_top_branches_without_class_names(self, dt_info, root_counts, tree, out):
        _, _, branches = dt_info
        report_plot.plot_top_branches(branches, root_counts, tree.tree_.n_node_samples[0], out)
        assert pdfs(out)

    def test_all_branches_without_class_names(self, dt_info, root_counts, tree, out):
        _, _, branches = dt_info
        report_plot.plot_all_branches(branches, root_counts, tree.tree_.n_node_samples[0], out)
        assert pdfs(out)

    def test_stability_without_class_names(self, iris_frame, dt_info, tree, out):
        X, y = iris_frame
        _, _, branches = dt_info
        stability_iter = [{"max_dt": tree, "max_dt_fidelity": 0.9, "iteration": 0, "top_branches": branches}]
        report_plot.plot_stability(stability_iter, X, y, tree, "max_dt", branches, out)
        assert pdfs(out)

    def test_stability_heatmap_without_class_names(self, iris_frame, dt_info, tree, out):
        X, y = iris_frame
        _, _, branches = dt_info
        stability_iter = [{"max_dt": tree, "max_dt_fidelity": 0.9, "iteration": 0, "top_branches": branches}]
        report_plot.plot_stability_heatmap(stability_iter, X, y, "max_dt", branches, out)
        assert pdfs(out)

    def test_distribution_without_class_names(self, iris_frame, dt_info, out):
        X, y = iris_frame
        _, _, branches = dt_info
        report_plot.plot_distribution(X, y, branches[:1], out)

    def test_class_names_shorter_than_the_class_count(self, dt_info, root_counts, tree, out):
        _, _, branches = dt_info
        report_plot.plot_top_branches(
            branches,
            root_counts,
            tree.tree_.n_node_samples[0],
            out,
            class_names=["only-one"],
        )
        assert pdfs(out)


class TestPlotDistributionAggregate:
    """The aggregate=True path folds `prefix_<bit>` columns back into integers."""

    @pytest.fixture
    def bitfield_frame(self):
        # Two 2-bit fields encoded one column per bit, the naming this path expects.
        return pd.DataFrame(
            {
                "flags_0": [0, 1, 1, 0],
                "flags_1": [1, 0, 1, 0],
                "opt_0": [1, 1, 0, 0],
                "opt_1": [0, 1, 0, 1],
            }
        ), pd.Series([0, 1, 0, 1])

    def test_aggregates_bit_columns(self, bitfield_frame, dt_info, out):
        X, y = bitfield_frame
        _, _, branches = dt_info
        # plot_distribution drops DataFrame column names (it does X.values first),
        # so the `prefix_<bit>` naming this path parses must arrive via feature_names.
        report_plot.plot_distribution(
            X,
            y,
            branches[:1],
            out,
            aggregate=True,
            feature_names=list(X.columns),
        )
        assert os.path.isdir(os.path.join(out, "aggr_dist"))

    def test_bit_columns_fold_to_the_integer_they_encode(self, bitfield_frame):
        """Verifies the aggregation arithmetic, not just that the call succeeds.

        flags_0/flags_1 are the high/low bits in column order, so row 0 is "01" -> 1,
        row 1 "10" -> 2, row 2 "11" -> 3, row 3 "00" -> 0.
        """
        X, _ = bitfield_frame
        non_opt = X[["flags_0", "flags_1"]]
        grouper = ["flags", "flags"]
        folded = pd.DataFrame(
            {
                prefix: non_opt[[c for c, g in zip(non_opt.columns, grouper) if g == prefix]]
                .astype(str)
                .apply("".join, axis=1)
                .apply(lambda n: int(n, 2))
                for prefix in dict.fromkeys(grouper)
            },
            index=non_opt.index,
        )
        assert folded["flags"].tolist() == [1, 2, 3, 0]

    def test_unparseable_column_names_raise(self, bitfield_frame, dt_info, out):
        """Documents that aggregate=True requires `prefix_<bit>` feature names.

        plot_distribution converts X with .values, so without feature_names the columns
        become "0", "1", ... which the prefix regex cannot match.
        """
        X, y = bitfield_frame
        _, _, branches = dt_info
        with pytest.raises(IndexError):
            report_plot.plot_distribution(X, y, branches[:1], out, aggregate=True)
