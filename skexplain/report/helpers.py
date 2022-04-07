import os
import math
import numbers
import numpy as np
from numpy.core.defchararray import isdigit
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


from scipy import spatial

from skexplain.utils import plot


def get_dt_info(dt):
    """Iterates through the given Decision Tree to collect relevant information for the trust report."""
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature
    thresholds = dt.tree_.threshold
    values = dt.tree_.value
    samples = dt.tree_.n_node_samples
    impurity = dt.tree_.impurity

    splits = []
    features_used = {}
    # sum of all samples in all non-leaf nodes
    samples_sum = np.sum(
        [node_sample if children_left[node] != children_right[node] else 0 for node, node_sample in enumerate(samples)]
    )

    def walk_tree(node, level, path):
        """Recursively iterates through all nodes in given decision tree and returns them as a list."""
        if children_left[node] == children_right[node]:  # if leaf node
            node_class = np.argmax(values[node][0])
            return [
                {
                    "level": level,
                    "path": path,
                    "class": node_class,
                    "prob": (values[node][0][node_class] / np.sum(values[node][0])) * 100,
                    "samples": samples[node],
                }
            ]

        feature = features[node]
        threshold = thresholds[node]
        left = children_left[node]
        right = children_right[node]

        if feature not in features_used:
            features_used[feature] = {"count": 0, "samples": 0}

        features_used[feature]["count"] += 1
        features_used[feature]["samples"] += samples[node]

        splits.append(
            {
                "idx": node,
                "level": level,
                "feature": feature,
                "threshold": threshold,
                "samples": samples[node],
                "values": values[node],
                "gini_split": (impurity[left], impurity[right]),
                "data_split": (np.sum(values[left]), np.sum(values[right])),
                "data_split_by_class": [
                    (c_left, c_right) for (c_left, c_right) in zip(values[left][0], values[right][0])
                ],
            }
        )

        return walk_tree(left, level + 1, path + [(feature, "<=", threshold)]) + walk_tree(
            right, level + 1, path + [(feature, ">", threshold)]
        )

    branches = walk_tree(0, 0, [])
    return features_used, splits, branches, samples_sum


def dt_similarity(dt_one, dt_two):
    """Compare decision tree elements and compute the similarity between them"""
    dt_one_matrix = np.array(
        [
            dt_one.tree_.children_left,
            dt_one.tree_.children_right,
            dt_one.tree_.feature,
            dt_one.tree_.threshold,
            [np.argmax(node[0]) for node in dt_one.tree_.value],
        ]
    )

    dt_two_matrix = np.array(
        [
            dt_two.tree_.children_left,
            dt_two.tree_.children_right,
            dt_two.tree_.feature,
            dt_two.tree_.threshold,
            [np.argmax(node[0]) for node in dt_two.tree_.value],
        ]
    )

    diff_tree_size = abs(len(dt_one.tree_.feature) - len(dt_two.tree_.feature))
    if len(dt_one.tree_.feature) > len(dt_two.tree_.feature):
        dt_two_matrix = np.pad(dt_two_matrix, [(0, 0), (0, diff_tree_size)], mode="constant")
    else:
        dt_one_matrix = np.pad(dt_one_matrix, [(0, 0), (0, diff_tree_size)], mode="constant")

    similarity_vector = [1 - spatial.distance.cosine(x, y) for x, y in zip(dt_one_matrix, dt_two_matrix)]
    similarity = np.mean(similarity_vector)

    return similarity, list(similarity_vector)


def plot_top_features(top_features, dt_sum_samples, dt_nodes, output_dir, feature_names=[]):
    """Uses top features information and plots CDF with it"""
    features = [feature_names[feat] if feature_names else str(feat) for (feat, _) in top_features]
    count = [(values["count"] / dt_nodes) * 100 for (_, values) in top_features]
    count_sum = np.cumsum(count)
    data = [(values["samples"] / dt_sum_samples) * 100 for (_, values) in top_features]
    data_sum = np.cumsum(data)

    plot.plot_lines(
        features,
        [count_sum, data_sum],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines.pdf",
    )
    plot.plot_bars(
        features,
        [count_sum, data_sum],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_bars.pdf",
    )

    plot.plot_lines_and_bars(
        features,
        [count_sum, data_sum],
        [count, data],
        y_lim=(0, 100),
        labels=["Nodes", "Samples"],
        path=f"{output_dir}/top_features_lines_bars.pdf",
    )


def plot_top_nodes(top_nodes, dt_samples_by_class, dt_samples, output_dir, feature_names=[], class_names=[]):
    """Uses top features information and plots CDF with it"""
    plot.plot_stacked_bars_split(
        [
            "{} <= {:.2f}".format(
                feature_names[node["feature"]] if feature_names else node["feature"],
                node["threshold"],
            )
            for node in top_nodes
        ],
        [[(node["data_split"][0] / dt_samples) * 100 for node in top_nodes]],
        [[(node["data_split"][1] / dt_samples) * 100 for node in top_nodes]],
        y_placeholder=[100],
        y_lim=(0, 100),
        path=f"{output_dir}/top_nodes.pdf",
    )

    plot.plot_stacked_bars_split(
        [
            "{} <= {:.2f}".format(
                feature_names[node["feature"]] if feature_names else node["feature"],
                node["threshold"],
            )
            for node in top_nodes
        ],
        [
            [(node["data_split_by_class"][idx][0] / dt_samples) * 100 for node in top_nodes]
            for idx in range(len(dt_samples_by_class))
        ],
        [
            [(node["data_split_by_class"][idx][1] / dt_samples) * 100 for node in top_nodes]
            for idx in range(len(dt_samples_by_class))
        ],
        y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
        y_lim=(0, 100),
        labels=class_names,
        path=f"{output_dir}/top_nodes_by_class.pdf",
    )


def plot_top_branches(top_branches, dt_samples_by_class, dt_samples, output_dir, class_names=[]):
    """Uses top features information and plots CDF with it"""
    colors = [
        "#d75d5b",
        "#524a47",
        "#8a4444",
        "#edeef0",
        "#c8c5c3",
        "#f5f0ed",
        "#a7c3cd",
    ]

    samples = []
    colors_by_class = {}
    colors_by_samples = []
    for branch in top_branches:
        class_label = class_names[branch["class"]] if class_names else branch["class"]
        if class_label not in colors_by_class:
            colors_by_class[class_label] = (
                colors.pop() if colors else "#%02x%02x%02x" % tuple(np.random.randint(256, size=3))
            )
        samples.append((branch["samples"] / dt_samples) * 100)
        colors_by_samples.append(colors_by_class[class_label])

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [np.cumsum(samples)],
        y_placeholder=[100],
        y_lim=(0, 100),
        path=f"{output_dir}/top_branches.pdf",
    )

    plot.plot_stacked_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [
            np.cumsum(
                [((branch["samples"] / dt_samples) * 100) if idx == branch["class"] else 0 for branch in top_branches]
            )
            for idx, _ in enumerate(dt_samples_by_class)
        ],
        y_placeholder=[(samples / dt_samples) * 100 for samples in dt_samples_by_class],
        y_lim=(0, 100),
        labels=class_names,
        path=f"{output_dir}/cum_top_branches_by_class.pdf",
    )

    plot.plot_lines_and_bars(
        [f"Top {idx + 1}" for idx in range(len(top_branches))],
        [np.cumsum(samples)],
        [samples],
        y_lim=(0, 100),
        legend={"CDF": "#d75d5b", **colors_by_class},
        colors_by_x=colors_by_samples,
        path=f"{output_dir}/top_branches_by_class.pdf",
    )


def plot_samples_by_level(dt_samples_by_level, dt_nodes_by_level, dt_samples, output_dir):
    """Uses dt information to plot number of samples per level"""
    samples = []
    for idx, level_samples in enumerate(dt_samples_by_level):
        if idx < len(dt_samples_by_level) - 1:
            samples.append(((level_samples - dt_samples_by_level[idx + 1]) / dt_samples) * 100)
        else:
            samples.append((level_samples / dt_samples) * 100)

    plot.plot_lines_and_bars(
        [f"Level {level}" for level, _ in enumerate(dt_samples_by_level)],
        [np.cumsum(samples)],
        [samples],
        y_lim=(0, 100),
        second_x_axis=dt_nodes_by_level,
        labels=["Samples"],
        legend={"CDF": "#d75d5b", "Samples": "#c8c5c3"},
        path=f"{output_dir}/samples_by_level.pdf",
    )


def plot_dts_fidelity_by_size(ccp_iter, max_depth_iter, max_leaves_iter, output_dir):
    """Uses pruning information to plot fidelity vs size of decision trees"""
    num_leaves_ccp = []
    depth_ccp = []
    fidelity_ccp = []

    num_leaves_max_depth = []
    depth_max_depth = []
    fidelity_max_depth = []

    num_leaves_max_leaves = []
    depth_max_leaves = []
    fidelity_max_leaves = []

    for i in ccp_iter:
        num_leaves_ccp.append(i["dt"].get_n_leaves())
        depth_ccp.append(i["dt"].get_depth())
        fidelity_ccp.append(i["fidelity"] * 100)

    for i in max_depth_iter:
        num_leaves_max_depth.append(i["dt"].get_n_leaves())
        depth_max_depth.append(i["dt"].get_depth())
        fidelity_max_depth.append(i["fidelity"] * 100)

    for i in max_leaves_iter:
        num_leaves_max_leaves.append(i["dt"].get_n_leaves())
        depth_max_leaves.append(i["dt"].get_depth())
        fidelity_max_leaves.append(i["fidelity"] * 100)

    plot.plot_lines(
        [num_leaves_ccp, num_leaves_max_depth, num_leaves_max_leaves],
        [fidelity_ccp, fidelity_max_depth, fidelity_max_leaves],
        y_lim=(0, 100),
        labels=["CCP", "Max Depth", "Max Leaves"],
        path=f"{output_dir}/dts_fidelity_x_leaves.pdf",
    )

    plot.plot_lines(
        [depth_ccp, depth_max_depth, depth_max_leaves],
        [fidelity_ccp, fidelity_max_depth, fidelity_max_leaves],
        y_lim=(0, 100),
        labels=["CCP", "Max Depth", "Max Leaves"],
        path=f"{output_dir}/dts_fidelity_x_depth.pdf",
    )


def plot_accuracy_by_feature_removed(whitebox_iter, output_dir, feature_names=[]):
    """Uses iterative analysis information to plot f1-score from the trained blackbox vs number of features removed"""
    blackbox_f1_scores = [i["f1"] * 100 for i in whitebox_iter]
    fidelity = [i["fidelity"] * 100 for i in whitebox_iter]
    features = [feature_names[i["feature_removed"]] if feature_names else i["feature_removed"] for i in whitebox_iter]
    plot.plot_lines(
        features,
        [blackbox_f1_scores, fidelity],
        y_lim=(0, 100),
        labels=["Blackbox F1-Score", "DT Fidelity"],
        path=f"{output_dir}/accuracy_by_feature_removed.pdf",
    )


def plot_distribution(X, y, top_branches, output_dir, feature_names=[], class_names=[]):
    """Plots the distribution of the data based on the top branches"""
    plots_output_dir = f"{output_dir}/dist"
    if not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)

    colors = [
        "#d75d5b",
        "#524a47",
        "#8a4444",
        "#edeef0",
        "#c8c5c3",
        "#f5f0ed",
        "#a7c3cd",
    ]

    df = pd.DataFrame(X, columns=feature_names if feature_names else None)
    if isinstance(df.columns[0], numbers.Number):
        df.columns = [str(i) for i in range(len(df.columns))]

    df["label"] = y
    if class_names and is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(lambda x: class_names[int(x)])

    num_classes = len(np.unique(y))
    split_dfs = [x for _, x in df.groupby("label")]

    for idx, branch in enumerate(top_branches):
        branch_class = class_names[branch["class"]] if class_names else branch["class"]
        branch_output_dir = f"{plots_output_dir}/{idx}_branch_{branch_class}/"

        if not os.path.exists(branch_output_dir):
            os.makedirs(branch_output_dir)

        filtered_dfs = [x.copy(deep=True) for _, x in df.groupby("label")]
        for rule_idx, (feat, op, thresh) in enumerate(branch["path"]):
            column = df.columns[int(feat)] if (isinstance(feat, numbers.Number) or feat.isdigit()) else feat

            plots_per_row = 5
            if num_classes > plots_per_row:
                n_rows = math.gcd(num_classes, plots_per_row)
                n_cols = num_classes if num_classes <= plots_per_row else int(num_classes / n_rows)
            else:
                n_rows = num_classes
                n_cols = 1

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                df_class = split_df["label"].unique()[0]

                ax = axes[df_idx]
                ax.hist(
                    split_df[column].values,
                    bins=50,
                    histtype="bar",
                    label="All" if df_idx == 0 else None,
                    color=colors[0],
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist_all.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                branch_filter = f"filtered_dfs[{df_idx}]['{column}'] {op} {thresh}"
                filtered_dfs[df_idx] = filtered_dfs[df_idx][eval(branch_filter)]
                df_class = split_df["label"].unique()[0]

                ax = axes[df_idx]
                ax.hist(
                    filtered_dfs[df_idx][column].values,
                    bins=50,
                    histtype="bar",
                    label=f"Branch ({branch_class})" if df_idx == 0 else None,
                    color=colors[-1],
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist_branch.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()

            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
            axes = axes.flatten()
            for df_idx, split_df in enumerate(split_dfs):
                df_class = split_df["label"].unique()[0]
                branch_filter = f"filtered_dfs[{df_idx}]['{column}'] {op} {thresh}"
                filtered_dfs[df_idx] = filtered_dfs[df_idx][eval(branch_filter)]

                ax = axes[df_idx]
                ax.hist(
                    split_df[column].values,
                    bins=50,
                    histtype="bar",
                    label="All" if df_idx == 0 else None,
                    color=colors[0],
                    # alpha=0.5,
                )
                ax.hist(
                    filtered_dfs[df_idx][column].values,
                    bins=50,
                    histtype="bar",
                    label=f"Branch ({branch_class})" if df_idx == 0 else None,
                    color=colors[-1],
                    # alpha=0.5,
                )
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=split_df.shape[0]))
                ax.tick_params(axis="both", labelsize=6)
                ax.set_title(df_class, fontsize=8)

            tlt = fig.suptitle(f"{column} {op} {thresh:.3f}")
            lgd = fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
            plt.tight_layout()
            plt.savefig(
                f"{branch_output_dir}/{rule_idx}_{column.replace('/', '_')}_hist.pdf",
                bbox_extra_artists=(lgd, tlt),
                bbox_inches="tight",
            )
            plt.close()
